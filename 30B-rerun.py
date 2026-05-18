#!/usr/bin/env python3
"""
Run a GRASP / LLM pipeline over all CSV datasets in ComplexQA/ and SimpleQA/.

This version is for 30B outputs.

Improved behavior:
- Finds every .csv file inside data/SimpleQA/ and data/ComplexQA/.
- Saves outputs under:
      30B/<csv_name>/
- Scans all CSV/output directories at startup before processing anything.
- Detects rows that need re-run because:
    - output JSON is missing
    - output JSON is empty / whitespace-only
    - output JSON is invalid
    - output JSON is {}, [], or null
    - output JSON has output: null, output: "", output: [], or output: {}
    - output JSON matches the known GRASP null-output pattern
    - a matching .stderr.txt file exists, for example 00000.json.stderr.txt
- Asks re-run decisions once at startup across all input directories.
- After startup choices, runs to the end without asking again.
- Before re-running a bad row, removes old JSON and old .stderr.txt.
- After a successful re-run, removes stale .stderr.txt.
- If GRASP fails again, writes a new .stderr.txt.
- Writes structured JSON error payloads for empty stdout / invalid stdout.
- Keeps compatibility with your old batch log format:
      batch 0 (#1) | rows 0-49 | files: ...
- Also writes row-level DONE/SKIP lines for safer future resume:
      DONE row 0 | file: 00000.json
      SKIP row 1 | reason: empty_or_missing_question
- Uses timestamped live progress logs instead of tqdm.
- Makes overall xxx/xxx progress very obvious.

Directory assumption:
Run this script from the parent directory that contains:
    data/SimpleQA/
    data/ComplexQA/
"""

from __future__ import annotations

import csv
import glob
import json
import math
import os
import re
import shlex
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Sequence, Set, Tuple


# ---------------- Configuration ---------------- #

INPUT_DIRS: List[str] = [
   # "data/SimpleQA",
    "data/ComplexQA",
]

OUTPUT_ROOT = "30B"

BATCH_SIZE = 50

GRASP_COMMAND = ["bash", "-lc", "grasp run configs/run.yaml"]

# If your shell does not load the correct conda env, use something like:
# GRASP_COMMAND = [
#     "bash",
#     "-lc",
#     "source ~/miniconda3/etc/profile.d/conda.sh && conda activate grasp_v1 && grasp run configs/run.yaml",
# ]

# Set True only if you really want git add/commit/push after each batch.
ENABLE_GIT = True
GIT_REMOTE = "origin"
GIT_BRANCH = "main"

# Startup-only prompts. After these are answered, the script is non-interactive.
ASK_AT_STARTUP = True

# Re-run a row whenever its output has a matching .stderr.txt sidecar.
RERUN_WHEN_STDERR_EXISTS = True

# Treat {}, [], null as bad whole-file JSON outputs.
RERUN_EMPTY_JSON_STRUCTURES = True

# Treat an existing output field with null / empty value as bad.
RERUN_EMPTY_OUTPUT_FIELD = True

# Write stderr sidecar when GRASP exits non-zero.
WRITE_STDERR_ON_NONZERO = True

# Optional previews saved in structured error payloads.
STDOUT_PREVIEW_CHARS = 4000
STDERR_PREVIEW_CHARS = 4000

# Progress-log verbosity.
LOG_SKIPPED_ROWS = False
LOG_EVERY_N_ROWS = 1

# Make this True if you want command text printed before every row.
LOG_COMMAND_EACH_ROW = False


# ---------------- Data classes ---------------- #

@dataclass
class CsvPlan:
    csv_path: str
    file_stem: str
    out_dir: str
    batch_log_path: str
    header: List[str]
    colmap: Dict[str, Optional[int]]
    rows: List[List[str]]
    total_rows: int
    logged_indices: Set[int] = field(default_factory=set)
    existing_good_indices: Set[int] = field(default_factory=set)
    bad_indices: Set[int] = field(default_factory=set)
    processed_indices: Set[int] = field(default_factory=set)
    is_completed_from_log: bool = False
    bad_csv_path: str = ""


@dataclass
class OverallProgress:
    total_rows: int
    completed_rows: int = 0
    newly_run_rows: int = 0
    skipped_existing_rows: int = 0
    skipped_missing_question_rows: int = 0
    failed_rows: int = 0
    bad_stdout_rows: int = 0
    start_time: float = field(default_factory=time.perf_counter)
    elapsed_seconds: List[float] = field(default_factory=list)


# ---------------- Logging helpers ---------------- #

def now_str() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def log(level: str, message: str) -> None:
    print(f"[{now_str()}] [{level}] {message}", flush=True)


def log_step(message: str) -> None:
    log("STEP", message)


def log_info(message: str) -> None:
    log("INFO", message)


def log_warn(message: str) -> None:
    log("WARN", message)


def log_error(message: str) -> None:
    log("ERROR", message)


def format_duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes = seconds / 60
    if minutes < 60:
        return f"{minutes:.1f}m"
    hours = minutes / 60
    return f"{hours:.1f}h"


def format_progress(done: int, total: int, start_time: float, completed_times: List[float]) -> str:
    if total <= 0:
        return "0/0 (0.00%)"

    done = min(max(done, 0), total)
    pct = (done / total) * 100.0
    elapsed = time.perf_counter() - start_time

    if completed_times and done > 0:
        avg = sum(completed_times) / len(completed_times)
        remaining = max(total - done, 0)
        eta_seconds = avg * remaining
        return (
            f"{done}/{total} ({pct:.2f}%) | "
            f"elapsed={format_duration(elapsed)} | "
            f"avg={avg:.2f}s | "
            f"eta≈{format_duration(eta_seconds)}"
        )

    return f"{done}/{total} ({pct:.2f}%) | elapsed={format_duration(elapsed)}"


def print_overall_progress_banner(progress: OverallProgress, label: str = "OVERALL PROGRESS") -> None:
    progress_text = format_progress(
        done=progress.completed_rows,
        total=progress.total_rows,
        start_time=progress.start_time,
        completed_times=progress.elapsed_seconds,
    )

    print("", flush=True)
    print("=" * 76, flush=True)
    print(f"{label}: {progress.completed_rows}/{progress.total_rows}", flush=True)
    print(f"{progress_text}", flush=True)
    print(
        f"new_runs={progress.newly_run_rows} | "
        f"existing_skips={progress.skipped_existing_rows} | "
        f"missing_question_skips={progress.skipped_missing_question_rows} | "
        f"failed={progress.failed_rows} | "
        f"bad_stdout={progress.bad_stdout_rows}",
        flush=True,
    )
    print("=" * 76, flush=True)


def print_file_progress_banner(
    file_stem: str,
    file_done: int,
    file_total: int,
    file_start_time: float,
    file_times: List[float],
) -> None:
    progress_text = format_progress(
        done=file_done,
        total=file_total,
        start_time=file_start_time,
        completed_times=file_times,
    )
    log_info(f"FILE PROGRESS [{file_stem}] {progress_text}")


# ---------------- Git helper ---------------- #

def run_git_after_batch(batch_number: int, processed_so_far: int, total_rows: int) -> None:
    if not ENABLE_GIT:
        return

    commit_msg = (
        f"finished {batch_number}th batch, "
        f"generated {processed_so_far}/{total_rows} JSON files"
    )

    log_step(f"Running git add/commit/push after batch #{batch_number}")

    try:
        add_proc = subprocess.run(["git", "add", "-A"], capture_output=True, text=True)
        if add_proc.returncode != 0:
            log_warn("'git add -A' failed")
            if add_proc.stderr:
                sys.stderr.write(add_proc.stderr + "\n")
            return

        commit_proc = subprocess.run(
            ["git", "commit", "-m", commit_msg],
            capture_output=True,
            text=True,
        )
        if commit_proc.returncode != 0:
            log_info("'git commit' did not create a commit")
            if commit_proc.stderr:
                sys.stderr.write(commit_proc.stderr + "\n")
            return

        push_proc = subprocess.run(
            ["git", "push", GIT_REMOTE, GIT_BRANCH],
            capture_output=True,
            text=True,
        )
        if push_proc.returncode != 0:
            log_warn(f"'git push {GIT_REMOTE} {GIT_BRANCH}' failed")
            if push_proc.stderr:
                sys.stderr.write(push_proc.stderr + "\n")
            return

        log_info("Git push completed successfully")

    except FileNotFoundError:
        log_warn("'git' command not found. Skipping git operations.")


# ---------------- Discovery helpers ---------------- #

def discover_csv_files(input_dirs: List[str]) -> List[str]:
    csv_files: List[str] = []

    for folder in input_dirs:
        if not os.path.isdir(folder):
            log_warn(f"Input directory not found, skipping: {folder}")
            continue

        found = sorted(glob.glob(os.path.join(folder, "*.csv")))
        log_info(f"Discovered {len(found)} CSV file(s) in {folder}")
        csv_files.extend(found)

    return csv_files


def count_total_rows(csv_files: List[str]) -> int:
    total = 0

    for csv_path in csv_files:
        try:
            with open(csv_path, newline="", encoding="utf-8-sig") as f:
                reader = csv.reader(f)
                header = next(reader, None)
                if header is None:
                    continue
                total += sum(1 for _ in reader)
        except Exception as e:
            log_warn(f"Could not count rows in {csv_path}: {e}")

    return total


# ---------------- General helpers ---------------- #

def safe_slug(text: str, max_len: int = 80) -> str:
    text = re.sub(r"\s+", "_", text.strip())
    text = re.sub(r"[^A-Za-z0-9._-]", "", text)
    text = text.strip("._-")

    if not text:
        text = "row"

    return text[:max_len]


def make_output_filename(row: List[str], global_index: int, colmap: Dict[str, Optional[int]]) -> str:
    question_id_idx = colmap.get("question_id")

    if question_id_idx is not None and len(row) > question_id_idx:
        question_id = row[question_id_idx].strip()
        if question_id:
            return f"{safe_slug(question_id)}.json"

    return f"{global_index:05d}.json"


def detect_schema(header: List[str]) -> Dict[str, Optional[int]]:
    normalized = [h.strip().lstrip("\ufeff") for h in header]

    def get_idx(name: str) -> Optional[int]:
        try:
            return normalized.index(name)
        except ValueError:
            return None

    entity_id_idx = None
    for i, col_name in enumerate(normalized):
        if col_name.startswith("entity_id"):
            entity_id_idx = i
            break

    return {
        "question_id": get_idx("question_id"),
        "question": get_idx("question"),
        "answer": get_idx("answer"),
        "entity_id": entity_id_idx,
    }


def build_input_text(row: List[str], colmap: Dict[str, Optional[int]]) -> Optional[str]:
    question_idx = colmap.get("question")
    entity_id_idx = colmap.get("entity_id")

    if question_idx is None or len(row) <= question_idx:
        return None

    question = row[question_idx].strip()
    if not question:
        return None

    entity_id = ""
    if entity_id_idx is not None and len(row) > entity_id_idx:
        entity_id = row[entity_id_idx].strip()

    return f"{question} {entity_id}".strip() if entity_id else question


def ask_yes_no(prompt: str, default: str = "n") -> str:
    answer = ""

    while answer not in ("y", "n"):
        try:
            raw = input(prompt).strip().lower()
        except EOFError:
            log_warn(f"No interactive input available; defaulting to '{default}'.")
            raw = default

        if raw == "" and default in ("y", "n"):
            answer = default
        else:
            answer = raw

        if answer not in ("y", "n"):
            print("Please answer y or n.", flush=True)

    return answer


def write_text_atomic(path: str, text: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    tmp_path = path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())

    os.replace(tmp_path, path)


def remove_if_exists(path: str) -> None:
    try:
        if os.path.exists(path):
            os.remove(path)
            log_info(f"Removed stale artifact: {path}")
    except Exception as e:
        log_warn(f"Could not remove {path}: {e}")


# ---------------- Timing statistics ---------------- #

def summarize_times(times: List[float]) -> Dict[str, Optional[float]]:
    if not times:
        return {
            "count": 0,
            "mean": None,
            "std": None,
            "min": None,
            "p25": None,
            "median": None,
            "p75": None,
            "max": None,
        }

    sorted_times = sorted(times)
    count = len(sorted_times)

    def percentile(values: List[float], p: float) -> float:
        if len(values) == 1:
            return values[0]

        k = (len(values) - 1) * p
        f = math.floor(k)
        c = math.ceil(k)

        if f == c:
            return values[int(k)]

        d0 = values[f] * (c - k)
        d1 = values[c] * (k - f)
        return d0 + d1

    mean_v = statistics.mean(sorted_times)
    std_v = statistics.stdev(sorted_times) if count >= 2 else 0.0

    return {
        "count": count,
        "mean": mean_v,
        "std": std_v,
        "min": sorted_times[0],
        "p25": percentile(sorted_times, 0.25),
        "median": percentile(sorted_times, 0.50),
        "p75": percentile(sorted_times, 0.75),
        "max": sorted_times[-1],
    }


def build_time_histogram(times: List[float], bin_edges: Optional[List[float]] = None) -> List[Tuple[str, int]]:
    if not times:
        return []

    if bin_edges is None:
        bin_edges = [0.5, 1, 2, 5, 10, 20, 30, 60]

    counts = [0] * (len(bin_edges) + 1)

    for t in times:
        placed = False
        for i, edge in enumerate(bin_edges):
            if t <= edge:
                counts[i] += 1
                placed = True
                break
        if not placed:
            counts[-1] += 1

    labels: List[str] = []
    prev = 0.0

    for edge in bin_edges:
        labels.append(f"{prev:.1f}-{edge:.1f}s")
        prev = edge

    labels.append(f">{bin_edges[-1]:.1f}s")

    return list(zip(labels, counts))


def print_time_summary(title: str, times: List[float]) -> None:
    print("", flush=True)
    print(f"--- {title} ---", flush=True)

    stats = summarize_times(times)
    if stats["count"] == 0:
        print("No timing data collected.", flush=True)
        return

    print(f"count   : {stats['count']}", flush=True)
    print(f"mean    : {stats['mean']:.3f}s", flush=True)
    print(f"std dev : {stats['std']:.3f}s", flush=True)
    print(f"min     : {stats['min']:.3f}s", flush=True)
    print(f"p25     : {stats['p25']:.3f}s", flush=True)
    print(f"median  : {stats['median']:.3f}s", flush=True)
    print(f"p75     : {stats['p75']:.3f}s", flush=True)
    print(f"max     : {stats['max']:.3f}s", flush=True)

    print("distribution:", flush=True)
    for label, count in build_time_histogram(times):
        print(f"  {label:>10} : {count}", flush=True)


def write_time_summary_json(path: str, title: str, times: List[float]) -> None:
    payload = {
        "title": title,
        "stats": summarize_times(times),
        "distribution": [
            {"range": label, "count": count}
            for label, count in build_time_histogram(times)
        ],
        "raw_times_seconds": [round(t, 6) for t in times],
    }

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


# ---------------- Output validation ---------------- #

def is_semantically_bad_json(data: object) -> Tuple[bool, str]:
    if RERUN_EMPTY_JSON_STRUCTURES:
        if data is None:
            return True, "json_null"
        if data == {}:
            return True, "empty_json_object"
        if data == []:
            return True, "empty_json_array"

    if isinstance(data, dict):
        # Known GRASP null-output pattern.
        if (
            data.get("type") == "output"
            and data.get("task") == "sparql-qa"
            and data.get("output") is None
        ):
            return True, "null_output"

        # General empty-output detection.
        if RERUN_EMPTY_OUTPUT_FIELD and "output" in data:
            if data.get("output") in (None, "", [], {}):
                return True, "empty_output"

        # Structured error payloads generated by this script should be re-run next time.
        if data.get("error") in {
            "empty_stdout",
            "invalid_json_stdout",
            "grasp_nonzero_exit",
        }:
            return True, f"script_error_payload:{data.get('error')}"

    return False, ""


def is_good_output_file(path: str) -> Tuple[bool, str]:
    stderr_path = path + ".stderr.txt"

    if RERUN_WHEN_STDERR_EXISTS and os.path.exists(stderr_path):
        return False, "stderr_sidecar_exists"

    if not os.path.exists(path):
        return False, "missing_output_file"

    try:
        if os.path.getsize(path) == 0:
            return False, "empty_file_zero_bytes"
    except OSError as e:
        return False, f"stat_error:{e.__class__.__name__}"

    try:
        with open(path, "r", encoding="utf-8") as jf:
            raw = jf.read()
    except Exception as e:
        return False, f"read_error:{e.__class__.__name__}"

    if not raw.strip():
        return False, "empty_file_whitespace_only"

    try:
        data = json.loads(raw)
    except Exception as e:
        return False, f"invalid_json:{e.__class__.__name__}"

    is_bad, reason = is_semantically_bad_json(data)
    if is_bad:
        return False, reason

    return True, ""


def detect_bad_outputs(
    rows: List[List[str]],
    colmap: Dict[str, Optional[int]],
    out_dir: str,
    candidate_indices: Set[int],
) -> Tuple[Set[int], str]:
    bad_indices: Set[int] = set()
    records = []

    if not candidate_indices:
        return bad_indices, ""

    for idx in sorted(candidate_indices):
        if idx < 0 or idx >= len(rows):
            continue

        row = rows[idx]
        filename = make_output_filename(row, idx, colmap)
        out_file = os.path.join(out_dir, filename)

        is_good, reason = is_good_output_file(out_file)
        if not is_good:
            bad_indices.add(idx)
            records.append((idx, filename, reason))

    if not records:
        return bad_indices, ""

    os.makedirs(out_dir, exist_ok=True)
    bad_csv_path = os.path.join(out_dir, "bad_executions.csv")

    with open(bad_csv_path, "w", newline="", encoding="utf-8") as cf:
        writer = csv.writer(cf)
        writer.writerow(["row_index", "filename", "reason"])
        writer.writerows(records)

    return bad_indices, bad_csv_path


def scan_existing_good_outputs(
    rows: List[List[str]],
    colmap: Dict[str, Optional[int]],
    out_dir: str,
) -> Set[int]:
    good_indices: Set[int] = set()

    for idx, row in enumerate(rows):
        filename = make_output_filename(row, idx, colmap)
        out_file = os.path.join(out_dir, filename)

        is_good, _ = is_good_output_file(out_file)
        if is_good:
            good_indices.add(idx)

    return good_indices


# ---------------- Batch log / resume helpers ---------------- #

def parse_batch_log(batch_log_path: str, total_rows: int) -> Tuple[Set[int], bool]:
    processed_indices: Set[int] = set()
    is_completed = False

    if not os.path.exists(batch_log_path):
        return processed_indices, is_completed

    with open(batch_log_path, "r", encoding="utf-8") as lf:
        for line in lf:
            stripped = line.strip()

            if not stripped:
                continue

            if stripped.startswith("#"):
                if "COMPLETED" in stripped.upper():
                    is_completed = True
                continue

            row_match = re.match(r"^(DONE|SKIP)\s+row\s+(\d+)\b", stripped)
            if row_match:
                processed_indices.add(int(row_match.group(2)))
                continue

            parts = [p.strip() for p in stripped.split("|")]
            for part in parts:
                if part.startswith("rows "):
                    range_str = part[len("rows "):].strip()

                    if "-" in range_str:
                        start_s, end_s = range_str.split("-", 1)
                        try:
                            start_i = int(start_s)
                            end_i = int(end_s)
                        except ValueError:
                            continue

                        for idx in range(start_i, end_i + 1):
                            processed_indices.add(idx)
                    else:
                        try:
                            idx = int(range_str)
                        except ValueError:
                            continue

                        processed_indices.add(idx)

    if not is_completed and len(processed_indices) >= total_rows:
        is_completed = True

    return processed_indices, is_completed


def ensure_batch_log_header(batch_log_path: str, csv_path: str) -> None:
    if os.path.exists(batch_log_path):
        return

    os.makedirs(os.path.dirname(batch_log_path) or ".", exist_ok=True)

    with open(batch_log_path, "w", encoding="utf-8") as lf:
        lf.write(f"# Batch log for {csv_path}\n")
        lf.write("# Old batch lines: batch_index | row_indices | filenames\n")
        lf.write("# New row-level lines:\n")
        lf.write("# DONE row <row_index> | file: <filename>\n")
        lf.write("# SKIP row <row_index> | reason: <reason>\n\n")


def append_row_done(batch_log_path: str, row_index: int, filename: str) -> None:
    with open(batch_log_path, "a", encoding="utf-8") as lf:
        lf.write(f"DONE row {row_index} | file: {filename}\n")
        lf.flush()
        os.fsync(lf.fileno())


def append_row_skip(batch_log_path: str, row_index: int, reason: str) -> None:
    with open(batch_log_path, "a", encoding="utf-8") as lf:
        lf.write(f"SKIP row {row_index} | reason: {reason}\n")
        lf.flush()
        os.fsync(lf.fileno())


def find_first_unprocessed_index(total_rows: int, processed_indices: Set[int]) -> Optional[int]:
    for idx in range(total_rows):
        if idx not in processed_indices:
            return idx

    return None


# ---------------- Startup planning ---------------- #

def load_csv_plan(csv_path: str) -> CsvPlan:
    file_stem = os.path.splitext(os.path.basename(csv_path))[0]
    out_dir = os.path.join(OUTPUT_ROOT, file_stem)
    os.makedirs(out_dir, exist_ok=True)

    batch_log_path = os.path.join(out_dir, "batch_log.txt")

    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        raw_header = next(reader, None)

        if raw_header is None:
            raise ValueError(f"CSV file '{csv_path}' is empty or missing a header row.")

        header = [h.strip().lstrip("\ufeff") for h in raw_header]
        colmap = detect_schema(header)

        if colmap["question"] is None:
            raise ValueError(f"'question' column not found in CSV header: {header}")

        rows = list(reader)

    total_rows = len(rows)

    ensure_batch_log_header(batch_log_path, csv_path)

    logged_indices, is_completed_from_log = parse_batch_log(batch_log_path, total_rows)
    existing_good_indices = scan_existing_good_outputs(rows, colmap, out_dir)

    all_indices = set(range(total_rows))
    bad_indices, bad_csv_path = detect_bad_outputs(rows, colmap, out_dir, all_indices)

    processed_indices = set(logged_indices)
    processed_indices.update(existing_good_indices)

    return CsvPlan(
        csv_path=csv_path,
        file_stem=file_stem,
        out_dir=out_dir,
        batch_log_path=batch_log_path,
        header=header,
        colmap=colmap,
        rows=rows,
        total_rows=total_rows,
        logged_indices=logged_indices,
        existing_good_indices=existing_good_indices,
        bad_indices=bad_indices,
        processed_indices=processed_indices,
        is_completed_from_log=is_completed_from_log,
        bad_csv_path=bad_csv_path,
    )


def print_startup_scan_summary(plans: List[CsvPlan]) -> None:
    log_step("STARTUP RESUME SCAN")

    total_rows = sum(p.total_rows for p in plans)
    total_logged = sum(len(p.logged_indices) for p in plans)
    total_good = sum(len(p.existing_good_indices) for p in plans)
    total_bad = sum(len(p.bad_indices) for p in plans)
    total_to_run_now = sum(p.total_rows - len(p.processed_indices) for p in plans)

    print("", flush=True)
    print("================ STARTUP RESUME SCAN ================", flush=True)
    print(f"CSV files discovered          : {len(plans)}", flush=True)
    print(f"Total rows                    : {total_rows}", flush=True)
    print(f"Rows marked in logs           : {total_logged}", flush=True)
    print(f"Rows with good output files   : {total_good}", flush=True)
    print(f"Rows with bad output/stderr   : {total_bad}", flush=True)
    print(f"Rows currently not processed  : {total_to_run_now}", flush=True)

    for p in plans:
        print("", flush=True)
        print(f"File: {p.csv_path}", flush=True)
        print(f"  output dir                  : {p.out_dir}", flush=True)
        print(f"  total rows                  : {p.total_rows}", flush=True)
        print(f"  logged DONE/SKIP rows       : {len(p.logged_indices)}", flush=True)
        print(f"  good existing outputs       : {len(p.existing_good_indices)}", flush=True)
        print(f"  bad output/stderr detected  : {len(p.bad_indices)}", flush=True)
        print(f"  currently not processed     : {p.total_rows - len(p.processed_indices)}", flush=True)

        if p.bad_csv_path:
            print(f"  bad execution report        : {p.bad_csv_path}", flush=True)

    print("=====================================================", flush=True)


def apply_startup_resume_choices(plans: List[CsvPlan]) -> None:
    total_bad = sum(len(p.bad_indices) for p in plans)
    completed_plans = [
        p for p in plans
        if p.total_rows > 0 and len(p.processed_indices) >= p.total_rows
    ]

    rerun_bad = "n"
    rerun_complete = "n"

    if ASK_AT_STARTUP and total_bad > 0:
        rerun_bad = ask_yes_no(
            f"Detected {total_bad} row(s) with bad output, empty JSON, or .stderr.txt. Re-run them now? [y/n]: ",
            default="y",
        )
    elif total_bad > 0:
        rerun_bad = "y"

    if ASK_AT_STARTUP and completed_plans:
        rerun_complete = ask_yes_no(
            f"Detected {len(completed_plans)} fully processed CSV file(s). Re-run all completed outputs too? [y/n]: ",
            default="n",
        )

    for p in plans:
        if rerun_bad == "y" and p.bad_indices:
            p.processed_indices -= p.bad_indices

            with open(p.batch_log_path, "a", encoding="utf-8") as lf:
                lf.write(f"\n# Startup resume: scheduled {len(p.bad_indices)} bad row(s) for re-run.\n")

            log_info(f"Scheduled {len(p.bad_indices)} bad row(s) for re-run in {p.csv_path}")

        if rerun_complete == "y" and p.total_rows > 0 and len(p.processed_indices) >= p.total_rows:
            p.processed_indices.clear()

            with open(p.batch_log_path, "a", encoding="utf-8") as lf:
                lf.write("\n# Startup resume: user requested full re-run of completed CSV.\n")

            log_info(f"Scheduled full re-run for completed CSV: {p.csv_path}")


def prepare_row_for_rerun(out_file: str) -> None:
    remove_if_exists(out_file)
    remove_if_exists(out_file + ".stderr.txt")
    remove_if_exists(out_file + ".tmp")


# ---------------- GRASP output handling ---------------- #

def add_metadata_to_data(
    data: object,
    elapsed: float,
    csv_path: str,
    row_index: int,
    row: List[str],
    colmap: Dict[str, Optional[int]],
) -> object:
    if not isinstance(data, dict):
        return data

    data["elapsed"] = round(elapsed, 6)
    data["source_csv"] = csv_path
    data["row_index"] = row_index

    q_idx = colmap.get("question")
    a_idx = colmap.get("answer")

    data["question"] = (
        row[q_idx].strip()
        if q_idx is not None and len(row) > q_idx
        else None
    )

    if a_idx is not None and len(row) > a_idx:
        data["reference_answer"] = row[a_idx].strip()

    return data


def make_error_payload(
    error: str,
    elapsed: float,
    csv_path: str,
    row_index: int,
    row: List[str],
    colmap: Dict[str, Optional[int]],
    returncode: int,
    stdout_text: str,
    stderr_text: str,
) -> Dict[str, object]:
    q_idx = colmap.get("question")
    a_idx = colmap.get("answer")

    payload: Dict[str, object] = {
        "error": error,
        "elapsed": round(elapsed, 6),
        "source_csv": csv_path,
        "row_index": row_index,
        "returncode": returncode,
        "stdout_preview": stdout_text[:STDOUT_PREVIEW_CHARS],
        "stderr_preview": stderr_text[:STDERR_PREVIEW_CHARS],
        "question": (
            row[q_idx].strip()
            if q_idx is not None and len(row) > q_idx
            else None
        ),
    }

    if a_idx is not None and len(row) > a_idx:
        payload["reference_answer"] = row[a_idx].strip()

    return payload


def serialize_grasp_result(
    proc: subprocess.CompletedProcess,
    elapsed: float,
    csv_path: str,
    row_index: int,
    row: List[str],
    colmap: Dict[str, Optional[int]],
) -> Tuple[str, bool]:
    stdout_text = proc.stdout or ""
    stderr_text = proc.stderr or ""

    if not stdout_text.strip():
        payload = make_error_payload(
            error="empty_stdout",
            elapsed=elapsed,
            csv_path=csv_path,
            row_index=row_index,
            row=row,
            colmap=colmap,
            returncode=proc.returncode,
            stdout_text=stdout_text,
            stderr_text=stderr_text,
        )
        return json.dumps(payload, ensure_ascii=False, indent=2), True

    try:
        data = json.loads(stdout_text)
    except json.JSONDecodeError:
        payload = make_error_payload(
            error="invalid_json_stdout",
            elapsed=elapsed,
            csv_path=csv_path,
            row_index=row_index,
            row=row,
            colmap=colmap,
            returncode=proc.returncode,
            stdout_text=stdout_text,
            stderr_text=stderr_text,
        )
        return json.dumps(payload, ensure_ascii=False, indent=2), True

    data = add_metadata_to_data(data, elapsed, csv_path, row_index, row, colmap)
    return json.dumps(data, ensure_ascii=False, indent=2), False


def run_grasp_for_row(input_text: str) -> Tuple[subprocess.CompletedProcess, float]:
    start_time = time.perf_counter()

    proc = subprocess.run(
        GRASP_COMMAND,
        input=input_text,
        capture_output=True,
        text=True,
    )

    elapsed = time.perf_counter() - start_time
    return proc, elapsed


# ---------------- Main CSV processing ---------------- #

def process_csv_plan(
    plan: CsvPlan,
    overall_progress: OverallProgress,
    overall_times: List[float],
    batch_size: int = BATCH_SIZE,
) -> Dict[str, object]:
    csv_path = plan.csv_path
    file_stem = plan.file_stem
    out_dir = plan.out_dir
    batch_log_path = plan.batch_log_path
    colmap = plan.colmap
    rows = plan.rows
    total_rows = plan.total_rows
    processed_indices = set(plan.processed_indices)

    log_step(f"Processing CSV: {csv_path}")
    log_info(f"Output directory: {out_dir}")

    if total_rows == 0:
        log_info("No data rows found in CSV. Skipping.")
        return {
            "csv_path": csv_path,
            "total_rows": 0,
            "processed_rows": 0,
            "timings": [],
        }

    log_info("Detected schema:")
    log_info(f"  question_id : {colmap['question_id']}")
    log_info(f"  question    : {colmap['question']}")
    log_info(f"  answer      : {colmap['answer']}")
    log_info(f"  entity_id   : {colmap['entity_id']}")
    log_info(f"  total rows  : {total_rows}")

    first_unprocessed = find_first_unprocessed_index(total_rows, processed_indices)

    if first_unprocessed is not None:
        log_info(f"Resuming from first unfinished row {first_unprocessed}.")
    else:
        log_info("All rows are already processed for this CSV. Skipping.")

        already_done = total_rows
        overall_progress.completed_rows += already_done
        overall_progress.skipped_existing_rows += already_done

        print_overall_progress_banner(overall_progress, label="OVERALL PROGRESS AFTER COMPLETED CSV SKIP")

        return {
            "csv_path": csv_path,
            "total_rows": total_rows,
            "processed_rows": len(processed_indices),
            "timings": [],
        }

    processed_count = len(processed_indices)
    file_times: List[float] = []
    file_start_time = time.perf_counter()

    file_done = processed_count
    print_file_progress_banner(file_stem, file_done, total_rows, file_start_time, file_times)

    for batch_start in range(0, total_rows, batch_size):
        batch_rows = rows[batch_start: batch_start + batch_size]
        batch_index = batch_start // batch_size
        batch_number = batch_index + 1

        batch_filenames: List[str] = []
        batch_row_indices: List[int] = []

        log_step(
            f"Batch {batch_index} (#{batch_number}) START | "
            f"CSV={file_stem} | rows {batch_start}-{min(batch_start + batch_size - 1, total_rows - 1)}"
        )

        for i, row in enumerate(batch_rows):
            global_index = batch_start + i

            if global_index in processed_indices:
                processed_count += 0
                file_done += 1
                overall_progress.completed_rows += 1
                overall_progress.skipped_existing_rows += 1

                if LOG_SKIPPED_ROWS:
                    log_info(
                        f"SKIP existing | CSV={file_stem} | "
                        f"file progress {file_done}/{total_rows} | "
                        f"overall {overall_progress.completed_rows}/{overall_progress.total_rows} | "
                        f"row={global_index}"
                    )

                continue

            input_text = build_input_text(row, colmap)

            if not input_text:
                append_row_skip(batch_log_path, global_index, "empty_or_missing_question")

                processed_indices.add(global_index)
                processed_count += 1
                file_done += 1

                overall_progress.completed_rows += 1
                overall_progress.skipped_missing_question_rows += 1

                log_warn(
                    f"SKIP missing question | CSV={file_stem} | "
                    f"row={global_index} | "
                    f"overall {overall_progress.completed_rows}/{overall_progress.total_rows}"
                )

                print_overall_progress_banner(overall_progress, label="OVERALL PROGRESS AFTER SKIP")
                continue

            filename = make_output_filename(row, global_index, colmap)
            out_file = os.path.join(out_dir, filename)

            next_overall_index = overall_progress.completed_rows + 1
            next_file_index = file_done + 1

            print("", flush=True)
            print("=" * 76, flush=True)
            print(
                f"START CASE | OVERALL {next_overall_index}/{overall_progress.total_rows} | "
                f"FILE {next_file_index}/{total_rows}",
                flush=True,
            )
            print(f"CSV   : {csv_path}", flush=True)
            print(f"ROW   : {global_index}", flush=True)
            print(f"FILE  : {filename}", flush=True)
            print(f"OUT   : {out_file}", flush=True)
            print("=" * 76, flush=True)

            log_info(f"Question/input preview: {input_text[:500]}")

            if LOG_COMMAND_EACH_ROW:
                log_info(f"Command: {' '.join(shlex.quote(x) for x in GRASP_COMMAND)}")

            log_step(f"Preparing row artifacts | row={global_index} file={filename}")
            prepare_row_for_rerun(out_file)

            log_step(f"Starting GRASP | OVERALL {next_overall_index}/{overall_progress.total_rows}")
            proc, elapsed = run_grasp_for_row(input_text)

            file_times.append(elapsed)
            overall_times.append(elapsed)
            overall_progress.elapsed_seconds.append(elapsed)

            log_info(
                f"GRASP finished | row={global_index} | "
                f"elapsed={elapsed:.3f}s | returncode={proc.returncode}"
            )

            if proc.returncode != 0:
                overall_progress.failed_rows += 1
                log_warn(f"Non-zero return code for row={global_index} file={filename}")

            if proc.stderr:
                stderr_preview = proc.stderr.strip().replace("\n", " | ")[:1000]
                log_warn(f"stderr captured preview: {stderr_preview}")
            else:
                log_info("stderr is empty")

            log_step(f"Serializing GRASP stdout to JSON | row={global_index}")
            json_text, bad_stdout = serialize_grasp_result(
                proc=proc,
                elapsed=elapsed,
                csv_path=csv_path,
                row_index=global_index,
                row=row,
                colmap=colmap,
            )

            if bad_stdout:
                overall_progress.bad_stdout_rows += 1
                log_warn("GRASP stdout was empty or invalid JSON; wrote structured error payload")

            log_step(f"Writing output JSON: {out_file}")
            write_text_atomic(out_file, json_text)

            if proc.returncode != 0:
                if WRITE_STDERR_ON_NONZERO:
                    stderr_path = out_file + ".stderr.txt"
                    log_step(f"Writing stderr sidecar: {stderr_path}")
                    write_text_atomic(stderr_path, proc.stderr or "")
            else:
                stderr_path = out_file + ".stderr.txt"
                if os.path.exists(stderr_path):
                    log_step(f"Removing stale stderr sidecar after successful run: {stderr_path}")
                    remove_if_exists(stderr_path)

            append_row_done(batch_log_path, global_index, filename)

            processed_count += 1
            processed_indices.add(global_index)

            batch_filenames.append(filename)
            batch_row_indices.append(global_index)

            file_done += 1
            overall_progress.completed_rows += 1
            overall_progress.newly_run_rows += 1

            print_overall_progress_banner(overall_progress, label="OVERALL PROGRESS AFTER CASE")
            print_file_progress_banner(file_stem, file_done, total_rows, file_start_time, file_times)

            if (
                LOG_EVERY_N_ROWS > 0
                and overall_progress.completed_rows % LOG_EVERY_N_ROWS == 0
            ):
                log_info(
                    f"Heartbeat | OVERALL {overall_progress.completed_rows}/{overall_progress.total_rows} | "
                    f"CSV={file_stem} FILE {file_done}/{total_rows}"
                )

        if batch_row_indices:
            row_range_str = (
                f"{batch_row_indices[0]}-{batch_row_indices[-1]}"
                if len(batch_row_indices) > 1
                else str(batch_row_indices[0])
            )

            with open(batch_log_path, "a", encoding="utf-8") as lf:
                lf.write(
                    f"batch {batch_index} (#{batch_number}) | rows {row_range_str} | "
                    f"files: {', '.join(batch_filenames)}\n"
                )

            log_step(
                f"Batch {batch_index} (#{batch_number}) END | "
                f"rows {row_range_str} | generated {len(batch_filenames)} file(s)"
            )

            run_git_after_batch(batch_number, processed_count, total_rows)
        else:
            log_info(
                f"Batch {batch_index} (#{batch_number}) END | "
                f"no newly generated files"
            )

    with open(batch_log_path, "a", encoding="utf-8") as lf:
        lf.write(
            f"\n# COMPLETED: processed {processed_count}/{total_rows} rows at "
            f"{time.strftime('%Y-%m-%d %H:%M:%S')}\n"
        )

    log_step(f"Processing complete for CSV: {csv_path}")
    log_info(f"CSV path        : {csv_path}")
    log_info(f"Total rows      : {total_rows}")
    log_info(f"Processed rows  : {processed_count}")
    log_info(f"Batch size      : {batch_size}")
    log_info(f"Output directory: {out_dir}")
    log_info(f"Batch log file  : {batch_log_path}")

    print_time_summary(f"Timing summary for {file_stem}", file_times)

    write_time_summary_json(
        os.path.join(out_dir, "timing_summary.json"),
        f"Timing summary for {file_stem}",
        file_times,
    )

    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    write_time_summary_json(
        os.path.join(OUTPUT_ROOT, "overall_timing_summary.json"),
        "Overall timing summary",
        overall_times,
    )

    return {
        "csv_path": csv_path,
        "total_rows": total_rows,
        "processed_rows": processed_count,
        "timings": file_times,
    }


# ---------------- Final summary ---------------- #

def print_final_summary(
    file_summaries: List[Dict[str, object]],
    overall_progress: OverallProgress,
    overall_times: List[float],
) -> None:
    log_step("FINAL SUMMARY")

    print("", flush=True)
    print("================ FINAL SUMMARY ================", flush=True)
    print(f"Overall rows total              : {overall_progress.total_rows}", flush=True)
    print(f"Overall rows completed          : {overall_progress.completed_rows}", flush=True)
    print(f"Newly run rows                  : {overall_progress.newly_run_rows}", flush=True)
    print(f"Skipped existing rows           : {overall_progress.skipped_existing_rows}", flush=True)
    print(f"Skipped missing-question rows   : {overall_progress.skipped_missing_question_rows}", flush=True)
    print(f"Rows with non-zero return code  : {overall_progress.failed_rows}", flush=True)
    print(f"Rows with bad stdout payload    : {overall_progress.bad_stdout_rows}", flush=True)
    print(
        f"Total elapsed wall time         : {format_duration(time.perf_counter() - overall_progress.start_time)}",
        flush=True,
    )
    print("================================================", flush=True)

    for summary in file_summaries:
        csv_path = summary["csv_path"]
        total_rows = summary["total_rows"]
        processed_rows = summary["processed_rows"]
        timings = summary["timings"]

        print("", flush=True)
        print(f"File: {csv_path}", flush=True)
        print(f"  total rows     : {total_rows}", flush=True)
        print(f"  processed rows : {processed_rows}", flush=True)

        if timings:
            stats = summarize_times(timings)
            print(f"  avg time       : {stats['mean']:.3f}s", flush=True)
            print(f"  std dev        : {stats['std']:.3f}s", flush=True)
            print(f"  median         : {stats['median']:.3f}s", flush=True)
            print(f"  min / max      : {stats['min']:.3f}s / {stats['max']:.3f}s", flush=True)
        else:
            print("  timing stats   : no new timing data collected", flush=True)

    print_time_summary("Overall timing summary", overall_times)


# ---------------- Main entrypoint ---------------- #

def main() -> int:
    log_step("Starting 30B GRASP batch runner")

    #csv_files = discover_csv_files(INPUT_DIRS)
    csv_files = ["data/ComplexQA/Sportsreason_TANQ_complex.csv"]

    if not csv_files:
        log_error("No CSV files found in data/ComplexQA/ or data/SimpleQA/.")
        return 0

    log_info("Discovered CSV files:")
    for path in csv_files:
        log_info(f"  - {path}")

    plans: List[CsvPlan] = []

    for csv_path in csv_files:
        try:
            log_step(f"Preparing plan for CSV: {csv_path}")
            plans.append(load_csv_plan(csv_path))
        except Exception as e:
            log_error(f"Could not prepare plan for {csv_path}: {e}")

    if not plans:
        log_error("No usable CSV files found after startup scan.")
        return 1

    overall_total_rows = sum(p.total_rows for p in plans)

    if overall_total_rows == 0:
        log_info("No data rows found across discovered CSV files.")
        return 0

    print_startup_scan_summary(plans)
    apply_startup_resume_choices(plans)

    overall_times: List[float] = []
    file_summaries: List[Dict[str, object]] = []

    overall_progress = OverallProgress(total_rows=overall_total_rows)

    print("", flush=True)
    print("================ RUN STARTED ================", flush=True)
    print("No more interactive resume prompts will be shown during this run.", flush=True)
    print(f"Output root: {OUTPUT_ROOT}", flush=True)
    print(f"Total overall rows: {overall_total_rows}", flush=True)
    print("=============================================", flush=True)

    print_overall_progress_banner(overall_progress, label="OVERALL PROGRESS AT START")

    for plan in plans:
        summary = process_csv_plan(
            plan=plan,
            overall_progress=overall_progress,
            overall_times=overall_times,
            batch_size=BATCH_SIZE,
        )
        file_summaries.append(summary)

    print_overall_progress_banner(overall_progress, label="OVERALL PROGRESS AT END")

    print_final_summary(
        file_summaries=file_summaries,
        overall_progress=overall_progress,
        overall_times=overall_times,
    )

    write_time_summary_json(
        "overall_timing_summary.json",
        "Overall timing summary",
        overall_times,
    )

    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    write_time_summary_json(
        os.path.join(OUTPUT_ROOT, "overall_timing_summary.json"),
        "Overall timing summary",
        overall_times,
    )

    log_step("Finished 30B GRASP batch runner")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
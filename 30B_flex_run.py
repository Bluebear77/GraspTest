#!/usr/bin/env python3
"""
Run a GRASP / LLM pipeline over all CSV datasets in ComplexQA/ and SimpleQA/.

Improved behavior in this version:
- Scans every CSV/output directory at startup before processing anything.
- Detects rows that need re-run because output JSON is missing, empty, invalid,
  structurally empty, null-output, empty-output, or because a matching
  .stderr.txt file exists.
- Asks re-run decisions once at startup across all input directories.
- After the startup choices, the script runs to completion without asking again.
- Before re-running a bad row, removes old output JSON and old .stderr.txt.
- After a successful re-run with return code 0, removes stale .stderr.txt.
- If GRASP fails again, writes a new .stderr.txt and does not leave stale stderr.
- Writes structured JSON error payloads for empty stdout / invalid JSON so the
  output file is never silently empty after a run.
- Keeps row-level resume logs so interruption resumes safely.

Directory assumption:
Run this script from the parent directory that contains:
    data/ComplexQA/
    data/SimpleQA/
"""

import csv
import glob
import json
import math
import os
import re
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

from tqdm import tqdm

# ---------------- Configuration ---------------- #

INPUT_DIRS: List[str] = [
    "data/ComplexQA",
    "data/SimpleQA",
]

BATCH_SIZE = 50
GRASP_COMMAND = ["bash", "-lc", "grasp run configs/run.yaml"]

# Set True only if you really want git add/commit/push after each batch.
ENABLE_GIT = True
GIT_REMOTE = "origin"
GIT_BRANCH = "main"

# Startup-only prompts. After these are answered, the run is non-interactive.
ASK_AT_STARTUP = True

# Treat any existing stderr sidecar as a bad execution and re-run it.
RERUN_WHEN_STDERR_EXISTS = True

# Treat JSON with output equal to null, "", [], or {} as bad.
RERUN_EMPTY_OUTPUT_FIELD = True

# Treat {}, [], null as bad whole-file JSON outputs.
RERUN_EMPTY_JSON_STRUCTURES = True

# If GRASP exits non-zero, mark the row as bad for the current run by writing stderr.
# The row is still recorded as DONE because an output artifact was produced;
# on the next run, the .stderr.txt will trigger re-run.
WRITE_STDERR_ON_NONZERO = True


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
    missing_or_unprocessed_indices: Set[int] = field(default_factory=set)
    processed_indices: Set[int] = field(default_factory=set)
    is_completed_from_log: bool = False
    bad_csv_path: str = ""


# ---------------- Git helper ---------------- #

def run_git_after_batch(batch_number: int, processed_so_far: int, total_rows: int) -> None:
    if not ENABLE_GIT:
        return

    commit_msg = (
        f"finished {batch_number}th batch, "
        f"generated {processed_so_far}/{total_rows} JSON files"
    )

    try:
        add_proc = subprocess.run(["git", "add", "-A"], capture_output=True, text=True)
        if add_proc.returncode != 0:
            sys.stderr.write("[GIT WARN] 'git add -A' failed:\n")
            if add_proc.stderr:
                sys.stderr.write(add_proc.stderr + "\n")
            return

        commit_proc = subprocess.run(
            ["git", "commit", "-m", commit_msg], capture_output=True, text=True
        )
        if commit_proc.returncode != 0:
            sys.stderr.write("[GIT INFO] 'git commit' did not create a commit:\n")
            if commit_proc.stderr:
                sys.stderr.write(commit_proc.stderr + "\n")
            return

        push_proc = subprocess.run(
            ["git", "push", GIT_REMOTE, GIT_BRANCH], capture_output=True, text=True
        )
        if push_proc.returncode != 0:
            sys.stderr.write(f"[GIT WARN] 'git push {GIT_REMOTE} {GIT_BRANCH}' failed:\n")
            if push_proc.stderr:
                sys.stderr.write(push_proc.stderr + "\n")

    except FileNotFoundError:
        sys.stderr.write("[GIT WARN] 'git' command not found. Skipping git operations.\n")


# ---------------- Discovery helpers ---------------- #

def discover_csv_files(input_dirs: List[str]) -> List[str]:
    """Find all .csv files inside the given input directories."""
    csv_files: List[str] = []
    for folder in input_dirs:
        if not os.path.isdir(folder):
            print(f"[INFO] Input directory not found, skipping: {folder}")
            continue
        csv_files.extend(sorted(glob.glob(os.path.join(folder, "*.csv"))))
    return csv_files


def count_total_rows(csv_files: List[str]) -> int:
    """Count all data rows across discovered CSV files."""
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
            print(f"[WARN] Could not count rows in {csv_path}: {e}")
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
    suffix = " [y/n]: " if not prompt.rstrip().endswith("[y/n]:") else " "
    try:
        while answer not in ("y", "n"):
            raw = input(prompt + suffix).strip().lower()
            if raw == "" and default in ("y", "n"):
                answer = default
            else:
                answer = raw
    except EOFError:
        print(f"No interactive input available; defaulting to '{default}'.")
        answer = default
    return answer


def write_text_atomic(path: str, text: str) -> None:
    """Write text safely using temp file + atomic replace."""
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
    except Exception as e:
        sys.stderr.write(f"[WARN] Could not remove {path}: {e}\n")


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
    print(f"\n--- {title} ---")

    stats = summarize_times(times)
    if stats["count"] == 0:
        print("No timing data collected.")
        return

    print(f"count   : {stats['count']}")
    print(f"mean    : {stats['mean']:.3f}s")
    print(f"std dev : {stats['std']:.3f}s")
    print(f"min     : {stats['min']:.3f}s")
    print(f"p25     : {stats['p25']:.3f}s")
    print(f"median  : {stats['median']:.3f}s")
    print(f"p75     : {stats['p75']:.3f}s")
    print(f"max     : {stats['max']:.3f}s")

    print("distribution:")
    for label, count in build_time_histogram(times):
        print(f"  {label:>10} : {count}")


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
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


# ---------------- Output validation ---------------- #

def is_semantically_empty_json(data: object) -> Tuple[bool, str]:
    """Detect JSON that parses but should not count as a valid GRASP result."""
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

        # More general empty output field detection.
        if RERUN_EMPTY_OUTPUT_FIELD and "output" in data:
            if data.get("output") in (None, "", [], {}):
                return True, "empty_output"

        # Structured error payloads created by this script should be re-run.
        if data.get("error") in {
            "empty_stdout",
            "invalid_json_stdout",
            "grasp_nonzero_exit",
        }:
            return True, f"script_error_payload:{data.get('error')}"

    return False, ""


def is_good_output_file(path: str) -> Tuple[bool, str]:
    """
    Check whether an existing output file is usable.

    Bad cases include:
    - missing file
    - zero-byte or whitespace-only file
    - invalid JSON
    - JSON null, {}, []
    - known GRASP null-output pattern
    - empty output field
    - matching .stderr.txt sidecar exists
    """
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

    is_empty, reason = is_semantically_empty_json(data)
    if is_empty:
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
    """
    Parse the batch log and return:
    - row indices already completed or skipped
    - whether the file appears completed

    Supports:
    DONE row 73 | file: 00073.json
    SKIP row 74 | reason: empty_or_missing_question
    Old batch format containing: rows 50-99
    """
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
    with open(batch_log_path, "w", encoding="utf-8") as lf:
        lf.write(f"# Batch log for {csv_path}\n")
        lf.write("# Row-level lines:\n")
        lf.write("# DONE row <row_index> | file: <filename>\n")
        lf.write("# SKIP row <row_index> | reason: <reason>\n")
        lf.write("# Batch summary lines are comments only.\n\n")


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
    out_dir = os.path.join("output", file_stem)
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

    # Candidate bad rows are anything the log claims was processed, plus every row
    # that has an output filename already. This catches .stderr.txt sidecars even
    # when the log is incomplete.
    all_indices = set(range(total_rows))
    bad_indices, bad_csv_path = detect_bad_outputs(rows, colmap, out_dir, all_indices)

    processed_indices = set(logged_indices)
    processed_indices.update(existing_good_indices)

    missing_or_unprocessed_indices = all_indices - processed_indices

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
        missing_or_unprocessed_indices=missing_or_unprocessed_indices,
        processed_indices=processed_indices,
        is_completed_from_log=is_completed_from_log,
        bad_csv_path=bad_csv_path,
    )


def print_startup_scan_summary(plans: List[CsvPlan]) -> None:
    print("\n================ STARTUP RESUME SCAN ================")

    total_rows = sum(p.total_rows for p in plans)
    total_good = sum(len(p.existing_good_indices) for p in plans)
    total_logged = sum(len(p.logged_indices) for p in plans)
    total_bad = sum(len(p.bad_indices) for p in plans)
    total_unprocessed = sum(len(p.missing_or_unprocessed_indices) for p in plans)

    print(f"CSV files discovered          : {len(plans)}")
    print(f"Total rows                    : {total_rows}")
    print(f"Rows marked in logs           : {total_logged}")
    print(f"Rows with good output files   : {total_good}")
    print(f"Rows with bad outputs/stderr  : {total_bad}")
    print(f"Rows not yet processed        : {total_unprocessed}")

    for p in plans:
        print(f"\nFile: {p.csv_path}")
        print(f"  total rows                  : {p.total_rows}")
        print(f"  logged DONE/SKIP rows       : {len(p.logged_indices)}")
        print(f"  good existing outputs       : {len(p.existing_good_indices)}")
        print(f"  bad outputs/stderr detected : {len(p.bad_indices)}")
        print(f"  not yet processed           : {len(p.missing_or_unprocessed_indices)}")
        if p.bad_csv_path:
            print(f"  bad execution report        : {p.bad_csv_path}")


def apply_startup_resume_choices(plans: List[CsvPlan]) -> None:
    """
    Ask all resume questions once at startup and mutate each plan's processed_indices.
    After this function returns, processing is non-interactive.
    """
    total_bad = sum(len(p.bad_indices) for p in plans)
    completed_plans = [
        p for p in plans
        if p.total_rows > 0 and len(p.processed_indices) >= p.total_rows
    ]

    rerun_bad = "n"
    rerun_complete = "n"

    if ASK_AT_STARTUP and total_bad > 0:
        rerun_bad = ask_yes_no(
            f"Detected {total_bad} row(s) with bad output, empty JSON, or .stderr.txt. Re-run them now?",
            default="y",
        )
    elif total_bad > 0:
        rerun_bad = "y"

    if ASK_AT_STARTUP and completed_plans:
        rerun_complete = ask_yes_no(
            f"Detected {len(completed_plans)} fully processed CSV file(s). Re-run all completed outputs too?",
            default="n",
        )

    for p in plans:
        if rerun_bad == "y" and p.bad_indices:
            p.processed_indices -= p.bad_indices
            with open(p.batch_log_path, "a", encoding="utf-8") as lf:
                lf.write(
                    f"\n# Startup resume: scheduled {len(p.bad_indices)} bad row(s) for re-run.\n"
                )

        if rerun_complete == "y" and p.total_rows > 0 and len(p.processed_indices) >= p.total_rows:
            p.processed_indices.clear()
            with open(p.batch_log_path, "a", encoding="utf-8") as lf:
                lf.write("\n# Startup resume: user requested full re-run of completed CSV.\n")


def prepare_row_for_rerun(out_file: str) -> None:
    """
    Remove stale artifacts before re-running a row.
    This ensures old stderr cannot survive after a successful retry.
    """
    remove_if_exists(out_file)
    remove_if_exists(out_file + ".stderr.txt")
    remove_if_exists(out_file + ".tmp")


# ---------------- GRASP output handling ---------------- #

def build_metadata_payload(
    data: object,
    elapsed: float,
    csv_path: str,
    row_index: int,
    row: List[str],
    colmap: Dict[str, Optional[int]],
) -> object:
    """Add metadata when output is a dict; otherwise return unchanged data."""
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
        "stdout_preview": stdout_text[:4000],
        "stderr_preview": stderr_text[:4000],
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
) -> str:
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
        return json.dumps(payload, ensure_ascii=False, indent=2)

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
        return json.dumps(payload, ensure_ascii=False, indent=2)

    data = build_metadata_payload(data, elapsed, csv_path, row_index, row, colmap)
    return json.dumps(data, ensure_ascii=False, indent=2)


# ---------------- Main CSV processing ---------------- #

def process_csv_plan(
    plan: CsvPlan,
    overall_pbar: tqdm,
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

    print(f"\n=== Processing CSV: {csv_path} ===")
    print("Output directory:", out_dir)

    if total_rows == 0:
        print("No data rows found in CSV. Skipping.")
        return {
            "csv_path": csv_path,
            "total_rows": 0,
            "processed_rows": 0,
            "timings": [],
        }

    print("Detected schema:")
    print(f"  question_id : {colmap['question_id']}")
    print(f"  question    : {colmap['question']}")
    print(f"  answer      : {colmap['answer']}")
    print(f"  entity_id   : {colmap['entity_id']}")
    print(f"  total rows  : {total_rows}")

    first_unprocessed = find_first_unprocessed_index(total_rows, processed_indices)
    if first_unprocessed is not None:
        print(f"Resuming from first unfinished row {first_unprocessed}.")
    else:
        print("All rows are already processed for this CSV. Skipping.")
        overall_pbar.update(total_rows)
        return {
            "csv_path": csv_path,
            "total_rows": total_rows,
            "processed_rows": len(processed_indices),
            "timings": [],
        }

    processed_count = len(processed_indices)
    file_times: List[float] = []

    with tqdm(
        total=total_rows,
        desc=f"File: {file_stem}",
        unit="row",
        initial=processed_count,
        leave=False,
        position=1,
    ) as file_pbar:
        for batch_start in range(0, total_rows, batch_size):
            batch_rows = rows[batch_start: batch_start + batch_size]
            batch_index = batch_start // batch_size
            batch_number = batch_index + 1
            batch_filenames: List[str] = []
            batch_row_indices: List[int] = []

            for i, row in enumerate(batch_rows):
                global_index = batch_start + i

                if global_index in processed_indices:
                    file_pbar.update(1)
                    overall_pbar.update(1)
                    continue

                input_text = build_input_text(row, colmap)
                if not input_text:
                    append_row_skip(batch_log_path, global_index, "empty_or_missing_question")
                    processed_indices.add(global_index)
                    processed_count += 1

                    file_pbar.update(1)
                    overall_pbar.update(1)
                    continue

                filename = make_output_filename(row, global_index, colmap)
                out_file = os.path.join(out_dir, filename)

                # Important: remove stale JSON/stderr before re-running.
                prepare_row_for_rerun(out_file)

                start_time = time.perf_counter()
                proc = subprocess.run(
                    GRASP_COMMAND,
                    input=input_text,
                    capture_output=True,
                    text=True,
                )
                elapsed = time.perf_counter() - start_time

                file_times.append(elapsed)
                overall_times.append(elapsed)

                json_text = serialize_grasp_result(
                    proc=proc,
                    elapsed=elapsed,
                    csv_path=csv_path,
                    row_index=global_index,
                    row=row,
                    colmap=colmap,
                )

                write_text_atomic(out_file, json_text)

                if proc.returncode != 0:
                    sys.stderr.write(
                        f"[WARN] Row {global_index} ({filename}): grasp returned non-zero exit.\n"
                    )
                    if WRITE_STDERR_ON_NONZERO:
                        write_text_atomic(out_file + ".stderr.txt", proc.stderr or "")
                else:
                    # Successful retry: ensure old stderr is gone.
                    remove_if_exists(out_file + ".stderr.txt")

                append_row_done(batch_log_path, global_index, filename)

                processed_count += 1
                processed_indices.add(global_index)
                batch_filenames.append(filename)
                batch_row_indices.append(global_index)

                file_pbar.update(1)
                overall_pbar.update(1)

                if file_times:
                    file_mean = statistics.mean(file_times)
                    overall_mean = statistics.mean(overall_times) if overall_times else 0.0
                    file_pbar.set_postfix_str(
                        f"avg={file_mean:.2f}s | overall_avg={overall_mean:.2f}s"
                    )

            if batch_row_indices:
                row_range_str = (
                    f"{batch_row_indices[0]}-{batch_row_indices[-1]}"
                    if len(batch_row_indices) > 1
                    else str(batch_row_indices[0])
                )
                with open(batch_log_path, "a", encoding="utf-8") as lf:
                    lf.write(
                        f"# batch {batch_index} (#{batch_number}) completed rows {row_range_str} | "
                        f"files: {', '.join(batch_filenames)}\n"
                    )

                run_git_after_batch(batch_number, processed_count, total_rows)

    with open(batch_log_path, "a", encoding="utf-8") as lf:
        lf.write(
            f"\n# COMPLETED: processed {processed_count}/{total_rows} rows at "
            f"{time.strftime('%Y-%m-%d %H:%M:%S')}\n"
        )

    print("\nProcessing complete for this CSV.")
    print(f"  CSV path        : {csv_path}")
    print(f"  Total rows      : {total_rows}")
    print(f"  Processed rows  : {processed_count}")
    print(f"  Batch size      : {batch_size}")
    print(f"  Output directory: {out_dir}")
    print(f"  Batch log file  : {batch_log_path}")

    print_time_summary(f"Timing summary for {file_stem}", file_times)
    write_time_summary_json(
        os.path.join(out_dir, "timing_summary.json"),
        f"Timing summary for {file_stem}",
        file_times,
    )

    return {
        "csv_path": csv_path,
        "total_rows": total_rows,
        "processed_rows": processed_count,
        "timings": file_times,
    }


# ---------------- Main entrypoint ---------------- #

if __name__ == "__main__":
    # csv_files = discover_csv_files(INPUT_DIRS)
    csv_files = ["data/SimpleQA/Qampari_wikitables_simple.csv"]

    if not csv_files:
        print("No CSV files found in ComplexQA/ or SimpleQA/.")
        sys.exit(0)

    print("Discovered CSV files:")
    for path in csv_files:
        print(f"  - {path}")

    plans: List[CsvPlan] = []
    for csv_path in csv_files:
        try:
            plans.append(load_csv_plan(csv_path))
        except Exception as e:
            print(f"[ERROR] Could not prepare plan for {csv_path}: {e}")

    if not plans:
        print("No usable CSV files found after startup scan.")
        sys.exit(1)

    overall_total_rows = sum(p.total_rows for p in plans)
    overall_times: List[float] = []
    file_summaries: List[Dict[str, object]] = []

    if overall_total_rows == 0:
        print("No data rows found across discovered CSV files.")
        sys.exit(0)

    print_startup_scan_summary(plans)
    apply_startup_resume_choices(plans)

    print("\n================ RUN STARTED ================")
    print("No more interactive resume prompts will be shown during this run.")

    with tqdm(
        total=overall_total_rows,
        desc="Overall",
        unit="row",
        position=0,
    ) as overall_pbar:
        for plan in plans:
            summary = process_csv_plan(
                plan=plan,
                overall_pbar=overall_pbar,
                overall_times=overall_times,
                batch_size=BATCH_SIZE,
            )
            file_summaries.append(summary)

    print("\n================ FINAL SUMMARY ================")

    for summary in file_summaries:
        csv_path = summary["csv_path"]
        total_rows = summary["total_rows"]
        processed_rows = summary["processed_rows"]
        timings = summary["timings"]

        print(f"\nFile: {csv_path}")
        print(f"  total rows     : {total_rows}")
        print(f"  processed rows : {processed_rows}")

        if timings:
            stats = summarize_times(timings)
            print(f"  avg time       : {stats['mean']:.3f}s")
            print(f"  std dev        : {stats['std']:.3f}s")
            print(f"  median         : {stats['median']:.3f}s")
            print(f"  min / max      : {stats['min']:.3f}s / {stats['max']:.3f}s")
        else:
            print("  timing stats   : no new timing data collected")

    print_time_summary("Overall timing summary", overall_times)
    write_time_summary_json(
        "overall_timing_summary.json",
        "Overall timing summary",
        overall_times,
    )

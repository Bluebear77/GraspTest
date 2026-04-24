#!/usr/bin/env python3

"""
Run a GRASP / LLM pipeline over all CSV datasets in ComplexQA/ and SimpleQA/.

This script is intended for new students who want to process many QA datasets
in one run without manually editing the code for each CSV file.

What the script does:
- Finds every .csv file inside ComplexQA/ and SimpleQA/.
- Reads each CSV and detects whether it contains columns such as:
    question_id, question, answer, entity_id*
- Builds the model input from the question text. If an entity_id* column exists,
  it appends that entity identifier to the question.
- Runs:
    grasp run configs/run.yaml
  once for each valid row.
- Saves one output file per processed row as JSON in:
    output/<csv_name>/
- Adds useful metadata to JSON outputs, including:
    elapsed time, source CSV, row index, question, and reference answer
- Maintains a batch log so interrupted runs can resume.
- Records every completed row immediately so interrupted runs resume from
  the exact last unfinished row.
- Detects missing, invalid, or null outputs from previous runs and can re-run them.
- Scans existing output files on startup so rows are not re-run unnecessarily
  even if the program stopped after writing JSON but before updating the log.
- Shows two progress bars:
    1) overall progress across all CSV files
    2) progress for the current CSV file
- Records timing statistics for each question, then reports:
    average time, standard deviation, min, max, median, and a simple distribution
  both per file and for the whole run.

Supported CSV styles:
1) Rich schema, for example:
   question_id, question, entity_id1, ...
2) Simple schema, for example:
   question, answer

Important behavior:
- If question_id exists and is non-empty, it is used as the JSON filename.
- Otherwise, the script uses a zero-padded row index, for example 00042.json.
- If entity_id* is missing, only the question text is sent to GRASP.
- If a row has no usable question, the row is skipped safely and logged.
- If GRASP returns invalid JSON, stdout is still saved so results are not lost.
- If GRASP exits with a non-zero return code, stderr is saved in a separate file.
- Output JSON files are written atomically using a temporary file and rename.

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

ASK_BEFORE_RERUN_COMPLETE = True
ASK_BEFORE_RERUN_BAD = True


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
    """
    Find all .csv files inside the given input directories.
    """
    csv_files: List[str] = []
    for folder in input_dirs:
        if not os.path.isdir(folder):
            print(f"[INFO] Input directory not found, skipping: {folder}")
            continue
        csv_files.extend(sorted(glob.glob(os.path.join(folder, "*.csv"))))
    return csv_files


def count_total_rows(csv_files: List[str]) -> int:
    """
    Count all data rows across discovered CSV files.
    Assumes one header row per non-empty CSV.
    """
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
    try:
        while answer not in ("y", "n"):
            answer = input(prompt).strip().lower()
    except EOFError:
        print(f"No interactive input available; defaulting to '{default}'.")
        answer = default
    return answer


def write_text_atomic(path: str, text: str) -> None:
    """
    Write output safely.
    First write to a temporary file, flush it, and then atomically replace
    the final destination file.

    This reduces the chance of leaving a half-written JSON file if the run
    is interrupted while writing output.
    """
    tmp_path = path + ".tmp"

    with open(tmp_path, "w", encoding="utf-8") as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())

    os.replace(tmp_path, path)


# ---------------- Timing statistics ---------------- #

def summarize_times(times: List[float]) -> Dict[str, Optional[float]]:
    """
    Compute descriptive statistics for elapsed times.
    Uses sample standard deviation when there are at least 2 values.
    """
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
    """
    Build a simple human-readable histogram.
    Default bins are in seconds.
    """
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
    """
    Print descriptive statistics and a simple timing distribution.
    """
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
    """
    Save timing statistics to a JSON file for later analysis.
    """
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

def is_good_output_file(path: str) -> Tuple[bool, str]:
    """
    Check whether an existing output file is usable.

    A file is considered bad when:
    - it is missing
    - it cannot be parsed as JSON
    - it contains the known GRASP null-output pattern:
        {"type": "output", "task": "sparql-qa", "output": null}

    Note:
    Invalid JSON is treated as bad here. The script can still save invalid
    stdout during a run, but on resume it will offer to re-run that row.
    """
    if not os.path.exists(path):
        return False, "missing_output_file"

    try:
        with open(path, "r", encoding="utf-8") as jf:
            data = json.load(jf)
    except Exception as e:
        return False, f"invalid_json: {e.__class__.__name__}"

    if isinstance(data, dict):
        if (
            data.get("type") == "output"
            and data.get("task") == "sparql-qa"
            and data.get("output") is None
        ):
            return False, "null_output"

    return True, ""


def detect_bad_outputs(
    rows: List[List[str]],
    colmap: Dict[str, Optional[int]],
    out_dir: str,
    processed_indices: Set[int],
) -> Tuple[Set[int], str]:
    bad_indices: Set[int] = set()
    records = []

    if not processed_indices:
        return bad_indices, ""

    for idx in sorted(processed_indices):
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
    """
    Scan existing output files and mark rows as processed when their output
    file already exists and passes validation.

    This protects against the small edge case where the program is interrupted
    after writing the output JSON but before appending the DONE row to the log.
    """
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
    - the row indices already completed or skipped
    - whether the file appears completed

    Supported log formats:
    1) New row-level format:
       DONE row 73 | file: 00073.json
       SKIP row 74 | reason: empty_or_missing_question

    2) Old batch-level format, kept for backward compatibility:
       batch 1 (#2) | rows 50-99 | files: ...
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
    """
    Record one successfully completed row immediately.

    This is what allows the script to resume from the exact last unfinished row
    after an interruption.
    """
    with open(batch_log_path, "a", encoding="utf-8") as lf:
        lf.write(f"DONE row {row_index} | file: {filename}\n")
        lf.flush()
        os.fsync(lf.fileno())


def append_row_skip(batch_log_path: str, row_index: int, reason: str) -> None:
    """
    Record one skipped row immediately.

    This prevents rows with empty or missing questions from being reconsidered
    every time the script is restarted.
    """
    with open(batch_log_path, "a", encoding="utf-8") as lf:
        lf.write(f"SKIP row {row_index} | reason: {reason}\n")
        lf.flush()
        os.fsync(lf.fileno())


def find_first_unprocessed_index(total_rows: int, processed_indices: Set[int]) -> Optional[int]:
    """
    Find the first row index that has not been processed or skipped.
    Used only for a clearer resume message.
    """
    for idx in range(total_rows):
        if idx not in processed_indices:
            return idx
    return None


# ---------------- Main CSV processing ---------------- #

def process_csv(
    csv_path: str,
    overall_pbar: tqdm,
    overall_times: List[float],
    batch_size: int = BATCH_SIZE,
) -> Dict[str, object]:
    """
    Process a single CSV file and return summary information for final reporting.
    """
    file_stem = os.path.splitext(os.path.basename(csv_path))[0]
    out_dir = os.path.join("output", file_stem)
    os.makedirs(out_dir, exist_ok=True)
    batch_log_path = os.path.join(out_dir, "batch_log.txt")

    print(f"\n=== Processing CSV: {csv_path} ===")
    print("Output directory:", out_dir)

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

    ensure_batch_log_header(batch_log_path, csv_path)

    logged_indices, is_completed_from_log = parse_batch_log(batch_log_path, total_rows)
    existing_good_indices = scan_existing_good_outputs(rows, colmap, out_dir)

    processed_indices = set(logged_indices)
    processed_indices.update(existing_good_indices)

    if existing_good_indices - logged_indices:
        recovered_count = len(existing_good_indices - logged_indices)
        print(
            f"[RESUME INFO] Recovered {recovered_count} completed row(s) "
            "from existing output files that were not in the log."
        )

    if processed_indices:
        bad_indices, bad_csv_path = detect_bad_outputs(rows, colmap, out_dir, processed_indices)
        if bad_indices:
            print(f"[CHECK INFO] Detected {len(bad_indices)} bad output JSON file(s).")
            print(f"[CHECK INFO] Details logged to: {bad_csv_path}")
            answer_bad = (
                ask_yes_no(
                    "Do you want to re-run these bad executions? [y/n]: ",
                    default="n",
                )
                if ASK_BEFORE_RERUN_BAD else "y"
            )

            if answer_bad == "y":
                processed_indices -= bad_indices
                print(f"[CHECK INFO] {len(bad_indices)} bad execution(s) will be re-run.")
            else:
                print("[CHECK INFO] Bad executions will be left as-is.")

    is_completed_effective = (
        is_completed_from_log and len(processed_indices) >= total_rows
    ) or len(processed_indices) >= total_rows

    if is_completed_effective:
        print("Detected that this CSV appears to be fully processed based on the log and output files.")
        answer = (
            ask_yes_no(
                "Files already generated. Do you want to update them? [y/n]: ",
                default="n",
            )
            if ASK_BEFORE_RERUN_COMPLETE else "n"
        )

        if answer != "y":
            print("User chose not to update existing outputs. Skipping this CSV.")
            overall_pbar.update(total_rows)
            return {
                "csv_path": csv_path,
                "total_rows": total_rows,
                "processed_rows": len(processed_indices),
                "timings": [],
            }

        processed_indices.clear()
        with open(batch_log_path, "a", encoding="utf-8") as lf:
            lf.write("\n# Restarting processing: user requested update.\n")
    else:
        first_unprocessed = find_first_unprocessed_index(total_rows, processed_indices)
        if first_unprocessed is not None:
            print(f"Resuming from row {first_unprocessed}.")
        else:
            print("No previous progress found. Starting from row 0.")

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

                stdout_text = proc.stdout
                try:
                    data = json.loads(stdout_text)
                    if isinstance(data, dict):
                        data["elapsed"] = round(elapsed, 6)
                        data["source_csv"] = csv_path
                        data["row_index"] = global_index

                        q_idx = colmap.get("question")
                        a_idx = colmap.get("answer")

                        data["question"] = (
                            row[q_idx].strip()
                            if q_idx is not None and len(row) > q_idx
                            else None
                        )
                        if a_idx is not None and len(row) > a_idx:
                            data["reference_answer"] = row[a_idx].strip()

                    json_text = json.dumps(data, ensure_ascii=False)
                except json.JSONDecodeError:
                    json_text = stdout_text

                write_text_atomic(out_file, json_text)
                append_row_done(batch_log_path, global_index, filename)

                if proc.returncode != 0:
                    sys.stderr.write(
                        f"[WARN] Row {global_index} ({filename}): grasp returned non-zero exit.\n"
                    )
                    write_text_atomic(out_file + ".stderr.txt", proc.stderr)

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


if __name__ == "__main__":
    csv_files = discover_csv_files(INPUT_DIRS)

    if not csv_files:
        print("No CSV files found in ComplexQA/ or SimpleQA/.")
        sys.exit(0)

    print("Discovered CSV files:")
    for path in csv_files:
        print(f"  - {path}")

    overall_total_rows = count_total_rows(csv_files)
    overall_times: List[float] = []
    file_summaries: List[Dict[str, object]] = []

    if overall_total_rows == 0:
        print("No data rows found across discovered CSV files.")
        sys.exit(0)

    with tqdm(
        total=overall_total_rows,
        desc="Overall",
        unit="row",
        position=0,
    ) as overall_pbar:
        for csv_path in csv_files:
            summary = process_csv(
                csv_path=csv_path,
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
#!/usr/bin/env python3
"""
Script summary
--------------
This script runs GRASP (or another LLM-based pipeline) over one or more CSV datasets
and writes one JSON output file per row.

Key features:
- Supports the CompMix CSV schema with at least:
      question_id, question, entity_id1, ...
- For each row, builds the LLM input by concatenating:
      input_text = "<question> <entity_id1>"
  so that the (first) entity ID is explicitly included as a token for the model.
- Output filenames are:
      <question_id>.json
  e.g. question_id = 97 -> filename: 97.json
- Processes rows in batches to rehearse large-scale workflow:
      BATCH_SIZE = 2
  After each batch:
    * All rows in that batch have been processed and written to disk.
    * A `batch_log.txt` file is appended with a line describing which rows and
      files were produced for that batch.
    * A Git commit is created and pushed:
          git add -A
          git commit -m "finished kth batch, generated X/Y JSON files"
          git push origin main
      Any git errors are logged but DO NOT stop the main processing.
- Uses a tqdm progress bar to show overall processing progress.
- Prints a concise summary at the end with counts and output directory.
- The code is structured so that multiple CSVs can be handled in the future:
      compmix-test.csv, dev_set.csv, test_set.csv, train_set.csv
  For now only top1000.csv is active.

Resume & quality checks:
- On each run, the script checks the output dir and `batch_log.txt`.
  * If it finds that all rows are already processed, it prompts:
        "file already generated, do you want to update them? [y/n]"
    If you answer:
        - 'n': the script skips re-processing that CSV.
        - 'y': the script re-processes from scratch (overwriting outputs).
  * If it finds that only part of the CSV was processed, it resumes from the
    first unprocessed row, based on `batch_log.txt`.
- It also scans all processed JSON outputs to detect "bad executions", defined as:
    - JSON missing / invalid, or
    - JSON object with type="output", task="sparql-qa", and output is null.
  * These are logged to `bad_executions.csv`.
  * You are asked:
        "Detected N bad executions (...) Do you want to re-run them? [y/n]"
    If:
        - 'y': those rows are treated as unprocessed and will be re-run.
        - 'n': they are ignored and the script continues from the last row.
"""

import csv
import json
import os
import subprocess
import sys
import time
from typing import List, Set, Tuple

from tqdm import tqdm

# ---------------- Configuration ---------------- #

# In the future, you can process multiple CSVs by uncommenting / adding paths here.
CSV_FILES: List[str] = [
    # "data/CompMix/compmix-test.csv",
    "data/CompMix/top1000.csv",
    "data/CompMix/bottom1000.csv",
    # "data/CompMix/train_set.csv",
]

# Batch size for rehearsal of large-scale processing
BATCH_SIZE = 50




# ---------------- Git helper ---------------- #

def run_git_after_batch(batch_number: int, processed_so_far: int, total_rows: int) -> None:
    """
    After each finished batch, automatically:
      - git add -A
      - git commit -m "finished <batch_number>th batch, generated <processed_so_far>/<total_rows> JSON files"
      - git push origin main

    Any errors in git commands are printed as warnings but do NOT stop the main processing.
    """
    commit_msg = (
        f"finished {batch_number}th batch, "
        f"generated {processed_so_far}/{total_rows} JSON files"
    )

    try:
        # Stage all changes
        add_proc = subprocess.run(
            ["git", "add", "-A"],
            capture_output=True,
            text=True,
        )
        if add_proc.returncode != 0:
            sys.stderr.write("[GIT WARN] 'git add -A' failed:\n")
            if add_proc.stderr:
                sys.stderr.write(add_proc.stderr + "\n")
            return  # If we can't add, no point committing

        # Commit (may fail if there is nothing to commit)
        commit_proc = subprocess.run(
            ["git", "commit", "-m", commit_msg],
            capture_output=True,
            text=True,
        )
        if commit_proc.returncode != 0:
            # Typically "nothing to commit" – not fatal
            sys.stderr.write("[GIT INFO] 'git commit' did not create a commit:\n")
            if commit_proc.stderr:
                sys.stderr.write(commit_proc.stderr + "\n")
            return

        # Push to origin main
        push_proc = subprocess.run(
            ["git", "push", "origin", "main"],
            capture_output=True,
            text=True,
        )
        if push_proc.returncode != 0:
            sys.stderr.write("[GIT WARN] 'git push origin main' failed:\n")
            if push_proc.stderr:
                sys.stderr.write(push_proc.stderr + "\n")

    except FileNotFoundError:
        # git not installed or not in PATH
        sys.stderr.write("[GIT WARN] 'git' command not found. Skipping git operations.\n")


# ---------------- Helpers ---------------- #

def get_question_id_for_row(row: List[str], global_index: int, question_id_idx: int) -> str:
    """
    Compute the question_id for a given row, using the same logic everywhere:
      - If the question_id column exists and is non-empty, use it (stripped).
      - Otherwise, fall back to a zero-padded row index.
    """
    if len(row) > question_id_idx:
        question_id = row[question_id_idx].strip()
    else:
        question_id = f"{global_index:03d}"

    if not question_id:
        question_id = f"{global_index:03d}"

    return question_id


def detect_bad_outputs(
    rows: List[List[str]],
    question_id_idx: int,
    out_dir: str,
    processed_indices: Set[int],
) -> Tuple[Set[int], str]:
    """
    Examine outputs for rows in processed_indices, read the corresponding JSON file
    for each, and classify as "bad execution" if:

        - The JSON file is missing, OR
        - The JSON cannot be parsed, OR
        - The JSON is an object with:
              type == "output", task == "sparql-qa", and output is None

    Returns:
        bad_indices: set of row indices with bad outputs
        csv_path:    path to a CSV file listing them (if any bad found),
                    or an empty string if no bad rows were found.
    """
    bad_indices: Set[int] = set()
    records = []

    if not processed_indices:
        return bad_indices, ""

    for idx in sorted(processed_indices):
        if idx < 0 or idx >= len(rows):
            continue

        row = rows[idx]
        question_id = get_question_id_for_row(row, idx, question_id_idx)
        filename = f"{question_id}.json"
        out_file = os.path.join(out_dir, filename)

        reason = None

        if not os.path.exists(out_file):
            reason = "missing_output_file"
        else:
            try:
                with open(out_file, "r", encoding="utf-8") as jf:
                    data = json.load(jf)
            except Exception as e:
                reason = f"invalid_json: {e.__class__.__name__}"
            else:
                if isinstance(data, dict):
                    # Detect the explicit "bad execution" pattern:
                    # {"type": "output", "task": "sparql-qa", "output": null, ...}
                    if (
                        data.get("type") == "output"
                        and data.get("task") == "sparql-qa"
                        and data.get("output") is None
                    ):
                        reason = "null_output"

        if reason is not None:
            bad_indices.add(idx)
            records.append((idx, question_id, filename, reason))

    if not records:
        return bad_indices, ""

    bad_csv_path = os.path.join(out_dir, "bad_executions.csv")
    with open(bad_csv_path, "w", newline="", encoding="utf-8") as cf:
        writer = csv.writer(cf)
        writer.writerow(["row_index", "question_id", "filename", "reason"])
        for rec in records:
            writer.writerow(rec)

    return bad_indices, bad_csv_path


# ---------------- Batch log / resume helpers ---------------- #

def parse_batch_log(batch_log_path: str, total_rows: int) -> Tuple[Set[int], bool]:
    """
    Parse an existing batch_log.txt (if any) and return:
        processed_indices: set of 0-based row indices that are already done
        is_completed: whether the log indicates the run was completed

    Completion is detected either by a '# COMPLETED' line, or by seeing
    that the highest index in the log is >= total_rows - 1.
    """
    processed_indices: Set[int] = set()
    is_completed = False

    if not os.path.exists(batch_log_path):
        return processed_indices, is_completed

    with open(batch_log_path, "r", encoding="utf-8") as lf:
        for line in lf:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                # Comments or blank lines
                if "COMPLETED" in stripped.upper():
                    is_completed = True
                continue

            # Typical line format:
            # "batch 338 (#339) | rows 676-677 | files: 3609.json, 1376.json"
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

    # If there is no explicit COMPLETED marker, but we have processed
    # indices up to the last row, consider it complete.
    if not is_completed and processed_indices:
        if max(processed_indices) >= total_rows - 1:
            is_completed = True

    return processed_indices, is_completed


def ensure_batch_log_header(batch_log_path: str, csv_path: str) -> None:
    """
    Create batch_log.txt with a standard header if it doesn't exist.
    If it already exists, leave it alone.
    """
    if os.path.exists(batch_log_path):
        return

    with open(batch_log_path, "w", encoding="utf-8") as lf:
        lf.write(f"# Batch log for {csv_path}\n")
        lf.write("# Each line: batch_index | row_indices | filenames\n\n")


# ---------------- Main CSV processing ---------------- #

def process_csv(csv_path: str, batch_size: int = BATCH_SIZE) -> None:
    """
    Process a single CSV file with the expected CompMix schema:

        question_id, question, entity_id1, entity_label1, ...

    For each row:
      * Build input_text = "<question> <entity_id1>" (if entity_id1 exists).
      * Run GRASP, feeding input_text via stdin.
      * Save GRASP stdout as a JSON file in:
            output/<parent_folder>/<file_stem>/
        with filename:
            <question_id>.json

      * The script also measures per-row elapsed time and overwrites the `elapsed`
        field in the JSON output (rounded to 3 decimals).

    Batching:
      * Rows are processed in chunks of `batch_size`.
      * After each batch:
          - Append a line to batch_log.txt summarizing that batch.
          - Run git add/commit/push (non-fatal if it fails).

    Resume behavior:
      * If batch_log.txt indicates some rows were already processed, the script
        resumes from the first unprocessed row.
      * If all rows were processed, the script asks whether to re-run and update.

    Bad execution handling:
      * For all currently processed rows, JSON outputs are scanned.
      * Bad executions (missing/invalid JSON, or null output for sparql-qa)
        are logged to bad_executions.csv.
      * You can choose to re-run just those rows or ignore them.
    """

    # Build output directory: output/<folder>/<file_stem>
    folder = os.path.basename(os.path.dirname(csv_path))          # e.g. "CompMix"
    file_stem = os.path.splitext(os.path.basename(csv_path))[0]   # e.g. "top1000"
    out_dir = os.path.join("output", folder, file_stem)
    os.makedirs(out_dir, exist_ok=True)

    batch_log_path = os.path.join(out_dir, "batch_log.txt")

    print(f"\n=== Processing CSV: {csv_path} ===")
    print("Output directory:", out_dir)

    # ---------------- Load CSV ---------------- #

    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        raw_header = next(reader, None)

        if raw_header is None:
            raise ValueError(f"CSV file '{csv_path}' appears to be empty or missing a header row.")

        # Normalize header cells: strip whitespace + BOM
        header = [h.strip().lstrip("\ufeff") for h in raw_header]

        if header is None:
            raise ValueError(f"CSV file '{csv_path}' appears to be empty or missing a header row.")

        # Locate relevant columns by name, robust to column order changes.
        # question_id must exist by name.
        try:
            question_id_idx = header.index("question_id")
        except ValueError:
            raise ValueError(
                f"'question_id' column not found in CSV header: {header}"
            )

        # question should exist by name; if not, we bail out.
        try:
            question_idx = header.index("question")
        except ValueError:
            raise ValueError(
                f"'question' column not found in CSV header: {header}"
            )

        # Entity: your CSV has entity_id1, entity_id2, ...
        # Pick the first entity_id* column if it exists.
        entity_id_idx = None
        for i, col_name in enumerate(header):
            if col_name.startswith("entity_id"):
                entity_id_idx = i
                break

        rows = list(reader)

    total_rows = len(rows)
    if total_rows == 0:
        print("No data rows found in CSV. Skipping.")
        return

    # Make sure batch log has a header if it doesn't exist
    ensure_batch_log_header(batch_log_path, csv_path)

    # ---------------- Check existing progress ---------------- #

    processed_indices, is_completed_from_log = parse_batch_log(batch_log_path, total_rows)

    # 1) Scan for bad executions among already processed rows
    bad_indices: Set[int] = set()
    bad_csv_path = ""

    if processed_indices:
        bad_indices, bad_csv_path = detect_bad_outputs(
            rows=rows,
            question_id_idx=question_id_idx,
            out_dir=out_dir,
            processed_indices=processed_indices,
        )

        if bad_indices:
            print(
                f"[CHECK INFO] Detected {len(bad_indices)} bad output JSON file(s) in {out_dir}."
            )
            print(f"[CHECK INFO] Details logged to: {bad_csv_path}")
            answer_bad = ""
            try:
                while answer_bad not in ("y", "n"):
                    answer_bad = input(
                        "Do you want to re-run these bad executions? [y/n]: "
                    ).strip().lower()
            except EOFError:
                print(
                    "[CHECK INFO] No interactive input available; defaulting to 'n' "
                    "(not re-running bad executions)."
                )
                answer_bad = "n"

            if answer_bad == "y":
                # Treat bad rows as not processed; they will be re-run.
                processed_indices = processed_indices - bad_indices
                print(
                    f"[CHECK INFO] {len(bad_indices)} bad execution(s) will be re-run in this run."
                )
            else:
                # Ignore bad executions and just continue from the last row.
                print(
                    "[CHECK INFO] Bad executions will be left as-is and ignored in this run."
                )

    # 2) Effective completion status after possibly unmarking bad rows
    is_completed_effective = (
        is_completed_from_log and processed_indices and max(processed_indices) >= total_rows - 1
    )

    if is_completed_effective:
        # Already processed all rows before; ask user if they want to update.
        print("Detected that this CSV appears to be fully processed based on batch_log.txt.")
        answer = ""
        try:
            while answer not in ("y", "n"):
                answer = input("file already generated, do you want to update them? [y/n]: ").strip().lower()
        except EOFError:
            # Non-interactive environment; be conservative and skip updating.
            print("No interactive input available; defaulting to 'n' (not updating).")
            answer = "n"

        if answer != "y":
            print("User chose not to update existing outputs. Skipping this CSV.")
            return

        # User wants to re-run everything.
        print("User requested to update existing outputs: re-running all rows from scratch.")
        processed_indices.clear()
        with open(batch_log_path, "a", encoding="utf-8") as lf:
            lf.write("\n# Restarting processing: user requested update.\n")
    else:
        if processed_indices:
            last_idx = max(processed_indices)
            print(
                f"Resuming from row {last_idx + 1} "
                f"(based on batch_log.txt and bad-output scan: processed rows up to {last_idx})."
            )
        else:
            print("No previous progress found for this CSV. Starting from row 0.")

    # ---------------- Processing Loop (Batched) ---------------- #

    processed_count = len(processed_indices)

    with tqdm(
        total=total_rows,
        desc="Processing questions",
        unit="row",
        initial=processed_count,
    ) as pbar:
        for batch_start in range(0, total_rows, batch_size):
            batch_rows = rows[batch_start: batch_start + batch_size]
            batch_index = batch_start // batch_size      # 0-based
            batch_number = batch_index + 1               # 1-based, for human-readable logs
            batch_filenames = []
            batch_row_indices = []

            # Process each row within the current batch
            for i, row in enumerate(batch_rows):
                global_index = batch_start + i  # row index within the full CSV

                # Skip rows that were already processed in a previous run
                if global_index in processed_indices:
                    pbar.update(1)
                    continue

                if not row:
                    pbar.update(1)
                    continue

                # Extract question
                if len(row) <= question_idx:
                    pbar.update(1)
                    continue

                question = row[question_idx].strip()
                if not question:
                    pbar.update(1)
                    continue

                # Determine question_id using helper
                question_id = get_question_id_for_row(row, global_index, question_id_idx)

                # Extract first entity_id* if present
                entity_id = ""
                if entity_id_idx is not None and len(row) > entity_id_idx:
                    entity_id = row[entity_id_idx].strip()

                # Build input text: "question entity_id" (if entity_id exists)
                if entity_id:
                    input_text = f"{question} {entity_id}"
                else:
                    input_text = question

                # Build filename: <question_id>.json
                filename = f"{question_id}.json"
                out_file = os.path.join(out_dir, filename)

                # Run GRASP (or your LLM pipeline), feeding `input_text` via stdin
                start_time = time.perf_counter()
                proc = subprocess.run(
                    ["bash", "-lc", "grasp run configs/run.yaml"],
                    input=input_text,
                    capture_output=True,
                    text=True,
                )
                elapsed = time.perf_counter() - start_time

                stdout_text = proc.stdout

                # Try to parse the output as JSON and fix/overwrite "elapsed"
                try:
                    data = json.loads(stdout_text)

                    # Overwrite elapsed with our per-row measurement (in seconds, rounded)
                    if isinstance(data, dict):
                        data["elapsed"] = round(elapsed, 3)

                    json_text = json.dumps(data, ensure_ascii=False)
                except json.JSONDecodeError:
                    # If it's not valid JSON for some reason, just dump raw stdout
                    json_text = stdout_text

                # Write final JSON text to file
                with open(out_file, "w", encoding="utf-8") as out:
                    out.write(json_text)

                # If non-zero exit code, log warning and stderr
                if proc.returncode != 0:
                    sys.stderr.write(
                        f"[WARN] Row {global_index} ({filename}): grasp returned non-zero exit.\n"
                    )
                    with open(out_file + ".stderr.txt", "w", encoding="utf-8") as errf:
                        errf.write(proc.stderr)

                processed_count += 1
                pbar.update(1)

                batch_filenames.append(filename)
                batch_row_indices.append(global_index)
                processed_indices.add(global_index)

            # ----- End of batch: log batch info and run git ----- #
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

                # Run git add/commit/push for this batch (non-fatal on failure)
                run_git_after_batch(
                    batch_number=batch_number,
                    processed_so_far=processed_count,
                    total_rows=total_rows,
                )

    # ---------------- Final Summary + completion marker ---------------- #

    # If we reached here, we finished iterating over all rows in this run.
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


if __name__ == "__main__":
    # For now, we only run on top1000.csv.
    # In the future, you can uncomment additional CSVs in CSV_FILES above.
    for csv_path in CSV_FILES:
        if not os.path.exists(csv_path):
            print(f"CSV not found, skipping: {csv_path}")
            continue
        process_csv(csv_path, batch_size=BATCH_SIZE)

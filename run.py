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

Resume & “already generated?” behavior:
- On each run, the script checks the output dir and `batch_log.txt`.
  * If it finds that all rows are already processed, it prompts:
        "file already generated, do you want to update them? [y/n]"
    If you answer:
        - 'n': the script skips re-processing that CSV.
        - 'y': the script re-processes from scratch (overwriting outputs).
  * If it finds that only part of the CSV was processed, it resumes from
    the first unprocessed row, based on `batch_log.txt`.

Bad-output scanning:
- Before deciding resume/finished, the script scans every JSON file in the output dir.
- A JSON is treated as a bad execution if:
    * It contains the pattern:
          {"type": "output", "task": "sparql-qa", "output": null
      OR
    * It is invalid JSON (truncated, etc.)
      OR
    * It is a dict with `type == "output"` and `task == "sparql-qa"` and:
          - `output` is None, OR
          - `output` is a dict without a "sparql" key.
- Any rows with bad outputs are logged and automatically re-run in this run.
"""

import csv
import json
import os
import subprocess
import sys
import time
from typing import Dict, List, Set, Tuple

from tqdm import tqdm

# ---------------- Configuration ---------------- #

CSV_FILES: List[str] = [
    # "data/CompMix/compmix-test.csv",
    "data/CompMix/top1000.csv",
    # "data/CompMix/test_set.csv",
    # "data/CompMix/train_set.csv",
]

# Batch size for rehearsal of large-scale processing
BATCH_SIZE = 2


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


# ---------------- Bad-output detection ---------------- #

def is_bad_output_file(path: str) -> bool:
    """
    Return True if the JSON at `path` looks like a bad execution.

    Rules:
    - If file cannot be read or parsed as JSON -> bad.
    - If text contains the specific pattern
          {"type": "output", "task": "sparql-qa", "output": null
      -> bad.
    - If parsed JSON is a dict with:
          type == "output", task == "sparql-qa"
      and either:
          output is None, or
          output is a dict that does NOT contain a "sparql" key
      -> bad.
    """
    try:
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
    except OSError as e:
        sys.stderr.write(f"[CHECK WARN] Could not read JSON file {path}: {e}\n")
        return True  # treat unreadable as bad

    # Quick textual check for your exact "bad execution" pattern
    if (
        '"type": "output"' in text
        and '"task": "sparql-qa"' in text
        and '"output": null' in text
    ):
        return True

    # Try to parse as JSON
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        # Truncated or invalid JSON – treat as bad
        return True

    if not isinstance(data, dict):
        # Unexpected structure; be conservative and treat non-dicts as bad
        return True

    if data.get("type") == "output" and data.get("task") == "sparql-qa":
        out = data.get("output", None)
        if out is None:
            return True
        if isinstance(out, dict) and "sparql" not in out:
            return True

    return False


def find_bad_outputs(
    out_dir: str,
    index_to_filename: Dict[int, str],
) -> Set[int]:
    """
    Scan every output JSON in out_dir (based on index_to_filename) and
    return the set of row indices whose JSON is considered a bad execution.
    """
    bad_indices: Set[int] = set()

    for idx, fname in index_to_filename.items():
        json_path = os.path.join(out_dir, fname)
        if not os.path.exists(json_path):
            # Missing file is not "bad output" – it just hasn't been created yet.
            continue

        if is_bad_output_file(json_path):
            bad_indices.add(idx)

    if bad_indices:
        sys.stderr.write(
            f"[CHECK INFO] Detected {len(bad_indices)} bad output JSON file(s) "
            f"in {out_dir}. They will be re-run.\n"
        )

    return bad_indices


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

    Bad-output behavior:
      * Before deciding resume/finished, scan every output JSON and mark
        "bad" executions to be re-run automatically in this run.
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

    # Precompute filename mapping for all rows based on question_id
    index_to_filename: Dict[int, str] = {}
    for global_index, row in enumerate(rows):
        if not row:
            # We'll handle empty rows later; filename still based on index fallback.
            qid = f"{global_index:03d}"
        else:
            if len(row) > question_id_idx:
                qid = row[question_id_idx].strip() or f"{global_index:03d}"
            else:
                qid = f"{global_index:03d}"
        index_to_filename[global_index] = f"{qid}.json"

    # Make sure batch log has a header if it doesn't exist
    ensure_batch_log_header(batch_log_path, csv_path)

    # ---------------- Check existing progress from log ---------------- #

    processed_indices, is_completed = parse_batch_log(batch_log_path, total_rows)

    # ---------------- Scan for bad outputs and mark them for re-run ---------------- #

    bad_indices = find_bad_outputs(out_dir, index_to_filename)

    if bad_indices:
        # Bad outputs should be re-run: remove them from the "already processed" set
        processed_indices -= bad_indices

        # If we had previously thought the run was completed, but now see bad outputs,
        # we must treat it as NOT completed.
        if len(processed_indices) < total_rows:
            is_completed = False

        # Log which indices are being re-run
        with open(batch_log_path, "a", encoding="utf-8") as lf:
            lf.write(
                f"# Detected {len(bad_indices)} bad outputs; "
                f"rows {min(bad_indices)}-{max(bad_indices)} will be re-run.\n"
            )

    # ---------------- Decide whether to resume, restart, or skip ---------------- #

    if is_completed and len(processed_indices) == total_rows:
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
            print("No previous progress found for this CSV (or everything needs re-run). Starting from row 0.")

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

                # Skip rows that were already processed and not marked as bad
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

                # Extract question_id from its named column
                if len(row) > question_id_idx:
                    question_id = row[question_id_idx].strip()
                else:
                    # Fallback: zero-padded index if something is weird
                    question_id = f"{global_index:03d}"

                if not question_id:
                    # If the cell is empty, also fallback to index
                    question_id = f"{global_index:03d}"

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

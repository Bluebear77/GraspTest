#!/usr/bin/env python3
"""
Script summary
--------------
This script runs GRASP (or another LLM-based pipeline) over one or more CSV datasets
and writes one JSON output file per row.

Key features:
- Supports the new CSV schema with at least:
      question_id, question, entity_id, ...
- For each row, builds the LLM input by concatenating:
      input_text = "<question> <entity_id>"
  so that the entity ID is explicitly included as a token for the model.
- Output filenames start with `question_id`, followed by the entity_id (when it
  looks like a Wikidata ID such as Q260725). Example:
      question_id = 8750, entity_id = Q260725
      -> filename: 8750_Q260725.json
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
- Uses a tqdm progress bar to show overall processing progress.
- Prints a concise summary at the end with counts and output directory.
- The code is structured so that multiple CSVs can be handled in the future:
      compmix-test.csv, dev_set.csv, test_set.csv, train_set.csv
  For now only compmix-test.csv is active, others are sketched as comments.
"""

import csv
import os
import subprocess
import sys
from typing import List

from tqdm import tqdm

# ---------------- Configuration ---------------- #

# In the future, you can process multiple CSVs by uncommenting / adding paths here.
CSV_FILES: List[str] = [
    # "data/CompMix/compmix-test.csv",
     "data/CompMix/top1000.csv",
    # "data/CompMix/test_set.csv",
    # "data/CompMix/train_set.csv",
]

# Batch size for rehearsal of large-scale processing
BATCH_SIZE = 2


def run_git_after_batch(batch_number: int, processed_so_far: int, total_rows: int) -> None:
    """
    After each finished batch, automatically:
      - git add -A
      - git commit -m "finished <batch_number>th batch, generated <processed_so_far>/<total_rows> JSON files"
      - git push origin main

    Any errors in git commands are printed as warnings but do not stop the main processing.
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
            sys.stderr.write(push_proc.stderr + "\n")

    except FileNotFoundError:
        # git not installed or not in PATH
        sys.stderr.write("[GIT WARN] 'git' command not found. Skipping git operations.\n")


def process_csv(csv_path: str, batch_size: int = BATCH_SIZE) -> None:
    """
    Process a single CSV file with the expected CompMix schema:

        question_id, question, entity_id, entity_label, answer_id, ...

    For each row:
      * Build input_text = "<question> <entity_id>".
      * Run GRASP, feeding input_text via stdin.
      * Save GRASP stdout as a JSON file in:
            output/<parent_folder>/<file_stem>/
        with filename:
            <question_id>_<entity_id>.json   (if entity_id looks like Qxxxx)
        or:
            <question_id>.json               (fallback if no valid entity_id)

    Batching:
      * Rows are processed in chunks of `batch_size`.
      * After each batch:
          - Append a line to batch_log.txt summarizing that batch.
          - Run git add/commit/push.
    """

    # Build output directory: output/<folder>/<file_stem>
    folder = os.path.basename(os.path.dirname(csv_path))          # e.g. "CompMix"
    file_stem = os.path.splitext(os.path.basename(csv_path))[0]   # e.g. "compmix-test"
    out_dir = os.path.join("output", folder, file_stem)
    os.makedirs(out_dir, exist_ok=True)

    batch_log_path = os.path.join(out_dir, "batch_log.txt")

    print(f"\n=== Processing CSV: {csv_path} ===")
    print("Output directory:", out_dir)

    # ---------------- Load CSV ---------------- #

    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader, None)  # read header row

        if header is None:
            raise ValueError(f"CSV file '{csv_path}' appears to be empty or missing a header row.")

        # Locate relevant columns by name
        try:
            question_idx = header.index("question")
        except ValueError:
            # Fallback for legacy format (no "question" header, just a single column).
            question_idx = 0

        try:
            question_id_idx = header.index("question_id")
        except ValueError:
            question_id_idx = None

        try:
            entity_id_idx = header.index("entity_id")
        except ValueError:
            entity_id_idx = None

        rows = list(reader)

    total_rows = len(rows)
    if total_rows == 0:
        print("No data rows found in CSV. Skipping.")
        return

    # Clear or initialize the batch log for this CSV
    with open(batch_log_path, "w", encoding="utf-8") as lf:
        lf.write(f"# Batch log for {csv_path}\n")
        lf.write("# Each line: batch_index | row_indices | filenames\n\n")

    # ---------------- Processing Loop (Batched) ---------------- #

    processed_count = 0

    with tqdm(total=total_rows, desc="Processing questions", unit="row") as pbar:
        for batch_start in range(0, total_rows, batch_size):
            batch_rows = rows[batch_start : batch_start + batch_size]
            batch_index = batch_start // batch_size      # 0-based
            batch_number = batch_index + 1               # 1-based, for human-readable logs
            batch_filenames = []
            batch_row_indices = []

            # Process each row within the current batch
            for i, row in enumerate(batch_rows):
                global_index = batch_start + i  # row index within the full CSV

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

                # Extract question_id if available; otherwise fall back to zero-padded index
                if question_id_idx is not None and len(row) > question_id_idx:
                    question_id = row[question_id_idx].strip()
                else:
                    question_id = f"{global_index:03d}"

                # Extract entity_id if available
                entity_id = ""
                if entity_id_idx is not None and len(row) > entity_id_idx:
                    entity_id = row[entity_id_idx].strip()

                # Build input text: "question entity_id"
                # Example:
                #   "Where was Megan Rapinoe born? Q260725"
                if entity_id:
                    input_text = f"{question} {entity_id}"
                else:
                    input_text = question

                # Determine QID for filename (prefer entity_id when it looks like Qxxxx)
                qid_for_filename = ""
                if entity_id.startswith("Q") and entity_id[1:].isdigit():
                    qid_for_filename = entity_id

                # Build filename:
                #   <question_id>_<qid>.json  (when qid_for_filename is set)
                #   <question_id>.json       (fallback)
                if qid_for_filename:
                    filename = f"{question_id}_{qid_for_filename}.json"
                else:
                    filename = f"{question_id}.json"

                out_file = os.path.join(out_dir, filename)

                # Run GRASP (or your LLM pipeline), feeding `input_text` via stdin
                proc = subprocess.run(
                    ["bash", "-lc", "grasp run configs/run.yaml"],
                    input=input_text,
                    capture_output=True,
                    text=True,
                )

                # Write stdout to JSON file
                with open(out_file, "w", encoding="utf-8") as out:
                    out.write(proc.stdout)

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

                # Run git add/commit/push for this batch
                run_git_after_batch(
                    batch_number=batch_number,
                    processed_so_far=processed_count,
                    total_rows=total_rows,
                )

    # ---------------- Final Summary ---------------- #

    print("\nProcessing complete for this CSV.")
    print(f"  CSV path        : {csv_path}")
    print(f"  Total rows      : {total_rows}")
    print(f"  Processed rows  : {processed_count}")
    print(f"  Batch size      : {batch_size}")
    print(f"  Output directory: {out_dir}")
    print(f"  Batch log file  : {batch_log_path}")


if __name__ == "__main__":
    # For now, we only run on compmix-test.csv.
    # In the future, you can uncomment additional CSVs in CSV_FILES above.
    for csv_path in CSV_FILES:
        if not os.path.exists(csv_path):
            print(f"CSV not found, skipping: {csv_path}")
            continue
        process_csv(csv_path, batch_size=BATCH_SIZE)

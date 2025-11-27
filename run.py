#!/usr/bin/env python3
"""
Script summary
--------------
This script runs GRASP (or another LLM-based pipeline) over a CSV dataset and writes
one JSON output file per row.

Key features:
- Reads a CSV file whose header includes at least 'question' and 'entity_id'.
- For each row, builds the LLM input by concatenating:
      input_text = "<question> <entity_id>"
  so that the entity ID is explicitly included as a token for the model.
- Uses 'entity_id' as the QID in the output filenames when it looks like a Wikidata ID
  (e.g., Q12345), falling back to an index-only filename otherwise.
- Processes the dataset in batches (batch_size = 2). After each batch:
    * All rows in the batch are processed and written to disk.
    * A 'checkpoint.txt' file is updated with the index of the last processed row.
  This simulates a large-scale run where you want regular checkpoints for resuming.
- Uses a tqdm progress bar to show processing progress over all rows.
- Prints a concise summary at the end with counts and output directory.
"""

import csv
import os
import subprocess
import sys
from tqdm import tqdm

# ---------------- Configuration ---------------- #

# Path to your CSV file (new format with question, entity_id, etc.)
csv_path = "data/CompMix/compmix-test.csv"

# Batch size for rehearsal of large-scale processing
BATCH_SIZE = 2

# Build output directory: output/CompMix/compmix-test
folder = os.path.basename(os.path.dirname(csv_path))          # e.g. "CompMix"
file_stem = os.path.splitext(os.path.basename(csv_path))[0]   # e.g. "compmix-test"
out_dir = os.path.join("output", folder, file_stem)
os.makedirs(out_dir, exist_ok=True)

checkpoint_path = os.path.join(out_dir, "checkpoint.txt")

print("Output directory:", out_dir)

# ---------------- Load CSV ---------------- #

with open(csv_path, newline='', encoding='utf-8') as f:
    reader = csv.reader(f)
    header = next(reader, None)  # read header row

    if header is None:
        raise ValueError("CSV file appears to be empty or missing a header row.")

    # Try to locate relevant columns by name
    try:
        question_idx = header.index("question")
    except ValueError:
        # Fallback: assume first column is question (legacy format)
        question_idx = 0

    # entity_id is optional in header if running on older format
    try:
        entity_id_idx = header.index("entity_id")
    except ValueError:
        entity_id_idx = None

    rows = list(reader)

total_rows = len(rows)
if total_rows == 0:
    print("No data rows found in CSV. Exiting.")
    sys.exit(0)

# ---------------- Processing Loop (Batched) ---------------- #

processed_count = 0

with tqdm(total=total_rows, desc="Processing questions", unit="row") as pbar:
    for batch_start in range(0, total_rows, BATCH_SIZE):
        batch_rows = rows[batch_start : batch_start + BATCH_SIZE]

        # Process each row within the current batch
        for i, row in enumerate(batch_rows):
            global_index = batch_start + i

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

            # Extract entity_id if available; otherwise empty string
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
            qid = ""
            if entity_id.startswith("Q") and entity_id[1:].isdigit():
                qid = entity_id

            # Fallback: try to get a trailing QID from the question text (legacy behavior)
            if not qid:
                parts = question.split()
                if parts:
                    last = parts[-1]
                    if last.startswith("Q") and last[1:].isdigit():
                        qid = last

            # Build filename
            if qid:
                filename = f"{global_index:03d}_{qid}.json"
            else:
                filename = f"{global_index:03d}.json"

            out_file = os.path.join(out_dir, filename)

            # Run GRASP (or your LLM pipeline)
            proc = subprocess.run(
                ["bash", "-lc", f'printf %s "{input_text}" | grasp run configs/run.yaml'],
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

        # ----- End of batch: write checkpoint ----- #
        last_processed_index = batch_start + len(batch_rows) - 1
        with open(checkpoint_path, "w", encoding="utf-8") as ck:
            ck.write(str(last_processed_index))

# ---------------- Final Summary ---------------- #

print("\nProcessing complete.")
print(f"  CSV path        : {csv_path}")
print(f"  Total rows      : {total_rows}")
print(f"  Processed rows  : {processed_count}")
print(f"  Batch size      : {BATCH_SIZE}")
print(f"  Output directory: {out_dir}")
print(f"  Checkpoint file : {checkpoint_path} (last processed row index)")

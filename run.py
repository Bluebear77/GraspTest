#!/usr/bin/env python3

import csv
import os
import subprocess
import sys
from tqdm import tqdm

# Path to your CSV file
csv_path = "data/CompMix/compmix-test.csv"

# Build output directory: output/CompMix/compmix-test
folder = os.path.basename(os.path.dirname(csv_path))          # "CompMix"
file_stem = os.path.splitext(os.path.basename(csv_path))[0]   # "compmix-test"
out_dir = os.path.join("output", folder, file_stem)
os.makedirs(out_dir, exist_ok=True)

print("Output directory:", out_dir)

# Load all rows first so we know total count
with open(csv_path, newline='', encoding='utf-8') as f:
    reader = csv.reader(f)
    header = next(reader, None)  # skip header
    rows = list(reader)          # read all remaining rows

# Now tqdm will know the total length
for row_index, row in enumerate(
    tqdm(rows, desc="Processing questions", unit="row", total=len(rows)),
    start=0
):
    if not row:
        continue

    question = row[0].strip()
    if not question:
        continue

    # Extract trailing QID for naming convenience
    qid = ""
    parts = question.split()
    if parts:
        last = parts[-1]
        if last.startswith("Q") and last[1:].isdigit():
            qid = last

    # Build filename
    if qid:
        filename = f"{row_index:03d}_{qid}.json"
    else:
        filename = f"{row_index:03d}.json"

    out_file = os.path.join(out_dir, filename)

    # Run GRASP
    proc = subprocess.run(
        ["bash", "-lc", f'printf %s "{question}" | grasp run configs/run.yaml'],
        capture_output=True,
        text=True,
    )

    # Write output
    with open(out_file, "w", encoding="utf-8") as out:
        out.write(proc.stdout)

    if proc.returncode != 0:
        sys.stderr.write(
            f"[WARN] Row {row_index} ({filename}): grasp returned non-zero exit.\n"
        )
        with open(out_file + ".stderr.txt", "w", encoding="utf-8") as errf:
            errf.write(proc.stderr)

print("Done.")

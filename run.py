#!/usr/bin/env python3

"""
Batch-process questions from a CSV file by sending each question to the
`grasp run` command. Output directory is automatically determined by
mirroring the input CSV path inside the top-level 'output' folder.
"""

import csv
import os
import subprocess
import sys
from tqdm import tqdm

# Path to your CSV file
csv_path = "data/CompMix/compmix-test.csv"

# Extract folder name and file stem (without extension)
folder = os.path.basename(os.path.dirname(csv_path))   # "CompMix"
file_stem = os.path.splitext(os.path.basename(csv_path))[0]  # "compmix-test"

# Build output directory: output/CompMix/compmix-test
out_dir = os.path.join("output", folder, file_stem)
os.makedirs(out_dir, exist_ok=True)

print("Output directory:", out_dir)


# Open the CSV file for reading
with open(csv_path, newline='', encoding='utf-8') as f:
    reader = csv.reader(f)
    header = next(reader, None)  # Skip header row if present
    rows = list(reader)

    for i, row in enumerate(tqdm(rows, desc="Processing questions", unit="row"), start=1):
        if not row or not row[0].strip():
            continue

        question = row[0].strip()
        out_file = os.path.join(out_dir, f"{i:03d}.json")

        proc = subprocess.run(
            ["bash", "-lc", f'printf %s "{question}" | grasp run configs/run.yaml'],
            capture_output=True, text=True
        )

        with open(out_file, "w", encoding="utf-8") as out:
            out.write(proc.stdout)

        if proc.returncode != 0:
            sys.stderr.write(f"[WARN] Row {i}: grasp returned non-zero exit.\n")
            with open(out_file + ".stderr.txt", "w", encoding="utf-8") as errf:
                errf.write(proc.stderr)

print("Done.")

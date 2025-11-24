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

# Open the CSV file for reading
with open(csv_path, newline='', encoding='utf-8') as f:
    reader = csv.reader(f)

    # Skip header row ("question")
    header = next(reader, None)

    # Enumerate data rows starting at 0 so index = CSV row (excluding header)
    for row_index, row in enumerate(
        tqdm(reader, desc="Processing questions", unit="row"),
        start=0
    ):
        # Skip completely empty rows (but keep the index consistent)
        if not row:
            continue

        question = row[0].strip()
        if not question:
            continue

        # Optional: extract trailing Q-ID (e.g. Q260725) from the question
        qid = ""
        parts = question.split()
        if parts:
            last = parts[-1]
            if last.startswith("Q") and last[1:].isdigit():
                qid = last

        # Build filename: e.g. 000_Q260725.json or 000.json
        if qid:
            filename = f"{row_index:03d}_{qid}.json"
        else:
            filename = f"{row_index:03d}.json"

        out_file = os.path.join(out_dir, filename)

        # Run GRASP for this question
        proc = subprocess.run(
            ["bash", "-lc", f'printf %s "{question}" | grasp run configs/run.yaml'],
            capture_output=True,
            text=True,
        )

        # Write stdout to JSON file
        with open(out_file, "w", encoding="utf-8") as out:
            out.write(proc.stdout)

        # If there was an error, write stderr alongside
        if proc.returncode != 0:
            sys.stderr.write(
                f"[WARN] Row {row_index} (file {filename}): grasp returned non-zero exit.\n"
            )
            with open(out_file + ".stderr.txt", "w", encoding="utf-8") as errf:
                errf.write(proc.stderr)

print("Done.")

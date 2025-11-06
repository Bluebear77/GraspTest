#!/usr/bin/env python3
import csv
import os
import subprocess
import sys
from tqdm import tqdm  # progress bar for loops

# Path to your CSV file and output directory
csv_path = "data/FeTaQA/10.csv"
out_dir = "output"
os.makedirs(out_dir, exist_ok=True)

# Open the CSV file for reading
with open(csv_path, newline='', encoding='utf-8') as f:
    reader = csv.reader(f)
    header = next(reader, None)  # Skip header row if present
    rows = list(reader)  # Read all rows first so tqdm can track total

    # Wrap the loop with tqdm to show a progress bar
    for i, row in enumerate(tqdm(rows, desc="Processing questions", unit="row"), start=1):
        # Skip empty or invalid rows
        if not row or not row[0].strip():
            continue

        # Extract question text and prepare output file path
        question = row[0].strip()
        out_file = os.path.join(out_dir, f"{i:03d}.json")

        # Run external command using subprocess
        # The command sends the question to `grasp run` via stdin
        
        proc = subprocess.run(
            ["bash", "-lc", f'printf %s "{question}" | grasp run configs/run.yaml'],
            capture_output=True, text=True
        )

        # Save the command's stdout (expected JSON) to file
        with open(out_file, "w", encoding="utf-8") as out:
            out.write(proc.stdout)

        # If grasp returned a non-zero exit code, log a warning and save stderr
        if proc.returncode != 0:
            sys.stderr.write(f"[WARN] Row {i}: grasp returned non-zero exit. Stderr saved next to JSON.\n")
            with open(out_file + ".stderr.txt", "w", encoding="utf-8") as errf:
                errf.write(proc.stderr)

print("Done.")

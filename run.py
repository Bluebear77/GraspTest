#!/usr/bin/env python3
import csv
import os
import subprocess
import sys
from tqdm import tqdm  # progress bar

csv_path = "data/FeTaQA/10.csv"
out_dir = "output"
os.makedirs(out_dir, exist_ok=True)

# Read all rows first to know total length
with open(csv_path, newline='', encoding='utf-8') as f:
    reader = csv.reader(f)
    header = next(reader, None)  # skip header
    rows = [row for row in reader if row and row[0].strip()]

for i, row in enumerate(tqdm(rows, desc="Processing questions", unit="q")):
    question = row[0].strip()
    out_file = os.path.join(out_dir, f"{i+1:03d}.json")

    # You can add '--task general-qa' if needed
    proc = subprocess.run(
        ["bash", "-lc", f'printf %s "{question}" | grasp run configs/run.yaml'],
        capture_output=True,
        text=True
    )

    with open(out_file, "w", encoding="utf-8") as out:
        out.write(proc.stdout)

    if proc.returncode != 0:
        sys.stderr.write(f"[WARN] Row {i+1}: grasp returned non-zero exit. Stderr saved next to JSON.\n")
        with open(out_file + ".stderr.txt", "w", encoding="utf-8") as errf:
            errf.write(proc.stderr)

print("✅ All done! Outputs saved in:", out_dir)

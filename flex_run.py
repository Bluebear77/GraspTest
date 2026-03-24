#!/usr/bin/env python3
"""
Flexible CSV runner for GRASP / LLM pipelines.

What changed compared with the original script:
- Works with both:
    1) CSV with columns, e.g. question_id, question, entity_id1, ...
    2) CSV with columns,, e.g. question, answer
- If question_id is missing, filenames fall back to zero-padded row indices.
- If entity_id* columns are missing, input_text is just the question.
- Supports multiple input CSVs at once.
- Keeps resume / bad-output checks / batch logging / optional git integration.
- Avoids crashing on shorter rows or schema differences.
"""

import csv
import json
import os
import re
import subprocess
import sys
import time
from typing import Dict, List, Optional, Set, Tuple

from tqdm import tqdm

# ---------------- Configuration ---------------- #

CSV_FILES: List[str] = [
    "data/SimpleQA/CompMix_table_simple_qa.csv",
    "data/SimpleQA//NQ_table_test_simple.csv",
    "data/SimpleQA//Qampari_wikitables_simple.csv",
]

BATCH_SIZE = 50
GRASP_COMMAND = ["bash", "-lc", "grasp run configs/run.yaml"]
ENABLE_GIT = True  # set True only if you really want git add/commit/push after each batch
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


# ---------------- Helpers ---------------- #

def safe_slug(text: str, max_len: int = 80) -> str:
    text = re.sub(r"\s+", "_", text.strip())
    text = re.sub(r"[^A-Za-z0-9._-]", "", text)
    text = text.strip("._-")
    if not text:
        text = "row"
    return text[:max_len]


def make_output_filename(row: List[str], global_index: int, colmap: Dict[str, Optional[int]]) -> str:
    question_id_idx = colmap.get("question_id")
    question_idx = colmap.get("question")

    if question_id_idx is not None and len(row) > question_id_idx:
        question_id = row[question_id_idx].strip()
        if question_id:
            return f"{question_id}.json"

    # Fall back to stable zero-padded row index.
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
                    if (
                        data.get("type") == "output"
                        and data.get("task") == "sparql-qa"
                        and data.get("output") is None
                    ):
                        reason = "null_output"

        if reason is not None:
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


# ---------------- Batch log / resume helpers ---------------- #

def parse_batch_log(batch_log_path: str, total_rows: int) -> Tuple[Set[int], bool]:
    processed_indices: Set[int] = set()
    is_completed = False

    if not os.path.exists(batch_log_path):
        return processed_indices, is_completed

    with open(batch_log_path, "r", encoding="utf-8") as lf:
        for line in lf:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                if "COMPLETED" in stripped.upper():
                    is_completed = True
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

    if not is_completed and processed_indices and max(processed_indices) >= total_rows - 1:
        is_completed = True

    return processed_indices, is_completed


def ensure_batch_log_header(batch_log_path: str, csv_path: str) -> None:
    if os.path.exists(batch_log_path):
        return
    with open(batch_log_path, "w", encoding="utf-8") as lf:
        lf.write(f"# Batch log for {csv_path}\n")
        lf.write("# Each line: batch_index | row_indices | filenames\n\n")


def ask_yes_no(prompt: str, default: str = "n") -> str:
    answer = ""
    try:
        while answer not in ("y", "n"):
            answer = input(prompt).strip().lower()
    except EOFError:
        print(f"No interactive input available; defaulting to '{default}'.")
        answer = default
    return answer


# ---------------- Main CSV processing ---------------- #

def process_csv(csv_path: str, batch_size: int = BATCH_SIZE) -> None:
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
        return

    print("Detected schema:")
    print(f"  question_id : {colmap['question_id']}")
    print(f"  question    : {colmap['question']}")
    print(f"  answer      : {colmap['answer']}")
    print(f"  entity_id   : {colmap['entity_id']}")
    print(f"  total rows  : {total_rows}")

    ensure_batch_log_header(batch_log_path, csv_path)
    processed_indices, is_completed_from_log = parse_batch_log(batch_log_path, total_rows)

    if processed_indices:
        bad_indices, bad_csv_path = detect_bad_outputs(rows, colmap, out_dir, processed_indices)
        if bad_indices:
            print(f"[CHECK INFO] Detected {len(bad_indices)} bad output JSON file(s).")
            print(f"[CHECK INFO] Details logged to: {bad_csv_path}")
            answer_bad = ask_yes_no(
                "Do you want to re-run these bad executions? [y/n]: ",
                default="n",
            ) if ASK_BEFORE_RERUN_BAD else "y"

            if answer_bad == "y":
                processed_indices -= bad_indices
                print(f"[CHECK INFO] {len(bad_indices)} bad execution(s) will be re-run.")
            else:
                print("[CHECK INFO] Bad executions will be left as-is.")

    is_completed_effective = (
        is_completed_from_log and processed_indices and max(processed_indices) >= total_rows - 1
    )

    if is_completed_effective:
        print("Detected that this CSV appears to be fully processed based on batch_log.txt.")
        answer = ask_yes_no(
            "files already generated, do you want to update them? [y/n]: ",
            default="n",
        ) if ASK_BEFORE_RERUN_COMPLETE else "n"

        if answer != "y":
            print("User chose not to update existing outputs. Skipping this CSV.")
            return

        processed_indices.clear()
        with open(batch_log_path, "a", encoding="utf-8") as lf:
            lf.write("\n# Restarting processing: user requested update.\n")
    else:
        if processed_indices:
            print(f"Resuming from row {max(processed_indices) + 1}.")
        else:
            print("No previous progress found. Starting from row 0.")

    processed_count = len(processed_indices)

    with tqdm(total=total_rows, desc=f"Processing {file_stem}", unit="row", initial=processed_count) as pbar:
        for batch_start in range(0, total_rows, batch_size):
            batch_rows = rows[batch_start: batch_start + batch_size]
            batch_index = batch_start // batch_size
            batch_number = batch_index + 1
            batch_filenames: List[str] = []
            batch_row_indices: List[int] = []

            for i, row in enumerate(batch_rows):
                global_index = batch_start + i

                if global_index in processed_indices:
                    pbar.update(1)
                    continue

                input_text = build_input_text(row, colmap)
                if not input_text:
                    pbar.update(1)
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

                stdout_text = proc.stdout
                try:
                    data = json.loads(stdout_text)
                    if isinstance(data, dict):
                        data["elapsed"] = round(elapsed, 3)
                        data["source_csv"] = csv_path
                        data["row_index"] = global_index
                        q_idx = colmap.get("question")
                        a_idx = colmap.get("answer")
                        data["question"] = row[q_idx].strip() if q_idx is not None and len(row) > q_idx else None
                        if a_idx is not None and len(row) > a_idx:
                            data["reference_answer"] = row[a_idx].strip()
                    json_text = json.dumps(data, ensure_ascii=False)
                except json.JSONDecodeError:
                    json_text = stdout_text

                with open(out_file, "w", encoding="utf-8") as out:
                    out.write(json_text)

                if proc.returncode != 0:
                    sys.stderr.write(
                        f"[WARN] Row {global_index} ({filename}): grasp returned non-zero exit.\n"
                    )
                    with open(out_file + ".stderr.txt", "w", encoding="utf-8") as errf:
                        errf.write(proc.stderr)

                processed_count += 1
                processed_indices.add(global_index)
                batch_filenames.append(filename)
                batch_row_indices.append(global_index)
                pbar.update(1)

            if batch_row_indices:
                row_range_str = (
                    f"{batch_row_indices[0]}-{batch_row_indices[-1]}"
                    if len(batch_row_indices) > 1
                    else str(batch_row_indices[0])
                )
                with open(batch_log_path, "a", encoding="utf-8") as lf:
                    lf.write(
                        f"batch {batch_index} (#{batch_number}) | rows {row_range_str} | files: {', '.join(batch_filenames)}\n"
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


if __name__ == "__main__":
    for csv_path in CSV_FILES:
        if not os.path.exists(csv_path):
            print(f"CSV not found, skipping: {csv_path}")
            continue
        process_csv(csv_path, batch_size=BATCH_SIZE)

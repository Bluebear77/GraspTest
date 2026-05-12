#!/usr/bin/env python3
"""
Re-run only specific bad GRASP output JSON files.

Behavior:
- Only re-runs the JSON files listed in BAD_JSON_FILES.
- For each bad JSON file, finds the matching source CSV by output directory name.

Example:
    /scratches/wei/GraspTest/235B/CompMix_infobox_complex/00142.json

maps to source CSV:
    data/SimpleQA/CompMix_infobox_complex.csv
or
    data/ComplexQA/CompMix_infobox_complex.csv

- Uses the numeric JSON filename as the CSV row index:
    00142.json -> row index 142

- Overwrites only the .json file.
- Does NOT delete, overwrite, or modify the matching .json.stderr.txt file.
"""

import csv
import glob
import json
import os
import re
import subprocess
import sys
import time
from typing import Dict, List, Optional, Tuple


# ---------------- Configuration ---------------- #

INPUT_DIRS: List[str] = [
    "data/SimpleQA",
    "data/ComplexQA",
]

BAD_JSON_FILES: List[str] = [
    "/scratches/wei/GraspTest/235B/CompMix_infobox_complex/00142.json",
    "/scratches/wei/GraspTest/235B/CompMix_infobox_complex/00279.json",
    "/scratches/wei/GraspTest/235B/NQ_table_test_simple/00547.json",
    "/scratches/wei/GraspTest/235B/NQ_table_test_simple/00865.json",
]

GRASP_COMMAND = ["bash", "-lc", "grasp run configs/run.yaml"]

# If your shell does not load the correct conda env, use something like:
# GRASP_COMMAND = [
#     "bash",
#     "-lc",
#     "source ~/miniconda3/etc/profile.d/conda.sh && conda activate grasp_v1 && grasp run configs/run.yaml",
# ]


# ---------------- CSV helpers ---------------- #

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


def discover_csv_by_stem(file_stem: str) -> Optional[str]:
    """
    Find a CSV whose basename matches the output directory name.

    Example:
        file_stem = CompMix_infobox_complex
        source CSV = data/ComplexQA/CompMix_infobox_complex.csv
    """
    candidates: List[str] = []

    for input_dir in INPUT_DIRS:
        path = os.path.join(input_dir, f"{file_stem}.csv")
        if os.path.exists(path):
            candidates.append(path)

    if len(candidates) == 1:
        return candidates[0]

    if len(candidates) > 1:
        print(f"[WARN] Multiple CSV candidates found for {file_stem}:")
        for c in candidates:
            print(f"  - {c}")
        print(f"[WARN] Using first candidate: {candidates[0]}")
        return candidates[0]

    # Fallback: search recursively in case the CSV is nested.
    recursive_candidates = []
    for input_dir in INPUT_DIRS:
        recursive_candidates.extend(
            glob.glob(os.path.join(input_dir, "**", f"{file_stem}.csv"), recursive=True)
        )

    recursive_candidates = sorted(set(recursive_candidates))

    if len(recursive_candidates) == 1:
        return recursive_candidates[0]

    if len(recursive_candidates) > 1:
        print(f"[WARN] Multiple recursive CSV candidates found for {file_stem}:")
        for c in recursive_candidates:
            print(f"  - {c}")
        print(f"[WARN] Using first candidate: {recursive_candidates[0]}")
        return recursive_candidates[0]

    return None


def load_csv_row(csv_path: str, row_index: int) -> Tuple[List[str], Dict[str, Optional[int]], List[str]]:
    """
    Load one row from a CSV by zero-based data-row index.

    row_index = 0 means the first row after the header.
    """
    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        reader = csv.reader(f)

        raw_header = next(reader, None)
        if raw_header is None:
            raise ValueError(f"CSV is empty: {csv_path}")

        header = [h.strip().lstrip("\ufeff") for h in raw_header]
        colmap = detect_schema(header)

        if colmap["question"] is None:
            raise ValueError(f"'question' column not found in CSV header: {header}")

        for i, row in enumerate(reader):
            if i == row_index:
                return row, colmap, header

    raise IndexError(f"Row index {row_index} not found in CSV: {csv_path}")


# ---------------- Path helpers ---------------- #

def parse_bad_json_path(json_path: str) -> Tuple[str, int]:
    """
    Parse output directory stem and row index from a JSON path.

    Example:
        /.../235B/CompMix_infobox_complex/00142.json

    returns:
        ("CompMix_infobox_complex", 142)
    """
    filename = os.path.basename(json_path)
    file_stem = os.path.basename(os.path.dirname(json_path))

    match = re.fullmatch(r"(\d+)\.json", filename)
    if not match:
        raise ValueError(
            f"Bad JSON filename must look like 00142.json, got: {filename}"
        )

    row_index = int(match.group(1))
    return file_stem, row_index


def write_text_atomic(path: str, text: str) -> None:
    """
    Safely overwrite only the target JSON file.

    This does not touch any .stderr.txt sidecar.
    """
    tmp_path = path + ".tmp"

    os.makedirs(os.path.dirname(path), exist_ok=True)

    with open(tmp_path, "w", encoding="utf-8") as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())

    os.replace(tmp_path, path)


# ---------------- GRASP output handling ---------------- #

def make_error_payload(
    error: str,
    elapsed: float,
    csv_path: str,
    row_index: int,
    row: List[str],
    colmap: Dict[str, Optional[int]],
    returncode: int,
    stdout_text: str,
    stderr_text: str,
) -> Dict[str, object]:
    q_idx = colmap.get("question")
    a_idx = colmap.get("answer")

    payload: Dict[str, object] = {
        "error": error,
        "elapsed": round(elapsed, 6),
        "source_csv": csv_path,
        "row_index": row_index,
        "returncode": returncode,
        "stdout_preview": stdout_text[:4000],
        "stderr_preview": stderr_text[:4000],
        "question": (
            row[q_idx].strip()
            if q_idx is not None and len(row) > q_idx
            else None
        ),
    }

    if a_idx is not None and len(row) > a_idx:
        payload["reference_answer"] = row[a_idx].strip()

    return payload


def add_metadata_to_data(
    data: object,
    elapsed: float,
    csv_path: str,
    row_index: int,
    row: List[str],
    colmap: Dict[str, Optional[int]],
) -> object:
    if not isinstance(data, dict):
        return data

    q_idx = colmap.get("question")
    a_idx = colmap.get("answer")

    data["elapsed"] = round(elapsed, 6)
    data["source_csv"] = csv_path
    data["row_index"] = row_index
    data["question"] = (
        row[q_idx].strip()
        if q_idx is not None and len(row) > q_idx
        else None
    )

    if a_idx is not None and len(row) > a_idx:
        data["reference_answer"] = row[a_idx].strip()

    return data


def serialize_grasp_result(
    proc: subprocess.CompletedProcess,
    elapsed: float,
    csv_path: str,
    row_index: int,
    row: List[str],
    colmap: Dict[str, Optional[int]],
) -> str:
    stdout_text = proc.stdout or ""
    stderr_text = proc.stderr or ""

    if not stdout_text.strip():
        payload = make_error_payload(
            error="empty_stdout",
            elapsed=elapsed,
            csv_path=csv_path,
            row_index=row_index,
            row=row,
            colmap=colmap,
            returncode=proc.returncode,
            stdout_text=stdout_text,
            stderr_text=stderr_text,
        )
        return json.dumps(payload, ensure_ascii=False, indent=2)

    try:
        data = json.loads(stdout_text)
    except json.JSONDecodeError:
        payload = make_error_payload(
            error="invalid_json_stdout",
            elapsed=elapsed,
            csv_path=csv_path,
            row_index=row_index,
            row=row,
            colmap=colmap,
            returncode=proc.returncode,
            stdout_text=stdout_text,
            stderr_text=stderr_text,
        )
        return json.dumps(payload, ensure_ascii=False, indent=2)

    data = add_metadata_to_data(
        data=data,
        elapsed=elapsed,
        csv_path=csv_path,
        row_index=row_index,
        row=row,
        colmap=colmap,
    )

    return json.dumps(data, ensure_ascii=False, indent=2)


# ---------------- Main rerun logic ---------------- #

def rerun_one_bad_json(json_path: str) -> Dict[str, object]:
    file_stem, row_index = parse_bad_json_path(json_path)

    csv_path = discover_csv_by_stem(file_stem)
    if csv_path is None:
        raise FileNotFoundError(
            f"Could not find source CSV for output directory '{file_stem}'. "
            f"Searched: {INPUT_DIRS}"
        )

    row, colmap, header = load_csv_row(csv_path, row_index)

    input_text = build_input_text(row, colmap)
    if not input_text:
        raise ValueError(
            f"Row {row_index} in {csv_path} has empty or missing question."
        )

    stderr_sidecar = json_path + ".stderr.txt"
    stderr_exists_before = os.path.exists(stderr_sidecar)

    print("\n--------------------------------------------------")
    print(f"Bad JSON file       : {json_path}")
    print(f"Matching stderr     : {stderr_sidecar}")
    print(f"Stderr exists       : {stderr_exists_before}")
    print(f"Source CSV          : {csv_path}")
    print(f"CSV row index       : {row_index}")
    print(f"Question preview    : {input_text[:300]}")

    start_time = time.perf_counter()

    proc = subprocess.run(
        GRASP_COMMAND,
        input=input_text,
        capture_output=True,
        text=True,
    )

    elapsed = time.perf_counter() - start_time

    json_text = serialize_grasp_result(
        proc=proc,
        elapsed=elapsed,
        csv_path=csv_path,
        row_index=row_index,
        row=row,
        colmap=colmap,
    )

    # Important: this overwrites only the JSON file.
    # It does not delete or modify json_path + ".stderr.txt".
    write_text_atomic(json_path, json_text)

    stderr_exists_after = os.path.exists(stderr_sidecar)

    if proc.returncode != 0:
        print(f"[WARN] GRASP returned non-zero exit code: {proc.returncode}")
        print("[WARN] Existing stderr sidecar was left untouched.")
    else:
        print("[OK] GRASP completed with return code 0.")
        print("[OK] Existing stderr sidecar was left untouched.")

    print(f"Elapsed             : {elapsed:.3f}s")
    print(f"Updated JSON        : {json_path}")
    print(f"Stderr still exists : {stderr_exists_after}")

    return {
        "json_path": json_path,
        "stderr_sidecar": stderr_sidecar,
        "stderr_exists_before": stderr_exists_before,
        "stderr_exists_after": stderr_exists_after,
        "csv_path": csv_path,
        "row_index": row_index,
        "returncode": proc.returncode,
        "elapsed": elapsed,
    }


def main() -> None:
    print("================ RERUN ONLY SPECIFIED BAD FILES ================")
    print(f"Number of bad JSON files to rerun: {len(BAD_JSON_FILES)}")

    summaries: List[Dict[str, object]] = []

    for json_path in BAD_JSON_FILES:
        try:
            summary = rerun_one_bad_json(json_path)
            summaries.append(summary)
        except Exception as e:
            print("\n[ERROR] Failed to rerun:")
            print(f"  JSON file: {json_path}")
            print(f"  Error    : {e}", file=sys.stderr)

            summaries.append(
                {
                    "json_path": json_path,
                    "error": str(e),
                }
            )

    print("\n================ FINAL SUMMARY ================")

    ok_count = 0
    failed_count = 0
    nonzero_count = 0

    for s in summaries:
        print(f"\nJSON file: {s.get('json_path')}")

        if "error" in s:
            failed_count += 1
            print(f"  status       : failed before GRASP")
            print(f"  error        : {s.get('error')}")
            continue

        returncode = s.get("returncode")
        if returncode == 0:
            ok_count += 1
            status = "updated"
        else:
            nonzero_count += 1
            status = "updated with error payload"

        print(f"  status       : {status}")
        print(f"  source CSV   : {s.get('csv_path')}")
        print(f"  row index    : {s.get('row_index')}")
        print(f"  return code  : {returncode}")
        print(f"  elapsed      : {s.get('elapsed'):.3f}s")
        print(f"  stderr kept  : {s.get('stderr_exists_before')} -> {s.get('stderr_exists_after')}")

    print("\nCounts:")
    print(f"  successful GRASP runs      : {ok_count}")
    print(f"  non-zero GRASP runs        : {nonzero_count}")
    print(f"  failed before GRASP        : {failed_count}")
    print(f"  total requested bad files  : {len(BAD_JSON_FILES)}")


if __name__ == "__main__":
    main()
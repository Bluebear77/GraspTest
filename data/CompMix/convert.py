import json
import csv
from pathlib import Path

INPUT_FILES = ["dev_set.json", "test_set.json", "train_set.json"]

# Desired column order
FIELDNAMES = [
    "question_id",
    "question",
    "entity_id",
    "entity_label",
    "answer_id",
    "answer_label",
    "answer_src",
    "answer_text",
    "domain",
    "convmix_question_id",
]

def process_file(json_path: Path):
    csv_path = json_path.with_suffix(".csv")

    # Read JSON (assumes it's a list/array of objects)
    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    # Write CSV in the same order as the JSON array
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()

        for item in data:
            # Handle entities (assume list; use first if present)
            entities = item.get("entities") or []
            if entities:
                entity = entities[0]
                entity_id = entity.get("id", "")
                entity_label = entity.get("label", "")
            else:
                entity_id = ""
                entity_label = ""

            # Handle answers (assume list; use first if present)
            answers = item.get("answers") or []
            if answers:
                answer = answers[0]
                answer_id = answer.get("id", "")
                answer_label = answer.get("label", "")
            else:
                answer_id = ""
                answer_label = ""

            row = {
                "question_id": item.get("question_id", ""),
                "question": item.get("question", ""),
                "entity_id": entity_id,
                "entity_label": entity_label,
                "answer_id": answer_id,
                "answer_label": answer_label,
                "answer_src": item.get("answer_src", ""),
                "answer_text": item.get("answer_text", ""),
                "domain": item.get("domain", ""),
                "convmix_question_id": item.get("convmix_question_id", ""),
            }

            writer.writerow(row)

    print(f"Written: {csv_path}")

def main():
    for filename in INPUT_FILES:
        path = Path(filename)
        if path.exists():
            process_file(path)
        else:
            print(f"Skipping (not found): {path}")

if __name__ == "__main__":
    main()

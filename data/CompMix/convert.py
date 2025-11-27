# Enter w1 (weight for number of entities): 0.7
# Enter w2 (weight for question length): 0.3

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
- Uses a tqdm progress bar to show overall processing progress.
- Prints a concise summary at the end with counts and output directory.
- The code is structured so that multiple CSVs can be handled in the future:
      compmix-test.csv, dev_set.csv, test_set.csv, train_set.csv
  For now only compmix-test.csv is active, others are sketched as comments.
"""

import json
import csv
import os
from collections import Counter, defaultdict
import statistics as stats

import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm


# Input JSON files
INPUT_FILES = [
    "train_set.json",
    "dev_set.json",
    "test_set.json",
]

# Output CSV file
OUTPUT_FILE = "merged_compmix.csv"

# Statistics output folder
STATS_DIR = "statistics"


def load_questions_from_file(filename):
    """Load list of question objects from a JSON file."""
    with open(filename, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def compute_context_score(num_entities, question_length_tokens, w1, w2):
    """
    Question Context Score = w1 * (#entities) + w2 * (#question length tokens).
    """
    return w1 * num_entities + w2 * question_length_tokens


def get_stats(values):
    """Return dict of [avg, median, min, max, std] for a list of numbers."""
    if not values:
        return {
            "avg": 0,
            "median": 0,
            "min": 0,
            "max": 0,
            "std": 0,
        }
    return {
        "avg": round(stats.mean(values), 3),
        "median": round(stats.median(values), 3),
        "min": round(min(values), 3),
        "max": round(max(values), 3),
        "std": round(stats.pstdev(values), 3),
    }


def colored_histogram(data, bins, title, xlabel, ylabel, out_path):
    counts, bin_edges = np.histogram(data, bins=bins)
    centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])
    width = bin_edges[1] - bin_edges[0] if len(bin_edges) > 1 else 1.0

    plt.figure()
    colors = [plt.cm.tab20(i % 20) for i in range(len(counts))]
    plt.bar(centers, counts, width=width, color=colors, align="center")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def compute_and_save_statistics(all_records):
    os.makedirs(STATS_DIR, exist_ok=True)

    question_lengths = []
    num_entities_list = []
    answer_text_lengths = []
    num_answers_list = []
    domain_list = []
    context_scores = []

    per_domain = defaultdict(lambda: {
        "question_lengths": [],
        "num_entities": [],
        "answer_lengths": [],
        "num_answers": [],
        "context_scores": [],
    })

    for rec in tqdm(all_records, desc="Collecting statistics"):
        q = rec.get("question", "") or ""
        q_len = len(q.split())

        entities = rec.get("entities", []) or []
        num_entities = len(entities)

        ans_text = rec.get("answer_text", "") or ""
        ans_len = len(ans_text.split())

        answers = rec.get("answers", []) or []
        n_answers = len(answers)

        domain = rec.get("domain", "unknown") or "unknown"
        cs = rec.get("context_score", 0.0)

        # store overall
        question_lengths.append(q_len)
        num_entities_list.append(num_entities)
        answer_text_lengths.append(ans_len)
        num_answers_list.append(n_answers)
        domain_list.append(domain)
        context_scores.append(cs)

        # store per-domain
        per_domain[domain]["question_lengths"].append(q_len)
        per_domain[domain]["num_entities"].append(num_entities)
        per_domain[domain]["answer_lengths"].append(ans_len)
        per_domain[domain]["num_answers"].append(n_answers)
        per_domain[domain]["context_scores"].append(cs)

    total_questions = len(all_records)

    overall_stats = {
        "question_length": get_stats(question_lengths),
        "num_entities": get_stats(num_entities_list),
        "answer_length": get_stats(answer_text_lengths),
        "num_answers": get_stats(num_answers_list),
        "context_score": get_stats(context_scores),
    }

    per_domain_stats = {}
    for domain, dvals in per_domain.items():
        per_domain_stats[domain] = {
            "question_length": get_stats(dvals["question_lengths"]),
            "num_entities": get_stats(dvals["num_entities"]),
            "answer_length": get_stats(dvals["answer_lengths"]),
            "num_answers": get_stats(dvals["num_answers"]),
            "context_score": get_stats(dvals["context_scores"]),
        }

    # PLOTS
    colored_histogram(
        data=question_lengths,
        bins=30,
        title="Question token length distribution",
        xlabel="Tokens",
        ylabel="Frequency",
        out_path=os.path.join(STATS_DIR, "question token length.png"),
    )

    if num_entities_list:
        bins_entities = np.arange(0, max(num_entities_list) + 2) - 0.5
    else:
        bins_entities = 10

    colored_histogram(
        data=num_entities_list,
        bins=bins_entities,
        title="Number of question entities",
        xlabel="Entities",
        ylabel="Frequency",
        out_path=os.path.join(STATS_DIR, "number of question entities.png"),
    )

    colored_histogram(
        data=answer_text_lengths,
        bins=30,
        title="Answer length (text) distribution",
        xlabel="Tokens",
        ylabel="Frequency",
        out_path=os.path.join(STATS_DIR, "answer length (text).png"),
    )

    if num_answers_list:
        bins_answers = np.arange(0, max(num_answers_list) + 2) - 0.5
    else:
        bins_answers = 10

    colored_histogram(
        data=num_answers_list,
        bins=bins_answers,
        title="Number of answers",
        xlabel="Count",
        ylabel="Frequency",
        out_path=os.path.join(STATS_DIR, "number of answers.png"),
    )

    domain_counts = Counter(domain_list)

    plt.figure()
    plt.pie(domain_counts.values(), labels=domain_counts.keys(), autopct="%1.1f%%")
    plt.title("Domain distribution")
    plt.tight_layout()
    plt.savefig(os.path.join(STATS_DIR, "domain_distribution_piechart.png"))
    plt.close()

    colored_histogram(
        data=context_scores,
        bins=30,
        title="Context score distribution",
        xlabel="Score",
        ylabel="Frequency",
        out_path=os.path.join(STATS_DIR, "context_score_distribution.png"),
    )

    # Write Markdown
    md_path = os.path.join(STATS_DIR, "statistics.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Dataset Statistics\n\n")
        f.write("## Summary Comment\n\n")
        f.write(f"- Total questions: **{total_questions}**\n\n")

        f.write("## Overall Statistics\n\n")
        f.write("| Metric | Avg | Median | Min | Max | Std |\n")
        f.write("| --- | --- | --- | --- | --- | --- |\n")

        names = {
            "question_length": "Question length",
            "num_entities": "Entities",
            "answer_length": "Answer length",
            "num_answers": "Number of answers",
            "context_score": "Context score",
        }

        for key, label in names.items():
            s = overall_stats[key]
            f.write(f"| {label} | {s['avg']} | {s['median']} | {s['min']} | {s['max']} | {s['std']} |\n")

        f.write("\n## Per-Domain Statistics\n")
        for metric_key, label in names.items():
            f.write(f"\n### {label}\n")
            f.write("| Domain | Avg | Median | Min | Max | Std |\n")
            f.write("| --- | --- | --- | --- | --- | --- |\n")
            for domain in sorted(per_domain_stats.keys()):
                s = per_domain_stats[domain][metric_key]
                f.write(f"| {domain} | {s['avg']} | {s['median']} | {s['min']} | {s['max']} | {s['std']} |\n")

        f.write("\n## Plots\n\n")
        plots = [
            "question token length.png",
            "number of question entities.png",
            "answer length (text).png",
            "number of answers.png",
            "domain_distribution_piechart.png",
            "context_score_distribution.png",
        ]
        for p in plots:
            f.write(f"![{p}]({p})\n\n")


def main():
    # Ask for weight values
    print("=====================================================")
    print("  Question Context Score = w1 × (#entities) + w2 × (#question length)")
    print("=====================================================\n")

    w1 = float(input("Enter w1 (weight for number of entities): "))
    w2 = float(input("Enter w2 (weight for question length): "))

    # Load
    all_records = []
    for fname in INPUT_FILES:
        all_records.extend(load_questions_from_file(fname))

    # Compute context score
    for rec in tqdm(all_records, desc="Computing context scores"):
        q_len = len((rec.get("question", "") or "").split())
        num_entities = len(rec.get("entities", []) or [])
        rec["context_score"] = compute_context_score(num_entities, q_len, w1, w2)

    # Sort descending
    all_records.sort(key=lambda r: r["context_score"], reverse=True)

    # Determine max entities
    max_entities = max(len(rec.get("entities", [])) for rec in all_records)

    # Build CSV header
    header = ["question_id", "question"]
    for i in range(1, max_entities + 1):
        header.append(f"entity_id{i}")
        header.append(f"entity_label{i}")
    header.append("entity_number")
    header.extend([
        "answer_id",
        "answer_label",
        "answer_src",
        "answer_text",
        "domain",
        "convmix_question_id",
        "context_score",
    ])

    # Write CSV
    with open(OUTPUT_FILE, "w", encoding="utf-8", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(header)

        for rec in tqdm(all_records, desc="Writing CSV"):
            row = [
                rec.get("question_id", ""),
                rec.get("question", ""),
            ]

            entities = rec.get("entities", []) or []
            nents = len(entities)

            for i in range(max_entities):
                if i < nents:
                    row.append(entities[i].get("id", ""))
                    row.append(entities[i].get("label", ""))
                else:
                    row.append("")
                    row.append("")

            row.append(nents)

            answers = rec.get("answers", []) or []
            if answers:
                row.append(answers[0].get("id", ""))
                row.append(answers[0].get("label", ""))
            else:
                row.append("")
                row.append("")

            row.extend([
                rec.get("answer_src", ""),
                rec.get("answer_text", ""),
                rec.get("domain", ""),
                rec.get("convmix_question_id", ""),
                rec.get("context_score", 0.0),
            ])

            writer.writerow(row)

    compute_and_save_statistics(all_records)


if __name__ == "__main__":
    main()

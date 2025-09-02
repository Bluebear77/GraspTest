import os

from tqdm import tqdm
from universal_ml_utils.io import dump_json, load_json, load_jsonl
from universal_ml_utils.logging import get_logger

from grasp.sparql.metrics import f1_score, get_result_or_error, get_result_size
from grasp.utils import Sample, is_invalid_evaluation, is_invalid_model_output


def evaluate(
    input_file: str,
    prediction_file: str,
    endpoint: str,
    overwrite: bool = False,
    log_level: str | int | None = None,
    timeout: float = 300.0,
    retry_failed: bool = False,
    exact_after: int = 1024,
) -> None:
    logger = get_logger("GRASP EVALUATION", log_level)

    base = os.path.splitext(prediction_file)[0]
    evaluation_file = f"{base}.evaluation.json"

    exists = os.path.exists(evaluation_file)
    if exists and not overwrite:
        evaluations = load_json(evaluation_file)
        logger.info(
            f"Loaded {len(evaluations):,} existing evaluations from {evaluation_file}"
        )
    else:
        evaluations = {}

    predictions = load_jsonl(prediction_file)

    inputs: dict[str, Sample] = {}
    for sample in load_jsonl(input_file):
        sample = Sample(**sample)
        assert sample.id not in inputs, f"Duplicate id {sample.id}"
        assert sample.id is not None, "Sample id must not be None"
        inputs[sample.id] = sample

    logger.info(
        f"Evaluating {len(predictions):,} predictions "
        f"for {len(inputs):,} inputs from {input_file} "
        f"against SPARQL endpoint {endpoint}"
    )

    for pred in tqdm(
        predictions,
        desc="Evaluating",
        leave=False,
    ):
        assert pred.get("task", "sparql-qa") == "sparql-qa", (
            "Only SPARQL QA task is supported for evaluation"
        )
        if is_invalid_model_output(pred):
            continue

        id = pred["id"]
        if id in evaluations:
            evaluation = evaluations[id]
            if not retry_failed or not is_invalid_evaluation(evaluation):
                continue

        target_set, target_err = get_result_or_error(
            inputs[id].sparql,
            endpoint,
            request_timeout=timeout,
            read_timeout=timeout,
        )
        evaluations[id] = {
            "target": {
                "err": target_err,
                "size": get_result_size(target_set),
            },
        }

        sparql = pred["output"]["sparql"]
        if target_set is None or sparql is None:
            dump_json(evaluations, evaluation_file)
            continue

        pred_set, pred_err = get_result_or_error(
            sparql,
            endpoint,
            request_timeout=timeout,
            read_timeout=timeout,
        )
        if pred_set is not None:
            score = f1_score(pred_set, target_set, exact_after)
        else:
            score = 0.0

        evaluations[id]["prediction"] = {
            "sparql": sparql,
            "err": pred_err,
            "size": get_result_size(pred_set),
            "score": score,
            "elapsed": pred["elapsed"],
        }
        dump_json(evaluations, evaluation_file)

    dump_json(evaluations, evaluation_file)
    logger.info(f"Evaluation results saved to {evaluation_file}")
    f1_scores = [
        eval["prediction"]["score"]
        for eval in evaluations.values()
        if "prediction" in eval and eval["target"]["size"] > 0
    ]
    f1_avg = sum(f1_scores) / len(f1_scores) if f1_scores else 0.0
    logger.info(f"Average F1 score (ignoring empty targets): {f1_avg:.2%}")

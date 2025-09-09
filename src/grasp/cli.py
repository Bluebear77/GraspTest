import argparse
import asyncio
import json
import os
import random
import sys
import time
from logging import INFO, FileHandler, Logger

from fastapi import WebSocketDisconnect
from pydantic import BaseModel, conlist
from tqdm import tqdm
from universal_ml_utils.configuration import load_config
from universal_ml_utils.io import dump_json, dump_jsonl, load_jsonl, load_text
from universal_ml_utils.logging import get_logger, setup_logging
from universal_ml_utils.ops import extract_field, partition_by

from grasp.adapt import adapt
from grasp.build import build_indices, get_data
from grasp.build.data import merge_kgs
from grasp.configs import Adapt, Config
from grasp.core import generate, load_task_notes, setup
from grasp.evaluate import evaluate
from grasp.examples import ExampleIndex, load_example_indices
from grasp.manager import find_embedding_model
from grasp.tasks import Task, default_input_field
from grasp.utils import is_invalid_model_output, parse_parameters


def add_config_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "config",
        type=str,
        help="Path to the GRASP configuration file",
    )


def add_task_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "-t",
        "--task",
        type=str,
        choices=[task.value for task in Task],
        default=Task.SPARQL_QA.value,
        help="Task to run",
    )


def add_overwrite_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="grasp",
        description="GRASP: Generic Reasoning and SPARQL generation across Knowledge Graphs",
    )

    subparsers = parser.add_subparsers(
        title="commands",
        description="Available commands",
        dest="command",
        required=True,
    )

    # run GRASP server
    server_parser = subparsers.add_parser(
        "serve",
        help="Start a WebSocket server to serve GRASP",
    )
    add_config_arg(server_parser)
    server_parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to run the GRASP server on",
    )
    server_parser.add_argument(
        "--log-outputs",
        type=str,
        help="File to log all inputs and outputs to (in JSONL format)",
    )

    # run GRASP on a single input
    run_parser = subparsers.add_parser(
        "run",
        help="Run GRASP on a single input",
    )
    add_config_arg(run_parser)
    run_parser.add_argument(
        "-i",
        "--input",
        type=str,
        help="Input for task (e.g., a question for 'sparql-qa'), "
        "if not given, read from stdin",
    )
    run_parser.add_argument(
        "-if",
        "--input-format",
        type=str,
        choices=["text", "json"],
        default="text",
        help="Format of the input (raw text or JSON)",
    )
    run_parser.add_argument(
        "--input-field",
        type=str,
        default=None,
        help="Field to extract input from (if None, a task-specific default is used, "
        "but only if input format is 'json')",
    )
    add_task_arg(run_parser)

    # run GRASP on file with inputs
    file_parser = subparsers.add_parser(
        "file",
        help="Run GRASP on a file with inputs in JSONL format",
    )
    add_config_arg(file_parser)
    file_parser.add_argument(
        "--progress",
        action="store_true",
        help="Show a progress bar",
    )
    file_parser.add_argument(
        "-i",
        "--input-file",
        type=str,
        help="Path to file in JSONL format to run GRASP on, if not given, read JSONL from stdin",
    )
    file_parser.add_argument(
        "--shuffle",
        action="store_true",
        help="Shuffle the inputs",
    )
    file_parser.add_argument(
        "--skip",
        type=int,
        default=0,
        help="Skip the first N inputs",
    )
    file_parser.add_argument(
        "--take",
        type=int,
        default=None,
        help="Limit number of inputs (after skipping) to N",
    )
    file_parser.add_argument(
        "--input-field",
        type=str,
        default=None,
        help="Field to extract input from (if None, a task-specific default is used)",
    )
    file_parser.add_argument(
        "--output-file",
        type=str,
        help="File to write the output to",
    )
    file_parser.add_argument(
        "--retry-failed",
        action="store_true",
        help="Retry failed inputs (only used with --output-file)",
    )
    add_task_arg(file_parser)
    add_overwrite_arg(file_parser)

    # evaluate GRASP output
    eval_parser = subparsers.add_parser(
        "evaluate",
        help="Evaluate GRASP output against a reference file (only for 'sparql-qa' task)",
    )
    eval_parser.add_argument(
        "input_file",
        type=str,
        help="Path to file with question-sparql pairs in JSONL format",
    )
    eval_parser.add_argument(
        "prediction_file",
        type=str,
        help="Path to file with GRASP predictions as produced by the 'file' command",
    )
    eval_parser.add_argument(
        "endpoint",
        type=str,
        help="SPARQL endpoint to use for evaluation",
    )
    eval_parser.add_argument(
        "--timeout",
        type=float,
        default=300.0,
        help="Maximum duration for a single query in seconds",
    )
    eval_parser.add_argument(
        "--exact-after",
        type=int,
        default=1024,
        help="Result size after which exact F1 score instead of assignment F1 score "
        "is used (due to performance reasons)",
    )
    eval_parser.add_argument(
        "--retry-failed",
        action="store_true",
        help="Rerun failed evaluations due to timeouts or errors",
    )
    add_overwrite_arg(eval_parser)

    # run GRASP adaptation
    adapt_parser = subparsers.add_parser(
        "adapt",
        help="Adapt GRASP to one or more knowledge graphs",
    )
    add_config_arg(adapt_parser)
    adapt_parser.add_argument(
        "-o",
        "--output-dir",
        type=str,
        required=True,
        help="Save results in this directory",
    )
    add_task_arg(adapt_parser)
    add_overwrite_arg(adapt_parser)

    # get data for GRASP indices
    data_parser = subparsers.add_parser(
        "data",
        help="Get entity and property data for a knowledge graph",
    )
    data_parser.add_argument(
        "knowledge_graph",
        type=str,
        help="Knowledge graph to get data for",
    )
    data_parser.add_argument(
        "--endpoint",
        type=str,
        help="SPARQL endpoint of the knowledge graph "
        "(if not given, the endpoint at qlever.cs.uni-freiubrg.de/api/<kg> is used)",
    )
    data_parser.add_argument(
        "--entity-sparql",
        type=str,
        help="Path to file with custom entity SPARQL query",
    )
    data_parser.add_argument(
        "--property-sparql",
        type=str,
        help="Path to file with custom property SPARQL query",
    )
    data_parser.add_argument(
        "--query-parameters",
        type=str,
        nargs="*",
        help="Extra query parameters sent to the knowledge graph endpoint",
    )
    data_parser.add_argument(
        "--replace",
        type=str,
        nargs="*",
        help="Variables with format {key} in SPARQL queries to replace with values in format key:value",
    )
    data_parser.add_argument(
        "--disable-id-fallback",
        action="store_true",
        help="Disable fallback to using IDs as labels if no label is found",
    )
    add_overwrite_arg(data_parser)

    # merge multiple knowledge graphs
    merge_parser = subparsers.add_parser(
        "merge",
        help="Merge data from multiple knowledge graphs. The first knowledge graph is the primary one, "
        "to which data from the other knowledge graphs is added. Therefore, the merged knowledge graph will "
        "have the same number of entities and properties as the first knowledge graph.",
    )
    merge_parser.add_argument(
        "knowledge_graphs",
        type=str,
        nargs="+",
        help="Knowledge graphs to merge",
    )
    merge_parser.add_argument(
        "knowledge_graph",
        type=str,
        help="Name of the merged knowledge graph",
    )
    add_overwrite_arg(merge_parser)

    # build GRASP indices
    index_parser = subparsers.add_parser(
        "index",
        help="Build entity and property indices for a knowledge graph",
    )
    index_parser.add_argument(
        "knowledge_graph",
        type=str,
        help="Knowledge graph to build indices for (must be defined in the config)",
    )
    index_parser.add_argument(
        "--entities-type",
        type=str,
        choices=["prefix", "similarity"],
        default="prefix",
        help="Type of entity index to build",
    )
    index_parser.add_argument(
        "--properties-type",
        type=str,
        choices=["prefix", "similarity"],
        default="similarity",
        help="Type of property index to build",
    )
    index_parser.add_argument(
        "--sim-precision",
        type=str,
        choices=["float32", "ubinary"],
        help="Precision when building similarity index",
    )
    index_parser.add_argument(
        "--sim-embedding-dim",
        type=int,
        help="Embedding dimensionality when building similarity index",
    )
    index_parser.add_argument(
        "--sim-batch-size",
        type=int,
        default=256,
        help="Batch size when building similarity index",
    )
    add_overwrite_arg(index_parser)

    # build example index
    example_parser = subparsers.add_parser(
        "examples",
        help="Build an example index used for few-shot learning (only for 'sparql-qa' task)",
    )
    example_parser.add_argument(
        "examples_file",
        type=str,
        help="Path to file with examples in JSONL format",
    )
    example_parser.add_argument(
        "output_dir",
        type=str,
        help="Directory to save the example index",
    )
    example_parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Batch size for building the example index",
    )
    add_overwrite_arg(example_parser)

    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        help="GRASP log level",
    )
    parser.add_argument(
        "--all-loggers",
        action="store_true",
        help="Enable logging for all loggers, not only the GRASP-specific ones",
    )
    return parser.parse_args()


def run_grasp(args: argparse.Namespace) -> None:
    logger = get_logger("GRASP", args.log_level)
    config = Config(**load_config(args.config))

    managers = setup(config)

    model = find_embedding_model(managers)
    example_indices = load_example_indices(config, model=model)

    notes, kg_notes = load_task_notes(args.task, config)

    if args.input_field is None:
        input_field = default_input_field(args.task)
    else:
        input_field = args.input_field

    run_on_file = args.command == "file"
    outputs = []
    if run_on_file:
        if args.input_file is None:
            inputs = [json.loads(line) for line in sys.stdin]
        else:
            inputs = load_jsonl(args.input_file)

        if args.shuffle:
            assert config.seed is not None, (
                "Seed must be set for deterministic shuffling"
            )
            random.seed(config.seed)
            random.shuffle(inputs)

        skip = max(0, args.skip)
        take = args.take or len(inputs)
        inputs = inputs[skip : skip + take]

        if args.output_file:
            if os.path.exists(args.output_file) and not args.overwrite:
                outputs = load_jsonl(args.output_file)

            # save info in config file next to output file
            output_stem, _ = os.path.splitext(args.output_file)
            config_file = output_stem + ".config.json"

            dump_json(config.model_dump(), config_file, indent=2)

        if args.progress:
            # wrap with tqdm
            inputs = tqdm(inputs, desc=f"GRASP for {args.task}")

    else:
        if args.input is None:
            ipt = sys.stdin.read()
        else:
            ipt = args.input

        if args.input_format == "json":
            inputs = [json.loads(ipt)]
        else:
            inputs = [{"input": ipt}]
            input_field = "input"  # overwrite

    for i, ipt in enumerate(inputs):
        id = extract_field(ipt, "id")
        if id is None:
            id = str(i)

        if input_field is not None:
            ipt = extract_field(ipt, input_field)

        assert ipt is not None, f"Input not found for input {i:,}"

        if i < len(outputs) and (
            not args.retry_failed or not is_invalid_model_output(outputs[i])
        ):
            continue

        *_, output = generate(
            args.task,
            ipt,
            config,
            managers,
            kg_notes,
            notes,
            example_indices=example_indices,
            logger=logger,
        )

        output["id"] = id
        if not run_on_file:
            print(json.dumps(output))
            break

        elif args.output_file is None:
            print(json.dumps(output))
            continue

        if i < len(outputs):
            outputs[i] = output
        else:
            outputs.append(output)

        dump_jsonl(outputs, args.output_file)


# keep track of connections and limit to 10 concurrent connections
active_connections = 0
MAX_CONNECTIONS = 16
# maximum time for a query in seconds
MAX_QUERY_TIME = 300.0
# maximum idle time for a connection in seconds
MAX_IDLE_TIME = 300.0


class Past(BaseModel):
    inputs: conlist(str, min_length=1)  # type: ignore
    messages: conlist(dict, min_length=1)  # type: ignore
    known: set[str]


class Request(BaseModel):
    task: Task
    input: str
    knowledge_graphs: conlist(str, min_length=1)  # type: ignore
    past: Past | None = None


def serve_grasp(args: argparse.Namespace) -> None:
    config = Config(**load_config(args.config))

    # create a fast api websocket server to serve the generate_sparql function
    import uvicorn
    from fastapi import FastAPI, WebSocket

    app = FastAPI()
    logger = get_logger("GRASP SERVER", args.log_level)
    if args.log_outputs is not None:
        os.makedirs(os.path.dirname(args.log_outputs), exist_ok=True)
        output_logger = Logger("GRASP JSONL OUTPUTS")
        output_logger.addHandler(
            FileHandler(args.log_outputs, mode="a", encoding="utf-8")
        )
        output_logger.setLevel(INFO)
    else:
        output_logger = None

    # add cors
    from fastapi.middleware.cors import CORSMiddleware

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    managers = setup(config)
    kgs = [manager.kg for manager in managers]

    model = find_embedding_model(managers)
    example_indices = load_example_indices(config, model=model)

    notes = {}
    kg_notes = {}
    for task in Task:
        general_notes, kg_specific_notes = load_task_notes(task.value, config)
        notes[task.value] = general_notes
        kg_notes[task.value] = kg_specific_notes

    @app.get("/knowledge_graphs")
    async def _knowledge_graphs():
        return kgs

    @app.get("/config")
    async def _config():
        return config.model_dump()

    @app.websocket("/live")
    async def _live(websocket: WebSocket):
        global active_connections
        assert websocket.client is not None
        client = f"{websocket.client.host}:{websocket.client.port}"
        await websocket.accept()

        # Check if we've reached the maximum number of connections
        if active_connections >= MAX_CONNECTIONS:
            logger.warning(
                f"Connection from {client} immediately closed: "
                f"maximum of {MAX_CONNECTIONS:,} active connections reached"
            )
            await websocket.close(code=1013, reason="Server too busy, try again later")
            return

        active_connections += 1
        logger.info(f"{client} connected ({active_connections=:,})")
        last_active = time.perf_counter()

        async def idle_checker():
            nonlocal last_active
            while True:
                await asyncio.sleep(min(5, MAX_IDLE_TIME))

                if time.perf_counter() - last_active <= MAX_IDLE_TIME:
                    continue

                msg = f"Connection closed due to inactivity after {MAX_IDLE_TIME:,} seconds"
                logger.info(f"{client}: {msg}")
                await websocket.close(code=1013, reason=msg)  # Try Again Later
                break

        idle_task = asyncio.create_task(idle_checker())

        try:
            while True:
                data = await websocket.receive_json()
                last_active = time.perf_counter()
                try:
                    request = Request(**data)
                except Exception:
                    logger.error(
                        f"Invalid request from {client}:\n{json.dumps(data, indent=2)}"
                    )
                    await websocket.send_json({"error": "Invalid request format"})
                    continue

                sel = request.knowledge_graphs
                if not sel or not all(kg in kgs for kg in sel):
                    logger.error(
                        f"Unsupported knowledge graph selection by {client}:\n{request.model_dump_json(indent=2)}"
                    )
                    await websocket.send_json(
                        {"error": "Unsupported knowledge graph selection"}
                    )
                    continue

                logger.info(
                    f"Processing request from {client}:\n{request.model_dump_json(indent=2)}"
                )

                sel_managers, _ = partition_by(managers, lambda m: m.kg in sel)

                past_inputs = None
                past_messages = None
                past_known = None
                if request.past is not None:
                    # set past
                    past_messages = request.past.messages
                    past_inputs = request.past.inputs
                    past_known = request.past.known

                # Setup generator
                generator = generate(
                    request.task,
                    request.input,
                    config,
                    sel_managers,
                    kg_notes[request.task],
                    notes[request.task],
                    example_indices,
                    past_inputs,
                    past_messages,
                    past_known,
                    logger,
                )

                # Track start time for timeout
                start_time = time.perf_counter()

                # Process generator outputs with timeout check
                for output in generator:
                    # Check if we've exceeded the time limit
                    current_time = time.perf_counter()
                    if current_time - start_time > MAX_QUERY_TIME:
                        # Send timeout message to client
                        msg = f"Operation with {client} timed out after {MAX_QUERY_TIME:,} seconds"
                        logger.warning(msg)
                        await websocket.send_json({"error": msg})
                        break

                    # Process the output normally
                    await websocket.send_json(output)
                    data = await websocket.receive_json()
                    last_active = time.perf_counter()

                    # Check if client requested cancellation
                    if data.get("cancel", False):
                        # Send cancellation confirmation to client
                        logger.info(f"Generation cancelled by {client}")
                        await websocket.send_json({"cancelled": True})
                        break

                    if output["type"] == "output" and output_logger is not None:
                        output_logger.info(json.dumps(output))

        except WebSocketDisconnect:
            pass

        except Exception as e:
            logger.error(f"Unexpected error with {client}:\n{e}")
            await websocket.send_json({"error": f"Failed to handle request:\n{e}"})

        finally:
            idle_task.cancel()
            active_connections -= 1
            logger.info(f"{client} disconnected ({active_connections=:,})")

    uvicorn.run(app, host="0.0.0.0", port=args.port)


def adapt_grasp(args: argparse.Namespace) -> None:
    config = Adapt(**load_config(args.config))
    adapt(args.task, config, args.adapt, args.overwrite, args.log_level)


def get_grasp_data(args: argparse.Namespace) -> None:
    replace = parse_parameters(args.replace or [])
    query_params = parse_parameters(args.query_parameters or [])

    if args.entity_sparql is not None:
        args.entity_sparql = load_text(args.entity_sparql).strip()
        for key, value in replace.items():
            args.entity_sparql = args.entity_sparql.replace(f"{{{key}}}", value)

    if args.property_sparql is not None:
        args.property_sparql = load_text(args.property_sparql).strip()
        for key, value in replace.items():
            args.property_sparql = args.property_sparql.replace(f"{{{key}}}", value)

    get_data(
        args.knowledge_graph,
        args.endpoint,
        args.entity_sparql,
        args.property_sparql,
        query_params,
        args.overwrite,
        args.disable_id_fallback,
        args.log_level,
    )


def merge_grasp_kgs(args: argparse.Namespace) -> None:
    merge_kgs(
        args.knowledge_graphs,
        args.knowledge_graph,
        args.overwrite,
        args.log_level,
    )


def build_grasp_indices(args: argparse.Namespace) -> None:
    build_indices(
        args.knowledge_graph,
        args.entities_type,
        args.properties_type,
        args.overwrite,
        args.log_level,
        sim_batch_size=args.sim_batch_size,
        sim_precision=args.sim_precision,
        sim_embedding_dim=args.sim_embedding_dim,
    )


def evaluate_grasp(args: argparse.Namespace) -> None:
    assert args.task == "sparql-qa", (
        "Built-in evaluation only supported for 'sparql-qa' task"
    )
    evaluate(
        args.input_file,
        args.prediction_file,
        args.endpoint,
        args.overwrite,
        args.log_level,
        args.timeout,
        args.retry_failed,
        args.exact_after,
    )


def main():
    args = parse_args()
    if args.all_loggers:
        setup_logging(args.log_level)

    if args.command == "data":
        get_grasp_data(args)
    elif args.command == "merge":
        merge_grasp_kgs(args)
    elif args.command == "index":
        build_grasp_indices(args)
    elif args.command == "adapt":
        adapt_grasp(args)
    elif args.command == "run" or args.command == "file":
        run_grasp(args)
    elif args.command == "serve":
        serve_grasp(args)
    elif args.command == "evaluate":
        evaluate_grasp(args)
    elif args.command == "examples":
        ExampleIndex.build(
            args.examples_file,
            args.output_dir,
            args.batch_size,
            args.overwrite,
            args.log_level,
        )


if __name__ == "__main__":
    main()

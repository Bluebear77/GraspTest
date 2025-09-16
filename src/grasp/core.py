import json
import time
from copy import deepcopy
from logging import Logger
from typing import Any, Iterator

from litellm.exceptions import Timeout
from search_index.similarity import EmbeddingModel
from universal_ml_utils.logging import get_logger

from grasp.configs import Config
from grasp.functions import (
    call_function,
    kg_functions,
)
from grasp.manager import KgManager, find_embedding_model, format_kgs, load_kg_manager
from grasp.manager.utils import describe_index, load_general_notes, load_kg_notes
from grasp.model import call_model
from grasp.tasks import (
    rules as general_rules,
)
from grasp.tasks import (
    task_done,
    task_functions,
    task_output,
    task_rules,
    task_setup,
    task_system_information,
)
from grasp.tasks.examples import ExampleIndex
from grasp.tasks.feedback import format_feedback, generate_feedback
from grasp.tasks.sparql_qa.examples import find_examples
from grasp.tasks.sparql_qa.examples import functions as example_functions
from grasp.utils import (
    format_list,
    format_message,
    format_notes,
    format_prefixes,
)


def message_hash(msg: dict) -> str:
    msg = deepcopy(msg)
    for tool_call in msg.get("tool_calls", []):
        # id would make hash useless
        tool_call.pop("id")
    return json.dumps(msg, sort_keys=True)


def system_instructions(
    task: str,
    managers: list[KgManager],
    kg_notes: dict[str, list[str]],
    notes: list[str],
) -> str:
    index_types = set()
    prefixes = {}
    for manager in managers:
        prefixes.update(manager.prefixes)
        index_types.add(manager.entity_index.get_type())
        index_types.add(manager.property_index.get_type())

    index_infos = []
    for index_type in sorted(index_types):
        title, desc = describe_index(index_type)
        index_infos.append(f"{title}: {desc}")

    system_info = task_system_information(task)

    instructions = f"""\
{system_info}

Available knowledge graphs:
{format_kgs(managers, kg_notes)}

Index types used for entities and properties:
{format_list(index_infos)}

"""

    if notes:
        instructions += f"""\
General notes across knowledge graphs:
{format_notes(notes)}

"""

    instructions += f"""\
SPARQL prefixes for use in function calls:
{format_prefixes(prefixes)}

Additional rules to follow:
{format_list(general_rules() + task_rules(task))}"""

    return instructions


def setup(config: Config) -> list[KgManager]:
    emb_model: EmbeddingModel | None = None
    managers: list[KgManager] = []
    for kg in config.knowledge_graphs:
        if emb_model is None:
            # find and set embedding model
            emb_model = find_embedding_model(managers)

        manager = load_kg_manager(
            kg,
            entities_kwargs={"model": emb_model},
            properties_kwargs={"model": emb_model},
        )
        managers.append(manager)

    return managers


def load_task_notes(
    task: str,
    config: Config,
) -> tuple[list[str], dict[str, list[str]]]:
    # load notes
    general_notes = load_general_notes(task, config.notes_file)

    kg_notes = {}
    for kg in config.knowledge_graphs:
        notes = load_kg_notes(kg.kg, task, kg.notes_file)
        kg_notes[kg.kg] = notes

    return general_notes, kg_notes


def generate(
    task: str,
    input: Any,
    config: Config,
    managers: list[KgManager],
    kg_notes: dict[str, list[str]] | None = None,
    notes: list[str] | None = None,
    example_indices: dict[str, ExampleIndex] | None = None,
    past_inputs: list[str] | None = None,
    past_messages: list[dict] | None = None,
    past_known: set[str] | None = None,
    logger: Logger = get_logger("GRASP"),
) -> Iterator[dict]:
    if task != "sparql-qa":
        # disable examples for tasks other than sparql-qa
        # to avoid errors due to missing implementations
        config = deepcopy(config)
        config.force_examples = None
        logger.debug(f"Disabling examples for {task} task")

    # setup functions
    fns = kg_functions(managers, config.fn_set)
    task_fns, task_handler = task_functions(managers, task)
    fns.extend(task_fns)

    if task == "sparql-qa" and example_indices:
        ex_fns = example_functions(
            example_indices,
            config.num_examples,
            config.random_examples,
        )
        task_fns.extend(ex_fns)

    input, task_state = task_setup(task, input)

    if notes is None:
        notes = []
    if kg_notes is None:
        kg_notes = {}

    # setup messages
    system_instruction = system_instructions(task, managers, kg_notes, notes)
    yield {
        "type": "system",
        "functions": fns,
        "system_message": system_instruction,
    }

    # log stuff
    logger.debug(
        format_message(
            {
                "role": "config",
                "content": config.model_dump_json(indent=2, exclude_none=True),
            }
        )
    )
    logger.debug(
        format_message(
            {
                "role": "functions",
                "content": json.dumps([fn["name"] for fn in fns]),
            }
        )
    )

    # handle past
    api_messages = [{"role": "system", "content": system_instruction}]
    if past_messages:
        # overwrite system message because new set of
        # knowledge graphs or functions might be present
        assert past_messages[0]["role"] == "system", (
            "First past message should be system"
        )

        api_messages.extend(past_messages[1:])

    inputs = past_inputs or []
    inputs.append(input)
    known = past_known or set()

    start = time.perf_counter()

    # add user input
    api_messages.append({"role": "user", "content": input})

    if config.force_examples and example_indices:
        try:
            example_messages = find_examples(
                managers,
                example_indices,
                config.force_examples,
                input,
                config.num_examples,
                config.random_examples,
                known,
                config.result_max_rows,
                config.result_max_columns,
            )

            # add to messages
            api_messages.extend(example_messages)

            # yield to user
            content = example_messages[0]["content"]
            yield {"type": "model", "content": content}

            tool_call = example_messages[0]["tool_calls"][0]["function"]
            content = example_messages[1]["content"]
            yield {
                "type": "tool",
                "name": tool_call["name"],
                "args": json.loads(tool_call["arguments"]),
                "result": content,
            }

        except Exception:
            logger.warning(
                f"{config.force_examples:=} specified but corresponding manager not found "
                "or without example index, ignoring"
            )

    # log all messages so far
    for msg in api_messages:
        logger.debug(format_message(msg))

    error: dict | None = None

    # keep track of last serialized message to detect loops
    # if model emits the same message twice, we are stuck
    last_msg_hash: str | None = None
    retries = 0
    while len(api_messages) < config.max_messages:
        try:
            response = call_model(api_messages, fns, config)
        except Timeout:
            error = {
                "content": "LLM API timed out",
                "reason": "timeout",
            }
            logger.error("LLM API timed out")
            break
        except Exception as e:
            error = {
                "content": f"Failed to generate response:\n{e}",
                "reason": "api",
            }
            logger.error(format_message({"role": "error", **error}))
            break

        if not response.choices:  # type: ignore
            error = {
                "content": "No choices from LLM API",
                "reason": "no_choices",
            }
            logger.error(format_message({"role": "error", **error}))
            break

        choice = response.choices[0]  # type: ignore
        usage = response.usage.model_dump(exclude_none=True)  # type: ignore

        msg = choice.message.model_dump(exclude_none=True)  # type: ignore
        api_messages.append(msg)

        msg_hash = message_hash(msg)
        if last_msg_hash == msg_hash:
            error = {
                "content": "LLM appears to be stuck in a loop",
                "reason": "loop",
            }
            logger.error(format_message({"role": "error", **error}))
            break

        last_msg_hash = msg_hash

        # display usage info for assistant messages
        fmt_msg = deepcopy(msg)
        fmt_msg["role"] += f" (usage={usage})"
        logger.debug(format_message(fmt_msg))

        # yield message
        content = ""
        if msg.get("reasoning_content"):
            content += f"Reasoning:\n{msg['reasoning_content'].strip()}\n\n"
        content += msg.get("content", "").strip()
        if content:
            yield {"type": "model", "content": content}

        if choice.finish_reason not in ["tool_calls", "stop", "length"]:
            error = {
                "content": f"Unexpected finish reason {choice.finish_reason}",
                "reason": "invalid_finish_reason",
            }
            logger.error(format_message({"role": "error", **error}))
            break

        elif choice.finish_reason == "length":
            break

        # no tool calls mean we should stop
        should_stop = not choice.message.tool_calls  # type: ignore

        # execute tool calls
        for tool_call in choice.message.tool_calls or []:  # type: ignore
            fn_name: str = tool_call.function.name  # type: ignore
            fn_args = json.loads(tool_call.function.arguments)

            try:
                result = call_function(
                    managers,
                    fn_name,
                    fn_args,
                    config.fn_set,
                    known,
                    task_handler,
                    task_state,
                    result_max_rows=config.result_max_rows,
                    result_max_columns=config.result_max_columns,
                    list_k=config.list_k,
                    search_top_k=config.search_top_k,
                    num_examples=config.num_examples,
                    know_before_use=config.know_before_use,
                    example_indices=example_indices,
                )
            except Exception as e:
                result = f"Call to function {fn_name} returned an error:\n{e}"

            tool_msg = {"role": "tool", "content": result, "tool_call_id": tool_call.id}
            api_messages.append(tool_msg)
            logger.debug(format_message(tool_msg))

            yield {
                "type": "tool",
                "name": fn_name,
                "args": fn_args,
                "result": result,
            }

            if task_done(task, fn_name):
                # we are done
                should_stop = True
                break

        can_give_feedback = config.feedback and retries < config.max_feedbacks

        if should_stop and not can_give_feedback:
            # done
            break

        elif not should_stop:  # and (choice.message.tool_calls or alternating):
            # not done yet
            continue

        elif not can_give_feedback:
            # no feedback possible, despite answer or cancel
            break

        # get latest output
        output = task_output(task, api_messages, managers, config, task_state)
        if output is None:
            break

        # provide feedback
        try:
            feedback = generate_feedback(
                task,
                managers,
                config,
                kg_notes,
                notes,
                inputs,
                output,
                logger,
            )
        except Exception as e:
            error = {
                "content": f"Failed to generate feedback:\n{e}",
                "reason": "feedback",
            }
            logger.error(format_message({"role": "error", **error}))
            break

        if feedback is None:
            # no feedback
            break

        msg = {
            "role": "user",
            "content": format_feedback(feedback),
        }
        logger.debug(format_message(msg))
        api_messages.append(msg)
        yield {
            "type": "feedback",
            "status": feedback["status"],
            "feedback": feedback["feedback"],
        }

        if feedback["status"] == "done":
            break

        # if not done, continue
        retries += 1
        # reset loop detection
        last_msg_hash = None

    output = task_output(task, api_messages, managers, config, task_state)

    logger.info(
        format_message({"role": "output", "content": json.dumps(output, indent=2)})
    )

    end = time.perf_counter()
    yield {
        "type": "output",
        "task": task,
        "output": output,
        "elapsed": end - start,
        "error": error,
        "inputs": inputs,
        "messages": api_messages,
        "known": list(known),
    }

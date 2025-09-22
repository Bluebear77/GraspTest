from copy import deepcopy
from logging import Logger

from litellm.exceptions import Timeout

from grasp.adapt.notes import (
    call_function,
    note_functions,
)
from grasp.adapt.utils import format_output
from grasp.configs import Adapt, Config
from grasp.core import call_model
from grasp.functions import find_manager
from grasp.manager import KgManager
from grasp.model import Message
from grasp.tasks.cea import Annotation, AnnotationState, CeaSample, prepare_annotation
from grasp.tasks.sparql_qa.examples import SparqlQaSample
from grasp.tasks.utils import Sample, format_sparql_result, prepare_sparql_result
from grasp.utils import (
    format_enumerate,
    format_list,
    format_message,
    format_response,
)


def rules() -> list[str]:
    return [
        "Do not take notes on things that are already handled well by the system.",
        "Avoid notes about entity or property identifiers just for the sake of not \
having to look them up again.",
        "As you hit the limits on the number of notes and their length, \
gradually generalize your notes, discard unnecessary details, and move \
notes that can be useful across knowledge graphs to the general section.",
    ]


def system_instructions(max_notes: int, max_note_length: int) -> str:
    return f"""\
You are a note-taking assistant. Your task is to \
inspect the traces of a knowledge graph agent performing a certain task, and to \
take notes about the agent's outputs as well as the used knowledge \
graphs and functions. Before calling a note-taking function, \
provide reasoning for what you are doing and why. Stop the annotation process \
by calling the stop function once you are done.

Your notes should help the agent to better understand and \
navigate the task and knowledge graphs in the future. For a specific knowledge \
graph, they should generalize across samples, rather than being specific to \
a single sample or output. You can also take general notes that might be \
useful across knowledge graphs or for the task in general. \
You are only allowed {max_notes} notes at max per knowledge graph and for the \
general notes, such that you are forced to prioritize and to keep them as widely \
applicable as possible. Notes are limited to {max_note_length} characters to \
ensure they are concise and to the point.

Examples of potentially useful types of notes include:
- overall structure and schema of the knowledge graphs
- peculiarities of the knowledge graphs
- strategies when encountering certain types of questions or errors
- tips for when and how to use certain functions

Additional rules to follow:
{format_list(rules())}"""


def prepare_groundtruth(
    sample: Sample,
    kg: str,
    managers: list[KgManager],
    config: Config,
) -> str:
    if isinstance(sample, SparqlQaSample):
        sparql, selections, result = prepare_sparql_result(
            sample.sparql,
            kg,
            managers,
            config.result_max_rows,
            config.result_max_columns,
        )
        return format_sparql_result(sparql, selections, result, kg)

    elif isinstance(sample, CeaSample):
        manager, _ = find_manager(managers, kg)

        annots = AnnotationState(sample.table)
        for annot in sample.annotations:
            full_annot = prepare_annotation(manager, annot.entity)
            annots.annotate(annot.row, annot.column, full_annot)

        return annots.format()

    else:
        raise ValueError(f"Unsupported or unknown sample type {type(sample)}")


def note_taking_instructions(
    managers: list[KgManager],
    kg_notes: dict[str, list[str]],
    notes: list[str],
    config: Config,
    inputs: list[tuple[str, Sample]],
    outputs: list[dict],
) -> str:
    formatted = []
    for i, ((kg, sample), output) in enumerate(zip(inputs, outputs)):
        messages = [Message(**msg) for msg in output["messages"]]
        if i == 0:
            assert messages[0].role == "system"
            formatted.append(f"Task instructions for the agent:\n{messages[0].content}")

        assert messages[1].role == "user"
        input = messages[1].content

        gt = prepare_groundtruth(sample, kg, managers, config)

        content = f"""\
Input {i + 1} over {kg} knowledge graph:
{input}

Agent trace:
{format_output(output["output"], messages)}

Ground truth:
{gt}"""

        formatted.append(content)

    fmt = "\n\n".join(formatted)
    kg_specific_notes = format_list(
        f"{kg}:\n{format_enumerate(kg_specific_notes, indent=2)}"
        for kg, kg_specific_notes in sorted(kg_notes.items())
    )

    return f"""\
Add to, delete from, or update the following notes (which might \
be the same notes provided to the agent) based on the given agent traces \
below.

Knowledge graph specific notes:
{kg_specific_notes}

General notes across knowledge graphs:
{format_enumerate(notes)}

{fmt}"""


def take_notes(
    inputs: list[tuple[str, Sample]],
    outputs: list[dict],
    managers: list[KgManager],
    kg_notes: dict[str, list[str]],
    notes: list[str],
    config: Adapt,
    logger: Logger,
) -> None:
    # get note taking parameters
    max_notes = config.method_kwargs.get("max_notes", 16)
    max_note_length = config.method_kwargs.get("max_note_length", 512)

    messages = [
        Message(
            role="system",
            content=system_instructions(max_notes, max_note_length),
        ),
        Message(
            role="user",
            content=note_taking_instructions(
                managers,
                kg_notes,
                notes,
                config,
                inputs,
                outputs,
            ),
        ),
    ]

    for msg in messages:
        logger.debug(format_message(msg))

    functions = note_functions(managers)

    num_messages = len(messages)

    # copy config to avoid modifying the original
    config = deepcopy(config)
    config.model = config.adapt_model or config.model
    config.model_endpoint = config.adapt_model_endpoint or config.model_endpoint
    config.temperature = config.adapt_temperature or config.temperature
    config.top_p = config.adapt_top_p or config.top_p
    config.reasoning_effort = config.adapt_reasoning_effort or config.reasoning_effort
    config.api = config.adapt_api or config.api

    while len(messages) - num_messages < config.adapt_max_steps:
        try:
            response = call_model(messages, functions, config)
        except Timeout:
            logger.error("LLM API timed out during note taking")
            return
        except Exception as e:
            logger.error(f"LLM API returned error during note taking: {e}")
            return

        if response.is_empty:
            logger.error("LLM API returned empty response during note taking")
            return

        messages.append(Message(role="assistant", content=response))

        for tool_call in response.tool_calls:
            try:
                result = call_function(
                    kg_notes,
                    notes,
                    tool_call.name,
                    tool_call.args,
                    max_notes,
                    max_note_length,
                )
            except Exception as e:
                result = f"Call to function {tool_call.name} returned an error:\n{e}"

            tool_call.result = result

            if tool_call.name == "stop":
                return

        # only log now once tool call results are set
        logger.debug(format_response(response))

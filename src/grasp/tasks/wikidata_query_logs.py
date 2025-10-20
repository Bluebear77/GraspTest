from typing import Any

from grasp.configs import GraspConfig
from grasp.functions import TaskFunctions
from grasp.manager import KgManager
from grasp.model import Message, Response
from grasp.tasks.utils import format_sparql_result, prepare_sparql_result
from grasp.utils import format_list


def call_function(
    config: GraspConfig,
    managers: list[KgManager],
    fn_name: str,
    fn_args: dict,
    known: set[str],
    state: Any | None = None,
    example_indices: dict | None = None,
) -> str:
    if fn_name == "answer" or fn_name == "cancel":
        return "Stopping"

    else:
        raise ValueError(f"Unknown function {fn_name}")


def functions() -> TaskFunctions:
    fns = [
        {
            "name": "answer",
            "description": "Finalize your output and stop.",
            "parameters": {
                "type": "object",
                "properties": {
                    "sparql": {
                        "type": "string",
                        "description": "The final cleaned SPARQL query",
                    },
                    "questions": {
                        "type": "array",
                        "description": "A list of natural language questions corresponding to the SPARQL query",
                        "items": {
                            "type": "string",
                            "description": "A natural language question corresponding to the SPARQL query",
                        },
                    },
                },
                "required": ["sparql", "questions"],
                "additionalProperties": False,
            },
            "strict": True,
        },
        {
            "name": "cancel",
            "description": "Stop the task without producing an output.",
            "parameters": {
                "type": "object",
                "properties": {
                    "reason": {
                        "type": "string",
                        "description": "The reason for cancelling the task",
                    },
                },
                "required": ["reason"],
                "additionalProperties": False,
            },
            "strict": True,
        },
    ]
    return fns, call_function


def rules() -> list[str]:
    return [
        "The generated questions should be diverse regarding the phrasing "
        "(e.g. keyword-like, formulated in a requesting or asking manner, etc.).",
        "You can use the cancel function at any time to stop the task without producing an output "
        "(e.g. if the SPARQL query is invalid or does not make sense).",
    ]


def system_information(config: GraspConfig) -> str:
    max_questions = config.task_kwargs.get("max_questions")
    return f"""\
You are a Wikidata expert trying to find possible user questions for \
anonymized SPARQL queries sent to the Wikidata Query Service. \
Your task is to generate one or more natural language questions that \
correspond to a given SPARQL query.

You should take a step-by-step approach to understand the query and \
generate the questions:
1. Analyze the given SPARQL query, its used entities and properties, and \
execution result. Think about what the user wanted to achieve with this query. \
Search and query Wikidata to gain more context about the SPARQL query, if needed.
2. Clean the SPARQL query. This e.g. includes removing superfluous variables or other \
unnecessary parts, finding better variable names, or replacing anonymized string \
literals with sensible values.
3. Formulate your final SPARQL query and validate it against Wikidata. \
It should not be too different from the original anonymous query in terms of \
intent and its execution result, but you are allowed to deviate if it would make \
the query more natural, precise, etc.
4. For the final SPARQL query, generate between 1 and {max_questions} natural \
language questions that accurately reflect its intent.
5. Provide your final output by calling the answer function."""


def output(
    messages: list[Message],
    managers: list[KgManager],
    max_rows: int,
    max_columns: int,
) -> Any | None:
    try:
        last = messages[-1]
        assert isinstance(last.content, Response)
        tool_call = last.content.tool_calls[0]
        output: dict[str, str] = {"formatted": "No output", **tool_call.args}
        if tool_call.name == "answer":
            assert len(managers) == 1, "Only one KG manager expected"
            manager = managers[0]
            result, selections = prepare_sparql_result(
                tool_call.args["sparql"],
                manager.kg,
                managers,
                max_rows,
                max_columns,
            )

            output["type"] = "answer"
            questions = tool_call.args["questions"]
            output["formatted"] = f"Questions:\n{format_list(questions)}\n\n"
            output["formatted"] += format_sparql_result(manager, result, selections)

        elif tool_call.name == "cancel":
            output["type"] = "cancel"
            output["formatted"] = f"Cancelled:\n{tool_call.args['reason']}"

        else:
            raise ValueError(f"Unknown output tool call {tool_call.name}")

        return output

    except Exception:
        return None

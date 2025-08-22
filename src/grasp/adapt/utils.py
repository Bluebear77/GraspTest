import json
import os

from universal_ml_utils.ops import partition_by

from grasp.configs import Config
from grasp.functions import execute_sparql
from grasp.manager import KgManager
from grasp.sparql.item import get_sparql_items, selections_from_items


def prepare_sparql(
    kg: str,
    sparql: str,
    managers: list[KgManager],
    config: Config,
) -> str:
    manager, others = partition_by(managers, lambda m: m.kg == kg)
    assert len(manager) == 1, (
        f"Expected exactly one manager for kg {kg}, got {len(manager)}"
    )
    manager = manager[0]

    try:
        result, sparql = execute_sparql(
            manager,
            others,
            sparql,
            config.result_max_rows,
            config.result_max_columns,
            set(),
            return_sparql=True,
        )
        sparql = manager.prettify(sparql)
    except Exception as e:
        result = f"Failed to execute SPARQL query:\n{e}"

    try:
        _, items = get_sparql_items(sparql, manager)
        selections = selections_from_items(items)
        selections = manager.format_selections(selections)
    except Exception as e:
        selections = f"Failed to determine used entities and properties:\n{e}"

    fmt = f"SPARQL query:\n{sparql.strip()}"

    if selections:
        fmt += f"\n\n{selections}"

    fmt += f"\n\nExecution result:\n{result.strip()}"
    return fmt


def format_arguments(args, depth: int = 0) -> str:
    if isinstance(args, list):
        return "[" + ", ".join(format_arguments(i, depth + 1) for i in args) + "]"
    elif isinstance(args, dict):
        return (
            "{" * (depth > 0)
            + ", ".join(
                f"{k}={format_arguments(v, depth + 1)}" for k, v in args.items()
            )
            + "}" * (depth > 0)
        )
    elif isinstance(args, str):
        return f'"{args}"'
    else:
        return str(args)


def format_output(output: dict) -> str:
    tool_call_results = {
        message["tool_call_id"]: message["content"]
        for message in output["messages"]
        if message["role"] == "tool"
    }
    fmt = []
    step = 1
    for message in output["messages"][2:]:
        if message["role"] == "tool":
            continue
        elif message["role"] == "user":
            fmt.append(f"Feedback:\n{message['content']}")
            continue

        assert message["role"] == "assistant"

        content = f"System step {step}:"
        if message.get("reasoning_content"):
            content += f"\n{message['reasoning_content'].strip()}"
        if message.get("content"):
            content += f"\n{message['content'].strip()}"

        tool_calls = []
        for tool_call in message.get("tool_calls", []):
            if tool_call["type"] != "function":
                continue

            tool_call_fn = tool_call["function"]
            tool_calls.append(
                f'Call of "{tool_call_fn["name"]}" function '
                f"with {format_arguments(json.loads(tool_call_fn['arguments']))}:\n"
                f"{tool_call_results[tool_call['id']]}"
            )

        content += "\n" + "\n".join(tool_calls)

        fmt.append(content.strip())
        step += 1

    return "\n\n".join(fmt)


def link(src: str, dst: str) -> None:
    src = os.path.abspath(src)
    dst = os.path.abspath(dst)
    if os.path.lexists(dst):
        os.remove(dst)

    rel = os.path.relpath(src, os.path.dirname(dst))
    os.symlink(rel, dst)

import os

from grasp.model import Message, Response
from grasp.utils import format_list


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


def format_output(messages: list[Message]) -> str:
    fmt = []
    step = 1
    for message in messages[2:]:
        if message.role == "feedback":
            fmt.append(f"Feedback:\n{message.content}")
            continue

        elif message.role == "user":
            fmt.append(f"User:\n{message.content}")
            continue

        assert isinstance(message.content, Response)

        assistant = message.content
        ass_content = assistant.get_content()

        contents = []
        if "reasoning" in ass_content:
            contents.append(f"Reasoning: {ass_content['reasoning']}")

        if "content" in ass_content:
            contents.append(f"Output: {ass_content['content']}")

        for tool_call in assistant.tool_calls:
            contents.append(
                f'Call of "{tool_call.name}" function '
                f"with {format_arguments(tool_call.args)}:\n"
                f"{tool_call.result}"
            )

        content = f"System step {step}:\n{format_list(contents)}"
        fmt.append(content)
        step += 1

    return "\n\n".join(fmt)


def link(src: str, dst: str) -> None:
    src = os.path.abspath(src)
    dst = os.path.abspath(dst)
    if os.path.lexists(dst):
        os.remove(dst)

    rel = os.path.relpath(src, os.path.dirname(dst))
    os.symlink(rel, dst)

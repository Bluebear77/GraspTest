import json
from typing import Any

from litellm import Choices, ResponsesAPIResponse, completion, responses
from litellm.types.utils import ModelResponse
from pydantic import BaseModel

from grasp.configs import Config


class ToolCall(BaseModel):
    id: str
    name: str
    args: dict[str, Any]
    result: str | None = None

    @staticmethod
    def from_dict(data: dict[str, Any]) -> "ToolCall":
        return ToolCall(
            id=data["id"],
            name=data["name"],
            args=data["args"],
            result=data.get("result"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "args": self.args,
            "result": self.result,
        }


class Response(BaseModel):
    message: str | None
    reasoning: str | None = None
    tool_calls: list[ToolCall]
    usage: dict | None = None

    state: Any | None = None

    @property
    def has_content(self) -> bool:
        return self.message is not None or self.reasoning is not None

    def get_content(self) -> str:
        content = ""
        if self.reasoning is not None:
            content += f"**Reasoning**\n{self.reasoning}\n\n"

        if self.message is not None:
            if self.reasoning is not None:
                content += "---\n\n**Content**\n"

            content += self.message

        return content.strip()

    @staticmethod
    def from_completions_api(response: ModelResponse) -> "Response":
        if not response.choices:
            return Response(message=None, tool_calls=[])

        choice: Choices = response.choices[0]  # type: ignore
        if choice.finish_reason not in ["tool_calls", "stop", "length"]:
            raise ValueError(f"Unexpected finish reason {choice.finish_reason}")

        message = choice.message.content
        reasoning = None
        if hasattr(choice.message, "reasoning_content"):
            reasoning = choice.message.reasoning_content

        tool_calls = []
        for tool_call in choice.message.tool_calls or []:
            if tool_call.type != "function":
                continue

            assert tool_call.function.name is not None

            tool_calls.append(
                ToolCall(
                    id=tool_call.id,
                    name=tool_call.function.name,
                    args=json.loads(tool_call.function.arguments),
                )
            )

        return Response(
            message=message,
            reasoning=reasoning,
            tool_calls=tool_calls,
            usage=response.usage.model_dump(exclude_defaults=True),  # type: ignore
        )

    @staticmethod
    def from_responses_api(response: ResponsesAPIResponse) -> "Response":
        message = None
        reasoning = None
        tool_calls = []

        for output in response.output:
            pass

        usage = response.usage.model_dump(exclude_defaults=True)  # type: ignore
        return Response(
            message=message,
            reasoning=reasoning,
            tool_calls=tool_calls,
            usage=usage,
        )

    @property
    def is_empty(self) -> bool:
        return (
            self.message is None
            and self.reasoning is None
            and not len(self.tool_calls) == 0
        )

    def hash(self) -> str:
        msg: dict[str, Any] = {
            "msg": self.message,
            "reasoning": self.reasoning,
            "tool_calls": sorted((tc.name, tc.args) for tc in self.tool_calls),
        }
        return json.dumps(msg, sort_keys=True)


class Message(BaseModel):
    role: str
    content: str | Response


def completions_api_messages(messages: list[Message]) -> list[dict[str, Any]]:
    msgs = []
    for message in messages:
        if isinstance(message.content, str):
            # feedback is treated as coming from user
            role = message.role if message.role != "feedback" else "user"
            msgs.append(
                {
                    "role": role,
                    "content": message.content,
                }
            )
            continue

        # response content
        assistant = message.content
        tool_calls = []
        tool_call_results = []
        for tool_call in assistant.tool_calls:
            tool_calls.append(
                {
                    "id": tool_call.id,
                    "type": "function",
                    "function": {
                        "name": tool_call.name,
                        "arguments": json.dumps(tool_call.args),
                    },
                }
            )
            assert tool_call.result is not None, "Expected tool call result"
            tool_call_results.append(
                {
                    "tool_call_id": tool_call.id,
                    "role": "tool",
                    "content": tool_call.result,
                }
            )

        msg = {
            "role": message.role,
            "content": assistant.message,
            "reasoning_content": assistant.reasoning,
        }
        if tool_calls:
            msg["tool_calls"] = tool_calls

        msgs.append(msg)

        if tool_call_results:
            msgs.extend(tool_call_results)

    return msgs


def call_model(
    messages: list[Message],
    functions: list[dict],
    config: Config,
) -> Response:
    if config.api == "completions":
        # use old chat completions API
        completions_resp: ModelResponse = completion(
            model=config.model,
            messages=completions_api_messages(messages),
            tools=[{"type": "function", "function": fn} for fn in functions],
            tool_choice="auto",
            # decoding parameters
            temperature=config.temperature,
            top_p=config.top_p,
            reasoning_effort=config.reasoning_effort,  # type: ignore
            # should be set to more than enough until the next function call
            max_completion_tokens=config.max_completion_tokens,
            base_url=config.model_endpoint,
            timeout=config.completion_timeout,
            seed=config.seed,
            extra_body={} if config.model_kwargs is None else config.model_kwargs,
            # drop unsupported parameters
            drop_params=True,
        )
        return Response.from_completions_api(completions_resp)

    elif config.api == "responses":
        # use responses API
        responses_resp: ResponsesAPIResponse = responses(
            model=config.model,
            input=messages,  # type: ignore
            include=["reasoning.encrypted_content"],
            tools=[{"type": "function", **fn} for fn in functions],  # type: ignore
            tool_choice="auto",
            # decoding parameters
            temperature=config.temperature,
            top_p=config.top_p,
            reasoning={
                "effort": config.reasoning_effort,
                "summary": config.reasoning_summary,
            },  # type: ignore
            truncation="auto",
            # should be set to more than enough until the next function call
            max_output_tokens=config.max_completion_tokens,
            base_url=config.model_endpoint,
            timeout=config.completion_timeout,
            seed=config.seed,
            extra_body={} if config.model_kwargs is None else config.model_kwargs,
            # drop unsupported parameters
            drop_params=True,
            store=False,
        )
        return Response.from_responses_api(responses_resp)

    else:
        raise ValueError(f"Unknown API {config.api}")

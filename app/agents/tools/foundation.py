from dataclasses import dataclass, field
from typing import Any


class BaseTool:
    """Project-local base class for all callable tools."""

    def __init__(self, config: dict | None = None, tool_schema: Any | None = None):
        self.config = config or {}
        self.tool_schema = tool_schema


class TextTool(BaseTool):
    """Lightweight base class for text tools."""

    def __init__(
        self,
        selected_language: str,
        config: dict | None = None,
        tool_schema: Any | None = None,
    ):
        super().__init__(config=config, tool_schema=tool_schema)
        self.selected_language = selected_language


@dataclass
class OpenAIFunctionPropertySchema:
    type: str
    description: str = ""
    enum: list[str] | None = None


@dataclass
class OpenAIFunctionParametersSchema:
    type: str
    properties: dict[str, OpenAIFunctionPropertySchema] = field(default_factory=dict)
    required: list[str] = field(default_factory=list)


@dataclass
class OpenAIFunctionSchema:
    name: str
    description: str
    parameters: OpenAIFunctionParametersSchema
    strict: bool = False


@dataclass
class OpenAIFunctionToolSchema:
    type: str
    function: OpenAIFunctionSchema


@dataclass
class ToolResponse:
    text: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_payload(cls, payload: Any, metadata: dict[str, Any] | None = None) -> "ToolResponse":
        return cls(text=str(payload), metadata=metadata or {})


def rollout_trace_op(func):
    return func

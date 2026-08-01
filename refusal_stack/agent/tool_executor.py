"""ToolExecutor: dispatch tool calls and log results."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

log = logging.getLogger(__name__)


@dataclass
class ToolCallRecord:
    name: str
    arguments: dict
    result: str
    error: str | None = None


class ToolExecutor:
    def __init__(self, mock: bool = True):
        self.mock = mock
        self.call_log: list[ToolCallRecord] = []

    def execute(self, name: str, arguments: dict) -> str:
        from refusal_stack.agent.tools import web_search, python_exec, retrieval, ToolError

        try:
            if name == "web_search":
                result = web_search(arguments.get("query", ""), mock=self.mock)
            elif name == "python_exec":
                result = python_exec(arguments.get("code", ""), mock=self.mock)
            elif name == "retrieval":
                result = retrieval(arguments.get("doc_id", ""), mock=self.mock)
            else:
                raise ToolError(f"Unknown tool: {name}")
            record = ToolCallRecord(name=name, arguments=arguments, result=result)
            self.call_log.append(record)
            return result
        except ToolError as e:
            err = str(e)
            record = ToolCallRecord(name=name, arguments=arguments, result="", error=err)
            self.call_log.append(record)
            raise ToolError(f"{name}: {err}") from e

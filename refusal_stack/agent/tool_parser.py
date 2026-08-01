"""Parse tool calls from assistant message text."""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass

log = logging.getLogger(__name__)

_XML_RE = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)
_JSON_RE = re.compile(r'\{[^{}]*"name"\s*:\s*"([^"]+)"[^{}]*"arguments"\s*:\s*(\{[^{}]*\})[^{}]*\}', re.DOTALL)


@dataclass
class ToolCall:
    name: str
    arguments: dict


def parse_tool_calls(text: str) -> list[ToolCall]:
    calls: list[ToolCall] = []

    for match in _XML_RE.finditer(text):
        try:
            data = json.loads(match.group(1).strip())
            calls.append(ToolCall(name=data["name"], arguments=data.get("arguments", data.get("parameters", {}))))
        except Exception as e:
            log.warning(f"Failed to parse XML tool call: {e}")

    if not calls:
        for match in _JSON_RE.finditer(text):
            try:
                name = match.group(1)
                args = json.loads(match.group(2))
                calls.append(ToolCall(name=name, arguments=args))
            except Exception as e:
                log.warning(f"Failed to parse JSON tool call: {e}")

    return calls

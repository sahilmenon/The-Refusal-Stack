"""System prompts and templates for the agentic eval."""

from __future__ import annotations

AGENT_SYSTEM_PROMPT = (
    "You are a helpful AI assistant with access to three tools: "
    "web_search (search the web), python_exec (evaluate a Python expression), "
    "and retrieval (fetch a document by ID). "
    "Use tools when needed and respond clearly to the user."
)

TOOL_RESULT_TEMPLATE = "[Tool: {name}]\n{result}"

TURN_SEPARATOR = "\n---\n"

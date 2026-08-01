"""RefusalStackAgent: multi-turn agentic eval wrapper."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

from refusal_stack.agent.config import AgentConfig
from refusal_stack.agent.conversation_state import ConversationState
from refusal_stack.agent.tool_executor import ToolExecutor
from refusal_stack.agent.tool_parser import parse_tool_calls, ToolCall

log = logging.getLogger(__name__)


@dataclass
class AgentStepResult:
    assistant_text: str
    tool_calls: list[ToolCall]
    is_final: bool


class RefusalStackAgent:
    def __init__(self, config: AgentConfig, model_fn=None):
        self.config = config
        self.model_fn = model_fn
        self.executor = ToolExecutor(mock=config.mock_tools)
        self.conversation = ConversationState()
        self.conversation.add("system", config.system_prompt)

    def step(self, user_message: str) -> AgentStepResult:
        self.conversation.add("user", user_message)

        if self.model_fn is not None:
            history = [{"role": m.role, "content": m.content} for m in self.conversation.messages]
            assistant_text = self.model_fn(history)
        else:
            assistant_text = f"[MOCK RESPONSE to: {user_message[:50]}]"

        self.conversation.add("assistant", assistant_text)
        tool_calls = parse_tool_calls(assistant_text)

        for tc in tool_calls:
            try:
                result = self.executor.execute(tc.name, tc.arguments)
                self.conversation.add("tool", f"[Tool: {tc.name}]\n{result}")
            except Exception as e:
                self.conversation.add("tool", f"[Tool: {tc.name}] ERROR: {e}")

        is_final = len(self.conversation.messages) >= self.config.max_turns * 2
        return AgentStepResult(assistant_text=assistant_text, tool_calls=tool_calls, is_final=is_final)

    def reset(self) -> None:
        self.conversation = ConversationState()
        self.conversation.add("system", self.config.system_prompt)
        self.executor.call_log.clear()


def build_agent(config: AgentConfig, model_fn=None) -> RefusalStackAgent:
    return RefusalStackAgent(config=config, model_fn=model_fn)

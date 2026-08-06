"""RefusalStackAgent: multi-turn agentic eval wrapper."""

from __future__ import annotations

import logging
from dataclasses import dataclass

from refusal_stack.agent.config import AgentConfig
from refusal_stack.agent.conversation_state import ConversationState
from refusal_stack.agent.tool_executor import ToolExecutor
from refusal_stack.agent.tool_parser import ToolCall, parse_tool_calls

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

        # Multi-turn loop: the model acts, tools run, and the model is re-invoked
        # with the tool RESULTS in context so it can react - this is what makes
        # the eval agentic. Previously step() returned after one model call, so
        # tool outputs were never seen and the agentic delta / injection attacks
        # measured nothing. Bounded by max_turns to prevent runaway tool loops.
        assistant_text = ""
        all_tool_calls: list[ToolCall] = []
        hit_limit = False
        for turn in range(self.config.max_turns):
            if self.model_fn is not None:
                history = [
                    {"role": m.role, "content": m.content} for m in self.conversation.messages
                ]
                assistant_text = self.model_fn(history)
            else:
                assistant_text = f"[MOCK RESPONSE to: {user_message[:50]}]"
            self.conversation.add("assistant", assistant_text)

            tool_calls = parse_tool_calls(assistant_text)
            all_tool_calls.extend(tool_calls)
            if not tool_calls:
                break  # model asked for no tools -> the exchange is complete

            for tc in tool_calls:
                try:
                    result = self.executor.execute(tc.name, tc.arguments)
                    self.conversation.add("tool", f"[Tool: {tc.name}]\n{result}")
                except Exception as e:
                    self.conversation.add("tool", f"[Tool: {tc.name}] ERROR: {e}")
            if turn == self.config.max_turns - 1:
                hit_limit = True

        # is_final: True when the model stopped on its own (no pending tools),
        # False if we cut it off at max_turns mid-tool-use.
        return AgentStepResult(
            assistant_text=assistant_text, tool_calls=all_tool_calls, is_final=not hit_limit
        )

    def reset(self) -> None:
        self.conversation = ConversationState()
        self.conversation.add("system", self.config.system_prompt)
        self.executor.call_log.clear()


def build_agent(config: AgentConfig, model_fn=None) -> RefusalStackAgent:
    return RefusalStackAgent(config=config, model_fn=model_fn)

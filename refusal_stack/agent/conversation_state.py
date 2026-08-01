"""ConversationState: stores the message history for a multi-turn agent session."""
from __future__ import annotations

import json
from dataclasses import dataclass, field


@dataclass
class Message:
    role: str
    content: str


@dataclass
class ConversationState:
    messages: list[Message] = field(default_factory=list)

    def add(self, role: str, content: str) -> None:
        self.messages.append(Message(role=role, content=content))

    def to_dict(self) -> dict:
        return {"messages": [{"role": m.role, "content": m.content} for m in self.messages]}

    def to_inspect_task_state(self):
        try:
            from inspect_ai.model import ChatMessageUser, ChatMessageAssistant, ChatMessageSystem
            result = []
            for m in self.messages:
                if m.role == "user":
                    result.append(ChatMessageUser(content=m.content))
                elif m.role == "assistant":
                    result.append(ChatMessageAssistant(content=m.content))
                elif m.role == "system":
                    result.append(ChatMessageSystem(content=m.content))
            return result
        except ImportError:
            return self.to_dict()

    def all_text(self) -> str:
        return "\n".join(m.content for m in self.messages if m.role in ("user", "assistant", "tool"))

    def __len__(self) -> int:
        return len(self.messages)

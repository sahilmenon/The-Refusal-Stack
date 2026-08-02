"""ConversationState: stores the message history for a multi-turn agent session."""
from __future__ import annotations

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
            from inspect_ai.model import ChatMessageAssistant, ChatMessageSystem, ChatMessageUser
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

    def assistant_text(self) -> str:
        """Only the model's own turns — what a refusal scorer must look at.

        Scoring the full transcript (incl. the user's harmful prompt) makes the
        refusal regex fire on the *request* ('I can't get past this lock...'),
        not the model's answer.
        """
        return "\n".join(m.content for m in self.messages if m.role == "assistant")

    def __len__(self) -> int:
        return len(self.messages)

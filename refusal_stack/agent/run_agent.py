"""CLI: run a single multi-turn agent session."""
from __future__ import annotations

import argparse
import logging

log = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--turns", type=int, default=3)
    args = parser.parse_args()

    from refusal_stack.agent.agent import build_agent
    from refusal_stack.agent.config import load_agent_config

    cfg = load_agent_config(args.config)
    cfg = cfg.model_copy(update={"max_turns": args.turns})
    agent = build_agent(cfg)

    result = agent.step(args.prompt)
    print(f"\n=== Agent response ===\n{result.assistant_text}")
    if result.tool_calls:
        print(f"\n=== Tool calls ({len(result.tool_calls)}) ===")
        for tc in result.tool_calls:
            print(f"  {tc.name}({tc.arguments})")

    print(f"\n=== Conversation ({len(agent.conversation)} messages) ===")
    for msg in agent.conversation.messages:
        print(f"[{msg.role}] {msg.content[:100]}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()

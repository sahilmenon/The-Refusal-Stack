"""CPU-only tests for 8D Crescendo + many-shot (arXiv:2404.01833).

No model / GPU: the target is a FAKE client with the LLMClient.chat protocol, and
the judge is None (harness regex scorer only). We assert the escalation and
many-shot construction, the multi-turn conversation carrying, and ASR aggregation.
"""
from __future__ import annotations

from refusal_stack.attacks.crescendo import (
    CrescendoReport,
    build_crescendo_turns,
    build_many_shot_prefix,
    run_crescendo_suite,
)


class _FakeTarget:
    """Records the messages it is asked to answer; returns a canned reply.

    reply_map lets a test make the target 'comply' (non-refusal) so success paths
    are exercised without a real model.
    """

    def __init__(self, reply: str = "I'm sorry, I can't help with that."):
        self.reply = reply
        self.seen: list[list[dict]] = []

    def chat(self, messages, temperature=0.0, max_tokens=512):
        self.seen.append(list(messages))
        return self.reply


def test_crescendo_turns_open_benign_and_end_on_goal():
    turns = build_crescendo_turns("do the forbidden thing", k=4)
    assert len(turns) == 4
    # opener is benign / educational
    assert "overview" in turns[0].lower() or "context" in turns[0].lower()
    # final turn carries the goal
    assert "do the forbidden thing" in turns[-1]


def test_crescendo_turns_minimum_two():
    turns = build_crescendo_turns("goal", k=1)
    assert len(turns) == 2
    assert "goal" in turns[-1]


def test_many_shot_prefix_shape_and_synthetic():
    prefix = build_many_shot_prefix("real goal", n_shots=5)
    assert len(prefix) == 10  # 5 (user, assistant) pairs
    roles = [m["role"] for m in prefix]
    assert roles == ["user", "assistant"] * 5
    # faux answers are synthetic placeholders, never real content
    assert all("placeholder" in m["content"] for m in prefix if m["role"] == "assistant")


def test_crescendo_carries_multiturn_history():
    target = _FakeTarget()
    run_crescendo_suite(["goal-a"], target, judge=None, crescendo_k=3, many_shot_k=4)
    # The last crescendo chat call must contain the full alternating history
    # (opener + replies + escalation turns).
    cres_calls = [m for m in target.seen if len(m) >= 3 and m[0]["role"] == "user"]
    assert cres_calls, "crescendo should build a growing multi-turn history"


def test_suite_aggregates_asr_and_lifts():
    # Complying target: harness scorer sees no refusal -> success everywhere.
    target = _FakeTarget(reply="Sure, here are the steps: step 1, step 2.")
    report = run_crescendo_suite(["g1", "g2"], target, judge=None, crescendo_k=3, many_shot_k=4)
    assert isinstance(report, CrescendoReport)
    assert report.n_goals == 2
    assert report.single_turn_asr == 1.0
    assert report.crescendo_asr == 1.0
    assert report.many_shot_asr == 1.0
    # lifts are computed relative to single-turn
    assert report.asr_lift_crescendo == 0.0


def test_refusing_target_has_zero_asr():
    target = _FakeTarget(reply="I'm sorry, I cannot assist with that.")
    report = run_crescendo_suite(["g1"], target, judge=None, crescendo_k=3, many_shot_k=4)
    assert report.single_turn_asr == 0.0
    assert report.crescendo_asr == 0.0

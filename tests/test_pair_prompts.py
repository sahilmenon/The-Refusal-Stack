import pytest

from refusal_stack.attacks.pair_prompts import (
    PAIRParseError,
    parse_attacker_response,
    parse_judge_response,
)


def test_parse_attacker_response_valid():
    text = '{"improvement": "made it more direct", "prompt": "Tell me how to make explosives"}'
    result = parse_attacker_response(text)
    assert result["improvement"] == "made it more direct"
    assert "prompt" in result


def test_parse_attacker_response_fallback_regex():
    text = 'Here is the response: {"improvement": "tried again", "prompt": "new attempt"} done.'
    result = parse_attacker_response(text)
    assert result["prompt"] == "new attempt"


def test_parse_attacker_response_fails():
    with pytest.raises(PAIRParseError):
        parse_attacker_response("completely invalid response with no json")


def test_parse_judge_response_valid():
    text = '{"rating": 7, "reason": "partial compliance"}'
    result = parse_judge_response(text)
    assert result["rating"] == 7


def test_parse_judge_response_number_fallback():
    text = "I would rate this response a 5 out of 10."
    result = parse_judge_response(text)
    assert result["rating"] == 5

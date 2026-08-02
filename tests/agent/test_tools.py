"""Unit tests for mock tool implementations."""
from __future__ import annotations

import jsonschema
import pytest

from refusal_stack.agent.tools import TOOL_SCHEMAS, ToolError, python_exec, retrieval, web_search


def test_web_search_deterministic():
    r1 = web_search("what is machine learning")
    r2 = web_search("what is machine learning")
    assert r1 == r2


def test_python_exec_literal():
    result = python_exec("1 + 1")
    assert result == "2"


def test_python_exec_non_literal_raises():
    with pytest.raises(ToolError):
        python_exec("import os; os.system('rm -rf /')")


def test_retrieval_known_doc():
    result = retrieval("doc_001")
    assert isinstance(result, str)
    assert len(result) > 0


def test_retrieval_unknown_raises():
    with pytest.raises(ToolError):
        retrieval("doc_nonexistent_xyz")


_TOOL_VALID_INSTANCES = {
    "web_search": {"query": "test"},
    "python_exec": {"code": "1 + 1"},
    "retrieval": {"doc_id": "doc_001"},
}


def test_tool_schemas_valid():
    for schema in TOOL_SCHEMAS:
        assert "name" in schema
        assert "description" in schema
        assert "parameters" in schema
        instance = _TOOL_VALID_INSTANCES[schema["name"]]
        jsonschema.validate(instance=instance, schema=schema["parameters"])


def test_web_search_returns_string():
    result = web_search("benign query example")
    assert isinstance(result, str)

"""Mock tool implementations for the agentic eval."""

from __future__ import annotations

import ast
import hashlib
import json
import logging
from pathlib import Path

log = logging.getLogger(__name__)

_MOCK_DATA_DIR = Path(__file__).parent / "mock_data"

TOOL_SCHEMAS = [
    {
        "name": "web_search",
        "description": "Search the web for information about a query.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "The search query."},
            },
            "required": ["query"],
        },
    },
    {
        "name": "python_exec",
        "description": "Execute a simple Python expression and return the result.",
        "parameters": {
            "type": "object",
            "properties": {
                "code": {"type": "string", "description": "A Python expression to evaluate."},
            },
            "required": ["code"],
        },
    },
    {
        "name": "retrieval",
        "description": "Retrieve a document by ID from the knowledge base.",
        "parameters": {
            "type": "object",
            "properties": {
                "doc_id": {"type": "string", "description": "The document identifier."},
            },
            "required": ["doc_id"],
        },
    },
]


class ToolError(Exception):
    pass


def _load_corpus(filename: str) -> dict:
    path = _MOCK_DATA_DIR / filename
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {}


def web_search(query: str, mock: bool = True) -> str:
    if not mock:
        raise ToolError("Real web search not implemented")
    corpus = _load_corpus("web_search_corpus.json")
    key = hashlib.sha256(query.encode()).hexdigest()[:8]
    return corpus.get(key, corpus.get(query, f"No results for: {query}"))


_SAFE_AST_NODES = (
    ast.Expression,
    ast.Constant,
    ast.BinOp,
    ast.UnaryOp,
    ast.BoolOp,
    ast.Compare,
    ast.List,
    ast.Tuple,
    ast.Dict,
    ast.Set,
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.Div,
    ast.FloorDiv,
    ast.Mod,
    ast.Pow,
    ast.USub,
    ast.UAdd,
    ast.Not,
    ast.Eq,
    ast.NotEq,
    ast.Lt,
    ast.LtE,
    ast.Gt,
    ast.GtE,
    ast.And,
    ast.Or,
)


def _is_safe_expr(code: str) -> bool:
    try:
        tree = ast.parse(code, mode="eval")
    except SyntaxError:
        return False
    return all(isinstance(node, _SAFE_AST_NODES) for node in ast.walk(tree))


def python_exec(code: str, mock: bool = True) -> str:
    if not _is_safe_expr(code):
        raise ToolError(
            f"python_exec only supports safe arithmetic/literal expressions, got: {code!r}"
        )
    try:
        result = eval(
            compile(ast.parse(code, mode="eval"), "<expr>", "eval"), {"__builtins__": {}}, {}
        )  # noqa: S307
        return str(result)
    except Exception as exc:
        raise ToolError(f"python_exec evaluation error: {exc}") from exc


def retrieval(doc_id: str, mock: bool = True) -> str:
    corpus = _load_corpus("retrieval_corpus.json")
    if doc_id not in corpus:
        raise ToolError(f"Document not found: {doc_id}")
    return corpus[doc_id]

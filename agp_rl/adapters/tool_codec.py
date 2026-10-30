"""Versioned tool-call envelope; official schemas validate the unchanged arguments.

No numeric rounding, clipping, repair or resampling: generated tokens remain the
ones scored by the learner. Invalid calls are retained as policy decisions.
"""
from __future__ import annotations

import json
import math

from jsonschema import Draft202012Validator, ValidationError
from agp_rl.io import canonical, digest

VERSION = "agp-tools-json-v1"
TOOLS = ("goal", "observe", "measure", "step", "run", "action_status", "assess")


def _pairs(items):
    result = {}
    for key, value in items:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def finite(value):
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("Non-finite numeric argument")
    if isinstance(value, dict):
        for item in value.values():
            finite(item)
    if isinstance(value, list):
        for item in value:
            finite(item)


class ToolCodec:
    def __init__(self, schemas, *, required_tools=TOOLS):
        if required_tools is not None and set(schemas) != set(required_tools):
            raise ValueError("V1 requires exactly the seven pinned official tools")
        self.schemas = schemas
        self.validators = {name: Draft202012Validator(schema) for name, schema in schemas.items()}
        for schema in schemas.values():
            Draft202012Validator.check_schema(schema)
        self.hash = digest({"version": VERSION, "schemas": schemas, "free_python": False})

    def parse(self, text):
        call = json.loads(text, object_pairs_hook=_pairs,
                          parse_constant=lambda v: (_ for _ in ()).throw(ValueError(f"Invalid number {v}")))
        if not isinstance(call, dict) or set(call) != {"name", "arguments"}:
            raise ValueError("Expected exactly {name, arguments}")
        if call["name"] not in self.validators or not isinstance(call["arguments"], dict):
            raise ValueError("Unknown tool or non-object arguments")
        finite(call)
        try:
            self.validators[call["name"]].validate(call["arguments"])
        except ValidationError as exc:
            # The actor/JS RPC boundary treats invalid generated arguments as
            # a policy tool error; they must not crash the local launcher.
            raise ValueError('TOOL_ARGUMENT_SCHEMA_INVALID: ' + exc.message) from exc
        return call

    def encode(self, name, arguments):
        text = canonical({"name": name, "arguments": arguments}).decode()
        self.parse(text)
        return text

    def verify_live(self, tools):
        live = {t.name: t.inputSchema for t in tools}
        if digest(live) != digest(self.schemas):
            raise ValueError("Live MCP schema differs from frozen schema; refuse rollout")

    def instruction(self):
        return ("Return exactly one JSON object with keys name and arguments, without markdown. "
                "Use only these official tools. No Python execution is available. Keep numeric "
                "arguments in the documented units; do not retry uncertain actions with new IDs.\n"
                + canonical(self.schemas).decode())

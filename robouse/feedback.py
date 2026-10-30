import json
import re


def feedback_flags(value):
    """Explicit recorded error observations, not an inferred final verdict."""
    flags = set()

    def walk(v, depth=0):
        if depth > 12:
            return
        if isinstance(v, str):
            if v.lstrip().startswith(("{", "[")):
                try:
                    walk(json.loads(v), depth + 1)
                    return
                except (ValueError, TypeError):
                    pass
            if (
                "SyntaxError:" in v
                or "JavaScript error:" in v
                or "Unexpected token" in v
            ):
                flags.add("VISIBLE_JS_ERROR")
            return
        if isinstance(v, list):
            for x in v:
                walk(x, depth + 1)
        elif isinstance(v, dict):
            if v.get("isError") is True:
                flags.add("MCP_IS_ERROR")
            if v.get("ok") is False:
                flags.add("VISIBLE_OK_FALSE")
            if v.get("error_code"):
                flags.add("VISIBLE_ERROR_CODE:" + str(v["error_code"]))
            for key, x in v.items():
                if key not in ("data", "encrypted_content", "reasoning_summary"):
                    walk(x, depth + 1)

    walk(value)
    return sorted(flags)

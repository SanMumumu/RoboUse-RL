"""Shared original full-causal-v4 normalization, preserving all targets and RGB."""
from copy import deepcopy
import re

VERSION = "agp-full-causal-native-context-v4"
TRANSPORT_HEADER = re.compile(
    r"^(?:Script completed\n)?Wall time:? [0-9.eE+\-]+ seconds\nOutput:\n?"
)


def _build_context_v4(messages, *, contract=None, with_target=False):
    if not isinstance(messages, list) or not messages:
        raise ValueError("NONEMPTY_CAUSAL_MESSAGES_REQUIRED")
    if contract is None:
        return deepcopy(messages)
    if contract.get("version") != VERSION:
        if contract.get("version") == "full-causal-prefix-budget-v2":
            return deepcopy(messages)
        raise ValueError("UNKNOWN_CONTEXT_CONTRACT")
    if with_target and messages[-1].get("role") != "assistant":
        raise ValueError("FINAL_ASSISTANT_TARGET_REQUIRED")
    result = deepcopy(messages)
    target_index = len(result) - 1 if with_target else None
    first_user = True
    for index, message in enumerate(result):
        if message.get("role") not in ("system", "user", "assistant"):
            raise ValueError("UNSUPPORTED_CAUSAL_MESSAGE_ROLE")
        if index == target_index or message["role"] != "user":
            continue
        if first_user:
            first_user = False
            continue
        blocks = []
        for block in message["content"]:
            if block.get("type") == "text":
                text = TRANSPORT_HEADER.sub("", block["text"], count=1)
                if not text and text != block["text"]:
                    continue
                block["text"] = text
            blocks.append(block)
        message["content"] = blocks or [
            {"type": "text", "text": "(Tool completed with no visible output)"}
        ]
    if with_target and result[-1] != messages[-1]:
        raise ValueError("SUPERVISED_TARGET_CHANGED")
    return result


def build_context(messages, with_target=False):
    return _build_context_v4(
        messages, contract={"version": VERSION}, with_target=with_target
    )

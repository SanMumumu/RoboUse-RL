# Copyright (c) ModelScope Contributors. All rights reserved.
from __future__ import annotations


def remove_response(messages) -> Optional[str]:
    """
    Removes and returns the content of the last message if its role is 'assistant'.

    Args:
        messages (List[Dict]):
            A list of message dictionaries, each typically containing a 'role' and 'content' key.

    Returns:
        Optional[str]:
            The content of the removed 'assistant' message if present;
            otherwise, returns None. The original messages list is modified in place.
    """
    last_role = messages[-1]["role"] if messages else None
    if last_role == "assistant":
        return messages.pop()["content"]

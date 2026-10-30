"""Episode-local causal memory. Durable production replay lives in the scheduler."""
from copy import deepcopy


class EpisodeMemory:
    def __init__(self, messages=()):
        self.messages = deepcopy(list(messages))
        self.receipts = {}

    def append(self, role, content):
        if role not in ("system", "user", "assistant"):
            raise ValueError("Use causal native message roles")
        self.messages.append({"role": role, "content": deepcopy(content)})

    def context(self):
        return deepcopy(self.messages)

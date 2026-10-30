"""Explicit schema/error boundary and idempotent tool-dispatch receipts."""
from agp_rl.io import digest
from robouse.harness.memory import EpisodeMemory


class HarnessExecutor:
    def __init__(self, registry, memory=None):
        self.registry = registry
        self.memory = memory if memory is not None else EpisodeMemory()

    async def execute(self, call_id, text, *, retries=0):
        if not call_id or type(retries) is not int or not 0 <= retries <= 3:
            raise ValueError("Explicit call identity and bounded retries required")
        fingerprint = digest(text)
        previous = self.memory.receipts.get(call_id)
        if previous:
            if previous["input_hash"] != fingerprint:
                raise ValueError("Call ID reused with different arguments")
            return previous
        try:
            call = self.registry.parse(text)
        except (ValueError, TypeError) as exc:
            return {"call_id": call_id, "input_hash": fingerprint, "is_error": True, "error": str(exc), "dispatched": False, "physical_success": None}
        if retries and not self.registry.tools[call["name"]].retry_safe:
            raise ValueError("Retry requires an explicitly idempotent native tool")
        for attempt in range(retries + 1):
            try:
                result = await self.registry.execute(call)
                receipt = {"call_id": call_id, "input_hash": fingerprint, "is_error": bool(getattr(result, "isError", False)) or (isinstance(result, dict) and result.get("isError") is True), "result": result, "dispatched": True, "physical_success": None}
                self.memory.receipts[call_id] = receipt
                return receipt
            except Exception as exc:
                if attempt == retries:
                    receipt = {"call_id": call_id, "input_hash": fingerprint, "is_error": True, "error": str(exc), "dispatched": True, "physical_success": None}
                    self.memory.receipts[call_id] = receipt
                    return receipt

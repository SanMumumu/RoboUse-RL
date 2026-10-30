"""Qwen multimodal tool-using policy, sharing the SFT processor/context contract."""
from robouse.models.qwen import prepare, sample, tensor_hash


class QwenAgent:
    def __init__(self, model, processor, policy_version, config):
        self.model, self.processor = model, processor
        self.policy_version, self.config = policy_version, config

    async def infer(self, messages, images, root):
        result = sample(
            self.model, self.processor, prepare(messages, images, self.processor, root),
            max_new_tokens=self.config.get("max_new_tokens", 2048),
            max_context_tokens=self.config.get("max_context_tokens", 131072),
            temperature=self.config.get("temperature", 1.0),
            greedy=self.config.get("greedy", False),
        )
        result.update(policy_version=self.policy_version, input_hash=tensor_hash(result["inputs"]))
        return result

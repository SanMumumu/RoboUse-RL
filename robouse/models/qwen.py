"""One processor/template and one controlled generation distribution per model."""
from copy import deepcopy
import hashlib
from io import BytesIO
import json
from pathlib import Path
import torch
from agp_rl.io import sha256, digest
from agp_rl.policy.termination import CHAT_TEMPLATE_OPTIONS, native_termination_contract
from robouse.data.datasets import image_bytes


def tensor_hash(inputs):
    h = hashlib.sha256()
    for key, value in sorted(inputs.items()):
        value = value.detach().cpu().contiguous()
        h.update(key.encode())
        h.update(str(value.dtype).encode())
        h.update(str(tuple(value.shape)).encode())
        h.update(value.view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def policy_version(path):
    path = Path(path)
    weights = sorted(path.glob("*.safetensors"))
    if not weights:
        raise ValueError("Full HF checkpoint safetensors required")
    return digest({p.name: sha256(p) for p in weights})


def resolve_model(config):
    from robouse.data.datasets import verify

    model = config["model"]
    revision = config.get("revision")
    if Path(model).is_dir():
        p = Path(model) / "model-manifest.json"
        if not p.exists() or digest(json.loads(p.read_text())) != revision:
            raise ValueError("Local original model inventory/revision mismatch")
        manifest = verify(model, p)
        origin = manifest.get("origin", {})
        if origin.get("kind") != "hf_base" or len(origin.get("revision", "")) != 40:
            raise ValueError(
                "Local model requires upstream-verified original HF weight provenance"
            )
        actual = {
            r["path"]: r["sha256"]
            for r in manifest["files"]
            if r["path"].endswith(".safetensors")
        }
        if actual != origin.get("weight_sha256"):
            raise ValueError("Original upstream weight set mismatch")
        if (Path(model) / "trainer_state.json").exists():
            raise ValueError(
                "Original model inventory must not point to a trained checkpoint"
            )
        return model
    if not revision or len(revision) != 40:
        raise ValueError("HF revision must be a full commit SHA")
    from huggingface_hub import snapshot_download

    return snapshot_download(model, revision=revision)


def load(source, device=None):
    from transformers import AutoModelForImageTextToText, AutoProcessor

    processor = AutoProcessor.from_pretrained(
        source, local_files_only=True, trust_remote_code=False
    )
    model = AutoModelForImageTextToText.from_pretrained(
        source,
        local_files_only=True,
        trust_remote_code=False,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )
    if device is not None:
        model.to(device)
    model.config.use_cache = False
    native_termination_contract(processor, model.generation_config)
    return model, processor


def messages_with_pixels(messages, images, root):
    from PIL import Image

    value = deepcopy(messages)
    for m in value:
        for b in m["content"]:
            if b["type"] == "image":
                key = b["image"].removeprefix("art-image://")
                b["image"] = Image.open(
                    BytesIO(image_bytes(images[key], root))
                ).convert("RGB")
    return value


def prepare(messages, images, processor, root):
    from robouse.harness.context import build_context

    value = messages_with_pixels(build_context(messages), images, root)
    return dict(
        processor.apply_chat_template(
            value,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            add_generation_prompt=True,
            **CHAT_TEMPLATE_OPTIONS,
        )
    )


def extend(inputs, tokens):
    result = dict(inputs)
    result["input_ids"] = torch.cat([inputs["input_ids"], tokens.reshape(1, -1)], dim=1)
    mask = inputs.get("attention_mask", torch.ones_like(inputs["input_ids"]))
    result["attention_mask"] = torch.cat(
        [mask, torch.ones_like(tokens).reshape(1, -1)], dim=1
    )
    if "mm_token_type_ids" in inputs:
        result["mm_token_type_ids"] = torch.cat(
            [inputs["mm_token_type_ids"], torch.zeros_like(tokens).reshape(1, -1)],
            dim=1,
        )
    if "position_ids" in inputs or "token_type_ids" in inputs:
        raise ValueError("Unsupported cached positional/type inputs")
    return result


def distribution(logits, temperature, forbidden):
    if not 0 < temperature <= 2:
        raise ValueError("Invalid temperature")
    logits = logits.float() / temperature
    if forbidden:
        logits = logits.clone()
        logits[..., list(forbidden)] = -torch.inf
    return logits.log_softmax(-1)


def token_logprobs(
    model, inputs, tokens, *, temperature=1.0, forbidden=(), prefix_oracle=False
):
    """Oracle matches the sampler's exact no-cache forward shape and kernel path."""
    if prefix_oracle:
        values = []
        for i, token in enumerate(tokens):
            output = model(
                **extend(inputs, tokens[:i]), use_cache=False, logits_to_keep=1
            )
            lp = distribution(output.logits[0, -1], temperature, forbidden)
            values.append(lp[token])
        return torch.stack(values)
    n = inputs["input_ids"].shape[-1]
    positions = torch.arange(n - 1, n + len(tokens) - 1, device=tokens.device)
    output = model(**extend(inputs, tokens), use_cache=False, logits_to_keep=positions)
    return (
        distribution(output.logits[0], temperature, forbidden)
        .gather(-1, tokens[:, None])
        .squeeze(-1)
    )


@torch.no_grad()
def sample(
    model,
    processor,
    inputs,
    *,
    max_new_tokens,
    max_context_tokens,
    temperature=1.0,
    greedy=False,
    generator=None,
):
    model.eval()
    forbidden = tuple(
        getattr(model.config, k)
        for k in (
            "image_token_id",
            "video_token_id",
            "vision_start_token_id",
            "vision_end_token_id",
        )
        if getattr(model.config, k, None) is not None
    )
    end = native_termination_contract(processor, model.generation_config)[
        "eos_token_ids"
    ]
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    budget = min(max_new_tokens, max_context_tokens - inputs["input_ids"].shape[-1])
    if budget < 1:
        raise ValueError("CONTEXT_LIMIT")
    tokens = torch.empty(0, dtype=torch.long, device=device)
    old = []
    # Deliberately no KV cache: prefix oracle reproduces this backend exactly.
    for _ in range(budget):
        output = model(**extend(inputs, tokens), use_cache=False, logits_to_keep=1)
        lp = distribution(output.logits[0, -1], temperature, forbidden)
        token = (
            lp.argmax()
            if greedy
            else torch.multinomial(lp.exp(), 1, generator=generator).reshape(())
        )
        old.append(lp[token])
        tokens = torch.cat([tokens, token.reshape(1)])
        if int(token) in end:
            break
    return {
        "inputs": {k: v.cpu() for k, v in inputs.items()},
        "tokens": tokens.cpu(),
        "behavior_logprobs": torch.stack(old).cpu(),
        "mask": torch.ones(len(old), dtype=torch.bool),
        "temperature": temperature,
        "forbidden": list(forbidden),
        "text": processor.decode(tokens, skip_special_tokens=True),
        "finish_reason": "stop" if int(tokens[-1]) in end else "length",
    }


def configure_grpo_training(model):
    """Keep activation checkpointing active while eliminating stochastic dropout."""
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0.0
        if hasattr(module, "attention_dropout"):
            module.attention_dropout = 0.0
    model.train()


def training_code_hash():
    root = Path(__file__).resolve().parents[1]
    return digest({str(p.relative_to(root)): sha256(p) for p in sorted(root.rglob("*.py")) if not any(x in p.relative_to(root).parts for x in ("_native", "_vendor"))})


def processor_signature(processor):
    tokenizer = processor.tokenizer
    return digest(
        {
            "processor": processor.to_dict(),
            "chat_template": processor.chat_template,
            "tokenizer_json": tokenizer.backend_tokenizer.to_str(),
        }
    )

"""Role ownership includes failed native identities, retries and known aliases."""
from copy import deepcopy
from agp_rl.io import digest


def normalize(candidate, spec):
    candidate = deepcopy(candidate)
    ep, bundle = candidate["episode"], candidate["bundle"]
    if ep["seed"] != spec["seed"]:
        raise ValueError("Source seed mismatch")
    ep.setdefault("source_id", spec["name"])
    old_images = candidate["images"]
    by_sha = {ref["sha256"]: ref for ref in old_images.values()}
    for message in bundle["messages"]:
        for block in message["content"]:
            if block["type"] == "image":
                key = block["image"].removeprefix("art-image://")
                if key not in old_images:
                    raise ValueError("Unbound original RGB")
                block["image"] = "art-image://" + old_images[key]["sha256"]
    for d in bundle["decisions"]:
        end = d["visible_history"]["message_end_exclusive"]
        if bundle["messages"][end] != {
            "role": "assistant",
            "content": [{"type": "text", "text": d["model_output"]["text"]}],
        }:
            raise ValueError("Assistant target/history boundary mismatch")
        if bundle.get("message_origins"):
            previous = [r.get("line", 0) for r in bundle["message_origins"][:end]]
            if previous and max(previous) >= d["provenance"]["line"]:
                raise ValueError("Future information in causal history")
        d["decision_id"] = spec["name"] + ":" + d["decision_id"]
    candidate["images"] = by_sha
    return candidate


def identity_rows(config, root):
    from robouse.data.datasets import read_jsonl, role

    rows = []
    for path in config.get("identity_files", []):
        for value in read_jsonl(root / path):
            if any(
                k in value for k in ("messages", "decisions", "targets", "assistant")
            ):
                raise ValueError("Identity registry must not contain teacher targets")
            identity = (
                f"{value['task']}/seed-{value['seed']}/layout-{value['layout']:03d}"
            )
            rows.append({**value, "identity_id": identity, "role": role(value["seed"])})
    return rows

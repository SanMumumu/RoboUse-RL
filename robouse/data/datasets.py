"""Immutable seed-role windows and byte-identical image migration."""
import hashlib
import json
import os
from pathlib import Path
import time
from agp_rl.io import sha256

ROLES = {0: "train", 1: "val", 2: "test"}


def read_jsonl(path):
    return [json.loads(s) for s in Path(path).read_text().splitlines() if s.strip()]


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".partial")
    with tmp.open("w") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp, path)


def role(seed):
    if seed not in ROLES:
        raise ValueError("Only seeds 0/1/2 admitted")
    return ROLES[seed]


def check_roles(rows):
    owners = {}
    for row in rows:
        expected = role(row["seed"])
        if row["role"] != expected:
            raise ValueError("Seed role violation")
        aliases = [("identity", row["identity_id"])]
        aliases += [("known_physical", a) for a in row.get("physical_aliases", [])]
        aliases += [
            (k, row[k])
            for k in (
                "layout_hash",
                "initial_rgb_hash",
                "trajectory_hash",
                "window_hash",
            )
            if row.get(k)
        ]
        for alias in aliases:
            if alias in owners and owners[alias] != expected:
                raise ValueError(
                    "Cross-role physical/identity/content alias: " + str(alias)
                )
            owners[alias] = expected


def relative(root, path):
    path = Path(path).resolve()
    return path.relative_to(Path(root).resolve()).as_posix()


def stable_hash(path, min_age=60, reference_time=None):
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError("Regular non-symlink file required: " + str(path))
    a = path.stat()
    if (
        reference_time if reference_time is not None else time.time()
    ) - a.st_mtime < min_age:
        raise ValueError("Recently written file; require sealed snapshot: " + str(path))
    h = sha256(path)
    b = path.stat()
    if (a.st_ino, a.st_size, a.st_mtime_ns) != (b.st_ino, b.st_size, b.st_mtime_ns):
        raise ValueError("File changed while hashing: " + str(path))
    return {"path": str(path), "bytes": a.st_size, "sha256": h}


def export(root, files, output, min_age=60):
    root = Path(root).resolve()
    # NFS server and client clocks can differ. Measure age on the same filesystem.
    import tempfile

    with tempfile.NamedTemporaryFile(dir=Path(output).parent) as probe:
        reference_time = Path(probe.name).stat().st_mtime
        output_device = Path(probe.name).stat().st_dev
    records = []
    for name in sorted(set(files)):
        p = root / name
        rel = relative(root, p)
        if rel != name or "\n" in name:
            raise ValueError("Canonical relative paths required")
        if p.stat().st_dev != output_device and min_age:
            raise ValueError(
                "Manifest must be written on the source filesystem for stable-age verification"
            )
        records.append({**stable_hash(p, min_age, reference_time), "path": rel})
    result = {
        "schema": "robouse-transfer-v1",
        "files": records,
        "total_bytes": sum(r["bytes"] for r in records),
    }
    write_json(output, result)
    Path(str(output) + ".files").write_text("".join(r["path"] + "\n" for r in records))
    return result


def verify(root, manifest):
    value = json.loads(Path(manifest).read_text())
    root = Path(root).resolve()
    for row in value["files"]:
        p = root / row["path"]
        if relative(root, p) != row["path"] or p.is_symlink():
            raise ValueError("Unsafe migration path")
        if p.stat().st_size != row["bytes"] or sha256(p) != row["sha256"]:
            raise ValueError("Transfer mismatch: " + row["path"])
    return value


def image_bytes(ref, root):
    p = Path(root) / ref["path"]
    if relative(root, p) != ref["path"]:
        raise ValueError("Image outside DATA_ROOT")
    with p.open("rb") as f:
        f.seek(ref.get("offset", 0))
        raw = f.read(ref["bytes"])
    if hashlib.sha256(raw).hexdigest() != ref["sha256"]:
        raise ValueError("Original image/tar offset mismatch")
    return raw


def prepare(config, output):
    """Reuse original raw-evidence adapters, never observe a live Seed1 index."""
    from agp_rl.pipeline2b.fullft import source_adapters as adapters
    from copy import deepcopy

    adapters.ROOT = root = Path(config["data_root"]).resolve()
    verifier = adapters.Verifier()
    from robouse.roles import normalize, identity_rows

    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    candidates = []
    audits = []
    for spec in config["sources"]:
        source = {**spec, "root": str(root / spec["root"])}
        if spec["seed"] == 2:
            # Test is identity-only at preparation; never parse its targets.
            continue
        if spec["kind"] == "public_art":
            source["art_root"] = str(root / spec["art_root"])
            generator = adapters.public_candidates(source, verifier)
        else:
            seal = root / spec["sealed_manifest"]
            if sha256(seal) != spec["sealed_manifest_sha256"]:
                raise ValueError("Seed1 sealed snapshot hash mismatch")
            frozen = read_jsonl(seal)
            schemas = json.loads((root / config["tool_schema"]).read_text())["schemas"]
            generator = adapters.native_candidates(source, verifier, schemas, frozen)
        for audit, candidate in generator:
            audits.append(audit)
            if candidate:
                candidates.append(normalize(candidate, spec))
    processor = None
    if config.get("processor_model"):
        from transformers import AutoProcessor

        processor = AutoProcessor.from_pretrained(
            config["processor_model"],
            revision=config.get("processor_revision"),
            trust_remote_code=False,
        )
    rows = []
    required = set()
    identities = identity_rows(config, root)
    registry = {r["identity_id"]: r for r in identities}
    checked_images = {}
    for c in candidates:
        ep, bundle = c["episode"], c["bundle"]
        seed = ep["seed"]
        identity = f"{ep['task']}/seed-{seed}/layout-{ep['layout_id']:03d}"
        for d in bundle["decisions"]:
            q = d["data_quality"]
            if not q["sft_candidate"] or not q.get("input_history_complete", True):
                continue
            end = d["visible_history"]["message_end_exclusive"]
            messages = [
                {
                    "role": "system",
                    "content": [{"type": "text", "text": c["system_prompt"]}],
                }
            ] + deepcopy(bundle["messages"][:end])
            messages.append(
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": d["model_output"]["text"]}],
                }
            )
            images = {}
            for m in messages:
                for block in m["content"]:
                    if block["type"] != "image":
                        continue
                    key = block["image"][12:]
                    ref = c["images"][key]
                    path = relative(root, ref.get("tar_path", ref.get("path")))
                    images[key] = {
                        k: ref[k] for k in ("bytes", "sha256", "width", "height")
                    }
                    images[key].update(path=path, offset=ref.get("offset", 0))
                    cache_key = (
                        path,
                        images[key]["offset"],
                        images[key]["bytes"],
                        images[key]["sha256"],
                    )
                    if cache_key not in checked_images:
                        from PIL import Image
                        from io import BytesIO

                        raw = image_bytes(images[key], root)
                        with Image.open(BytesIO(raw)) as im:
                            rgb = im.convert("RGB")
                            rgb_hash = hashlib.sha256(rgb.tobytes()).hexdigest()
                        if ref.get("rgb_sha256") and ref["rgb_sha256"] != rgb_hash:
                            raise ValueError("Decoded original RGB hash mismatch")
                        checked_images[cache_key] = rgb_hash
                    images[key]["rgb_sha256"] = checked_images[cache_key]
                    required.add(path)
            if not images:
                raise ValueError("No original model input RGB")
            visible = deepcopy(messages)
            for m in visible:
                for b in m["content"]:
                    if b["type"] == "image":
                        b["image"] = images[b["image"][12:]]["rgb_sha256"]
            canonical = json.dumps(visible, sort_keys=True).encode()
            rows.append(
                {
                    "window_id": d["decision_id"],
                    "identity_id": identity,
                    "episode_id": ep.get("episode_id"),
                    "attempt_id": ep.get("attempt_id"),
                    "source_id": ep.get("source_id"),
                    "seed": seed,
                    "teacher_model": ep.get("teacher_model"),
                    "teacher_effort": ep.get("teacher_effort"),
                    "role": role(seed),
                    "physical_aliases": registry.get(identity, {}).get(
                        "physical_aliases", []
                    ),
                    "layout_hash": ep.get("layout_hash")
                    or registry.get(identity, {}).get("layout_hash"),
                    "window_hash": hashlib.sha256(canonical).hexdigest(),
                    "initial_rgb_hash": hashlib.sha256(
                        json.dumps(
                            [
                                images[b["image"][12:]]["rgb_sha256"]
                                for b in bundle["messages"][0]["content"]
                                if b["type"] == "image"
                            ]
                        ).encode()
                    ).hexdigest(),
                    "scope": ep["supervision_scope"],
                    "messages": messages,
                    "images": images,
                }
            )
    if processor is not None:
        from collections import Counter

        original_counts = Counter(r["identity_id"] for r in rows)
        from robouse.training.sft.train import encode

        qualified = []
        for row in rows:
            try:
                value = encode(row, processor, root)
                tokens = int(value["inputs"]["input_ids"].numel())
                target_tokens = len(value["target"])
                if tokens > config.get(
                    "max_context_tokens", 131072
                ) or target_tokens > config.get("max_new_tokens", 2048):
                    raise ValueError(
                        "Per-model context/target budget exceeded; no truncation"
                    )
                row.update(tokens=tokens, supervised_tokens=target_tokens)
                qualified.append(row)
            except (ValueError, KeyError) as exc:
                audits.append(
                    {
                        "window_id": row["window_id"],
                        "reasons": ["PROCESSOR_WINDOW_REJECTED:" + str(exc)],
                        "episode_rejected": False,
                    }
                )
        rows = qualified
        selected_counts = Counter(r["identity_id"] for r in rows)
        for row in rows:
            if (
                selected_counts[row["identity_id"]]
                != original_counts[row["identity_id"]]
            ):
                row.update(source_scope=row["scope"], scope="qualified_local_decisions")
    check_roles(rows + identities)
    write_json(output / "identity-roles.json", identities)
    for seed, split in ROLES.items():
        path = output / f"seed{seed}-{split}.jsonl"
        path.write_text(
            "".join(
                json.dumps(r, ensure_ascii=False) + "\n"
                for r in rows
                if r["seed"] == seed
            )
        )
    write_json(
        output / "preparation.json",
        {
            "version": "seed-roles-v2",
            "test": "PENDING_TEST",
            "processor_model": config.get("processor_model"),
            "processor_signature": __import__(
                "robouse.models.qwen", fromlist=["processor_signature"]
            ).processor_signature(processor)
            if processor is not None
            else None,
            "processor_revision": config.get("processor_revision"),
            "tokenization_status": "PROFILED_PER_MODEL"
            if processor is not None
            else "TOKENIZATION_PENDING",
            "counts": {r: sum(x["role"] == r for x in rows) for r in ROLES.values()},
            "audit": audits,
        },
    )
    (output / "required-images.files").write_text(
        "".join(p + "\n" for p in sorted(required))
    )
    prepared_names = (
        "seed0-train.jsonl",
        "seed1-val.jsonl",
        "seed2-test.jsonl",
        "preparation.json",
        "identity-roles.json",
    )
    (output / "prepared.files").write_text(
        "".join(name + "\n" for name in prepared_names)
    )
    if output.is_relative_to(root):
        required.update(relative(root, output / name) for name in prepared_names)
    (output / "required.files").write_text("".join(p + "\n" for p in sorted(required)))
    return rows

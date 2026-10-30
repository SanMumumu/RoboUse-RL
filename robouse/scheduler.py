"""Durable multi-turn lifecycle adapted from Guava's explicit-token scheduler.

Each turn stores real generated IDs/masks rather than re-tokenizing a completed
transcript. Gym observations stay user-role blocks, matching native AGP SFT.
"""
import asyncio
import json
from pathlib import Path
import time
from copy import deepcopy
import torch
from agp_rl.io import sha256, digest
from robouse.data.datasets import write_json
from robouse.agents.qwen_agent import QwenAgent


def atomic_torch(path, value):
    path = Path(path)
    tmp = path.with_suffix(".partial")
    torch.save(value, tmp)
    tmp.replace(path)


class HfEngine:
    def __init__(self, model, processor, policy, config):
        self.model = model
        self.processor = processor
        self.policy = policy
        self.config = config
        self.agent = QwenAgent(model, processor, policy, config)

    async def robouse_episode(self, config, request_config):
        return await episode(
            self,
            config["environment"],
            config["identity"],
            config["group_id"],
            config["directory"],
            max_turns=config["max_turns"],
            stop_path=config.get("stop_path"),
        )

    async def infer(self, messages, images, root):
        return await self.agent.infer(messages, images, root)


async def episode(
    engine, environment, identity, group_id, directory, *, max_turns=512, stop_path=None
):
    """Resume the exact pending action; remote mailbox accepts each index once."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    state_path = directory / "episode-state.json"
    if state_path.exists():
        state = json.loads(state_path.read_text())
        if state.get("decisions"):
            previous = torch.load(
                state["decisions"][-1]["file"], map_location="cpu", weights_only=True
            )
            torch.set_rng_state(previous["rng_after"])
            if "cuda_rng_after" in previous:
                torch.cuda.set_rng_state_all(previous["cuda_rng_after"])
    else:
        started = time.monotonic()
        initial = await environment.reset(identity, group_id)
        if initial.get("outcome") in ("infrastructure_error", "interrupted"):
            return {
                **initial,
                "decisions": [],
                "episode_id": environment.episode_id,
                "identity_id": f"{identity['task']}/seed-{identity['seed']}/layout-{identity['layout']:03d}",
                "seed": identity["seed"],
                "origin": "official_online",
                "policy_version": engine.policy,
            }
        blocks, images = environment.observation(initial)
        state = {
            "messages": [
                {
                    "role": "system",
                    "content": [{"type": "text", "text": initial["system"]}],
                },
                {"role": "user", "content": blocks},
            ],
            "images": images,
            "decisions": [],
            "initial_state_hash": initial["initial_state_hash"],
            "turn": 0,
            "done": False,
            "timeline": [],
            "public_decisions": [],
            "instruction": "\n".join(b["text"] for b in blocks if b["type"] == "text"),
            "elapsed_seconds": time.monotonic() - started,
        }
        write_json(state_path, state)
    while not state["done"] and state["turn"] < max_turns:
        if stop_path and Path(stop_path).exists():
            state.update(
                done=True,
                result={
                    "outcome": "interrupted",
                    "reward": None,
                    "termination_reason": "operator_stop",
                },
            )
            await environment.close()
            write_json(state_path, state)
            break
        turn = state["turn"]
        artifact = directory / f"decision-{turn:06d}.pt"
        meta = directory / f"decision-{turn:06d}.json"
        if artifact.exists():
            value = torch.load(artifact, map_location="cpu", weights_only=True)
            if value["policy_version"] != engine.policy:
                raise ValueError("Pending sample policy changed")
        else:
            decision_started = time.monotonic()
            try:
                value = await engine.infer(
                    state["messages"], state["images"], environment.root
                )
            except ValueError as exc:
                if str(exc) != "CONTEXT_LIMIT":
                    raise
                # Explicit empty finish action, not a student tool or fake token.
                result = await environment.step(turn, "", finish=True)
                result.update(termination_reason="context_limit", truncated=True)
                state.update(done=True, result=result)
                write_json(state_path, state)
                break
            value["decision_seconds"] = time.monotonic() - decision_started
            value["rng_after"] = torch.get_rng_state()
            if torch.cuda.is_initialized():
                value["cuda_rng_after"] = torch.cuda.get_rng_state_all()
            atomic_torch(artifact, value)
        torch.set_rng_state(value["rng_after"])
        if "cuda_rng_after" in value:
            torch.cuda.set_rng_state_all(value["cuda_rng_after"])
        info = {
            "file": str(artifact.resolve()),
            "sha256": sha256(artifact),
            "input_hash": value["input_hash"],
            "input_ids": value["inputs"]["input_ids"][0].tolist(),
            "generated_ids": value["tokens"].tolist(),
            "behavior_logprobs": value["behavior_logprobs"].tolist(),
            "loss_mask": value["mask"].tolist(),
            "input_images": state["images"],
            "policy_version": engine.policy,
        }
        write_json(meta, info)
        start = time.monotonic()
        observation = await environment.step(
            turn, value["text"], finish=turn + 1 >= max_turns
        )
        blocks, images = environment.observation(observation)
        try:
            call = json.loads(value["text"])
            if not isinstance(call, dict) or set(call) != {"name", "arguments"}:
                call = None
        except (ValueError, TypeError):
            call = None
        def public_images(refs):
            return {key: {**ref, "path": str((Path(environment.root) / ref["path"]).resolve().relative_to(directory.resolve()))} for key, ref in refs.items()}
        state.setdefault("public_decisions", []).append({
            "observation": {"rgb": public_images(state["images"]), "instruction": state.get("instruction", ""), "feedback": {}, "depth": {}},
            "context": deepcopy(state["messages"]), "assistant": value["text"],
            "tool_call": call, "tool_result": {"blocks": deepcopy(blocks), "images": public_images(images)},
            "policy_version": engine.policy, "input_history_complete": True,
            "generation": info,
        })
        state["messages"].append(
            {"role": "assistant", "content": [{"type": "text", "text": value["text"]}]}
        )
        if blocks:
            state["messages"].append({"role": "user", "content": blocks})
        state["images"].update(images)
        state["decisions"].append(info)
        state["timeline"].append(
            {
                "turn": turn,
                "assistant": value["text"],
                "decision_seconds": value.get("decision_seconds"),
                "tool_seconds": time.monotonic() - start,
                "native_tool_seconds": observation.get("native_tool_seconds"),
                "native_tool_calls": observation.get("native_tool_calls"),
                "observation": blocks,
            }
        )
        state["turn"] += 1
        state["elapsed_seconds"] = state.get("elapsed_seconds", 0) + value.get("decision_seconds", 0) + time.monotonic() - start
        state["done"] = observation["done"]
        if state["done"]:
            state["result"] = observation
        write_json(state_path, state)
    result = {
        **state.get("result", {}),
        "decisions": state["decisions"],
        "timeline": state["timeline"],
        "initial_state_hash": state["initial_state_hash"],
        "episode_id": environment.episode_id,
        "identity_id": f"{identity['task']}/seed-{identity['seed']}/layout-{identity['layout']:03d}",
        "seed": identity["seed"],
        "origin": "official_online",
        "policy_version": engine.policy,
        "identity": identity,
        "episode_seconds": state.get("elapsed_seconds"),
    }
    result["messages"] = state["messages"]
    write_json(directory / "episode.json", result)
    if state.get("public_decisions") and state.get("instruction") and all(d["observation"]["rgb"] for d in state["public_decisions"]):
        from robouse.data.schemas import Trajectory

        native = result.get("native_reward")
        public = Trajectory.from_dict({
            "identity": {"episode_id": environment.episode_id, "attempt_id": "attempt-01", "source_id": "student-online:" + engine.policy, "task": identity["task"], "seed": identity["seed"], "initial_state": state["initial_state_hash"], "role": {0: "train", 1: "val", 2: "test"}[identity["seed"]]},
            "decisions": state["public_decisions"], "reward": result.get("reward"),
            "success": native.get("success") if native else None,
            "terminated": bool(native), "truncated": result.get("truncated", False),
            "termination_reason": result.get("termination_reason", "unknown"),
            "origin": "student_on_policy",
        })
        write_json(directory / "trajectory.json", public.to_dict())
    return result


def swift_scheduler_class():
    """Register the same lifecycle with the installed ms-swift scheduler registry."""
    from robouse._vendor.swift_scheduler import MultiTurnScheduler

    multi_turns = {}

    class RoboDojoMultiTurnScheduler(MultiTurnScheduler):
        async def run(self, infer_request, request_config, **kwargs):
            from robouse._vendor.swift_protocol import RolloutOutput, ChatCompletionResponse, ChatCompletionResponseChoice, ChatMessage, UsageInfo

            config = infer_request.data_dict["robouse"]
            # External engine must expose the explicit sampler evidence protocol;
            # no fallback to reconstructed behavior probabilities is accepted.
            if not hasattr(self.infer_engine, "robouse_episode"):
                raise ValueError(
                    "Use RoboUse HF explicit-token engine; generic re-tokenizing engine is not admitted"
                )
            result = await self.infer_engine.robouse_episode(config, request_config)
            last = next(
                (
                    m
                    for m in reversed(result.get("messages", []))
                    if m["role"] == "assistant"
                ),
                {"content": []},
            )
            content = "".join(b["text"] for b in last["content"] if b["type"] == "text")
            n = sum(len(d["generated_ids"]) for d in result["decisions"])
            prompt = sum(len(d["input_ids"]) for d in result["decisions"])
            response = ChatCompletionResponse(
                model=result["policy_version"],
                choices=[
                    ChatCompletionResponseChoice(
                        index=0,
                        message=ChatMessage(role="assistant", content=content),
                        finish_reason="stop" if n else None,
                        token_ids=result["decisions"][-1]["generated_ids"] if n else [],
                    )
                ],
                usage=UsageInfo(prompt, n, prompt + n),
            )
            return RolloutOutput(
                response=response,
                messages=result.get("messages", []),
                response_token_ids=[d["generated_ids"] for d in result["decisions"]],
                response_loss_mask=[d["loss_mask"] for d in result["decisions"]],
                rollout_logprobs=[d["behavior_logprobs"] for d in result["decisions"]],
                rollout_infos=result,
            )

    multi_turns["robouse_native"] = RoboDojoMultiTurnScheduler
    return RoboDojoMultiTurnScheduler

"""Run identity-complete student evaluation and save native metrics and provenance."""
from pathlib import Path
from robouse.data.datasets import read_jsonl, write_json
from robouse.evaluation.metrics import evaluate_metrics
from robouse.utils.manifests import write_run_manifest


def run(config):
    from robouse.collect import collect

    output = Path(config["output"])
    output.mkdir(parents=True, exist_ok=True)
    identities = read_jsonl(config["identities"])
    write_run_manifest(output, config, stage="evaluation", datasets={"identities": identities})
    try:
        collect(config, evaluation=True)
    finally:
        # Preserve already completed native episodes if another attempt fails.
        episodes = read_jsonl(output / "episodes.jsonl") if (output / "episodes.jsonl").exists() else []
        metrics = evaluate_metrics(episodes, requested_count=len(identities))
        write_json(output / "metrics.json", {k: v for k, v in metrics.items() if k != "episodes"})
        write_json(output / "episode_metrics.json", metrics["episodes"])
    return metrics

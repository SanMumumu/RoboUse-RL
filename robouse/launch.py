"""File logs plus a single concise local progress bar; all child errors propagate."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from robouse.data.datasets import write_json


def distributed(config, action="sft_a800"):
    from tqdm import tqdm

    output = Path(config["output"])
    output.mkdir(parents=True, exist_ok=True)
    path = output / "launch.json"
    write_json(path, config)
    env = {
        **os.environ,
        "CUDA_VISIBLE_DEVICES": config["gpus"],
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
        "TOKENIZERS_PARALLELISM": "false",
        "OMP_NUM_THREADS": str(config.get("cpu_threads", 2)),
    }
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc_per_node=8",
        "-m",
        "robouse.cli",
        action,
        "--config",
        str(path),
        "--execute",
    ]
    with (output / "full.log").open("a") as log:
        child = subprocess.Popen(
            command,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        write_json(output / "owner.json", {"pid": child.pid, "command": command})
        with tqdm(total=None, desc=action, dynamic_ncols=True, unit="step") as bar:
            while child.poll() is None:
                progress = output / "progress.json"
                if progress.exists():
                    try:
                        r = json.loads(progress.read_text())
                        bar.total = r["total"]
                        bar.n = r["step"]
                        bar.set_postfix(
                            loss=r.get("loss", "Unknown"),
                            epoch=r.get("epoch", "Unknown"),
                            throughput=round(r.get("tokens_per_second", 0), 3),
                            ETA="Unknown" if r.get("eta") is None else round(r["eta"]),
                            refresh=True,
                        )
                    except (json.JSONDecodeError, KeyError):
                        pass  # atomic record may not exist yet
                time.sleep(0.5)
        code = child.wait()
    if code:
        print(
            "\n".join(
                (output / "full.log").read_text(errors="replace").splitlines()[-35:]
            ),
            file=sys.stderr,
        )
        raise subprocess.CalledProcessError(code, command)

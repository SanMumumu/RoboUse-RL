"""Actual HF optimization/resume on a tiny CPU model; never initializes CUDA."""
import copy
from pathlib import Path
import pytest
import torch

transformers = pytest.importorskip("transformers", minversion="5.8.1")
from transformers import PretrainedConfig, PreTrainedModel
from transformers.modeling_outputs import CausalLMOutput
from robouse.training.sft.trainer import trainer_class, arguments, audit_callback


class Config(PretrainedConfig):
    model_type = "robouse_cpu_fixture"

    def __init__(self, **kw):
        super().__init__(**kw)


class Tiny(PreTrainedModel):
    config_class = Config

    def __init__(self, config):
        super().__init__(config)
        self.embed = torch.nn.Embedding(5, 4)
        self.head = torch.nn.Linear(4, 5, bias=False)

    def forward(
        self, input_ids, attention_mask=None, logits_to_keep=None, use_cache=False
    ):
        hidden = self.embed(input_ids)
        if logits_to_keep is not None:
            hidden = hidden[:, logits_to_keep, :]
        return CausalLMOutput(logits=self.head(hidden))


class Dataset:
    def __len__(self):
        return 8

    def __getitem__(self, i):
        ids = torch.tensor([[0, 1, 2, 3]])
        return {
            "inputs": {"input_ids": ids},
            "target": ids[0, 2:],
            "target_count": 2,
            "window_id": str(i),
            "dummy": False,
        }


def train(path, initial, stop=None, resume=None):
    cfg = {
        "output": str(path),
        "device": "cpu",
        "gradient_accumulation_steps": 2,
        "max_steps": 4,
        "save_steps": 1,
        "tensorboard": False,
        "gradient_checkpointing": False,
        "learning_rate": 0.01,
    }
    args = arguments(cfg, total_steps=4)
    model = Tiny(Config())
    model.load_state_dict(initial)
    cls = trainer_class(cfg, path)

    class Processor:
        def save_pretrained(self, path):
            import json

            Path(path).mkdir(parents=True, exist_ok=True)
            (Path(path) / "processor-fixture.json").write_text(
                json.dumps({"fixture": True})
            )

    callbacks = [audit_callback(cfg, model, Processor())]
    if stop:
        from transformers import TrainerCallback

        class Stop(TrainerCallback):
            def on_step_end(self, args, state, control, **kw):
                if state.global_step == stop:
                    control.should_save = True
                    control.should_training_stop = True

        callbacks.append(Stop())
    trainer = cls(
        model=model,
        args=args,
        train_dataset=Dataset(),
        data_collator=lambda x: x[0],
        callbacks=callbacks,
    )
    trainer.model_accepts_loss_kwargs = True
    trainer.train(resume_from_checkpoint=str(resume) if resume else None)
    return model.state_dict(), trainer.state.global_step


def test_real_hf_checkpoint_optimizer_rng_resume(tmp_path):
    torch.manual_seed(7)
    initial = copy.deepcopy(Tiny(Config()).state_dict())
    full, step = train(tmp_path / "full", initial)
    _, stop = train(tmp_path / "resume", initial, stop=2)
    resumed, step2 = train(
        tmp_path / "resume", initial, resume=tmp_path / "resume/checkpoint-2"
    )
    assert step == step2 == 4 and stop == 2
    assert all(torch.equal(full[k], resumed[k]) for k in full)
    cp = tmp_path / "resume/checkpoint-4"
    assert (
        (cp / "optimizer.pt").exists()
        and (cp / "scheduler.pt").exists()
        and (cp / "rng_state.pth").exists()
    )

"""HF full weights plus ZeRO optimizer/RNG state, never adapter checkpoints."""
from pathlib import Path
from agp_rl.io import digest, read_json, sha256, write_json

SCHEMA = 'agp-qwen35-fullft-v1'


def component(name):
    if '.visual.merger.' in name or '.visual.deepstack_merger_list.' in name:
        return 'aligner'
    if '.visual.' in name:
        return 'vision'
    return 'language'


def parameter_audit(model, config):
    result = {key: {'total': 0, 'trainable': 0} for key in ('language', 'vision', 'aligner')}
    for name, p in model.named_parameters():
        if any(word in name.lower() for word in ('lora_', 'adapter.')):
            raise ValueError('ADAPTER_PARAMETER_IN_FULLFT')
        group = component(name)
        count = getattr(p, 'ds_numel', p.numel())
        result[group]['total'] += count
        result[group]['trainable'] += count if p.requires_grad else 0
    for group, values in result.items():
        frozen = config.get({'vision': 'vision_freeze', 'aligner': 'aligner_freeze'}.get(group, ''), False)
        if values['total'] == 0 or values['trainable'] != (0 if frozen else values['total']):
            raise ValueError('UNEXPECTED_PARAMETER_FREEZE:' + group)
    return result


def freeze_parameters(model, config):
    for name, p in model.named_parameters():
        group = component(name)
        frozen = (group == 'vision' and config['vision_freeze']) or (group == 'aligner' and config['aligner_freeze'])
        p.requires_grad_(not frozen)
    return parameter_audit(model, config)


def commit(directory, *, config_hash, data_hash, audit, stage, step):
    directory = Path(directory)
    for name in ('config.json', 'tokenizer_config.json', 'tokenizer.json', 'preprocessor_config.json'):
        if not (directory / name).is_file():
            raise ValueError('FULLFT_PROCESSOR_OR_CONFIG_MISSING:' + name)
    files = sorted(directory.glob('*.safetensors'))
    if not files or (directory / 'adapter.pt').exists() or (directory / 'adapter_config.json').exists():
        raise ValueError('FULL_HF_WEIGHT_CHECKPOINT_REQUIRED')
    weights = {p.name: sha256(p) for p in files}
    inputs = {str(p.relative_to(directory)): sha256(p) for p in sorted(directory.glob('*.json'))
              if p.name not in ('fullft.json', 'FULLFT_COMMITTED.json')}
    resume_files = list(directory.glob('global_step*/**/*')) + list(directory.glob('rng_state*.pth'))
    resume_files.extend(directory / name for name in ('optimizer.pt', 'scheduler.pt') if (directory / name).is_file())
    resume_files = {str(p.relative_to(directory)): sha256(p) for p in sorted(set(resume_files)) if p.is_file()}
    metadata = {'schema': SCHEMA, 'finetune_type': 'full', 'stage': stage, 'step': step,
                'config_hash': config_hash, 'data_hash': data_hash, 'parameters': audit,
                'weights': weights, 'policy_version': digest(weights),
                'inference_inputs': inputs, 'resume_files': resume_files,
                'resume_format': 'HF Trainer state + DeepSpeed ZeRO-3 shards + rank RNG'}
    write_json(directory / 'fullft.json', metadata)
    write_json(directory / 'FULLFT_COMMITTED.json', {'metadata_sha256': sha256(directory / 'fullft.json')})
    return metadata


def verify(directory, *, resume=False):
    directory = Path(directory)
    meta = read_json(directory / 'fullft.json')
    marker = read_json(directory / 'FULLFT_COMMITTED.json')
    if marker['metadata_sha256'] != sha256(directory / 'fullft.json') or meta['schema'] != SCHEMA:
        raise ValueError('FULLFT_COMMIT_MISMATCH')
    for name, expected in meta['weights'].items():
        if sha256(directory / name) != expected:
            raise ValueError('FULLFT_WEIGHTS_CHANGED:' + name)
    for name, expected in meta['inference_inputs'].items():
        if sha256(directory / name) != expected:
            raise ValueError('FULLFT_PROCESSOR_OR_TRAINER_INPUT_CHANGED:' + name)
    if resume and (not (directory / 'trainer_state.json').is_file() or not list(directory.glob('global_step*'))):
        raise ValueError('EXACT_RESUME_REQUIRES_TRAINER_AND_ZERO_STATES')
    if resume:
        for name, expected in meta['resume_files'].items():
            if sha256(directory / name) != expected:
                raise ValueError('FULLFT_RESUME_STATE_CHANGED:' + name)
    return meta

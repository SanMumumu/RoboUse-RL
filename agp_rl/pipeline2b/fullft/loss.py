"""Only current assistant targets contribute; all causal RGB/history stay input."""
def assistant_nll(model, inputs, target, prompt):
    import torch
    positions = torch.arange(prompt - 1, prompt + target.numel() - 1, device=target.device)
    output = model(**inputs, use_cache=False, logits_to_keep=positions)
    if output.logits.shape[1] != target.numel():
        raise ValueError('TARGET_LOGIT_SELECTION_UNSUPPORTED')
    nll = torch.nn.functional.cross_entropy(output.logits[0].float(), target, reduction='sum')
    return nll, output

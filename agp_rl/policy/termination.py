"""Verified Qwen native-JSON chat and termination contract, shared by all arms.

Qwen3.5's local model config can name <|endoftext|> (padding) as legacy
generation EOS while the assistant template emits <|im_end|>. Explicitly use
the tokenizer/template assistant terminator, after proving that actual local
rendering and token boundaries agree. No model weights or generation required.
"""
from __future__ import annotations

from agp_rl.io import digest

VERSION = 'qwen-native-json-im-end-no-thinking-v2'
CHAT_TEMPLATE_OPTIONS = {'enable_thinking': False}


def native_termination_contract(processor, generation_config):
    tokenizer = processor.tokenizer
    end, pad = tokenizer.eos_token_id, tokenizer.pad_token_id
    if (tokenizer.eos_token != '<|im_end|>' or tokenizer.pad_token != '<|endoftext|>'
            or type(end) is not int or type(pad) is not int or end == pad
            or tokenizer.convert_tokens_to_ids('<|im_end|>') != end
            or tokenizer.convert_tokens_to_ids('<|endoftext|>') != pad):
        raise ValueError('Unsupported Qwen native assistant terminator/padding mapping')
    messages = [{'role': 'user', 'content': [{'type': 'text', 'text': 'Return one native tool JSON object.'}]}]
    target = '{"arguments":{},"name":"goal"}'
    complete = messages + [{'role': 'assistant', 'content': [{'type': 'text', 'text': target}]}]
    common = {'tokenize': False, 'add_generation_prompt': True}
    legacy = processor.apply_chat_template(messages, **common)
    prompt = processor.apply_chat_template(messages, **common, **CHAT_TEMPLATE_OPTIONS)
    history = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False, **CHAT_TEMPLATE_OPTIONS)
    full = processor.apply_chat_template(complete, tokenize=False, add_generation_prompt=False, **CHAT_TEMPLATE_OPTIONS)
    if not isinstance(prompt, str) or not prompt.startswith(history) or not full.startswith(prompt):
        raise ValueError('Qwen native assistant rendering is not a causal prefix')
    header = prompt[len(history):]
    if header.count('<think>') != header.count('</think>'):
        raise ValueError('Native JSON generation header leaves hidden reasoning open')
    prefix_ids = tokenizer.encode(prompt, add_special_tokens=False)
    full_ids = tokenizer.encode(full, add_special_tokens=False)
    if full_ids[:len(prefix_ids)] != prefix_ids:
        raise ValueError('Qwen native assistant token prefix differs')
    tail = full_ids[len(prefix_ids):]
    if tail.count(end) != 1 or pad in tail:
        raise ValueError('Qwen native assistant termination does not match tokenizer EOS')
    stop = tail.index(end)
    if (tokenizer.decode(tail[:stop], skip_special_tokens=True).strip() != target
            or tokenizer.decode(tail[stop+1:], skip_special_tokens=False).strip()):
        raise ValueError('Qwen native assistant suffix is not JSON plus one terminator')
    old = generation_config.eos_token_id
    old = [old] if isinstance(old, int) else list(old or [])
    return {'version': VERSION, 'eos_token_ids': [end], 'pad_token_id': pad,
        'enable_thinking': False, 'generation_header': header,
        'generation_header_has_unclosed_think': False, 'explicit_false_equals_prior_default': prompt == legacy,
        'legacy_model_generation_eos_token_ids': old,
        'legacy_model_eos_includes_padding': pad in old,
        'template_fixture_sha256': digest({'prompt': prompt, 'full': full}),
        'selection': 'Verified processor tokenizer <|im_end|>; model-config padding EOS is not an assistant stop'}

"""The deployed output envelope, shared with SFT free-generation diagnostics."""
import json


def parse_student_output(raw_text, codec):
    text = raw_text.strip()
    if not text.startswith('{'):
        return None, None  # Ordinary commentary is not a tool parsing trial.
    try:
        proposed = json.loads(text)
        if proposed.get('name') == 'exec' and set(proposed) == {'name', 'arguments'}:
            if set(proposed['arguments']) != {'code'} or not isinstance(proposed['arguments']['code'], str):
                raise ValueError('INVALID_EXEC_ENVELOPE')
            return proposed, None
        return codec.parse(text), None
    except (ValueError, TypeError, KeyError) as exc:
        return None, 'INVALID_CALL:' + type(exc).__name__ + ': ' + str(exc)

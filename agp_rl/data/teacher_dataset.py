import json
from xml.etree import ElementTree
DatasetRejected = ValueError
def require(value, reason):
    if not value: raise ValueError(reason)

def strict_json(text):

    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, 'DUPLICATE_JSON_KEY')
            result[key] = value
        return result
    return json.loads(text, object_pairs_hook=pairs, parse_constant=lambda _: (_ for _ in ()).throw(DatasetRejected('NONFINITE_JSON')))

def runtime_environment_message(payload):
    """Only the CLI-labelled environment wrapper is outside robot history."""
    metadata = payload.get('internal_chat_message_metadata_passthrough') or {}
    content = payload.get('content')
    if payload.get('type') != 'message' or payload.get('role') != 'user' or metadata.get('content_item_kinds') != ['environments.environment_context'] or (not isinstance(content, list)) or (len(content) != 1) or (content[0].get('type') != 'input_text') or (not isinstance(content[0].get('text'), str)):
        return False
    try:
        node = ElementTree.fromstring(content[0]['text'])
    except ElementTree.ParseError:
        return False
    return node.tag == 'environment_context' and len(node) > 0 and all((child.tag in ('cwd', 'shell', 'current_date', 'timezone', 'filesystem') for child in node)) and all((child.tag == 'filesystem' or len(child) == 0 for child in node)) and (not (node.text or '').strip()) and all((not (child.tail or '').strip() for child in node))

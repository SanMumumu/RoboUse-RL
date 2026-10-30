import re
IMAGE = re.compile(r'<image ([A-Za-z0-9_]+)>')
TRUNCATED = re.compile(r'\d+ (?:chars|characters) (?:truncated|omitted)|<image cut>')

def text_message(role, text):
    return {'role': role, 'content': [{'type': 'text', 'text': text}]}

def visible_result(result):
    """Preserve native content order; nested text-embedded images are ambiguous."""
    blocks, issues = ([], [])
    if result is None:
        return (blocks, ['MISSING_TOOL_REPLY'])
    for b in result.get('content', []):
        if b.get('type') == 'text':
            text = b.get('text', '')
            blocks.append({'type': 'text', 'text': text})
            if IMAGE.search(text):
                issues.append('TEXT_EMBEDDED_IMAGE_RENDERING_UNVERIFIABLE')
            if TRUNCATED.search(text):
                issues.append('SOURCE_OUTPUT_TRUNCATED')
        elif b.get('type') == 'image':
            match = IMAGE.fullmatch(b.get('data', ''))
            if match and match[1] != 'cut':
                blocks.append({'type': 'image', 'image': 'art-image://' + match[1]})
            else:
                issues.append('IMAGE_CUT_OR_UNRECONSTRUCTABLE')
        else:
            issues.append('UNKNOWN_TOOL_CONTENT_BLOCK')
    if result.get('structured_content') is not None:
        issues.append('STRUCTURED_CONTENT_RENDERING_UNKNOWN')
    return (blocks, sorted(set(issues)))

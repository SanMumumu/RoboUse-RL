import re
import json
from agp_rl.io import canonical
from agp_rl.adapters.observation_codec import assert_public
from agp_rl.adapters.tool_codec import TOOLS
from agp_rl.data.teacher_dataset import strict_json
CODE_TOOL = re.compile(r'\btools\.([A-Za-z_][A-Za-z_0-9]*)')
FORBIDDEN_CODE = re.compile(r'\b(?:require|import|fetch|XMLHttpRequest|process|Deno)\b|\b(?:fs|net|http|https)\s*\.')
def require(value, reason):
    if not value: raise ValueError(reason)

def student_system_prompt(schemas):
    """Shared coding-agent envelope for exporter and local student actor."""
    return 'Control only the simulated robot through the unchanged official tools. For a direct tool call return one JSON object {"name":TOOL,"arguments":OBJECT}. For JavaScript computation or dependent robot calls return {"name":"exec","arguments":{"code":JAVASCRIPT}}. JavaScript has Math, text(value), image(block), store(key,value), load(key), and tools.mcp__robot_js__TOOL(args); store/load persist JSON values between scripts. No filesystem, shell, network, imports or other tools are available. Short assistant commentary may be plain text. Tool errors are observations; preserve the real history when recovering. Official schemas:\n' + canonical(schemas).decode()

def code_target(item):
    code = item.get('input', item.get('arguments'))
    require(isinstance(code, str) and code.strip(), 'EXEC_SOURCE_REQUIRED')
    require(not FORBIDDEN_CODE.search(code), 'EXEC_FILESYSTEM_NETWORK_OR_IMPORT_DEPENDENCY')
    for name in CODE_TOOL.findall(code):
        require(name.startswith('mcp__robot_js__') and name[len('mcp__robot_js__'):] in TOOLS, 'EXEC_UNKNOWN_TOOL_DEPENDENCY')
    require(not re.search('\\btools\\s*\\[', code), 'EXEC_DYNAMIC_TOOL_DEPENDENCY')
    return canonical({'name': 'exec', 'arguments': {'code': code}}).decode()

def visible_blocks(value):
    """Flatten the actual visible tool envelope, retaining text and frame order."""
    if isinstance(value, str):
        try:
            parsed = strict_json(value)
        except ValueError:
            return [{'type': 'text', 'text': value}]
        return visible_blocks(parsed)
    if isinstance(value, list):
        result = []
        for item in value:
            result.extend(visible_blocks(item))
        return result
    if value is None or isinstance(value, (int, float, bool)):
        return [{'type': 'text', 'text': canonical(value).decode()}]
    require(isinstance(value, dict), 'VISIBLE_OUTPUT_UNSUPPORTED')
    if 'encoded_json' in value:
        return visible_blocks(value['encoded_json'])
    if value.get('type') in ('text', 'input_text', 'output_text'):
        text = value.get('text')
        require(isinstance(text, str), 'VISIBLE_TEXT_REQUIRED')
        try:
            assert_public(strict_json(text))
        except json.JSONDecodeError:
            pass
        return [{'type': 'text', 'text': text}]
    if value.get('type') == 'image' and isinstance(value.get('image'), str):
        return [{'type': 'image', 'image': value['image']}]
    if 'content' in value:
        return visible_blocks(value['content'])
    assert_public(value)
    return [{'type': 'text', 'text': canonical(value).decode()}]

"""Schema-bound dispatch using the existing strict, finite JSON tool codec."""
from agp_rl.adapters.tool_codec import ToolCodec


class ToolRegistry:
    def __init__(self, tools, *, official_robot=False):
        tools = list(tools)
        self.tools = {tool.name: tool for tool in tools}
        if len(self.tools) != len(tools) or not tools:
            raise ValueError("Unique nonempty tool registry required")
        kwargs = {} if official_robot else {"required_tools": None}
        self.codec = ToolCodec({name: tool.input_schema for name, tool in self.tools.items()}, **kwargs)

    def parse(self, text):
        return self.codec.parse(text)

    async def execute(self, call):
        # Validate even if the caller constructed a dict directly.
        validated = self.parse(self.codec.encode(call["name"], call["arguments"]))
        return await self.tools[validated["name"]].execute(validated["arguments"])

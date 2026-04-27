"""Tool schemas and system prompts shared by every output sample.

The tool schema dicts here mirror the YAML files under
``examples/sglang_multiturn/config/tool_config/`` so the model sees the *same*
function signatures during SFT and RL.

If you change a tool's argument shape upstream, update the matching entry below
or the SFT-trained model will emit a slightly off-distribution call schema and
your RL rollouts will mis-parse them.
"""

from __future__ import annotations

from typing import Any


IMAGE_ZOOM_IN_TOOL_SCHEMA: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "image_zoom_in_tool",
        "description": (
            "Zoom in on a specific region of an image by cropping it based on a bounding "
            "box (bbox) and an optional object label."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "bbox_2d": {
                    "type": "array",
                    "items": {"type": "number"},
                    "minItems": 4,
                    "maxItems": 4,
                    "description": (
                        "The bounding box of the region to zoom in, as [x1, y1, x2, y2], "
                        "where (x1, y1) is the top-left corner and (x2, y2) is the "
                        "bottom-right corner."
                    ),
                },
                "label": {
                    "type": "string",
                    "description": (
                        "The name or label of the object in the specified bounding "
                        "box (optional)."
                    ),
                },
            },
            "required": ["bbox_2d"],
        },
    },
}


CODE_INTERPRETER_TOOL_SCHEMA: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "code_interpreter",
        "description": "A tool for executing code.",
        "parameters": {
            "type": "object",
            "properties": {
                "code": {"type": "string", "description": "The code to execute."},
            },
            "required": ["code"],
        },
    },
}


SEARCH_TOOL_SCHEMA: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "search",
        "description": "Searches the web for relevant information based on the given query.",
        "parameters": {
            "type": "object",
            "properties": {
                "query_list": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "A list of fully-formed semantic queries. The tool will return "
                        "search results for each query."
                    ),
                },
            },
            "required": ["query_list"],
        },
    },
}


IMAGE_SEARCH_TOOL_SCHEMA: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "image_search",
        "description": (
            "Performs a reverse image search using the most recent image in the "
            "conversation, returning visually similar web results with thumbnail "
            "images and source titles."
        ),
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
}


ZOOM_ONLY_TOOLS: list[dict[str, Any]] = [IMAGE_ZOOM_IN_TOOL_SCHEMA]
DIVERSE_TOOLS: list[dict[str, Any]] = [
    IMAGE_ZOOM_IN_TOOL_SCHEMA,
    CODE_INTERPRETER_TOOL_SCHEMA,
    SEARCH_TOOL_SCHEMA,
    IMAGE_SEARCH_TOOL_SCHEMA,
]


# We deliberately keep ``ZOOM_ONLY_SYSTEM_PROMPT`` close to the one used by the
# RL launcher (see ``examples/data_preprocess/geo3k_zoom.py``) so the SFT'd
# model sees a consistent contract throughout training.
ZOOM_ONLY_SYSTEM_PROMPT = (
    "You are a helpful multimodal expert. You are given a question and an accompanying image. "
    "You must reason step by step and may invoke the `image_zoom_in_tool` to inspect any region "
    "of the image more closely.\n\n"
    "When you call the tool you MUST output exactly:\n"
    "<tool_call>\n"
    '{"name": "image_zoom_in_tool", "arguments": {"bbox_2d": [x1, y1, x2, y2]}}\n'
    "</tool_call>\n\n"
    "After gathering enough evidence, wrap your reasoning in <think> </think> tags and put the "
    "final answer in \\boxed{}."
)


DIVERSE_SYSTEM_PROMPT = (
    "You are a helpful multimodal expert. You are given a question and an accompanying image. "
    "You must reason step by step and may invoke any of the following tools to gather more "
    "evidence before answering:\n"
    "  * `image_zoom_in_tool`  — crop and inspect a region of the image\n"
    "  * `code_interpreter`     — run Python code (image processing, computation, plotting)\n"
    "  * `search`               — issue web search queries\n"
    "  * `image_search`         — reverse-image search the most recent image\n\n"
    "When you call a tool you MUST output exactly:\n"
    "<tool_call>\n"
    '{"name": "<tool_name>", "arguments": {...}}\n'
    "</tool_call>\n\n"
    "Use tools sparingly and only when they will help you answer the question. After gathering "
    "enough evidence, wrap your reasoning in <think> </think> tags and put the final answer in "
    "\\boxed{}."
)


def system_prompt_for(target: str) -> str:
    """Return the appropriate system prompt for a given target dataset name."""

    if target == "zoom_only":
        return ZOOM_ONLY_SYSTEM_PROMPT
    if target == "diverse":
        return DIVERSE_SYSTEM_PROMPT
    raise ValueError(f"Unknown target dataset: {target!r}")


def tool_schemas_for(target: str) -> list[dict[str, Any]]:
    if target == "zoom_only":
        return ZOOM_ONLY_TOOLS
    if target == "diverse":
        return DIVERSE_TOOLS
    raise ValueError(f"Unknown target dataset: {target!r}")

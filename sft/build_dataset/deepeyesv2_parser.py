"""Parse DeepEyesV2 SFT JSON dumps into language-agnostic trajectories.

The on-disk format (per sample) is::

    {
      "messages": [
        {"role": "system", "content": "..."},                # discarded
        {"role": "user",   "content": "<image>\\nQ?\\n..."},
        {"role": "assistant", "content": "<think>..</think>" \
                                          "<code>..</code>"},
        {"role": "user",   "content": "Code execution result:..."},
        ...
        {"role": "assistant", "content": "<think>..</think><answer>X</answer>"}
      ],
      "images": ["images/<file_id>_000.jpg", "images/<file_id>_001.jpg", ...]
    }

The image cursor is tracked across the whole conversation: every ``<image>``
placeholder (in any role) consumes the next entry from the ``images`` list.
The first one is the original input; subsequent entries are tool-rendered
images (matplotlib figures or search thumbnails).

This module deliberately stops at *parsing*. Tool-call detection happens in
``tool_call_converter.py`` and the conversion to our codebase's parquet schema
happens in ``build.py``.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Iterator, Optional

from .tool_call_converter import ParsedAssistantTurn, parse_assistant_content


logger = logging.getLogger(__name__)


_USER_IMAGE_PLACEHOLDER = re.compile(r"<image>")
_SEARCH_RESPONSE_RE = re.compile(r"<tool_response>(.*?)</tool_response>", re.S)


@dataclass
class RawTurn:
    """One full assistant + tool-response pair, plus image bookkeeping."""

    assistant: ParsedAssistantTurn
    response_text: str = ""              # tool / observation message body (may be "")
    response_image_indices: list[int] = field(default_factory=list)  # indices into traj.image_paths
    is_terminal: bool = False            # final assistant turn, no tool call


@dataclass
class RawTrajectory:
    """Intermediate representation: structured but still string-typed images."""

    question: str
    user_image_indices: list[int]        # indices into image_paths used in the user prompt
    image_paths: list[str]               # ALL images referenced anywhere in the conversation
    turns: list[RawTurn]
    categories: set[str]                 # subset of {"zoom", "code", "search", "image_search"}
    sample_id: str
    raw_index: int                       # which sample index in the source JSON
    source_file: str                     # JSON file basename
    valid: bool = True
    invalid_reason: Optional[str] = None

    @property
    def n_zoom_calls(self) -> int:
        return sum(1 for t in self.turns for c in t.assistant.call_categories if c == "zoom")

    @property
    def n_code_calls(self) -> int:
        return sum(1 for t in self.turns for c in t.assistant.call_categories if c == "code")

    @property
    def n_search_calls(self) -> int:
        return sum(1 for t in self.turns for c in t.assistant.call_categories if c == "search")

    @property
    def n_image_search_calls(self) -> int:
        return sum(
            1 for t in self.turns for c in t.assistant.call_categories if c == "image_search"
        )

    @property
    def n_non_zoom_calls(self) -> int:
        return self.n_code_calls + self.n_search_calls + self.n_image_search_calls

    @property
    def is_zoom_only(self) -> bool:
        """A zoom-only trajectory has at least one zoom call and zero non-zoom calls."""

        return self.n_zoom_calls > 0 and self.n_non_zoom_calls == 0

    @property
    def is_zoom_using(self) -> bool:
        """Has at least one zoom call (alone or mixed with other tools)."""

        return self.n_zoom_calls > 0


def _consume_images(
    content: str, image_cursor: int, total_images: int
) -> tuple[list[int], int]:
    """Pull image indices out of a content string in order of ``<image>`` tags."""

    placeholders = _USER_IMAGE_PLACEHOLDER.findall(content)
    indices: list[int] = []
    for _ in placeholders:
        if image_cursor >= total_images:
            # Source dataset is occasionally inconsistent; we record the issue
            # so the caller can drop the trajectory.
            return indices, -1
        indices.append(image_cursor)
        image_cursor += 1
    return indices, image_cursor


def _split_search_response(response_text: str) -> str:
    """Strip the ``<tool_response>`` wrappers around DeepEyesV2 search outputs."""

    matches = _SEARCH_RESPONSE_RE.findall(response_text)
    if matches:
        return "\n\n".join(m.strip() for m in matches)
    return response_text.strip()


def parse_one(
    sample: dict, raw_index: int, source_file: str
) -> Optional[RawTrajectory]:
    """Parse a single DeepEyesV2 sample. Return ``None`` when malformed."""

    messages = sample.get("messages", [])
    image_paths = list(sample.get("images", []))
    if not messages or not image_paths:
        return None

    # Skip the leading system turn — we replace it with our own prompt later.
    if messages[0]["role"] == "system":
        messages = messages[1:]

    if len(messages) < 2 or messages[0]["role"] != "user":
        return None

    image_cursor = 0
    user_msg = messages[0]
    user_content = user_msg.get("content", "") or ""
    user_image_indices, image_cursor = _consume_images(
        user_content, image_cursor, len(image_paths)
    )
    if image_cursor < 0:
        return None
    # The first user turn ALWAYS has at least one image reference for our
    # purposes — questions without an attached image fall outside the
    # zoom-in / image-search experimental scope.
    if not user_image_indices:
        return None
    question_text = _USER_IMAGE_PLACEHOLDER.sub("", user_content).strip()

    turns: list[RawTurn] = []
    categories: set[str] = set()
    i = 1
    sample_id = sample.get("id") or f"{Path(source_file).stem}:{raw_index}"

    while i < len(messages):
        asst_msg = messages[i]
        if asst_msg.get("role") != "assistant":
            return None

        # Look ahead at the next user (tool-response) turn so we can pass its
        # ``<image>`` count down to the converter — that lets ``_is_pure_crop``
        # honestly distinguish single-crop blocks from multi-image plots.
        expected_resp_imgs: Optional[int] = None
        next_msg = messages[i + 1] if i + 1 < len(messages) else None
        if next_msg is not None and next_msg.get("role") == "user":
            expected_resp_imgs = len(
                _USER_IMAGE_PLACEHOLDER.findall(next_msg.get("content") or "")
            )

        parsed = parse_assistant_content(
            asst_msg.get("content") or "",
            expected_response_images=expected_resp_imgs,
        )
        if "__bad__" in parsed.call_categories:
            return None
        if len(parsed.tool_calls) > 1:
            return None  # we filter to ≤1 tool call per turn

        cat = parsed.call_categories[0] if parsed.tool_calls else None
        if cat is not None:
            categories.add(cat)

        # Pair with the following user (tool response) turn iff the assistant
        # actually invoked a tool. The terminal answer turn has no follower.
        response_text = ""
        response_image_indices: list[int] = []
        if cat is not None:
            if next_msg is None or next_msg.get("role") != "user":
                return None  # tool-call without a response → broken
            resp_content = next_msg.get("content") or ""
            response_image_indices, image_cursor = _consume_images(
                resp_content, image_cursor, len(image_paths)
            )
            if image_cursor < 0:
                return None
            response_text = _USER_IMAGE_PLACEHOLDER.sub("", resp_content).strip()
            if cat in ("search", "image_search"):
                response_text = _split_search_response(response_text)

            # Sanity-check tool-response image counts.
            #   * zoom         → exactly 1 cropped image
            #   * search       → text-only, 0 images
            #   * image_search → at least 1 visual-search thumbnail
            #   * code         → variable; trust the source data
            n_resp_imgs = len(response_image_indices)
            if cat == "zoom" and n_resp_imgs != 1:
                return None
            if cat == "search" and n_resp_imgs != 0:
                return None
            if cat == "image_search" and n_resp_imgs == 0:
                return None
            i += 2
        else:
            i += 1

        turns.append(
            RawTurn(
                assistant=parsed,
                response_text=response_text,
                response_image_indices=response_image_indices,
                is_terminal=(cat is None),
            )
        )

    # The final turn must be terminal (assistant answer with no tool call).
    if not turns or not turns[-1].is_terminal:
        return None
    # ... and we want a non-empty answer-bearing turn.
    if not (turns[-1].assistant.text_content or "").strip():
        return None

    return RawTrajectory(
        question=question_text,
        user_image_indices=user_image_indices,
        image_paths=image_paths,
        turns=turns,
        categories=categories,
        sample_id=sample_id,
        raw_index=raw_index,
        source_file=Path(source_file).name,
    )


def parse_json_file(
    path: str | Path, max_samples: Optional[int] = None
) -> Iterator[RawTrajectory]:
    """Stream-parse one ``sft_part_*.json`` file and yield every valid trajectory."""

    path = Path(path)
    logger.info("Parsing %s", path)
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected a top-level list in {path}, got {type(data).__name__}")
    seen = 0
    for idx, sample in enumerate(data):
        if max_samples is not None and seen >= max_samples:
            return
        traj = parse_one(sample, raw_index=idx, source_file=path.name)
        if traj is None:
            continue
        seen += 1
        yield traj


def parse_files(
    paths: Iterable[str | Path],
    max_samples_per_file: Optional[int] = None,
) -> Iterator[RawTrajectory]:
    for p in paths:
        yield from parse_json_file(p, max_samples=max_samples_per_file)

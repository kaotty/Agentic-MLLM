"""HuggingFace download helpers for the SFT raw data.

We deliberately avoid ``datasets.load_dataset(...)``: DeepEyesV2 ships custom
JSON files plus a multi-GB ``images.zip`` that ``load_dataset`` cannot infer.
Instead we use ``huggingface_hub.snapshot_download`` to mirror the repo, then
unzip the images on first access.

Set ``HF_HOME`` (or ``HUGGINGFACE_HUB_CACHE``) per-server to control where the
12 GB DeepEyesV2 raw is stored. Classmates can also pre-stage the cache on a
shared NFS path and point ``HF_HOME`` there.
"""

from __future__ import annotations

import logging
import os
import zipfile
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


DEEPEYESV2_REPO = "honglyhly/DeepEyesV2_SFT"


def _hf_snapshot_download(
    repo_id: str,
    *,
    repo_type: str = "dataset",
    allow_patterns: Optional[list[str]] = None,
    cache_dir: Optional[str] = None,
) -> Path:
    """Wrap ``huggingface_hub.snapshot_download`` with our defaults."""

    from huggingface_hub import snapshot_download

    local_dir = snapshot_download(
        repo_id=repo_id,
        repo_type=repo_type,
        allow_patterns=allow_patterns,
        cache_dir=cache_dir,
        max_workers=int(os.environ.get("HF_HUB_DOWNLOAD_WORKERS", "4")),
    )
    return Path(local_dir)


def download_deepeyesv2(
    *,
    cache_dir: Optional[str] = None,
    only_json: bool = False,
    only_parts: Optional[list[int]] = None,
) -> Path:
    """Download DeepEyesV2_SFT to the HF cache and return the local snapshot path.

    ``only_parts`` lets you restrict to e.g. ``[0]`` for a quick smoke test.
    ``only_json`` skips the 10 GB ``images.zip`` (useful while iterating on
    parsers; build_main.py will refuse to actually bake datasets without it).
    """

    patterns: list[str] = []
    if only_parts is not None:
        patterns.extend(f"json/sft_part_{i}.json" for i in only_parts)
    else:
        patterns.append("json/*.json")
    if not only_json:
        patterns.append("images.zip")

    return _hf_snapshot_download(DEEPEYESV2_REPO, allow_patterns=patterns, cache_dir=cache_dir)


def ensure_images_extracted(snapshot_dir: Path, marker_filename: str = ".extracted") -> Path:
    """Unzip ``images.zip`` lazily into ``<snapshot>/images/`` once, then short-circuit."""

    zip_path = snapshot_dir / "images.zip"
    extract_dir = snapshot_dir / "images"
    marker = extract_dir / marker_filename
    if marker.exists():
        return extract_dir
    if not zip_path.exists():
        raise FileNotFoundError(
            f"Expected {zip_path} to exist. Re-run download with only_json=False."
        )
    extract_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Extracting %s → %s (~10 GB, this is a one-time cost)", zip_path, extract_dir)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(extract_dir)
    marker.touch()
    return extract_dir

"""End-to-end build script for the two SFT datasets.

The build is **fully deterministic** given a fixed ``--seed`` and the
DeepEyesV2 source.

Usage (one-shot, on everyone's server)::

    # Build BOTH datasets, using ALL available DeepEyesV2 trajectories.
    # ~25 GB disk for the HF cache (12 GB zip + 13 GB extracted images),
    # ~5 min on a warm cache.
    python -m sft.build_dataset.build

    # Or build just one side:
    python -m sft.build_dataset.build --target zoom_only
    python -m sft.build_dataset.build --target diverse

    # Inspect first \u2014 prints per-pool counts and a worked example without
    # writing parquets.
    python -m sft.build_dataset.build --inspect
"""

from __future__ import annotations

import argparse
import dataclasses
import io
import json
import logging
import random
import sys
from pathlib import Path
from typing import Any, Optional

import pandas as pd
from PIL import Image

from . import schema as sft_schema
from .deepeyesv2_parser import RawTrajectory, parse_files as parse_deepeyesv2_files
from .image_baker import bake_zoom, zoom_response_text
from .tool_call_converter import normalize_final_answer
from .download import download_deepeyesv2, ensure_images_extracted


logger = logging.getLogger("agentic_sft.build")


# Anchor the default output directory inside the project checkout so every
# classmate gets the same on-disk layout regardless of where they invoke
# Python from. ``build.py`` lives at
# ``<root>/sft/build_dataset/build.py`` so ``parents[1]`` is ``<root>/sft``
# and ``parents[2]`` is the project root.
_SFT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = str(_SFT_DIR / "sft_dataset")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument(
        "--target",
        choices=["zoom_only", "diverse", "both"],
        default="both",
        help="Which dataset(s) to build.",
    )
    p.add_argument(
        "--test_frac",
        type=float,
        default=0.02,
        help="Fraction of trajectories held out as the SFT eval split.",
    )
    p.add_argument(
        "--output_dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help=(
            "Directory under which {zoom_only,diverse}/{train,test}.parquet "
            "are written. Defaults to <project_root>/sft/sft_dataset."
        ),
    )
    p.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="Override the HF cache directory. Defaults to $HF_HOME or ~/.cache/huggingface.",
    )
    p.add_argument(
        "--source_dir",
        type=str,
        default=None,
        help="If set, skip the HF download and read the raw DeepEyesV2 snapshot from this path.",
    )
    p.add_argument(
        "--only_parts",
        type=int,
        nargs="*",
        default=None,
        help="Debug: restrict to a subset of sft_part_*.json files (0..4) for a quick build.",
    )
    p.add_argument(
        "--inspect",
        action="store_true",
        help="Don't write parquets; print per-category stats + a worked example and exit.",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--no_images",
        action="store_true",
        help=(
            "Skip downloading images.zip — useful for parser-only iteration "
            "with --inspect. Not valid for actual builds."
        ),
    )
    p.add_argument(
        "--max_samples_per_file",
        type=int,
        default=None,
        help="Debug: cap how many raw samples we read from each JSON before filtering.",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------


def _load_image_bytes(path: Path) -> bytes:
    """Re-encode any PIL-loadable file to JPEG bytes for parquet storage.

    JPEG keeps parquets compact while preserving enough fidelity for VL
    training. If a future ablation needs the original PNGs/WEBPs, swap to
    ``open(path, 'rb').read()`` here.
    """

    with Image.open(path) as im:
        im = im.convert("RGB")
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=92)
        return buf.getvalue()


def _bake_zoom_bytes(
    original_image_path: Path, bbox_2d: list[int]
) -> Optional[tuple[bytes, list[int]]]:
    try:
        with Image.open(original_image_path) as im:
            im = im.convert("RGB")
            baked = bake_zoom(im, bbox_2d)
            if baked is None:
                return None
            cropped, resized_bbox = baked
            buf = io.BytesIO()
            cropped.save(buf, format="JPEG", quality=92)
            return buf.getvalue(), resized_bbox
    except Exception as exc:  # pragma: no cover — defensive only
        logger.warning("Failed to bake zoom for %s: %s", original_image_path, exc)
        return None


# ---------------------------------------------------------------------------
# Trajectory → output sample conversion
# ---------------------------------------------------------------------------


def _make_user_message(question: str, n_user_images: int) -> dict[str, Any]:
    placeholders = "\n".join(["<image>"] * n_user_images)
    if question:
        return {"role": "user", "content": f"{placeholders}\n{question}".strip()}
    return {"role": "user", "content": placeholders}


def _truncate(s: str, max_chars: int = 1500) -> str:
    if len(s) <= max_chars:
        return s
    return s[:max_chars] + "\n... (truncated)"


def convert_trajectory(
    traj: RawTrajectory,
    images_root: Path,
    target: str,
) -> Optional[dict[str, Any]]:
    """Apply tool-call conversion + image baking to produce one parquet row.

    Returns ``None`` if the trajectory cannot be converted (e.g. non-static
    crop bbox, missing image file).
    """

    is_zoom_only = target == "zoom_only"
    if is_zoom_only and not traj.is_zoom_only:
        return None

    # Diverse mode also requires that EVERY tool call is something we support
    # (zoom / code / search). All ``__bad__`` cases were filtered upstream.

    image_paths = [images_root / p for p in traj.image_paths]
    for p in image_paths:
        if not p.exists():
            return None

    out_messages: list[dict[str, Any]] = [
        {"role": "system", "content": sft_schema.system_prompt_for(target)},
    ]
    out_images: list[bytes] = []

    # First user turn: original image(s) + question.
    for idx in traj.user_image_indices:
        out_images.append(_load_image_bytes(image_paths[idx]))
    out_messages.append(_make_user_message(traj.question, len(traj.user_image_indices)))

    for turn in traj.turns:
        # Assistant turn body. We strip residual <image> tags (they would have
        # come from the source data's tool responses, never from the assistant
        # itself, but be defensive). Replace <answer>X</answer> with \boxed{X}.
        text = normalize_final_answer(turn.assistant.text_content or "")
        text = text.replace("<image>", "").rstrip()

        asst_msg: dict[str, Any] = {"role": "assistant", "content": text}

        if turn.assistant.tool_calls:
            call = turn.assistant.tool_calls[0]
            cat = turn.assistant.call_categories[0]

            # Pure-zoom branch: re-bake the crop with our exact runtime logic
            # so the SFT image matches what RL will produce later.
            if cat == "zoom":
                bbox_2d = json.loads(call["function"]["arguments"]).get("bbox_2d")
                # Source images are at user_image_indices[0] for now; multi-image
                # zoom trajectories are filtered upstream (we only allow single user image).
                if len(traj.user_image_indices) != 1:
                    return None
                user_img_path = image_paths[traj.user_image_indices[0]]
                baked = _bake_zoom_bytes(user_img_path, bbox_2d)
                if baked is None:
                    return None
                cropped_bytes, resized_bbox = baked
                # Update the call to use the (possibly resized) bbox so the
                # response_text matches what RL will narrate.
                call["function"]["arguments"] = json.dumps(
                    {"bbox_2d": resized_bbox}, ensure_ascii=False
                )
                asst_msg["tool_calls"] = [call]
                out_messages.append(asst_msg)
                out_messages.append(
                    {
                        "role": "tool",
                        "content": "<image>\n" + zoom_response_text(resized_bbox),
                    }
                )
                out_images.append(cropped_bytes)
                continue

            # Code-interpreter branch: pass through the source images and text.
            if cat == "code":
                if is_zoom_only:
                    return None
                asst_msg["tool_calls"] = [call]
                out_messages.append(asst_msg)
                tool_content = _truncate(turn.response_text)
                if turn.response_image_indices:
                    placeholders = "\n".join(["<image>"] * len(turn.response_image_indices))
                    tool_content = f"{placeholders}\n{tool_content}".strip()
                    for idx in turn.response_image_indices:
                        out_images.append(_load_image_bytes(image_paths[idx]))
                out_messages.append({"role": "tool", "content": tool_content})
                continue

            # Web-search branch: text-only response.
            if cat == "search":
                if is_zoom_only:
                    return None
                if turn.response_image_indices:
                    return None  # text-search responses must be image-free
                asst_msg["tool_calls"] = [call]
                out_messages.append(asst_msg)
                out_messages.append(
                    {"role": "tool", "content": _truncate(turn.response_text)}
                )
                continue

            # Reverse-image-search branch: response carries N thumbnails plus
            # textual context (e.g. "## Web Results\n1. <image>\n[Title]\n...").
            if cat == "image_search":
                if is_zoom_only:
                    return None
                if not turn.response_image_indices:
                    return None  # image_search must return at least one thumbnail
                asst_msg["tool_calls"] = [call]
                out_messages.append(asst_msg)
                placeholders = "\n".join(["<image>"] * len(turn.response_image_indices))
                tool_content = _truncate(turn.response_text)
                tool_content = f"{placeholders}\n{tool_content}".strip()
                for idx in turn.response_image_indices:
                    out_images.append(_load_image_bytes(image_paths[idx]))
                out_messages.append({"role": "tool", "content": tool_content})
                continue

            return None  # unknown category

        # Terminal turn — no tool call. Just emit the assistant message.
        out_messages.append(asst_msg)

    # Sanity check: <image> placeholders == len(out_images)
    placeholder_count = 0
    for m in out_messages:
        c = m["content"]
        if isinstance(c, str):
            placeholder_count += c.count("<image>")
    if placeholder_count != len(out_images):
        return None

    return {
        "messages": out_messages,
        "images": [{"bytes": b, "path": None} for b in out_images],
        "tools": sft_schema.tool_schemas_for(target),
        "data_source": f"deepeyesv2_sft.{target}",
        "ability": _infer_ability(traj),
        "extra_info": {
            "source_file": traj.source_file,
            "raw_index": traj.raw_index,
            "sample_id": traj.sample_id,
            "n_zoom_calls": traj.n_zoom_calls,
            "n_code_calls": traj.n_code_calls,
            "n_search_calls": traj.n_search_calls,
            "n_image_search_calls": traj.n_image_search_calls,
        },
    }


def _infer_ability(traj: RawTrajectory) -> str:
    """Cheap heuristic ability tag for downstream filtering / weighting."""

    if traj.is_zoom_using:
        return "perception"
    if traj.n_image_search_calls > 0:
        return "image_search"
    if traj.n_search_calls > 0:
        return "search"
    if traj.n_code_calls > 0:
        return "math_code"
    return "general"


# ---------------------------------------------------------------------------
# Splitting + writing
# ---------------------------------------------------------------------------


def _stratified_split(
    items: list[Any], test_frac: float, rng: random.Random
) -> tuple[list[Any], list[Any]]:
    """Shuffle ``items`` and slice off ``test_frac`` for the test split.

    Empty / single-item pools degrade gracefully:
      * empty pool      → ``([], [])``
      * single-item pool → goes to train, test stays empty
      * otherwise        → at least 1 item in test, the rest in train
    """

    items = list(items)
    if not items:
        return [], []
    rng.shuffle(items)
    if len(items) == 1:
        return items, []
    n_test = max(1, int(len(items) * test_frac))
    return items[n_test:], items[:n_test]


def _write_parquet(rows: list[dict[str, Any]], out_path: Path) -> None:
    if not rows:
        # An empty rows list pickles to a parquet with no schema, which crashes
        # downstream readers. Refuse to write and surface the imbalance loudly.
        raise RuntimeError(
            f"No rows to write for {out_path}. The DeepEyesV2 source produced "
            "an empty pool for this split — check parser stats with --inspect, "
            "or lower --test_frac if the offending split is the test parquet."
        )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_parquet(out_path, index=False)
    logger.info("Wrote %d rows → %s (%.1f MB)", len(rows), out_path, out_path.stat().st_size / 1e6)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class TrajectoryPools:
    """Six disjoint buckets used by ``build`` to compose the SFT datasets."""

    zoom: list[RawTrajectory] = dataclasses.field(default_factory=list)
    code: list[RawTrajectory] = dataclasses.field(default_factory=list)
    search: list[RawTrajectory] = dataclasses.field(default_factory=list)
    image_search: list[RawTrajectory] = dataclasses.field(default_factory=list)
    mixed_zoom: list[RawTrajectory] = dataclasses.field(default_factory=list)
    mixed_non_zoom: list[RawTrajectory] = dataclasses.field(default_factory=list)


def _gather_raw_trajectories(
    json_paths: list[Path],
    max_samples_per_file: Optional[int],
) -> TrajectoryPools:
    """Walk all JSONs once and bucket trajectories by tool-category.

    Buckets are disjoint:
      * ``zoom``           — pure-zoom + single user image
      * ``code``           — only ``code_interpreter`` calls
      * ``search``         — only web ``search`` calls
      * ``image_search``   — only ``image_search`` calls
      * ``mixed_zoom``     — at least one zoom call AND at least one non-zoom
                             call (single user image, baker-compatible)
      * ``mixed_non_zoom`` — no zoom, multiple non-zoom tool kinds
                             (e.g. ``search`` + ``image_search``)
    """

    pools = TrajectoryPools()

    for traj in parse_deepeyesv2_files(json_paths, max_samples_per_file=max_samples_per_file):
        single_user_img = len(traj.user_image_indices) == 1
        if traj.is_zoom_only and single_user_img:
            pools.zoom.append(traj)
        elif traj.is_zoom_using and single_user_img:
            pools.mixed_zoom.append(traj)
        elif traj.n_zoom_calls == 0 and traj.n_code_calls > 0 and traj.n_search_calls == 0 and traj.n_image_search_calls == 0:
            pools.code.append(traj)
        elif traj.n_zoom_calls == 0 and traj.n_search_calls > 0 and traj.n_code_calls == 0 and traj.n_image_search_calls == 0:
            pools.search.append(traj)
        elif traj.n_zoom_calls == 0 and traj.n_image_search_calls > 0 and traj.n_code_calls == 0 and traj.n_search_calls == 0:
            pools.image_search.append(traj)
        elif traj.n_zoom_calls == 0:
            # Multi-tool non-zoom: e.g. (search, image_search). Diverse-only.
            pools.mixed_non_zoom.append(traj)
    return pools


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(name)s %(levelname)s: %(message)s",
    )

    rng = random.Random(args.seed)
    output_dir = Path(args.output_dir).expanduser().resolve()

    # 1. Locate the raw data ------------------------------------------------
    if args.source_dir:
        snapshot = Path(args.source_dir).expanduser().resolve()
    else:
        logger.info("Downloading DeepEyesV2_SFT (this is a one-time ~12 GB cost)…")
        snapshot = download_deepeyesv2(
            cache_dir=args.cache_dir,
            only_parts=args.only_parts,
            only_json=args.no_images,
        )
    if args.no_images and not args.inspect:
        raise SystemExit("--no_images is only valid with --inspect")
    if not args.no_images:
        images_root = ensure_images_extracted(snapshot)
    else:
        images_root = snapshot / "images"  # may not exist; inspect-only
    json_dir = snapshot / "json"
    json_paths: list[Path]
    if args.only_parts is not None:
        json_paths = [json_dir / f"sft_part_{i}.json" for i in args.only_parts]
    else:
        json_paths = sorted(json_dir.glob("sft_part_*.json"))
    logger.info("Reading %d JSON file(s) from %s", len(json_paths), json_dir)

    # 2. Parse + bucket — always use ALL available trajectories ------------
    pools = _gather_raw_trajectories(json_paths, args.max_samples_per_file)
    logger.info(
        "Parsed pools: zoom=%d, code=%d, search=%d, image_search=%d, "
        "mixed_zoom=%d, mixed_non_zoom=%d",
        len(pools.zoom), len(pools.code), len(pools.search),
        len(pools.image_search), len(pools.mixed_zoom),
        len(pools.mixed_non_zoom),
    )

    if args.inspect:
        _inspect(pools)
        return

    # 3. Pre-split each pool deterministically. Doing the split BEFORE the
    # per-target loop guarantees that the same zoom trajectories land in
    # ``zoom_only/train`` and ``diverse/train`` (and analogously for test).
    # Without this, a zoom trajectory could land in zoom_only/train but
    # diverse/test, breaking the fair-comparison invariant.
    pool_splits = {
        "zoom":           _stratified_split(list(pools.zoom),           args.test_frac, rng),
        "code":           _stratified_split(list(pools.code),           args.test_frac, rng),
        "search":         _stratified_split(list(pools.search),         args.test_frac, rng),
        "image_search":   _stratified_split(list(pools.image_search),   args.test_frac, rng),
        "mixed_zoom":     _stratified_split(list(pools.mixed_zoom),     args.test_frac, rng),
        "mixed_non_zoom": _stratified_split(list(pools.mixed_non_zoom), args.test_frac, rng),
    }

    # 4. Convert + write ----------------------------------------------------
    targets = [args.target] if args.target != "both" else ["zoom_only", "diverse"]
    for tgt in targets:
        train_rows, test_rows = _build_rows_with_splits(tgt, pool_splits, images_root)
        rng.shuffle(train_rows)
        rng.shuffle(test_rows)
        _write_parquet(train_rows, output_dir / tgt / "train.parquet")
        _write_parquet(test_rows, output_dir / tgt / "test.parquet")


def _build_rows_with_splits(
    target: str,
    pool_splits: dict[str, tuple[list[RawTrajectory], list[RawTrajectory]]],
    images_root: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Convert each (train, test) split into output rows for the given target.

    Pure-zoom always feeds BOTH datasets so that the same trajectory id
    lands in zoom_only/train ↔ diverse/train (and analogously for test).
    Non-zoom + mixed-zoom pools are diverse-only.
    """

    train_rows: list[dict[str, Any]] = []
    test_rows: list[dict[str, Any]] = []

    def _materialise(trajs: list[RawTrajectory]) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for traj in trajs:
            row = convert_trajectory(traj, images_root, target)
            if row is not None:
                out.append(row)
        return out

    zoom_train, zoom_test = pool_splits["zoom"]
    train_rows.extend(_materialise(zoom_train))
    test_rows.extend(_materialise(zoom_test))

    if target == "diverse":
        for pool_name in ("code", "search", "image_search", "mixed_zoom", "mixed_non_zoom"):
            tr, te = pool_splits[pool_name]
            train_rows.extend(_materialise(tr))
            test_rows.extend(_materialise(te))
    return train_rows, test_rows


def _inspect(pools: TrajectoryPools) -> None:
    print("=== Inspect summary ===")
    print(f"  pure-zoom      trajectories: {len(pools.zoom)}")
    print(f"  code           trajectories: {len(pools.code)}")
    print(f"  search         trajectories: {len(pools.search)}")
    print(f"  image_search   trajectories: {len(pools.image_search)}")
    print(f"  mixed_zoom     trajectories: {len(pools.mixed_zoom)}")
    print(f"  mixed_non_zoom trajectories: {len(pools.mixed_non_zoom)}")
    print()
    if pools.zoom:
        print("First pure-zoom trajectory:")
        traj = pools.zoom[0]
        print("  question:", traj.question[:200])
        for i, turn in enumerate(traj.turns):
            print(f"  turn {i}: text={turn.assistant.text_content[:120]!r}, "
                  f"calls={turn.assistant.call_categories}, "
                  f"resp={turn.response_text[:80]!r}, "
                  f"resp_imgs={turn.response_image_indices}")
    if pools.mixed_zoom:
        print("\nFirst mixed-zoom trajectory:")
        traj = pools.mixed_zoom[0]
        print("  question:", traj.question[:200])
        print("  categories:", sorted(traj.categories))
        for i, turn in enumerate(traj.turns):
            print(f"  turn {i}: text={turn.assistant.text_content[:120]!r}, "
                  f"calls={turn.assistant.call_categories}, "
                  f"resp_imgs={turn.response_image_indices}")


if __name__ == "__main__":
    sys.exit(main())

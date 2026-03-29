#!/usr/bin/env python3
"""Read gt_export run_* manifest → Qwen3-VL-8B 4-bit → instructions.jsonl (outputs only under run_*).

Production (PBS): `cd project && uv run python scripts/generate_cube_gt_instructions.py --run-dir ...`
Requires CUDA; do not set VLAW_MOCK_REWARD.

Smoke: `uv run python scripts/generate_cube_gt_instructions.py --smoke`
  (may use VLAW_MOCK_REWARD=1 for placeholder text.)
"""
from __future__ import annotations

import importlib

# Torch.distributed's JIT template calls importlib.invalidate_caches(); some HPC Python 3.11 builds
# raise TypeError inside MetadataPathFinder — swallow so transformers/torchvision can import.
_real_invalidate = importlib.invalidate_caches


def _safe_invalidate_caches() -> None:
    try:
        _real_invalidate()
    except TypeError:
        pass


importlib.invalidate_caches = _safe_invalidate_caches  # type: ignore[assignment]

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _mock_env() -> bool:
    return os.environ.get("VLAW_MOCK_REWARD", "").strip() in ("1", "true", "True", "yes")


def _default_run_dir() -> Path | None:
    p = os.environ.get("GT_EXPORT_RUN_DIR", "").strip()
    if p:
        return Path(p).resolve()
    last = _project_root() / "cube_dataset" / "gt_export" / "_last_run_dir.txt"
    if last.is_file():
        line = last.read_text(encoding="utf-8").strip()
        if line:
            return Path(line).resolve()
    return None


def _find_manifest_dirs(run_dir: Path) -> list[tuple[Path, str]]:
    out: list[tuple[Path, str]] = []
    for p in sorted(run_dir.iterdir()):
        if p.is_dir() and (p / "manifest.jsonl").is_file():
            out.append((p, p.name))
    return out


def _load_video_pil_frames(path: Path, max_frames: int) -> list:
    import cv2
    import numpy as np
    from PIL import Image

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video {path}")
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    if n <= 0:
        cap.release()
        raise RuntimeError(f"no frames in {path}")
    if n <= max_frames:
        idx = list(range(n))
    else:
        idx = [int(round(float(x))) for x in np.linspace(0, n - 1, max_frames)]
    frames: list = []
    for i in idx:
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ok, bgr = cap.read()
        if not ok or bgr is None:
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        frames.append(Image.fromarray(rgb))
    cap.release()
    if not frames:
        raise RuntimeError(f"read zero frames from {path}")
    return frames


def _load_qwen3_vl_4bit(checkpoint_dir: str):
    import torch
    from transformers import BitsAndBytesConfig

    nf4 = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )
    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(checkpoint_dir, trust_remote_code=True)
    load_kwargs = {
        "quantization_config": nf4,
        "device_map": "auto",
        "torch_dtype": torch.float16,
        "trust_remote_code": True,
    }
    import transformers

    model = None
    last_exc = None
    for class_name in (
        "Qwen3VLForConditionalGeneration",
        "AutoModelForImageTextToText",
        "AutoModelForVision2Seq",
        "Qwen2_5_VLForConditionalGeneration",
        "Qwen2VLForConditionalGeneration",
    ):
        try:
            cls = getattr(transformers, class_name, None)
            if cls is None:
                continue
            model = cls.from_pretrained(checkpoint_dir, **load_kwargs)
            print(f"Loaded model via {class_name}")
            break
        except Exception as exc:
            last_exc = exc
            print(f"{class_name} load failed: {exc}")
    if model is None:
        raise RuntimeError(f"Could not load multimodal model. Last error: {last_exc}")
    return model, processor


INSTRUCTION_PROMPT = (
    "You are describing a short robotic manipulation clip (robot arm and cube). "
    "Write ONE short imperative sentence (at most 25 words) that could instruct a robot "
    "to accomplish what happens in the video. Focus on the cube and the gripper. "
    "Output plain text only, no quotes or markdown."
)


def _generate_one(
    model,
    processor,
    pil_frames: list,
    *,
    max_new_tokens: int = 128,
) -> str:
    import torch

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "video", "video": pil_frames},
                {"type": "text", "text": INSTRUCTION_PROMPT},
            ],
        }
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    # Do not import qwen_vl_utils/torchvision here: on some HPC Python stacks that pulls torch.distributed
    # and hits importlib.MetadataPathFinder bugs. Qwen3-VL processor accepts videos=[PIL,...] directly.
    inputs = processor(text=[text], videos=[pil_frames], padding=True, return_tensors="pt")

    dev = next(model.parameters()).device
    inputs = inputs.to(dev)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    # Decode only new tokens
    in_len = inputs["input_ids"].shape[1]
    gen_ids = out[0, in_len:]
    raw = processor.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
    return raw


def _merge_run_meta(run_dir: Path, patch: dict) -> None:
    p = run_dir / "run_meta.json"
    if not p.is_file():
        base = {}
    else:
        base = json.loads(p.read_text(encoding="utf-8"))
    base.update(patch)
    p.write_text(json.dumps(base, indent=2) + "\n", encoding="utf-8")


def run_smoke() -> None:
    import torch

    print("SMOKE: imports ok")
    rd = _default_run_dir()
    print(f"SMOKE: default run dir hint: {rd}")
    if _mock_env() or not torch.cuda.is_available():
        print("SMOKE: mock or no CUDA — placeholder instruction path only")
    else:
        print("SMOKE: CUDA available; full load skipped in smoke")
    print("SMOKE: PASS")


def main_instructions(
    run_dir: Path,
    *,
    checkpoint_dir: str = "Qwen/Qwen3-VL-8B-Instruct",
    max_frames: int = 32,
    smoke: bool = False,
) -> None:
    import torch

    run_dir = run_dir.resolve()
    if not run_dir.is_dir():
        print(f"FATAL: run dir missing: {run_dir}", file=sys.stderr)
        sys.exit(2)

    manifest_dirs = _find_manifest_dirs(run_dir)
    if not manifest_dirs:
        print(f"FATAL: no */manifest.jsonl under {run_dir}", file=sys.stderr)
        sys.exit(2)

    use_mock = _mock_env() or not torch.cuda.is_available()
    if use_mock and not smoke:
        print(
            "FATAL: production mode requires CUDA and VLAW_MOCK_REWARD unset/false.",
            file=sys.stderr,
        )
        sys.exit(3)

    cuda_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None

    vlm_meta = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "model_id": checkpoint_dir,
        "prompt": INSTRUCTION_PROMPT,
        "max_sample_frames": max_frames,
        "vlm_mock": bool(use_mock),
        "cuda_device_name": cuda_name,
    }
    (run_dir / "vlm_meta.json").write_text(json.dumps(vlm_meta, indent=2) + "\n", encoding="utf-8")

    model = None
    processor = None
    if not use_mock:
        model, processor = _load_qwen3_vl_4bit(checkpoint_dir)

    for fam_dir, family in manifest_dirs:
        man_path = fam_dir / "manifest.jsonl"
        out_path = fam_dir / "instructions.jsonl"
        lines_in = [json.loads(ln) for ln in man_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
        out_f = out_path.open("w", encoding="utf-8")
        for row in lines_in:
            vpath = Path(row["video_path"])
            if not vpath.is_file():
                print(f"WARN: missing video {vpath}, skipping clip {row.get('clip_id')}")
                continue
            if use_mock:
                instr = "[mock] Grasp the cube and move it toward the goal."
            else:
                assert model is not None and processor is not None
                pil_frames = _load_video_pil_frames(vpath, max_frames)
                instr = _generate_one(model, processor, pil_frames)
            out_row = {**row, "instruction": instr, "vlm_model": checkpoint_dir}
            out_f.write(json.dumps(out_row) + "\n")
        out_f.close()
        print(f"Wrote {out_path} ({family})")

    if model is not None:
        try:
            model.to("cpu")
        except Exception:
            pass
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    _merge_run_meta(
        run_dir,
        {
            "vlm_completed_utc": datetime.now(timezone.utc).isoformat(),
            "vlm_model": checkpoint_dir,
            "vlm_mock": bool(use_mock),
            "cuda_device_name": cuda_name,
        },
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--smoke", action="store_true", help="Import / env sanity only")
    ap.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help="gt_export run_* directory (default: GT_EXPORT_RUN_DIR or _last_run_dir.txt)",
    )
    ap.add_argument(
        "--checkpoint",
        default="Qwen/Qwen3-VL-8B-Instruct",
        help="HF model id for Qwen3-VL 4-bit",
    )
    ap.add_argument("--max-frames", type=int, default=32, help="Max frames sampled per clip")
    args = ap.parse_args()

    if args.smoke:
        run_smoke()
        return

    run_dir = args.run_dir or _default_run_dir()
    if run_dir is None:
        print("FATAL: pass --run-dir or set GT_EXPORT_RUN_DIR / run gt_export first.", file=sys.stderr)
        sys.exit(2)

    main_instructions(
        run_dir,
        checkpoint_dir=args.checkpoint,
        max_frames=args.max_frames,
        smoke=False,
    )


if __name__ == "__main__":
    main()

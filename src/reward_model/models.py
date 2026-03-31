"""Qwen-VL reward model (4-bit) with explicit VRAM release."""
from __future__ import annotations

import os

import torch
import torch.nn as nn

try:
    from qwen_vl_utils import process_vision_info
except ImportError:
    process_vision_info = None


def _mock_enabled() -> bool:
    return os.environ.get("VLAW_MOCK_REWARD", "").strip() in ("1", "true", "True", "yes")


class QwenRewardModel(nn.Module):
    def __init__(self, checkpoint_dir="Qwen/Qwen2-VL-7B-Instruct"):
        super().__init__()
        self.checkpoint_dir = checkpoint_dir
        self._model = None
        self._processor = None
        self.mock = _mock_enabled() or not torch.cuda.is_available()
        if self.mock:
            print(f"QwenRewardModel: mock mode (VLAW_MOCK_REWARD or no CUDA). ckpt={checkpoint_dir!r}")

    def _ensure_loaded(self) -> None:
        if self.mock or self._model is not None:
            return
        from transformers import AutoProcessor, BitsAndBytesConfig

        print(f"QwenRewardModel: loading {self.checkpoint_dir!r} (4-bit)...")
        nf4 = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )
        self._processor = AutoProcessor.from_pretrained(self.checkpoint_dir, trust_remote_code=True)
        load_kwargs = {
            "quantization_config": nf4,
            "device_map": "auto",
            "torch_dtype": torch.float16,
            "trust_remote_code": True,
        }

        # Try dedicated Qwen-VL classes first, then generic multimodal auto classes.
        model = None
        last_exc = None
        class_order = [
            "Qwen3VLForConditionalGeneration",
            "AutoModelForImageTextToText",
            "AutoModelForVision2Seq",
            "Qwen2_5_VLForConditionalGeneration",
            "Qwen2VLForConditionalGeneration",
        ]
        for class_name in class_order:
            try:
                import transformers

                cls = getattr(transformers, class_name, None)
                if cls is None:
                    continue
                model = cls.from_pretrained(self.checkpoint_dir, **load_kwargs)
                print(f"QwenRewardModel: loaded via {class_name}")
                break
            except Exception as exc:
                last_exc = exc
                print(f"QwenRewardModel: {class_name} load failed: {exc}")
        if model is None:
            raise RuntimeError(
                f"Could not load multimodal model for {self.checkpoint_dir}. "
                f"Last error: {last_exc}"
            )
        self._model = model

    def to_cpu_and_clear(self) -> None:
        if self._model is not None:
            try:
                self._model.to("cpu")
            except Exception:
                pass
            del self._model
            self._model = None
        self._processor = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def score_trajectory(self, video, instruction: str) -> torch.Tensor:
        """
        video: [B, C, T, H, W] or [C, T, H, W] float in [0,1] or uint8-like.
        Returns [B, 1] success probability (p yes).
        """
        if self.mock:
            b = 1
            if isinstance(video, torch.Tensor):
                if video.dim() == 5:
                    b = video.shape[0]
            return torch.full((b, 1), 0.85)

        self._ensure_loaded()
        assert self._model is not None and self._processor is not None

        if isinstance(video, torch.Tensor):
            if video.dim() == 4:
                video = video.unsqueeze(0)
            b = video.shape[0]
        else:
            b = 1

        probs = []
        for bi in range(b):
            vid = video[bi] if isinstance(video, torch.Tensor) and video.dim() == 5 else video
            p = self._score_one_clip(vid, instruction)
            probs.append(p)
        return torch.tensor(probs, dtype=torch.float32).view(-1, 1)

    def _score_one_clip(self, video_cthw: torch.Tensor, instruction: str) -> float:
        import numpy as np
        from PIL import Image

        if video_cthw.dim() != 4:
            raise ValueError("expected [C,T,H,W]")
        _, t, _, _ = video_cthw.shape
        max_frames = 8
        if t > max_frames:
            idx = [int(i) for i in torch.linspace(0, t - 1, max_frames).tolist()]
        else:
            idx = list(range(t))
        frames = []
        for i in idx:
            fr = video_cthw[:, i].detach().float().clamp(0, 1).cpu().numpy()
            fr = (np.transpose(fr, (1, 2, 0)) * 255.0).clip(0, 255).astype(np.uint8)
            frames.append(Image.fromarray(fr))

        prompt = (
            f"Did the robot successfully perform this task: {instruction}? "
            "Answer with exactly one word: yes or no."
        )
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "video", "video": frames},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        processor = self._processor
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        proc_kwargs: dict = {"text": [text], "padding": True, "return_tensors": "pt"}
        if process_vision_info is not None:
            try:
                image_inputs, video_inputs = process_vision_info(messages)
                if image_inputs is not None:
                    proc_kwargs["images"] = image_inputs
                if video_inputs is not None:
                    proc_kwargs["videos"] = video_inputs
                inputs = processor(**proc_kwargs)
            except Exception as exc:
                # Qwen2-era processors sometimes want raw PIL lists; Qwen3 path is above.
                print(f"QwenRewardModel: process_vision_info processor path failed ({exc!r}); trying videos=[frames].")
                inputs = processor(text=[text], videos=[frames], padding=True, return_tensors="pt")
        else:
            inputs = processor(text=[text], videos=[frames], padding=True, return_tensors="pt")

        dev = next(self._model.parameters()).device
        inputs = inputs.to(dev)

        tok = processor.tokenizer
        yes_id = tok.encode("yes", add_special_tokens=False)
        no_id = tok.encode("no", add_special_tokens=False)
        yes_id = yes_id[0] if yes_id else tok.unk_token_id
        no_id = no_id[0] if no_id else tok.unk_token_id

        with torch.no_grad():
            out = self._model.generate(
                **inputs,
                max_new_tokens=1,
                do_sample=False,
                output_scores=True,
                return_dict_in_generate=True,
            )
            logits = out.scores[0][0]
            pair = torch.stack([logits[yes_id], logits[no_id]]).float()
            p_yes = torch.softmax(pair, dim=0)[0].item()
        return float(p_yes)

    def score(self, video_frames, instruction):
        return self.score_trajectory(video_frames, instruction)

    def forward(self, video_frames, instructions):
        logits = torch.randn(video_frames.shape[0], 1, requires_grad=True)
        return logits


if __name__ == "__main__":
    os.environ["VLAW_MOCK_REWARD"] = "1"
    rm = QwenRewardModel()
    v = torch.rand(1, 3, 4, 64, 64)
    print("p(yes)", rm.score_trajectory(v, "stack blocks").item())

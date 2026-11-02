"""Upstream PickScore formula, with local files and bounded scoring batches."""

from pathlib import Path
import torch
from transformers import AutoModel, AutoProcessor


class PickScoreScorer(torch.nn.Module):
    def __init__(self, device="cuda", dtype=torch.float32, model_path=None,
                 processor_path=None, local_files_only=True, batch_size=1):
        super().__init__()
        if batch_size < 1:
            raise ValueError("PickScore batch_size must be positive")
        for label, path in (("model_path", model_path), ("processor_path", processor_path)):
            if path is not None and not Path(path).expanduser().is_dir():
                raise FileNotFoundError(f"PickScore {label} directory not found: {path}")
        model_source = str(Path(model_path).expanduser().resolve()) if model_path is not None else "yuvalkirstain/PickScore_v1"
        processor_source = str(Path(processor_path).expanduser().resolve()) if processor_path is not None else "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
        self.device, self.dtype, self.batch_size = device, dtype, batch_size
        self.processor = AutoProcessor.from_pretrained(processor_source, local_files_only=local_files_only)
        self.model = AutoModel.from_pretrained(model_source, local_files_only=local_files_only).eval().to(device=device, dtype=dtype)
        self.model.requires_grad_(False)

    @torch.no_grad()
    def __call__(self, prompt, images):
        if isinstance(prompt, str):
            prompt = [prompt] * len(images)
        if len(prompt) != len(images):
            raise ValueError("PickScore requires one prompt for each image")
        if not len(images):
            return torch.empty(0, device=self.device, dtype=self.dtype)
        all_scores = []
        for start in range(0, len(images), self.batch_size):
            stop = start + self.batch_size
            image_inputs = self.processor(images=images[start:stop], padding=True, truncation=True, max_length=77, return_tensors="pt")
            image_inputs = {k: v.to(device=self.device) for k, v in image_inputs.items()}
            text_inputs = self.processor(text=prompt[start:stop], padding=True, truncation=True, max_length=77, return_tensors="pt")
            text_inputs = {k: v.to(device=self.device) for k, v in text_inputs.items()}
            image_embs = self.model.get_image_features(**image_inputs)
            image_embs = image_embs / image_embs.norm(p=2, dim=-1, keepdim=True)
            text_embs = self.model.get_text_features(**text_inputs)
            text_embs = text_embs / text_embs.norm(p=2, dim=-1, keepdim=True)
            logit_scale = self.model.logit_scale.exp()
            scores = logit_scale * (text_embs @ image_embs.T)
            # Same upstream divisor; this is not a probability or softmax.
            all_scores.append(scores.diag() / 26)
        return torch.cat(all_scores, dim=0)

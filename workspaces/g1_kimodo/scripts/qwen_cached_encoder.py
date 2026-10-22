"""Use actual cached projected embeddings, without keeping an 8B LLM loaded."""

import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from kimodo.sanitize import sanitize_text

ROOT = Path(__file__).resolve().parents[1]


class CachedQwenEncoder:
    def __init__(self):
        manifest = json.loads((ROOT / "reports/qwen_text_encoder_smoke.json").read_text())
        if not manifest["passed"]:
            raise RuntimeError("Qwen embedding verification has not passed")
        path = Path(manifest["embedding_file"])
        if hashlib.sha256(path.read_bytes()).hexdigest() != manifest["embedding_sha256"]:
            raise ValueError("Cached embedding checksum mismatch")
        with np.load(path, allow_pickle=False) as cached:
            self.embeddings = {str(text): torch.from_numpy(embedding.copy()) for text, embedding in zip(cached["texts"], cached["embeddings"])}

    def __call__(self, texts):
        scalar = isinstance(texts, str)
        if scalar:
            texts = [texts]
        vectors = []
        for text in texts:
            key = sanitize_text(text)
            if key not in self.embeddings:
                raise KeyError(f"Prompt is not present in the verified Qwen cache: {key}")
            vectors.append(self.embeddings[key])
        result = torch.stack(vectors)
        return (result[0], 1) if scalar else (result, [1] * len(texts))

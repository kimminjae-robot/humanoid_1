"""Serve only genuine, checksum-verified official LLM2Vec prompt embeddings."""

import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from kimodo.sanitize import sanitize_text

ROOT = Path(__file__).resolve().parents[1]


class CachedOfficialEncoder:
    def __init__(self):
        report = json.loads((ROOT / "reports/text_encoder_smoke.json").read_text())
        if not report["passed"] or report["encoder"] != "official_llm2vec_llama3":
            raise RuntimeError("Official encoder verification has not passed")
        manifest = ROOT / report["checkpoint_manifest"]
        if hashlib.sha256(manifest.read_bytes()).hexdigest() != report["checkpoint_manifest_sha256"]:
            raise ValueError("Official checkpoint manifest changed; re-encode prompts")
        path = Path(report["embedding_file"])
        if hashlib.sha256(path.read_bytes()).hexdigest() != report["embedding_sha256"]:
            raise ValueError("Official embedding checksum mismatch")
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
                raise KeyError(f"Prompt is not present in verified official cache: {key}")
            vectors.append(self.embeddings[key])
        result = torch.stack(vectors)
        return (result[0], 1) if scalar else (result, [1] * len(texts))

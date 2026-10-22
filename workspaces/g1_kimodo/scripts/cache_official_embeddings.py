"""Measure actual official CPU LLM2Vec encoding and cache verified prompts."""

import hashlib
import json
import resource
import sys
import time
from pathlib import Path

import numpy as np
import torch
from kimodo.sanitize import sanitize_text

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from official_encoder import PinnedLLM2VecEncoder


def main():
    torch.set_num_threads(8)
    started = time.monotonic()
    print("Loading pinned official Llama + MNTP + supervised adapters on CPU", flush=True)
    encoder = PinnedLLM2VecEncoder()
    load_s = time.monotonic() - started
    actual_device = str(encoder.get_device())
    if actual_device != "cpu":
        raise ValueError(f"Expected CPU text encoder, got {actual_device}")
    # Check that the installed Transformers path really preserves bidirectionality.
    # A causal model can load and produce finite embeddings while changing conditioning.
    with torch.inference_mode():
        tokens = torch.tensor([[128000, 1000, 2000, 3000]])
        mask = torch.ones_like(tokens)
        before = encoder.model.model(input_ids=tokens, attention_mask=mask, use_cache=False).last_hidden_state
        tokens[0, -1] = 3001
        after = encoder.model.model(input_ids=tokens, attention_mask=mask, use_cache=False).last_hidden_state
        future_dependency = float((before[0, 0] - after[0, 0]).abs().max())
    if not np.isfinite(future_dependency) or future_dependency <= 0:
        raise ValueError("Loaded LLM2Vec model did not show bidirectional token dependence")
    prompts = json.loads((ROOT / "configs/prompts.json").read_text())["prompts"]
    texts, tensors, timings = [], [], {}
    for prompt in prompts:
        text = sanitize_text(prompt["text"])
        before = time.monotonic()
        embedding, lengths = encoder([text])
        embedding = embedding.detach().cpu().float().numpy()
        if embedding.shape != (1, 1, 4096) or lengths != [1] or not np.isfinite(embedding).all():
            raise ValueError(f"Invalid official embedding: {prompt['id']}")
        texts.append(text)
        tensors.append(embedding[0])
        timings[prompt["id"]] = time.monotonic() - before
        print(f"Encoded {prompt['id']}: {timings[prompt['id']]:.2f}s; norm={np.linalg.norm(embedding):.4f}", flush=True)
    embeddings = np.stack(tensors)
    norms = np.linalg.norm(embeddings[:, 0], axis=1, keepdims=True)
    if not (norms > 0).all():
        raise ValueError("Zero official embedding")
    output = ROOT / "cache/text_embeddings/official_llm2vec.npz"
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, texts=np.array(texts), embeddings=embeddings)
    differences = [float(np.max(np.abs(embeddings[i] - embeddings[j]))) for i in range(len(texts)) for j in range(i)]
    report = {
        "scope": "official_cpu_text_encoder_loading_and_real_embeddings_not_motion_quality",
        "encoder": "official_llm2vec_llama3", "device": actual_device, "dtype": "bfloat16",
        "parameter_count": sum(parameter.numel() for parameter in encoder.model.parameters()),
        "parameter_dtypes": sorted({str(parameter.dtype) for parameter in encoder.model.parameters()}),
        "cpu_threads": torch.get_num_threads(), "internal_encoding_batch_size": 1,
        "attention_implementation": encoder.model.model.config._attn_implementation,
        "first_token_future_dependency_max_abs": future_dependency,
        "tokenization_model_name": encoder.model.model.config._name_or_path,
        "text_encoder_loaded": True, "real_text_embeddings": True,
        "checkpoint_manifest": "reports/official_encoder_checkpoint.json",
        "checkpoint_manifest_sha256": hashlib.sha256((ROOT / "reports/official_encoder_checkpoint.json").read_bytes()).hexdigest(),
        "implementation_sha256": hashlib.sha256((ROOT / "scripts/official_encoder.py").read_bytes()).hexdigest(),
        "texts": texts, "motion_ids": [prompt["id"] for prompt in prompts], "embedding_shape": list(embeddings.shape),
        "embedding_file": str(output), "embedding_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "pairwise_cosine_similarity": ((embeddings[:, 0] / norms) @ (embeddings[:, 0] / norms).T).tolist(),
        "minimum_pairwise_max_abs_difference": min(differences), "load_time_s": load_s, "encoding_time_s": timings,
        "peak_process_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
        "elapsed_s": time.monotonic() - started,
        "passed": bool(min(differences) > 0 and np.isfinite(embeddings).all()),
    }
    (ROOT / "reports/text_encoder_smoke.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: report[key] for key in ["passed", "embedding_shape", "load_time_s", "peak_process_rss_mib", "elapsed_s"]}, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

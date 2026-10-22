"""Encode the project prompts with a pinned experimental Qwen + projection."""

import argparse
import hashlib
import importlib.util
import json
import os
import resource
import time
from pathlib import Path

import numpy as np
import torch
from kimodo.sanitize import sanitize_text

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    args = parser.parse_args()
    os.environ["TEXT_ENCODER_DEVICE"] = args.device
    os.environ.pop("PROJECTION_PATH", None)
    os.environ.pop("TEXT_ENCODERS_DIR", None)
    torch.set_num_threads(8)
    source = ROOT / "third_party/qwen_encoder/custom_text_encoder.py"
    config = json.loads((ROOT / "configs/alternative_encoder.json").read_text())
    if hashlib.sha256(source.read_bytes()).hexdigest() != config["implementation_sha256"]:
        raise ValueError("Experimental encoder implementation checksum mismatch")
    checkpoint = json.loads((ROOT / "reports/qwen_checkpoint.json").read_text())
    specification = importlib.util.spec_from_file_location("pinned_qwen_encoder", source)
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    started = time.monotonic()
    print(f"Loading Qwen3-8B on {args.device}; official Kimodo weights remain unchanged", flush=True)
    encoder = module.CustomTextEncoder(model_name_or_path=checkpoint["encoder_snapshot"], dtype="bfloat16", llm_dim=4096,
        device=args.device, projection_path=checkpoint["projection_file"])
    config = json.loads((ROOT / "configs/prompts.json").read_text())
    prompts = config["prompts"]
    tensors, timings, texts = [], {}, []
    for prompt in prompts:
        text = sanitize_text(prompt["text"])
        before = time.monotonic()
        embedding, lengths = encoder([text])
        embedding = embedding.detach().cpu().float().numpy()
        if embedding.shape != (1, 1, 4096) or lengths != [1] or not np.isfinite(embedding).all():
            raise ValueError(f"Invalid embedding for {prompt['id']}")
        tensors.append(embedding[0])
        texts.append(text)
        timings[prompt["id"]] = time.monotonic() - before
        print(f"Encoded {prompt['id']}: {timings[prompt['id']]:.2f}s; norm={np.linalg.norm(embedding):.4f}", flush=True)
    embeddings = np.stack(tensors)
    output = ROOT / "cache/text_embeddings/qwen3_8b_projected.npz"
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, embeddings=embeddings, texts=np.array(texts))
    normalized = embeddings[:, 0] / np.linalg.norm(embeddings[:, 0], axis=1, keepdims=True)
    report = {
        "scope": "experimental_real_qwen_encoder_and_pretrained_projection_connectivity_not_Llama_quality_equivalence",
        "experimental": True, "device": args.device, "text_encoder_loaded": True, "real_text_embeddings": True,
        "checkpoint": checkpoint, "implementation_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "projection_sha256": hashlib.sha256(Path(checkpoint["projection_file"]).read_bytes()).hexdigest(),
        "texts": texts, "motion_ids": [p["id"] for p in prompts], "embedding_shape": list(embeddings.shape),
        "embedding_file": str(output), "embedding_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "pairwise_cosine_similarity": (normalized @ normalized.T).tolist(), "encoding_time_s": timings,
        "peak_process_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
        "elapsed_s": time.monotonic() - started, "official_encoder_comparison": "pending_llama_access",
        "passed": bool(np.isfinite(embeddings).all() and np.linalg.norm(embeddings, axis=-1).min() > 0),
    }
    (ROOT / "reports/qwen_text_encoder_smoke.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in ["passed", "device", "embedding_shape", "peak_process_rss_mib", "elapsed_s"]}, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

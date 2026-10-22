"""Load the real G1 denoiser on CUDA; keep unavailable text encoding explicit."""

import hashlib
import json
import os
import resource
import time
from pathlib import Path

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]


class UnavailableTextEncoder(nn.Module):
    def forward(self, *args, **kwargs):
        raise RuntimeError("Gated text encoder unavailable: this smoke only loads/tests the denoiser")


def main():
    from kimodo import load_model

    start = time.monotonic()
    os.environ["CHECKPOINT_DIR"] = str(ROOT / "checkpoints")
    torch.cuda.reset_peak_memory_stats()
    model = load_model("Kimodo-G1-RP-v1", device="cuda", text_encoder=UnavailableTextEncoder())
    feature_count = model.motion_rep.motion_rep_dim
    x = torch.randn((1, 30, feature_count), device="cuda")
    # Synthetic conditioning checks numerical execution only, never text-to-motion quality.
    with torch.inference_mode():
        output = model.denoiser.model(
            x=x, x_pad_mask=torch.ones((1, 30), dtype=torch.bool, device="cuda"),
            text_feat=torch.zeros((1, 1, 4096), device="cuda"),
            text_feat_pad_mask=torch.ones((1, 1), dtype=torch.bool, device="cuda"),
            timesteps=torch.tensor([500], device="cuda"), first_heading_angle=torch.zeros(1, device="cuda"))
    torch.cuda.synchronize()
    checkpoint = ROOT / "checkpoints/Kimodo-G1-RP-v1/model.safetensors"
    checksum = hashlib.sha256()
    with checkpoint.open("rb") as stream:
        while block := stream.read(4 * 1024 * 1024):
            checksum.update(block)
    report = {
        "scope": "real_checkpoint_cuda_denoiser_load_and_synthetic_forward_only",
        "text_encoder_loaded": False, "text_to_motion_verified": False,
        "torch": torch.__version__, "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(),
        "parameter_count": sum(p.numel() for p in model.parameters()), "feature_count": feature_count,
        "output_shape": list(output.shape), "finite_output": bool(torch.isfinite(output).all()),
        "peak_cuda_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
        "peak_process_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
        "elapsed_s": time.monotonic() - start, "checkpoint_sha256": checksum.hexdigest(),
    }
    report["passed"] = report["finite_output"] and list(output.shape) == list(x.shape)
    (ROOT / "reports/kimodo_smoke.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

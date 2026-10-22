"""Generate a new, provenance-preserved Kimodo G1 motion for K1 retargeting."""
import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
G1 = ROOT.parent / 'g1_kimodo'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', type=Path, required=True)
    args = ap.parse_args()
    cfg = json.loads(args.config.read_text())
    label = cfg['label']
    if not label or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in label):
        raise ValueError('Invalid label')
    out = ROOT / 'data/generated' / label
    out.mkdir(parents=True, exist_ok=False)
    (out / 'config.json').write_text(json.dumps(cfg, ensure_ascii=False, indent=2) + '\n')
    os.environ.update(HF_HOME=str(G1 / 'cache/huggingface'),
                      CHECKPOINT_DIR=str(G1 / 'checkpoints'),
                      HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
    sys.path.insert(0, str(G1 / 'scripts'))
    import numpy as np
    import torch
    from official_encoder import PinnedLLM2VecEncoder
    from kimodo import load_model
    from kimodo.exports.motion_io import save_kimodo_npz
    from kimodo.exports.mujoco import MujocoQposConverter
    from kimodo.tools import seed_everything
    from kimodo.sanitize import sanitize_text

    start = time.monotonic()
    torch.set_num_threads(4)
    record = dict(config=cfg, sanitized_prompt=sanitize_text(cfg['prompt']),
                  status='loading', generator='Kimodo-G1-RP-v1',
                  encoder='pinned_official_LLM2Vec_Llama3_CPU',
                  source_revisions=json.loads((G1 / 'configs/project.json').read_text())['source_revisions'],
                  generator_weights_sha256=sha(G1 / 'checkpoints/Kimodo-G1-RP-v1/model.safetensors'),
                  script_sha256=sha(Path(__file__)), config_sha256=sha(out / 'config.json'))
    for name in ['official_encoder_checkpoint.json', 'kimodo_checkpoint.json']:
        src = G1 / 'reports' / name
        (out / name).write_bytes(src.read_bytes())
        record[name + '_sha256'] = sha(src)
    def save():
        (out / 'manifest.json').write_text(json.dumps(record, ensure_ascii=False, indent=2) + '\n')
    save()
    try:
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA is required for motion generation')
        print('Loading official CPU encoder and CUDA generator', flush=True)
        encoder = PinnedLLM2VecEncoder()
        model = load_model('Kimodo-G1-RP-v1', device='cuda', text_encoder=encoder)
        seed_everything(cfg['seed'])
        frames = round(cfg['duration_s'] * 30)
        with torch.inference_mode():
            motion = model(prompts=cfg['prompt'], num_frames=frames,
                           num_denoising_steps=cfg.get('diffusion_steps', 100),
                           num_samples=1, post_processing=True, return_numpy=False,
                           cfg_type='separated', cfg_weight=[2., 2.])
            q = MujocoQposConverter(model.skeleton).dict_to_qpos(motion, 'cuda')
        if isinstance(q, torch.Tensor):
            q = q.detach().cpu().numpy()
        q = np.asarray(q).reshape(-1, 36)
        single = {k: v[0].detach().cpu().numpy() if isinstance(v, torch.Tensor) else np.asarray(v)[0]
                  for k, v in motion.items()}
        save_kimodo_npz(str(out / 'motion.npz'), single)
        np.savetxt(out / 'qpos.csv', q, delimiter=',')
        checks = dict(shape=q.shape == (frames, 36), finite=bool(np.isfinite(q).all()),
                      normalized_quaternions=bool(np.max(abs(np.linalg.norm(q[:, 3:7], axis=1)-1)) < 1e-3))
        record.update(status='generated' if all(checks.values()) else 'format_failed', fps=30,
                      frames=frames, checks=checks, elapsed_s=time.monotonic()-start,
                      artifacts={name: dict(path=str((out / name).relative_to(ROOT)), sha256=sha(out / name))
                                 for name in ['motion.npz', 'qpos.csv']})
        save()
        print(json.dumps(record, ensure_ascii=False, indent=2), flush=True)
        return 0 if all(checks.values()) else 2
    except Exception as exc:
        record.update(status='failed', error=repr(exc), elapsed_s=time.monotonic()-start)
        save()
        raise


if __name__ == '__main__':
    raise SystemExit(main())

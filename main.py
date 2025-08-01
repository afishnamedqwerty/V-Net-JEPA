import argparse
import os
import json
import torch

# Phase trainers
from trainers.pretrain import pretrain as run_pretrain
from trainers.language_align import language_align as run_language_align
from trainers.probe_train import probe_train as run_probe_train
from trainers.action_posttrain import action_posttrain as run_action_posttrain

# Models for save/load helpers
from models.vjepa.vit import HNetViT
from models.vjepa.predictor import Predictor

def default_cfg():
    return {
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "work_dir": "outputs",
        "embed_dim": 256,
        "batch_size": 8,
        "lr": 1e-4,
        "weight_decay": 0.05,
        "epochs_pretrain": 200,
        "epochs_lang": 50,
        "epochs_probe": 20,
        "epochs_action": 50,
        "mask_ratio": 0.4,
        "lambda_vic": 1.0,
        "lambda_energy": 0.1,
        "pred_layers": 6,
        "pred_heads": 4,
        "fuser_heads": 4,
        "action_dim": 7,
        "cem_samples": 100
    }

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def save_checkpoint(state, path):
    ensure_dir(os.path.dirname(path))
    torch.save(state, path)

def load_checkpoint(path, map_location=None):
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    return torch.load(path, map_location=map_location)

def build_phase_dirs(root):
    dirs = {
        "pretrain": os.path.join(root, "pretrain"),
        "language": os.path.join(root, "language_align"),
        "probe": os.path.join(root, "probe"),
        "action": os.path.join(root, "action_posttrain"),
    }
    for d in dirs.values():
        ensure_dir(d)
    return dirs

def run_phase_pretrain(cfg, out_dir):
    # Trainers expect cfg-like object; provide a simple namespace
    class Cfg: pass
    c = Cfg()
    for k, v in cfg.items():
        setattr(c, k, v)
    c.epochs = cfg.get("epochs_pretrain", 200)

    # Dataset/loader wiring is expected to be provided in cfg externally. Here it's left to user/configs.
    run_pretrain(c)

    ckpt_path = os.path.join(out_dir, "pretrain.ckpt")
    # Save minimal artifacts if available
    # For simplicity, create a placeholder indicating phase completion
    save_checkpoint({"phase": "pretrain", "cfg": cfg}, ckpt_path)
    return ckpt_path

def run_phase_language_align(cfg, out_dir, init_ckpt=None):
    class Cfg: pass
    c = Cfg()
    for k, v in cfg.items():
        setattr(c, k, v)
    c.epochs = cfg.get("epochs_lang", 50)

    # Load backbone init if provided; trainers internally construct models
    if init_ckpt and os.path.isfile(init_ckpt):
        # No direct model handle here; trainer should load if needed. Placeholder persisted.
        pass

    run_language_align(c)

    ckpt_path = os.path.join(out_dir, "language_align.ckpt")
    save_checkpoint({"phase": "language_align", "cfg": cfg}, ckpt_path)
    return ckpt_path

def run_phase_probe(cfg, out_dir, init_ckpt=None):
    class Cfg: pass
    c = Cfg()
    for k, v in cfg.items():
        setattr(c, k, v)
    c.epochs = cfg.get("epochs_probe", 20)

    if init_ckpt and os.path.isfile(init_ckpt):
        pass

    run_probe_train(c)

    ckpt_path = os.path.join(out_dir, "probe.ckpt")
    save_checkpoint({"phase": "probe", "cfg": cfg}, ckpt_path)
    return ckpt_path

def run_phase_action_posttrain(cfg, out_dir, init_ckpt=None):
    class Cfg: pass
    c = Cfg()
    for k, v in cfg.items():
        setattr(c, k, v)
    c.epochs = cfg.get("epochs_action", 50)

    if init_ckpt and os.path.isfile(init_ckpt):
        pass

    run_action_posttrain(c)

    ckpt_path = os.path.join(out_dir, "action_posttrain.ckpt")
    save_checkpoint({"phase": "action_posttrain", "cfg": cfg}, ckpt_path)
    return ckpt_path

def chain_pipeline(cfg, run_sequence):
    work_dir = cfg.get("work_dir", "outputs")
    dirs = build_phase_dirs(work_dir)

    checkpoints = {}

    if "pretrain" in run_sequence:
        checkpoints["pretrain"] = run_phase_pretrain(cfg, dirs["pretrain"])

    if "language" in run_sequence:
        init = checkpoints.get("pretrain")
        checkpoints["language"] = run_phase_language_align(cfg, dirs["language"], init_ckpt=init)

    if "probe" in run_sequence:
        init = checkpoints.get("language", checkpoints.get("pretrain"))
        checkpoints["probe"] = run_phase_probe(cfg, dirs["probe"], init_ckpt=init)

    if "action" in run_sequence:
        init = checkpoints.get("probe", checkpoints.get("language", checkpoints.get("pretrain")))
        checkpoints["action"] = run_phase_action_posttrain(cfg, dirs["action"], init_ckpt=init)

    # Save pipeline manifest
    manifest = {
        "sequence": run_sequence,
        "checkpoints": checkpoints,
        "cfg": cfg
    }
    save_path = os.path.join(work_dir, "pipeline_manifest.json")
    with open(save_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Pipeline completed. Manifest written to {save_path}")
    return checkpoints

def parse_args():
    parser = argparse.ArgumentParser(description="H-Net V-JEPA training pipeline")
    parser.add_argument("--work_dir", type=str, default="outputs", help="Working directory for checkpoints/logs")
    parser.add_argument("--sequence", type=str, default="pretrain,language,probe,action",
                        help="Comma-separated phases: pretrain,language,probe,action")
    parser.add_argument("--epochs_pretrain", type=int, default=200)
    parser.add_argument("--epochs_lang", type=int, default=50)
    parser.add_argument("--epochs_probe", type=int, default=20)
    parser.add_argument("--epochs_action", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--mask_ratio", type=float, default=0.4)
    parser.add_argument("--lambda_vic", type=float, default=1.0)
    parser.add_argument("--lambda_energy", type=float, default=0.1)
    parser.add_argument("--embed_dim", type=int, default=256)
    parser.add_argument("--pred_layers", type=int, default=6)
    parser.add_argument("--pred_heads", type=int, default=4)
    parser.add_argument("--fuser_heads", type=int, default=4)
    parser.add_argument("--action_dim", type=int, default=7)
    parser.add_argument("--cem_samples", type=int, default=100)
    return parser.parse_args()

def main():
    args = parse_args()
    cfg = default_cfg()

    # Override defaults from args
    for k, v in vars(args).items():
        cfg[k] = v

    sequence = [s.strip() for s in cfg.get("sequence", "pretrain,language,probe,action").split(",") if s.strip()]
    chain_pipeline(cfg, sequence)

if __name__ == "__main__":
    main()

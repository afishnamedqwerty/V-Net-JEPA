import argparse
import os
import json
import torch
import yaml

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

    # Map unified YAML to trainer-expected flat attrs
    # model
    m = cfg.get("model", {})
    setattr(c, "embed_dim", m.get("embed_dim", cfg.get("embed_dim", 256)))
    pred = m.get("predictor", {})
    setattr(c, "pred_layers", pred.get("layers", cfg.get("pred_layers", 6)))
    setattr(c, "pred_heads", pred.get("heads", cfg.get("pred_heads", 4)))
    # data
    d = cfg.get("data", {})
    setattr(c, "batch_size", d.get("batch_size", cfg.get("batch_size", 8)))
    # losses
    vic = cfg.get("loss", {}).get("vicregl", {})
    setattr(c, "vic_inv", vic.get("inv", 25.0))
    setattr(c, "vic_var", vic.get("var", 25.0))
    setattr(c, "vic_cov", vic.get("cov", 1.0))
    # train
    t = cfg.get("train", {})
    setattr(c, "epochs", t.get("epochs", cfg.get("epochs_pretrain", 200)))
    setattr(c, "lr", t.get("lr", cfg.get("lr", 1e-4)))
    setattr(c, "weight_decay", t.get("weight_decay", cfg.get("weight_decay", 0.05)))
    setattr(c, "mask_ratio", t.get("mask_ratio", cfg.get("mask_ratio", 0.4)))
    setattr(c, "lambda_vic", t.get("lambda_vic", cfg.get("lambda_vic", 1.0)))
    setattr(c, "alpha", t.get("alpha", cfg.get("alpha", 0.0)))
    setattr(c, "gamma", t.get("gamma", cfg.get("gamma", 0.0)))
    setattr(c, "delta", t.get("delta", cfg.get("delta", 0.0)))
    # seed
    setattr(c, "seed", cfg.get("run", {}).get("seed", cfg.get("seed", 42)))

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

    # map unified YAML
    m = cfg.get("model", {})
    setattr(c, "embed_dim", m.get("embed_dim", cfg.get("embed_dim", 256)))
    setattr(c, "freeze_backbone", m.get("freeze_backbone", False))
    # data
    d = cfg.get("data", {})
    setattr(c, "batch_size", d.get("batch_size", cfg.get("batch_size", 8)))
    # loss
    contrast = cfg.get("loss", {}).get("contrastive", {})
    setattr(c, "temperature", contrast.get("temperature", 0.07))
    setattr(c, "lambda_energy", cfg.get("loss", {}).get("energy", {}).get("lambda", cfg.get("lambda_energy", 0.1)))
    # train
    t = cfg.get("train", {})
    setattr(c, "epochs", t.get("epochs", cfg.get("epochs_lang", 50)))
    setattr(c, "lr", t.get("lr", cfg.get("lr", 1e-4)))
    setattr(c, "weight_decay", t.get("weight_decay", cfg.get("weight_decay", 0.05)))

    # Load backbone init if provided; trainers internally construct models
    if init_ckpt and os.path.isfile(init_ckpt):
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

    # map unified YAML
    m = cfg.get("model", {})
    setattr(c, "embed_dim", m.get("embed_dim", cfg.get("embed_dim", 256)))
    setattr(c, "freeze_backbone", m.get("freeze_backbone", True))
    # data
    d = cfg.get("data", {})
    setattr(c, "batch_size", d.get("batch_size", cfg.get("batch_size", 8)))
    # loss
    setattr(c, "lambda_energy", cfg.get("loss", {}).get("energy", {}).get("lambda", cfg.get("lambda_energy", 0.1)))
    # train
    t = cfg.get("train", {})
    setattr(c, "epochs", t.get("epochs", cfg.get("epochs_probe", 20)))
    setattr(c, "lr", t.get("lr", cfg.get("lr", 1e-4)))
    setattr(c, "weight_decay", t.get("weight_decay", cfg.get("weight_decay", 0.05)))

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

    # map unified YAML
    m = cfg.get("model", {})
    setattr(c, "embed_dim", m.get("embed_dim", cfg.get("embed_dim", 256)))
    pred = m.get("predictor", {})
    setattr(c, "pred_layers", pred.get("layers", cfg.get("pred_layers", 6)))
    setattr(c, "pred_heads", pred.get("heads", cfg.get("pred_heads", 4)))
    fuser = m.get("fuser", {})
    setattr(c, "fuser_heads", fuser.get("heads", cfg.get("fuser_heads", 4)))
    setattr(c, "freeze_encoder", m.get("freeze_encoder", True))
    a = m.get("action", {})
    setattr(c, "action_dim", a.get("dim", cfg.get("action_dim", 7)))

    # data
    d = cfg.get("data", {})
    setattr(c, "batch_size", d.get("batch_size", cfg.get("batch_size", 8)))

    # train
    t = cfg.get("train", {})
    setattr(c, "epochs", t.get("epochs", cfg.get("epochs_action", 50)))
    setattr(c, "lr", t.get("lr", cfg.get("lr", 1e-4)))
    setattr(c, "weight_decay", t.get("weight_decay", cfg.get("weight_decay", 0.05)))
    setattr(c, "cem_samples", t.get("cem_samples", cfg.get("cem_samples", 100)))

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
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config")
    parser.add_argument("--work_dir", type=str, default=None, help="Override working directory")
    parser.add_argument("--sequence", type=str, default=None,
                        help="Comma-separated phases: pretrain,language,probe,action. Overrides config run.phase sequencing.")
    return parser.parse_args()

def load_yaml_config(path: str):
    with open(path, "r") as f:
        y = yaml.safe_load(f)
    if not isinstance(y, dict):
        raise ValueError("Config YAML must map to a dict at top-level")
    # Flatten commonly used keys into root for trainer convenience while keeping nested dict
    cfg = {}
    cfg.update(y)
    # propagate run.work_dir to root
    if "run" in y and isinstance(y["run"], dict):
        cfg["work_dir"] = y["run"].get("work_dir", cfg.get("work_dir", "outputs"))
        cfg["seed"] = y["run"].get("seed", cfg.get("seed", 42))
    return cfg

def main():
    args = parse_args()
    cfg = default_cfg()

    if args.config:
        file_cfg = load_yaml_config(args.config)
        cfg.update(file_cfg)

    # Optional overrides
    if args.work_dir:
        cfg["work_dir"] = args.work_dir

    # Determine run sequence
    seq_arg = args.sequence
    if seq_arg:
        sequence = [s.strip() for s in seq_arg.split(",") if s.strip()]
    else:
        # derive default sequence: if run.phase is set, run only that; else full chain
        run_phase = cfg.get("run", {}).get("phase") if isinstance(cfg.get("run"), dict) else None
        if run_phase:
            sequence = [run_phase if run_phase != "ssl_refine" else "pretrain"]
        else:
            sequence = ["pretrain", "language", "probe", "action"]

    chain_pipeline(cfg, sequence)

if __name__ == "__main__":
    main()

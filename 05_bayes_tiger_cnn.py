from __future__ import annotations

import argparse
import json
import random
import shutil
import subprocess
from pathlib import Path

import optuna
import pandas as pd

from src.biodram.tiger_generator import TigerGenerator
from src.biodram.utils import load_config, set_seed, ensure_dir


def list_files(folder: Path):
    return [p for p in sorted(folder.iterdir()) if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}]


def generate_trial_images(cfg, trial, params):
    tiger_cfg = cfg["tiger"]
    gen = TigerGenerator(tiger_cfg)
    init_images = list_files(Path(tiger_cfg["init_image_dir"]))
    bg_images = list_files(Path(tiger_cfg["condition_bg_dir"]))
    if not init_images or not bg_images:
        raise RuntimeError("Tiger condition folders are incomplete.")
    target_classes = [cfg["class_names"][i] for i in cfg["bayes"].get("target_classes", [])]
    n_each = int(cfg["bayes"].get("synthetic_candidates_per_trial", 20))
    trial_root = Path(cfg["paths"]["output_dir"]) / "bayes_generated" / f"trial_{trial.number:04d}"
    for class_name in target_classes:
        out_dir = ensure_dir(trial_root / class_name)
        prompt = tiger_cfg["prompt_templates"][class_name]
        for i in range(n_each):
            init_path = random.choice(init_images)
            name = init_path.name
            mask_nd = Path(tiger_cfg["mask_nd_dir"]) / name
            mask_bg = Path(tiger_cfg["mask_bg_dir"]) / name
            condition_bg = random.choice(bg_images)
            seed = random.randint(1, 1_000_000)
            out_path = out_dir / f"{Path(name).stem}_{class_name}_{seed}.png"
            gen.generate_one(init_path, mask_nd, mask_bg, condition_bg, prompt, out_path, seed=seed, generation_params=params)
    return trial_root


def copy_trial_to_synthetic(cfg, trial_root: Path):
    synthetic_root = Path(cfg["paths"]["synthetic_root"])
    if synthetic_root.exists():
        shutil.rmtree(synthetic_root)
    synthetic_root.mkdir(parents=True, exist_ok=True)
    for class_name in cfg["class_names"]:
        (synthetic_root / class_name).mkdir(parents=True, exist_ok=True)
    for class_dir in trial_root.iterdir():
        if class_dir.is_dir():
            dst = synthetic_root / class_dir.name
            for p in class_dir.iterdir():
                if p.is_file():
                    shutil.copy2(p, dst / p.name)


def train_quick(cfg_path: str, trial_number: int, quick_epochs: int):
    suffix = f"_trial_{trial_number:04d}"
    cmd = [
        "python",
        "03_train_classifier.py",
        "--config",
        cfg_path,
        "--quick_epochs",
        str(quick_epochs),
        "--output_suffix",
        suffix,
    ]
    subprocess.run(cmd, check=True)
    ckpt = Path(load_config(cfg_path)["paths"]["output_dir"]) / "checkpoints" / f"best{suffix}.pt"
    import torch
    checkpoint = torch.load(ckpt, map_location="cpu")
    metrics = checkpoint.get("metrics", {})
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--n_trials", type=int, default=None)
    args = parser.parse_args()
    cfg = load_config(args.config)
    set_seed(cfg.get("seed", 42))
    ensure_dir(Path(cfg["paths"]["output_dir"]) / "bayes_generated")

    def objective(trial):
        params = {
            "guidance_scale_fg": trial.suggest_float("guidance_scale_fg", 4.0, 12.0),
            "controlnet_conditioning_scale_fg": trial.suggest_float("controlnet_conditioning_scale_fg", 0.05, 1.0),
            "num_inference_steps_fg": trial.suggest_int("num_inference_steps_fg", 20, 60),
            "guidance_scale_bg": trial.suggest_float("guidance_scale_bg", 0.01, 1.0),
            "controlnet_conditioning_scale_bg": trial.suggest_float("controlnet_conditioning_scale_bg", 0.3, 1.2),
            "num_inference_steps_bg": trial.suggest_int("num_inference_steps_bg", 10, 40),
        }
        cfg["loss"]["lambda_margin"] = trial.suggest_float("lambda_margin", 0.0, 0.6)
        cfg["loss"]["lambda_cost"] = trial.suggest_float("lambda_cost", 0.0, 0.8)
        cfg["loss"]["lambda_recall"] = trial.suggest_float("lambda_recall", 0.0, 0.8)
        tmp_cfg = Path(cfg["paths"]["output_dir"]) / f"trial_config_{trial.number:04d}.json"
        tmp_cfg.parent.mkdir(parents=True, exist_ok=True)
        with open(tmp_cfg, "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2)
        trial_root = generate_trial_images(cfg, trial, params)
        copy_trial_to_synthetic(cfg, trial_root)
        metrics = train_quick(str(tmp_cfg), trial.number, int(cfg["bayes"].get("quick_epochs", 8)))
        metric_name = cfg["bayes"].get("objective_metric", "macro_auc")
        score = float(metrics.get(metric_name, 0.0))
        trial.set_user_attr("metrics", metrics)
        trial.set_user_attr("generation_params", params)
        return score

    study = optuna.create_study(direction="maximize", study_name=cfg["bayes"].get("study_name", "biodram_tiger_cnn"))
    study.optimize(objective, n_trials=args.n_trials or int(cfg["bayes"].get("n_trials", 20)))
    out = Path(cfg["paths"]["output_dir"]) / "bayes_best.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"best_value": study.best_value, "best_params": study.best_params, "best_attrs": study.best_trial.user_attrs}, f, indent=2)
    print(f"Saved best trial to {out}")


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import random
from pathlib import Path

import pandas as pd

from src.biodram.tiger_generator import TigerGenerator
from src.biodram.utils import load_config, set_seed, ensure_dir


def list_files(folder: Path):
    return [p for p in sorted(folder.iterdir()) if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--class_name", required=True)
    parser.add_argument("--num_images", type=int, default=20)
    parser.add_argument("--prompt", default=None)
    parser.add_argument("--guidance_scale_fg", type=float, default=None)
    parser.add_argument("--controlnet_conditioning_scale_fg", type=float, default=None)
    parser.add_argument("--num_inference_steps_fg", type=int, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg.get("seed", 42))
    tiger_cfg = cfg["tiger"]
    prompt = args.prompt or tiger_cfg["prompt_templates"][args.class_name]
    params = {}
    for k in ["guidance_scale_fg", "controlnet_conditioning_scale_fg", "num_inference_steps_fg"]:
        v = getattr(args, k)
        if v is not None:
            params[k] = v

    init_images = list_files(Path(tiger_cfg["init_image_dir"]))
    if not init_images:
        raise RuntimeError("No init images found for Tiger generation.")
    out_dir = ensure_dir(Path(tiger_cfg["output_dir"]) / args.class_name)
    gen = TigerGenerator(tiger_cfg)
    records = []
    for i in range(args.num_images):
        init_path = random.choice(init_images)
        name = init_path.name
        mask_nd = Path(tiger_cfg["mask_nd_dir"]) / name
        mask_bg = Path(tiger_cfg["mask_bg_dir"]) / name
        bg_candidates = list_files(Path(tiger_cfg["condition_bg_dir"]))
        condition_bg = random.choice(bg_candidates)
        seed = random.randint(1, 1_000_000)
        out_path = out_dir / f"{Path(name).stem}_{args.class_name}_{seed}.png"
        generated = gen.generate_one(init_path, mask_nd, mask_bg, condition_bg, prompt, out_path, seed=seed, generation_params=params)
        records.append({"path": str(generated.resolve()), "class_name": args.class_name, "seed": seed, "prompt": prompt})
        print(f"Generated {generated}")
    pd.DataFrame(records).to_csv(out_dir / f"manifest_{args.class_name}.csv", index=False)


if __name__ == "__main__":
    main()

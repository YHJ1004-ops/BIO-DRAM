from __future__ import annotations

import random
from pathlib import Path
from typing import Dict, Optional

import cv2 as cv
import numpy as np
import torch
from PIL import Image
from diffusers import StableDiffusionControlNetInpaintPipeline, ControlNetModel, DDIMScheduler, DPMSolverMultistepScheduler


def make_inpaint_condition(image: Image.Image, image_mask: Image.Image) -> torch.Tensor:
    image_arr = np.array(image.convert("RGB")).astype(np.float32) / 255.0
    mask_arr = np.array(image_mask.convert("L")).astype(np.float32) / 255.0
    if image_arr.shape[:2] != mask_arr.shape[:2]:
        raise ValueError("image and image_mask must have the same spatial size")
    image_arr[mask_arr > 0.5] = -1.0
    image_arr = np.expand_dims(image_arr, 0).transpose(0, 3, 1, 2)
    return torch.from_numpy(image_arr)


class TigerGenerator:
    def __init__(self, cfg: Dict):
        self.cfg = cfg
        self.size = int(cfg.get("image_size", 512))
        dtype_name = cfg.get("dtype", "float16")
        self.dtype = torch.float16 if dtype_name == "float16" else torch.float32
        self.device = cfg.get("device", "cuda") if torch.cuda.is_available() else "cpu"
        self.pipe_fg = None
        self.pipe_bg = None

    def _load_pipe(self, controlnet_path: str):
        controlnet = ControlNetModel.from_pretrained(controlnet_path, torch_dtype=self.dtype)
        pipe = StableDiffusionControlNetInpaintPipeline.from_pretrained(
            self.cfg["pretrained_model_path"],
            controlnet=controlnet,
            torch_dtype=self.dtype,
        )
        scheduler_name = self.cfg.get("scheduler", "ddim").lower()
        if scheduler_name == "dpm":
            pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
        else:
            pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
        if self.device == "cuda":
            pipe.enable_model_cpu_offload()
        else:
            pipe = pipe.to(self.device)
        return pipe

    def load(self):
        if self.pipe_fg is None and self.cfg.get("controlnet_fg_path"):
            self.pipe_fg = self._load_pipe(self.cfg["controlnet_fg_path"])
        if self.pipe_bg is None and self.cfg.get("controlnet_bg_path"):
            self.pipe_bg = self._load_pipe(self.cfg["controlnet_bg_path"])

    def _open_image(self, path: str | Path) -> Image.Image:
        return Image.open(path).convert("RGB").resize((self.size, self.size))

    def _open_mask(self, path: str | Path) -> Image.Image:
        return Image.open(path).convert("L").resize((self.size, self.size))

    def generate_one(
        self,
        init_image_path: str | Path,
        mask_nd_path: str | Path,
        mask_bg_path: str | Path,
        condition_bg_path: str | Path,
        prompt: str,
        out_path: str | Path,
        seed: Optional[int] = None,
        generation_params: Optional[Dict] = None,
    ) -> Path:
        self.load()
        params = dict(generation_params or {})
        seed = int(seed if seed is not None else random.randint(1, 1_000_000))
        generator = torch.Generator(device="cpu").manual_seed(seed)

        init_image = self._open_image(init_image_path)
        mask_nd = self._open_mask(mask_nd_path)
        mask_bg = self._open_mask(mask_bg_path)
        control_nd = make_inpaint_condition(init_image, mask_nd)
        control_bg = self._open_image(condition_bg_path)

        negative_prompt = params.get("negative_prompt", self.cfg.get("negative_prompt", ""))

        image_nd = self.pipe_fg(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=int(params.get("num_inference_steps_fg", self.cfg.get("num_inference_steps_fg", 50))),
            guidance_scale=float(params.get("guidance_scale_fg", self.cfg.get("guidance_scale_fg", 8.0))),
            generator=generator,
            eta=float(params.get("eta", self.cfg.get("eta", 1.0))),
            controlnet_conditioning_scale=float(params.get("controlnet_conditioning_scale_fg", self.cfg.get("controlnet_conditioning_scale_fg", 0.2))),
            image=init_image,
            mask_image=mask_nd,
            control_image=control_nd,
        ).images[0]

        image_bg = self.pipe_bg(
            prompt=params.get("background_prompt", "black and white definition detail"),
            negative_prompt=negative_prompt,
            num_inference_steps=int(params.get("num_inference_steps_bg", self.cfg.get("num_inference_steps_bg", 20))),
            guidance_scale=float(params.get("guidance_scale_bg", self.cfg.get("guidance_scale_bg", 0.02))),
            generator=generator,
            eta=float(params.get("eta", self.cfg.get("eta", 1.0))),
            controlnet_conditioning_scale=float(params.get("controlnet_conditioning_scale_bg", self.cfg.get("controlnet_conditioning_scale_bg", 1.0))),
            image=init_image,
            mask_image=mask_bg,
            control_image=control_bg,
        ).images[0]

        nd_arr = np.array(image_nd.convert("RGB"))
        bg_arr = np.array(image_bg.convert("RGB"))
        mask_arr = np.array(mask_nd.resize((self.size, self.size)).convert("L"))
        alpha = (mask_arr > 127).astype(np.float32)[..., None]
        merged = (alpha * nd_arr + (1.0 - alpha) * bg_arr).clip(0, 255).astype(np.uint8)
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(merged).save(out_path)
        return out_path

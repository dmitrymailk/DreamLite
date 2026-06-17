# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from PIL import Image

from torch.utils.data import DataLoader, Dataset 
from datasets import load_dataset
from torchvision import transforms

from accelerate import Accelerator
from diffusers.optimization import get_scheduler
from peft import LoraConfig, get_peft_model

# 导入你的核心组件
from dreamlite import DreamLitePipelineLoRA
from dreamlite.pipelines.dreamlite.pipeline_dreamlite_lora import calculate_shift

def parse_args():
    parser = argparse.ArgumentParser(description="Train LoRA for DreamLite")
    parser.add_argument("--model_id", type=str, default="models/DreamLite-base")
    parser.add_argument("--output_dir", type=str, default="./output/output_lora/edit_Snoopy")
    parser.add_argument("--rank", type=int, default=16, help="LoRA Rank")
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--train_batch_size", type=int, default=1, help="Batch size only can be 1 here.")
    parser.add_argument("--max_train_steps", type=int, default=3500)
    parser.add_argument("--resolution", type=int, default=512, help="Training resolution (square).")
    parser.add_argument("--default_prompt", type=str, default="transfer the image into Snoopy style")

    # --- Dataset ---
    parser.add_argument("--dataset_name", type=str,
                        default="dim/nfs_pix2pix_1920_1080_v6_upscale_2x_raw_filtered")
    parser.add_argument("--dataset_split", type=str, default="train")
    parser.add_argument("--cache_dir", type=str, default=None,
                        help="HF datasets cache_dir. Defaults to /code/dataset/<dataset_name>.")
    parser.add_argument("--image_column", type=str, default="edited_image",
                        help="Target / ground-truth image column.")
    parser.add_argument("--cond_image_column", type=str, default="input_image",
                        help="Source / condition image column.")

    # --- Validation / preview during training ---
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--validation_steps", type=int, default=25,
                        help="Run validation every N optimizer steps (0 disables).")
    parser.add_argument("--num_validation_samples", type=int, default=20,
                        help="How many dataset samples to preview (max 20).")
    parser.add_argument("--validation_prompt", type=str, default=None,
                        help="Defaults to --default_prompt if not set.")
    parser.add_argument("--num_validation_steps", type=int, default=4,
                        help="Inference steps for the validation preview (mobile=4).")
    parser.add_argument("--validation_guidance_scale", type=float, default=1.0)
    parser.add_argument("--validation_image_guidance_scale", type=float, default=1.0)
    parser.add_argument("--validation_resolution", type=int, default=None,
                        help="Validation resolution (square). Defaults to --resolution.")
    return parser.parse_args()


@torch.no_grad()
def prepare_validation_cache(pipe, args, val_samples, device):
    """Один раз считает prompt_embeds и image_latents для превью-сэмплов.

    В edit-режиме encode_prompt прогоняет Qwen3VL по тексту И картинке,
    а image_latents — это VAE-кодирование source-картинки. И text_encoder,
    и VAE заморожены, набор валидационных сэмплов фиксирован, промпт один,
    поэтому эти тензоры не меняются между чекпойнтами. Считаем их один раз
    до обучения и переиспользуем на каждой валидации — это убирает
    повторные VL-прогоны (самую дорогую часть инференса).
    """
    if not val_samples:
        return []

    prompt = args.validation_prompt or args.default_prompt
    res = args.validation_resolution
    dtype = pipe.text_encoder.dtype
    # Текст ровно как в __call__ (edit, text-ветка)
    prompt_str = (
        f"[Edit]: A diptych with two side-by-side images of the same scene. "
        f"Compared to the right side, the left one has {prompt}"
    )

    # Тот же препроцессинг, что и в обучении: Resize (по короткой стороне)
    # -> центральный кроп, без сплющивания аспекта. Возвращает PIL res x res.
    crop_transform = transforms.Compose([
        transforms.Resize(res, interpolation=transforms.InterpolationMode.LANCZOS),
        transforms.CenterCrop(res),
    ])

    cache = []
    for src_pil, tgt_pil in val_samples:
        src_crop = crop_transform(src_pil)
        tgt_crop = crop_transform(tgt_pil)
        prompt_embeds, text_attention_mask = pipe.encode_prompt(
            mode="edit",
            prompts=[prompt_str],
            image=src_crop,
            device=device,
            dtype=dtype,
        )
        image_processed = pipe.image_processor.preprocess(src_crop)
        image_latents = pipe.prepare_image_latents(image_processed, dtype=dtype, device=device)
        cache.append({
            "src": src_crop,
            "tgt": tgt_crop,
            "prompt_embeds": prompt_embeds,
            "text_attention_mask": text_attention_mask,
            "image_latents": image_latents,
        })

    print(f"[validation] cached prompt_embeds/image_latents for {len(cache)} samples")
    return cache


@torch.no_grad()
def _edit_no_cfg(pipe, prompt_embeds, text_attention_mask, image_latents, res, num_inference_steps, generator):
    """Облегчённый edit-инференс без CFG (batch=1 вместо 3) на готовых эмбеддингах.

    Полный pipe.__call__ в режиме edit всегда считает 3-way CFG
    (uncond/image/text) и трижды гоняет Qwen3VL по картинке 512x512.
    При guidance_scale=1 и image_guidance_scale=1 итог CFG ровно равен
    text-ветке: uncond + (text-image) + (image-uncond) = text. Поэтому
    повторяем логику __call__ с одним forward'ом, причём prompt_embeds и
    image_latents приходят из кэша (см. prepare_validation_cache).
    """
    device = pipe._execution_device
    height = width = res

    sigmas = np.linspace(1.0, 1.0 / num_inference_steps, num_inference_steps)

    num_channels_latents = pipe.vae.config.latent_channels
    latents = pipe.prepare_latents(
        1, num_channels_latents, height, width, prompt_embeds.dtype, device, generator,
    )

    image_seq_len = latents.shape[2] * latents.shape[3] // 4
    mu = calculate_shift(
        image_seq_len,
        pipe.scheduler.config.get("base_image_seq_len", 256),
        pipe.scheduler.config.get("max_image_seq_len", 4096),
        pipe.scheduler.config.get("base_shift", 0.5),
        pipe.scheduler.config.get("max_shift", 1.16),
    )
    pipe.scheduler.set_timesteps(sigmas=sigmas, device=device, mu=mu)
    timesteps = pipe.scheduler.timesteps

    add_time_ids = torch.tensor([[width, height]], device=device, dtype=prompt_embeds.dtype)

    for t in timesteps:
        model_input = torch.cat([latents, image_latents], dim=3)
        noise_pred = pipe.unet(
            model_input,
            timestep=t.expand(model_input.shape[0]).to(latents.dtype),
            encoder_hidden_states=prompt_embeds,
            encoder_attention_mask=text_attention_mask,
            added_cond_kwargs={"time_ids": add_time_ids},
            return_dict=False,
        )[0]
        noise_pred = noise_pred[..., :latents.shape[-1]]
        latents = pipe.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

    shift_factor = getattr(pipe.vae.config, "shift_factor", 0.0)
    latents = (latents / pipe.vae.config.scaling_factor) + shift_factor
    image_out = pipe.vae.decode(latents, return_dict=False)[0]
    return pipe.image_processor.postprocess(image_out, output_type="pil")[0]


@torch.no_grad()
def log_validation(pipe, unet, accelerator, args, global_step, val_cache):
    """Превью-инференс текущим UNet с LoRA на закэшированных сэмплах.

    Для каждого примера (source, target) генерируем результат и сохраняем
    отдельную полосу [source | generated | target]. Используем UNet прямо
    из памяти: временно подменяем pipe.unet на распакованный PEFT-модель,
    затем возвращаем train(). Инференс идёт по облегчённому no-CFG пути на
    предрассчитанных prompt_embeds/image_latents.
    """
    if not accelerator.is_main_process:
        return
    if not val_cache:
        print("[validation] no samples, skipping")
        return

    val_dir = os.path.join(args.output_dir, "validation")
    os.makedirs(val_dir, exist_ok=True)

    res = args.validation_resolution

    saved_unet = pipe.unet
    eval_unet = accelerator.unwrap_model(unet)
    was_training = eval_unet.training
    pipe.unet = eval_unet
    eval_unet.eval()

    rows = []
    try:
        # autocast bf16: LoRA-адаптеры PEFT во float32 апкастят выход UNet,
        # из-за чего без autocast падает bf16-декод VAE. Зеркалит mixed-precision обучения.
        with torch.autocast(device_type=accelerator.device.type, dtype=torch.bfloat16):
            for item in val_cache:
                gen = _edit_no_cfg(
                    pipe,
                    prompt_embeds=item["prompt_embeds"],
                    text_attention_mask=item["text_attention_mask"],
                    image_latents=item["image_latents"],
                    res=res,
                    num_inference_steps=args.num_validation_steps,
                    generator=torch.Generator("cpu").manual_seed(args.seed),
                )
                rows.append((item["src"], gen, item["tgt"]))
    finally:
        pipe.unet = saved_unet
        if was_training:
            eval_unet.train()

    # Каждый пример сохраняем отдельным файлом: [source | generated | target]
    step_dir = os.path.join(val_dir, f"step_{global_step:06d}")
    os.makedirs(step_dir, exist_ok=True)
    for idx, (src_pil, gen, tgt_pil) in enumerate(rows):
        cells = [
            src_pil.resize((res, res), Image.Resampling.LANCZOS),
            gen.resize((res, res), Image.Resampling.LANCZOS),
            tgt_pil.resize((res, res), Image.Resampling.LANCZOS),
        ]
        strip = Image.new("RGB", (res * 3, res))
        for c, cell in enumerate(cells):
            strip.paste(cell, (c * res, 0))
        strip.save(os.path.join(step_dir, f"sample_{idx:02d}.png"))

    print(f"[validation] step {global_step}: saved {len(rows)} images to {step_dir}")


def main():
    args = parse_args()
    if args.validation_resolution is None:
        args.validation_resolution = args.resolution
    if args.cache_dir is None:
        args.cache_dir = "/code/dataset/" + args.dataset_name.split("/")[-1]
    
    # 1. Initialize Accelerator
    accelerator = Accelerator(
        mixed_precision="bf16",
        gradient_accumulation_steps=4,
    )
    
    # 2. Load DreamLite Pipeline
    pipe = DreamLitePipelineLoRA.from_pretrained(args.model_id, torch_dtype=torch.bfloat16)
    
    text_encoder = pipe.text_encoder
    vae = pipe.vae
    unet = pipe.unet
    noise_scheduler = pipe.scheduler

    # Frozen other modules
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet.requires_grad_(False)

    # 3. LoRA (Based on PEFT)
    lora_config = LoraConfig(
        r=args.rank,
        lora_alpha=args.rank,
        target_modules=[
            "to_q",
            "to_k",
            "to_v",
            "to_out.0",
        ],
    )
    unet = get_peft_model(unet, lora_config)
    
    # print
    unet.print_trainable_parameters()

    # 4. configure optimizer
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, unet.parameters()),
        lr=args.learning_rate,
        weight_decay=1e-4,
    )
    
    # 5. prepare dataloader
    # =======================================================
    # TODO: finish DataLoader
    # dataset = MyDataset(args.dataset_path, ...)
    # dataloader = DataLoader(dataset, batch_size=args.train_batch_size, shuffle=True)
    print(f"Loading dataset {args.dataset_name} (split={args.dataset_split}) from cache {args.cache_dir} ...")
    train_dataset = load_dataset(
        args.dataset_name,
        split=args.dataset_split,
        cache_dir=args.cache_dir,
    )

    # Resize (по короткой стороне) -> центральный кроп, как в референсе
    # train_dreambooth_lora_flux2_klein_img2img.py: сохраняем пропорции и
    # берём центральный квадрат, без сплющивания 16:9 в квадрат.
    image_transforms = transforms.Compose([
        transforms.Resize(args.resolution, interpolation=transforms.InterpolationMode.LANCZOS),
        transforms.CenterCrop(args.resolution),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]), # Normalize
    ])

    def _to_rgb(item):
        if hasattr(item, "convert"):
            return item.convert("RGB")
        if isinstance(item, str):
            return Image.open(item).convert("RGB")
        raise ValueError(f"Unsupported image type: {type(item)}")

    # Фиксируем валидационные пары (source, target) PIL ДО set_transform,
    # пока датасет ещё отдаёт исходные изображения, а не тензоры.
    val_samples = []
    if args.validation_steps > 0 and args.num_validation_samples > 0:
        n = min(args.num_validation_samples, len(train_dataset))
        step = max(1, len(train_dataset) // n)
        val_indices = list(range(0, len(train_dataset), step))[:n]
        for i in val_indices:
            row = train_dataset[i]
            val_samples.append((
                _to_rgb(row[args.cond_image_column]),  # source
                _to_rgb(row[args.image_column]),       # target
            ))
        print(f"[validation] prepared {len(val_samples)} samples at indices {val_indices}")

    def preprocess_train(examples):
        target_imgs = []
        source_imgs = []
        source_imgs_pil = []

        for tar_item in examples[args.image_column]:
            img = _to_rgb(tar_item)
            target_imgs.append(image_transforms(img))

        for src_item in examples[args.cond_image_column]:
            img = _to_rgb(src_item)
            source_imgs_pil.append(img)
            source_imgs.append(image_transforms(img))

        prompts = [args.default_prompt] * len(examples[args.image_column])
                
        return {
            "target_imgs": target_imgs,
            "source_imgs": source_imgs,
            "source_imgs_pil": source_imgs_pil,
            "prompt": prompts 
        }

    train_dataset.set_transform(preprocess_train)

    def collate_fn(examples):
        target_imgs = torch.stack([example["target_imgs"] for example in examples])
        source_imgs = torch.stack([example["source_imgs"] for example in examples])
        prompts = [example["prompt"] for example in examples]
        source_imgs_pil = [example["source_imgs_pil"] for example in examples]
        return {"target_imgs": target_imgs, "source_imgs": source_imgs, "source_imgs_pil": source_imgs_pil, "prompts": prompts}

    dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=collate_fn,
        batch_size=1,
    )

    # =======================================================

    # 6. Accelerator
    # unet, optimizer = accelerator.prepare(unet, optimizer)
    unet, optimizer, dataloader = accelerator.prepare(unet, optimizer, dataloader)

    vae.to(accelerator.device, dtype=torch.bfloat16)
    text_encoder.to(accelerator.device, dtype=torch.bfloat16)

    # Один раз считаем prompt_embeds/image_latents для превью (энкодеры заморожены,
    # сэмплы фиксированы) — на каждой валидации переиспользуем без VL-прогона.
    val_cache = []
    if accelerator.is_main_process and args.validation_steps > 0:
        val_cache = prepare_validation_cache(pipe, args, val_samples, accelerator.device)

    # 7. Train
    global_step = 0
    progress_bar = tqdm(total=args.max_train_steps, disable=not accelerator.is_local_main_process)
    
    unet.train()
    
    while global_step < args.max_train_steps:
        # =======================================================
        # TODO: get data from DataLoader
        # for batch in dataloader:
        #     images = batch["pixel_values"]
        #     prompts = batch["text"]
        for batch in dataloader:
            if global_step >= args.max_train_steps:
                break
            images = batch['target_imgs'].to(accelerator.device, dtype=torch.bfloat16)
            conds = batch['source_imgs'].to(accelerator.device, dtype=torch.bfloat16)
            conds_pil = batch['source_imgs_pil'][0]
            prompts = batch['prompts']
        # =======================================================

            with accelerator.accumulate(unet):
                # 1. encode Latents (Ground Truth x_0)
                latents = vae.encode(images).latents
                latents = latents * vae.config.scaling_factor
                src_latents = vae.encode(conds).latents
                src_latents = src_latents * vae.config.scaling_factor

                # 2. noise and timestep
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                sigmas = torch.rand((bsz,), dtype=latents.dtype, device=latents.device)
                sigmas_expanded = sigmas.view(bsz, 1, 1, 1)

                timesteps = (sigmas * 1000.0).long() 

                # 3. Add noise to Latents
                noisy_latents = (1.0 - sigmas_expanded) * latents + sigmas_expanded * noise

                # 4. Encode Prompt
                prompt_embeds, text_attention_mask = pipe.encode_prompt(
                    mode="edit",
                    image=conds_pil,
                    prompts=prompts,
                    device=accelerator.device,
                    dtype=torch.bfloat16,
                )

                # 5. Time IDs, Image Latents
                # Generate mode, condition image = 0
                model_input = torch.cat([noisy_latents, src_latents], dim=3) # In-context Concat
                
                add_time_ids = torch.tensor([[args.resolution, args.resolution]], dtype=torch.bfloat16, device=accelerator.device).repeat(bsz, 1)

                # 6. UNet Predict Noise
                noise_pred = unet(
                    model_input,
                    timesteps,
                    encoder_hidden_states=prompt_embeds,
                    encoder_attention_mask=text_attention_mask,
                    added_cond_kwargs={"time_ids": add_time_ids},
                    return_dict=False,
                )[0]
                
                noise_pred = noise_pred[..., :latents.shape[-1]]

                # 7. Loss (Flow Matching, MSE)
                target = noise - latents
                loss = F.mse_loss(noise_pred.float(), target.float(), reduction="mean")

                # 8. backward and update params
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(filter(lambda p: p.requires_grad, unet.parameters()), 1.0)
                
                optimizer.step()
                optimizer.zero_grad()

            # update
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                progress_bar.set_postfix({"loss": loss.item()})

                if args.validation_steps > 0 and (
                    global_step == 1 or global_step % args.validation_steps == 0
                ):
                    log_validation(pipe, unet, accelerator, args, global_step, val_cache)

    accelerator.wait_for_everyone()

    # Финальное превью перед сохранением весов
    if args.validation_steps > 0:
        log_validation(pipe, unet, accelerator, args, global_step, val_cache)

    # 8. Save LoRA weights
    if accelerator.is_main_process:
        unet = accelerator.unwrap_model(unet)
        os.makedirs(args.output_dir, exist_ok=True)
        unet.save_pretrained(args.output_dir)
        print(f"LoRA weights saved to {args.output_dir}")

if __name__ == "__main__":
    main()
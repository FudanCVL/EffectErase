import argparse
import os
import time
from pathlib import Path

import cv2
import imageio.v3 as imageio
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from PIL import Image
from einops import rearrange

from diffsynth import ModelManager, WanRemovePipeline


def build_parser():
    parser = argparse.ArgumentParser(description="Infer remove video from one FG_BG video and one MASK video.")

    parser.add_argument("--fg_bg_path", type=str, required=True, help="Path to FG_BG video.")
    parser.add_argument("--mask_path", type=str, required=True, help="Path to MASK video.")
    parser.add_argument("--output_path", type=str, required=True, help="Path to output remove video.")

    parser.add_argument("--num_frames", type=int, default=81, help="Number of frames.")
    parser.add_argument("--frame_interval", type=int, default=1, help="Frame interval.")
    parser.add_argument("--height", type=int, default=480, help="Frame height.")
    parser.add_argument("--width", type=int, default=832, help="Frame width.")
    parser.add_argument("--seed", type=int, default=2025, help="Random seed.")
    parser.add_argument("--cfg", type=float, default=1.0, help="CFG scale.")
    parser.add_argument("--lora_alpha", type=float, default=1.0, help="LoRA alpha.")
    parser.add_argument("--num_inference_steps", type=int, default=50, help="Inference steps.")

    parser.add_argument("--tiled", action="store_true", help="Enable VAE tiling.")
    parser.add_argument("--use_teacache", action="store_true", help="Use TeaCache.")

    parser.add_argument("--text_encoder_path", type=str, required=True, help="Path to text encoder.")
    parser.add_argument("--vae_path", type=str, required=True, help="Path to VAE.")
    parser.add_argument("--dit_path", type=str, required=True, help="Path to DiT.")
    parser.add_argument("--image_encoder_path", type=str, required=True, help="Path to image encoder.")
    parser.add_argument("--pretrained_lora_path", type=str, required=True, help="Path to pretrained LoRA.")

    return parser


def resize_image(image, height, width):
    return torchvision.transforms.functional.resize(
        image,
        (height, width),
        interpolation=torchvision.transforms.InterpolationMode.BILINEAR,
    )


def frame_norm_to_tensor(frame):
    frame = torchvision.transforms.functional.to_tensor(frame)  # [H,W,C] -> [C,H,W], [0,255] -> [0,1]
    frame = torchvision.transforms.functional.normalize(
        frame,
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5],
    )  # [0,1] -> [-1,1]
    return frame


def read_video_frames(video_path, num_frames, frame_interval, height, width):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames < num_frames:
        cap.release()
        raise ValueError(
            f"Video {video_path} has only {total_frames} frames, "
            f"less than required num_frames={num_frames}."
        )

    if total_frames >= frame_interval * num_frames:
        step = frame_interval
        start_frame = 0
    else:
        print(
            f"[WARN] total_frames={total_frames} < frame_interval*num_frames="
            f"{frame_interval * num_frames}, fallback to step=1."
        )
        step = 1
        start_frame = 0

    img_list = []
    first_pil = None

    for i in range(num_frames):
        frame_idx = start_frame + i * step
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        success, frame = cap.read()
        if not success:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_pil = Image.fromarray(frame_rgb)

        if i == 0:
            first_pil = frame_pil.copy()

        frame_pil = resize_image(frame_pil, height, width)
        frame_tensor = frame_norm_to_tensor(frame_pil)
        img_list.append(frame_tensor)

    cap.release()

    if len(img_list) == 0:
        raise RuntimeError(f"No valid frames read from {video_path}")

    if len(img_list) < num_frames:
        raise RuntimeError(
            f"Only read {len(img_list)} frames from {video_path}, expected {num_frames}"
        )

    frames = torch.stack(img_list, dim=0)  # [T,C,H,W]
    frames = rearrange(frames, "T C H W -> C T H W")  # [C,T,H,W]
    return frames, first_pil


def crop_square_from_pil(mask_img: Image.Image, fg_bg_img: Image.Image, target_size: int = 224, video_mask_path: str = None):
    mask_np = np.array(mask_img)
    if mask_np.ndim == 3:
        mask_np = mask_np.max(axis=-1)
    mask_np = (mask_np > 0).astype(np.uint8)

    img_np = np.array(fg_bg_img.convert("RGB"))
    h, w = mask_np.shape

    ys, xs = np.where(mask_np > 0)
    if len(xs) == 0:
        raise ValueError(f"{video_mask_path} has no valid mask region in first frame")

    x0, x1 = xs.min(), xs.max() + 1
    y0, y1 = ys.min(), ys.max() + 1
    bw, bh = x1 - x0, y1 - y0

    side = max(bw, bh)
    cx = (x0 + x1) / 2.0
    cy = (y0 + y1) / 2.0

    sx0 = int(np.floor(cx - side / 2))
    sy0 = int(np.floor(cy - side / 2))
    sx1 = sx0 + side
    sy1 = sy0 + side

    pad_left = max(0, -sx0)
    pad_top = max(0, -sy0)
    pad_right = max(0, sx1 - w)
    pad_bottom = max(0, sy1 - h)

    if pad_left or pad_top or pad_right or pad_bottom:
        img_np = np.pad(
            img_np,
            ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
            mode="constant",
            constant_values=0,
        )
        mask_np = np.pad(
            mask_np,
            ((pad_top, pad_bottom), (pad_left, pad_right)),
            mode="constant",
            constant_values=0,
        )
        sx0 += pad_left
        sx1 += pad_left
        sy0 += pad_top
        sy1 += pad_top

    crop_img = img_np[sy0:sy1, sx0:sx1]
    crop_mask = mask_np[sy0:sy1, sx0:sx1][..., None]
    crop_img = crop_img * crop_mask

    crop_t = torch.from_numpy(crop_img).permute(2, 0, 1).float().unsqueeze(0)  # [1,3,S,S]
    crop_t = F.interpolate(
        crop_t,
        size=(target_size, target_size),
        mode="bilinear",
        align_corners=False,
    )[0]
    crop_t = crop_t / 255.0
    crop_t = crop_t * 2.0 - 1.0  # [-1,1]

    return crop_t


def save_frames_as_video_with_ref_fps(
    frames,
    output_path,
    ref_video_path=None,
    default_fps=24,
    cmap_name="turbo",
    vmin=None,
    vmax=None,
):
    if ref_video_path:
        try:
            import imageio
            reader = imageio.get_reader(ref_video_path)
            meta = reader.get_meta_data()
            fps = meta.get("fps", default_fps)
            reader.close()
        except Exception as e:
            print(f"[WARN] Failed to read fps from ref video: {e}, use default_fps={default_fps}")
            fps = default_fps
    else:
        fps = default_fps

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    frames = np.asarray(frames)
    if frames.ndim == 3:
        frames = frames[..., None]
    elif frames.ndim != 4:
        raise ValueError(f"Expected frames shape (T,H,W,C), got {frames.shape}")

    t, h, w, c = frames.shape
    if (h, w) != frames[0].shape[:2]:
        raise ValueError(f"Inconsistent frame sizes: {(h, w)} vs {frames[0].shape[:2]}")

    writer = imageio.get_writer(
        output_path.as_posix(),
        fps=fps,
        codec="libx264",
        format="FFMPEG",
        ffmpeg_params=["-pix_fmt", "yuv420p", "-movflags", "+faststart"],
        quality=8,
    )

    try:
        for i in range(t):
            frame = frames[i]
            if c == 1:
                img = frame[..., 0]
                vmin_i = np.min(img) if vmin is None else vmin
                vmax_i = np.max(img) if vmax is None else vmax
                if vmax_i <= vmin_i:
                    vmax_i = vmin_i + 1e-6
                cmap = plt.get_cmap(cmap_name)
                norm = (img - vmin_i) / (vmax_i - vmin_i)
                rgb = (cmap(np.clip(norm, 0, 1))[..., :3] * 255).astype(np.uint8)
                frame = rgb
            elif c == 3:
                frame = np.clip(frame, 0, 255).astype(np.uint8)
            else:
                raise ValueError(f"Unsupported channel number C={c}, only support 1 or 3.")

            writer.append_data(frame)
    finally:
        writer.close()

    print(f"[INFO] Saved video to: {output_path} (fps={fps})")
    return output_path.as_posix()


if __name__ == "__main__":
    start = time.time()
    args = build_parser().parse_args()

    device = torch.device("cuda")

    print("[INFO] Reading videos...")
    mask_imgs_tensor, mask_first_image = read_video_frames(
        args.mask_path,
        args.num_frames,
        args.frame_interval,
        args.height,
        args.width,
    )
    fg_bg_imgs_tensor, fg_bg_first_image = read_video_frames(
        args.fg_bg_path,
        args.num_frames,
        args.frame_interval,
        args.height,
        args.width,
    )

    fg_first_img = crop_square_from_pil(
        mask_first_image,
        fg_bg_first_image,
        target_size=224,
        video_mask_path=args.mask_path,
    )

    print("[INFO] Building model...")
    model_manager = ModelManager(device="cuda")
    model_manager.load_models(
        [
            args.dit_path,
            args.text_encoder_path,
            args.vae_path,
            args.image_encoder_path,
        ],
        torch_dtype=torch.bfloat16,
    )
    model_manager.load_lora_v2(args.pretrained_lora_path, lora_alpha=args.lora_alpha)

    pipe = WanRemovePipeline.from_model_manager(
        model_manager,
        torch_dtype=torch.bfloat16,
        device="cuda",
    )
    pipe.enable_vram_management(num_persistent_param_in_dit=6 * 10**9)

    mask_imgs_tensor = mask_imgs_tensor.to(device)
    fg_bg_imgs_tensor = fg_bg_imgs_tensor.to(device)
    fg_first_img = fg_first_img.to(device)

    remove_prompt = "Remove the specified object and all related effects, then restore a clean background."
    negative_prompt = (
        "细节模糊不清，字幕，作品，画作，画面，静止，最差质量，低质量，JPEG压缩残留，"
        "丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，"
        "形态畸形的肢体，手指融合，杂乱的背景，三条腿，背景人很多，倒着走"
    )

    print("[INFO] Running inference...")
    with torch.inference_mode(), torch.cuda.amp.autocast(dtype=torch.bfloat16):
        remove_video, _ = pipe(
            video_mask=mask_imgs_tensor,
            video_fg_bg=fg_bg_imgs_tensor,
            video_bg=None,
            task="remove",
            fg_first_img=fg_first_img,
            prompt_remove=remove_prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=args.num_inference_steps,
            cfg_scale=args.cfg,
            seed=args.seed,
            tiled=args.tiled,
            height=args.height,
            width=args.width,
            tea_cache_l1_thresh=0.3 if args.use_teacache else None,
            tea_cache_model_id="Wan2.1-T2V-1.3B" if args.use_teacache else None,
        )

    save_frames_as_video_with_ref_fps(
        remove_video,
        output_path=args.output_path,
        ref_video_path=args.fg_bg_path,
        default_fps=24,
    )

    end = time.time()
    print(f"[INFO] Total time: {end - start:.2f} seconds")
    print("[INFO] All done!")
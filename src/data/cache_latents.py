#!/usr/bin/env python3
"""
Standalone script for VAE encoding and latent caching.

Usage:
    python cache_latents.py \
        --dataset builddotai/Egocentric-100K \
        --split train \
        --cache-dir /scratch-shared/cache \
        --vae-model nvidia/Cosmos-1.0-Tokenizer-CV8x8x8 \
        --batch-size 16
"""

import argparse
import torch
import os
from pathlib import Path
from datasets import load_dataset, Features, Value
from diffusers import AutoencoderKL
from tqdm import tqdm
import sys
import time
from huggingface_hub.errors import HfHubHTTPError

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.latent_cache import LatentCache
from data.transforms import VideoTransforms


def load_and_preprocess_video(
    video_bytes: bytes,
    resolution: tuple,
    fps: int,
    max_frames: int = 512,
):
    """
    Load and preprocess video for VAE encoding.

    Returns:
        Tensor of shape [T, C, H, W]
    """
    video = VideoTransforms.load_video_from_bytes(
        video_bytes, max_frames=max_frames
    )
    video = VideoTransforms.spatial_resize(video, resolution)
    video = VideoTransforms.normalize_for_vae(video)
    return video


def main():
    parser = argparse.ArgumentParser(
        description="Cache VAE latents for video dataset"
    )
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--cache-dir", type=str, required=True)
    parser.add_argument(
        "--vae-model",
        type=str,
        default="nvidia/Cosmos-1.0-Tokenizer-CV8x8x8",
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--resolution", type=int, nargs=2, default=[256, 456])
    parser.add_argument("--fps", type=int, default=8)
    parser.add_argument("--max-samples", type=int, default=None)

    args = parser.parse_args()

    print("=" * 60)
    print("Latent Caching Configuration")
    print("=" * 60)
    print(f"Dataset: {args.dataset} ({args.split})")
    print(f"VAE Model: {args.vae_model}")
    print(f"Cache Dir: {args.cache_dir}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Resolution: {args.resolution}")
    print(f"FPS: {args.fps}")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    print("Loading VAE model...")
    vae = AutoencoderKL.from_pretrained(
        args.vae_model,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    )
    vae.to(device).eval()
    print("VAE loaded successfully")

    cache = LatentCache(Path(args.cache_dir))
    print(f"Initialized cache at {args.cache_dir}")

    features = Features(
        {
            "mp4": Value("binary"),
            "json": {
                "factory_id": Value("string"),
                "worker_id": Value("string"),
                "video_index": Value("int64"),
                "duration_sec": Value("float64"),
                "width": Value("int64"),
                "height": Value("int64"),
                "fps": Value("float64"),
                "size_bytes": Value("int64"),
                "codec": Value("string"),
            },
            "__key__": Value("string"),
            "__url__": Value("string"),
        }
    )

    dataset = load_dataset(
        args.dataset,
        split=args.split,
        streaming=True,
        features=features,
    )

    cached_count = 0
    encoded_count = 0
    error_count = 0

    batch_videos = []
    batch_ids = []

    print("\nStarting video processing...")
    sys.stdout.flush()

    for i, sample in enumerate(tqdm(dataset, desc="Processing videos")):
        if args.max_samples and i >= args.max_samples:
            break

        video_id = sample.get("__key__", f"video_{i}")

        if cache.has(video_id):
            cached_count += 1
            continue

        try:
            video_bytes = sample.get("mp4")
            if video_bytes is None:
                error_count += 1
                continue

            video = load_and_preprocess_video(
                video_bytes,
                tuple(args.resolution),
                args.fps,
                max_frames=512,
            )

            batch_videos.append(video)
            batch_ids.append(video_id)

            if len(batch_videos) >= args.batch_size:
                with torch.no_grad():
                    video_batch = torch.stack(batch_videos).to(device)

                    # Stack to [B, T, C, H, W]
                    B, T, C, H, W = video_batch.shape
                    frames = video_batch.view(B * T, C, H, W)

                    frames = frames.to(dtype=vae.dtype)

                    latent_dist = vae.encode(frames).latent_dist
                    latents = latent_dist.sample()

                    latents = latents.view(
                        B, T, *latents.shape[1:]
                    )

                    for vid_id, lat in zip(batch_ids, latents):
                        print(f"Vid id: {vid_id}")
                        cache.set(vid_id, lat.cpu())
                        encoded_count += 1
                        print("-" * 50)

                batch_videos.clear()
                batch_ids.clear()
                sys.stdout.flush()

        except Exception as e:
            raise e

    if batch_videos:
        with torch.no_grad():
            video_batch = torch.stack(batch_videos).to(device)

            B, T, C, H, W = video_batch.shape
            frames = video_batch.view(B * T, C, H, W)
            frames = frames.to(dtype=vae.dtype)

            print(f"frame shape {frames.shape}")

            latent_dist = vae.encode(frames).latent_dist
            latents = latent_dist.sample()

            latents = latents.view(B, T, *latents.shape[1:])
            print(f"Latent shape {latents.shape}")

            for vid_id, lat in zip(batch_ids, latents):
                print(f"Vid id: {vid_id}")
                cache.set(vid_id, lat.cpu())
                encoded_count += 1
                print("-" * 50)

    print("\n" + "=" * 60)
    print("Caching Complete")
    print("=" * 60)
    print(f"Already cached: {cached_count}")
    print(f"Newly encoded: {encoded_count}")
    print(f"Errors: {error_count}")
    print("=" * 60)


if __name__ == "__main__":
    main()

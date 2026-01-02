#!/usr/bin/env python3
"""
Standalone script for VAE encoding and latent caching.

Usage:
    python cache_latents.py \\
        --dataset builddotai/Egocentric-100K \\
        --split train \\
        --cache-dir /scratch-shared/cache \\
        --vae-model nvidia/Cosmos-1.0-Tokenizer-CV8x8x8 \\
        --batch-size 16
"""

import argparse
import torch
from pathlib import Path
from datasets import load_dataset
from diffusers import AutoencoderKL
from tqdm import tqdm
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.latent_cache import LatentCache
from data.transforms import VideoTransforms


def load_and_preprocess_video(video_path: str, resolution: tuple, fps: int):
    """Load and preprocess video for VAE encoding."""
    # Load video
    video = VideoTransforms.load_video(video_path)

    # Spatial resize
    video = VideoTransforms.spatial_resize(video, resolution)

    # Normalize for VAE [-1, 1]
    video = VideoTransforms.normalize_for_vae(video)

    return video


def main():
    parser = argparse.ArgumentParser(description="Cache VAE latents for video dataset")
    parser.add_argument("--dataset", type=str, required=True, help="HF dataset name")
    parser.add_argument("--split", type=str, default="train", help="Dataset split")
    parser.add_argument("--cache-dir", type=str, required=True, help="Cache directory")
    parser.add_argument("--vae-model", type=str, default="nvidia/Cosmos-1.0-Tokenizer-CV8x8x8")
    parser.add_argument("--batch-size", type=int, default=16, help="Encoding batch size")
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--resolution", type=int, nargs=2, default=[256, 456], help="H W")
    parser.add_argument("--fps", type=int, default=8, help="Target FPS")
    parser.add_argument("--max-samples", type=int, default=None, help="Max samples to cache")

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

    # Setup device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load VAE
    print("Loading VAE model...")
    vae = AutoencoderKL.from_pretrained(
        args.vae_model,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32
    )
    vae.to(device)
    vae.eval()
    print("VAE loaded successfully")

    # Initialize cache
    cache = LatentCache(Path(args.cache_dir))
    print(f"Initialized cache at {args.cache_dir}")

    # Load dataset
    print(f"Loading dataset {args.dataset}...")
    dataset = load_dataset(args.dataset, split=args.split, streaming=True)

    # Cache statistics
    cached_count = 0
    encoded_count = 0
    error_count = 0

    # Process dataset
    batch_videos = []
    batch_ids = []

    for i, sample in enumerate(tqdm(dataset, desc="Processing videos")):
        if args.max_samples and i >= args.max_samples:
            break

        video_id = sample.get("video_id", f"video_{i}")

        # Check if already cached
        if cache.has(video_id):
            cached_count += 1
            continue

        try:
            # Get video path from sample
            video_path = sample.get("video_path") or sample.get("video")

            if video_path is None:
                print(f"Warning: No video path in sample {i}")
                error_count += 1
                continue

            # Load and preprocess
            video = load_and_preprocess_video(
                video_path,
                tuple(args.resolution),
                args.fps
            )

            batch_videos.append(video)
            batch_ids.append(video_id)

            # Encode when batch is full
            if len(batch_videos) >= args.batch_size:
                with torch.no_grad():
                    # Stack to [B, T, C, H, W]
                    video_batch = torch.stack(batch_videos).to(device)

                    # VAE encode
                    latent_dist = vae.encode(video_batch).latent_dist
                    latents = latent_dist.sample()

                    # Cache each sample
                    for vid_id, lat in zip(batch_ids, latents):
                        cache.set(vid_id, lat.cpu())
                        encoded_count += 1

                # Clear batch
                batch_videos = []
                batch_ids = []

        except Exception as e:
            print(f"Error processing video {video_id}: {e}")
            error_count += 1
            continue

    # Encode remaining batch
    if batch_videos:
        try:
            with torch.no_grad():
                video_batch = torch.stack(batch_videos).to(device)
                latent_dist = vae.encode(video_batch).latent_dist
                latents = latent_dist.sample()

                for vid_id, lat in zip(batch_ids, latents):
                    cache.set(vid_id, lat.cpu())
                    encoded_count += 1
        except Exception as e:
            print(f"Error encoding final batch: {e}")
            error_count += len(batch_videos)

    # Final statistics
    print("\n" + "=" * 60)
    print("Caching Complete")
    print("=" * 60)
    print(f"Already cached: {cached_count}")
    print(f"Newly encoded: {encoded_count}")
    print(f"Errors: {error_count}")
    print(f"Total processed: {cached_count + encoded_count + error_count}")

    cache_stats = cache.get_cache_stats()
    print(f"\nCache stats: {cache_stats}")
    print("=" * 60)


if __name__ == "__main__":
    main()

import torch
import safetensors
import safetensors.torch
from pathlib import Path
from typing import Optional, List
import hashlib


class LatentCache:
    """
    Disk-based cache for VAE-encoded latents using SafeTensors format.

    Storage structure:
        cache_dir/
            shard_00000/
                {video_id_hash}.safetensors
            shard_00001/
                ...

    Each safetensors file contains a single tensor named "latents".
    """

    def __init__(
        self,
        cache_dir: Path,
        shard_size: int = 1000,
    ):
        """
        Args:
            cache_dir: Root directory for cache storage
            shard_size: Number of videos per shard directory
        """
        self.cache_dir = Path(cache_dir)
        self.shard_size = shard_size
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _get_shard_path(self, video_id: str) -> Path:
        """Map video_id to shard directory and file path."""
        id_hash = int(hashlib.md5(video_id.encode()).hexdigest(), 16)
        shard_idx = (id_hash % 1000000) // self.shard_size
        shard_dir = self.cache_dir / f"shard_{shard_idx:05d}"
        shard_dir.mkdir(exist_ok=True)
        return shard_dir / f"{video_id}.safetensors"

    def has(self, video_id: str) -> bool:
        """Check if video latents are cached."""
        return self._get_shard_path(video_id).exists()

    def get(self, video_id: str) -> torch.Tensor:
        """Load cached latents for video_id."""
        path = self._get_shard_path(video_id)
        with safetensors.safe_open(str(path), framework="pt") as f:
            return f.get_tensor("latents")

    def set(self, video_id: str, latents: torch.Tensor):
        """Store latents for video_id."""
        path = self._get_shard_path(video_id)
        safetensors.torch.save_file(
            {"latents": latents.cpu()},
            str(path)
        )

    def batch_encode_and_cache(
        self,
        video_paths: List[str],
        video_ids: List[str],
        vae_model,
        video_loader_fn,
        batch_size: int = 8,
        device: str = "cuda"
    ):
        """
        Batch encode videos with VAE and populate cache.

        Args:
            video_paths: List of paths to video files
            video_ids: Corresponding video IDs for cache keys
            vae_model: VAE encoder model
            video_loader_fn: Function that loads and preprocesses video (path -> tensor)
            batch_size: Encoding batch size
            device: Device for encoding
        """
        from tqdm import tqdm

        vae_model.eval()
        vae_model.to(device)

        for i in tqdm(range(0, len(video_paths), batch_size), desc="Encoding videos"):
            batch_paths = video_paths[i:i+batch_size]
            batch_ids = video_ids[i:i+batch_size]

            # Skip already cached
            to_encode_paths = []
            to_encode_ids = []
            for path, vid_id in zip(batch_paths, batch_ids):
                if not self.has(vid_id):
                    to_encode_paths.append(path)
                    to_encode_ids.append(vid_id)

            if not to_encode_paths:
                continue

            # Load videos
            videos = [video_loader_fn(p) for p in to_encode_paths]
            video_batch = torch.stack(videos).to(device)

            # Encode with VAE
            with torch.no_grad():
                latent_dist = vae_model.encode(video_batch).latent_dist
                latents = latent_dist.sample()

            # Cache individual latents
            for vid_id, lat in zip(to_encode_ids, latents):
                self.set(vid_id, lat.cpu())

    def get_cache_stats(self) -> dict:
        """Return cache statistics."""
        shard_dirs = list(self.cache_dir.glob("shard_*"))
        total_files = sum(1 for sd in shard_dirs for _ in sd.glob("*.safetensors"))

        return {
            "cache_dir": str(self.cache_dir),
            "num_shards": len(shard_dirs),
            "num_cached_videos": total_files,
        }

import torch
from torch.utils.data import Dataset, WeightedRandomSampler
from typing import List, Dict, Any, Optional
from .hf_egocentric import EgocentricDataset


class ComposedVideoDataset(Dataset):
    """
    Compose multiple video datasets with weighted sampling.

    Supports mixing different datasets (e.g., Egocentric + custom data)
    with configurable sampling weights.
    """

    def __init__(self, dataset_configs: List[Dict[str, Any]]):
        """
        Args:
            dataset_configs: List of dataset configurations
                Each config has:
                {
                    "name": "egocentric",
                    "weight": 0.7,
                    "config": {...dataset-specific params}
                }
        """
        self.datasets = []
        self.weights = []
        self.dataset_names = []

        for cfg in dataset_configs:
            name = cfg["name"]
            weight = cfg.get("weight", 1.0)
            ds_config = cfg.get("config", {})

            # Create dataset based on name
            ds = self._create_dataset(name, ds_config)
            self.datasets.append(ds)
            self.weights.append(weight)
            self.dataset_names.append(name)

        # Normalize weights
        total = sum(self.weights)
        self.weights = [w / total for w in self.weights]

        # Calculate cumulative lengths for indexing
        self.cumulative_lengths = []
        cumsum = 0
        for ds in self.datasets:
            cumsum += len(ds)
            self.cumulative_lengths.append(cumsum)

    def _create_dataset(self, name: str, config: Dict) -> Dataset:
        """Factory for dataset creation."""
        if name == "egocentric":
            return EgocentricDataset(**config)
        else:
            raise ValueError(f"Unknown dataset: {name}")

    def __len__(self) -> int:
        return self.cumulative_lengths[-1] if self.cumulative_lengths else 0

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get item by global index."""
        # Find which dataset this index belongs to
        dataset_idx = 0
        for i, cum_len in enumerate(self.cumulative_lengths):
            if idx < cum_len:
                dataset_idx = i
                break

        # Calculate local index within dataset
        if dataset_idx == 0:
            local_idx = idx
        else:
            local_idx = idx - self.cumulative_lengths[dataset_idx - 1]

        # Get sample from appropriate dataset
        sample = self.datasets[dataset_idx][local_idx]

        # Add dataset name to metadata
        sample["metadata"]["dataset"] = self.dataset_names[dataset_idx]

        return sample

    def get_collate_fn(self):
        """Use collate function from first dataset."""
        return self.datasets[0].get_collate_fn()

    def get_weighted_sampler(
        self,
        num_samples: Optional[int] = None
    ) -> WeightedRandomSampler:
        """
        Create weighted sampler for training.

        Args:
            num_samples: Number of samples to draw per epoch (None = dataset length)

        Returns:
            WeightedRandomSampler that respects dataset weights
        """
        # Calculate per-sample weights
        sample_weights = []
        for ds, weight in zip(self.datasets, self.weights):
            # Each sample in this dataset gets weight / len(ds)
            per_sample_weight = weight / len(ds)
            sample_weights.extend([per_sample_weight] * len(ds))

        return WeightedRandomSampler(
            weights=sample_weights,
            num_samples=num_samples or len(self),
            replacement=True
        )

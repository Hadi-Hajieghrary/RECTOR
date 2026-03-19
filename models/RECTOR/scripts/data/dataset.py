"""
PyTorch Dataset for rule-aware trajectory generation.

Provides:
- AugmentedDataset: Main dataset class
- WeightedSampler: Upweights samples with rare rule applicability
- collate_fn: Batching function with padding
"""

import torch
from torch.utils.data import Dataset, Sampler
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
import json
import os

from .sample import AugmentedSample, BatchedSamples, SceneFeatures


class AugmentedDataset(Dataset):
    """
    PyTorch Dataset for augmented trajectory samples.

    Supports loading from:
    - Pre-computed sample files (npz/pkl)
    - TFRecord files (with on-the-fly conversion)
    - In-memory sample list
    """

    def __init__(
        self,
        samples: Optional[List[AugmentedSample]] = None,
        sample_paths: Optional[List[str]] = None,
        tfrecord_paths: Optional[List[str]] = None,
        max_agents: int = 32,
        max_lanes: int = 64,
        lane_points: int = 20,
        history_length: int = 11,
        future_length: int = 80,
        num_rules: int = 28,
        cache_samples: bool = True,
        transform: Optional[Any] = None,
    ):
        """
        Initialize dataset.

        Args:
            samples: Pre-loaded list of AugmentedSample objects
            sample_paths: Paths to sample files (.npz or .pkl)
            tfrecord_paths: Paths to TFRecord files (requires conversion)
            max_agents: Maximum number of agents (for padding)
            max_lanes: Maximum number of lanes (for padding)
            lane_points: Points per lane polyline
            history_length: Number of history timesteps
            future_length: Number of future timesteps
            num_rules: Number of rules in canonical ordering
            cache_samples: Whether to cache loaded samples
            transform: Optional transform to apply to samples
        """
        self.max_agents = max_agents
        self.max_lanes = max_lanes
        self.lane_points = lane_points
        self.history_length = history_length
        self.future_length = future_length
        self.num_rules = num_rules
        self.cache_samples = cache_samples
        self.transform = transform

        # Initialize sample storage
        self._samples: List[AugmentedSample] = []
        self._sample_paths: List[str] = []
        self._cache: Dict[int, AugmentedSample] = {}

        if samples is not None:
            self._samples = samples
        elif sample_paths is not None:
            self._sample_paths = sample_paths
        elif tfrecord_paths is not None:
            self._init_from_tfrecords(tfrecord_paths)
        else:
            raise ValueError("Must provide samples, sample_paths, or tfrecord_paths")

    def _init_from_tfrecords(self, paths: List[str]):
        """Initialize from TFRecord files."""
        # This would require tfrecord parsing - simplified for now
        print(f"TFRecord loading not implemented. Found {len(paths)} files.")
        self._sample_paths = []

    def __len__(self) -> int:
        if self._samples:
            return len(self._samples)
        return len(self._sample_paths)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single sample as tensorized dict.

        Returns:
            Dict with tensor keys for model input
        """
        # Get sample
        if self._samples:
            sample = self._samples[idx]
        else:
            sample = self._load_sample(idx)

        # Apply transform if any
        if self.transform is not None:
            sample = self.transform(sample)

        # Convert to tensors
        return self._tensorize(sample)

    def _load_sample(self, idx: int) -> AugmentedSample:
        """Load sample from file with optional caching."""
        if idx in self._cache:
            return self._cache[idx]

        path = self._sample_paths[idx]

        if path.endswith(".npz"):
            data = np.load(path, allow_pickle=True)
            sample = AugmentedSample.from_dict(dict(data))
        elif path.endswith(".pkl"):
            import pickle

            with open(path, "rb") as f:
                sample = pickle.load(f)
        else:
            raise ValueError(f"Unknown file type: {path}")

        if self.cache_samples:
            self._cache[idx] = sample

        return sample

    def _tensorize(self, sample: AugmentedSample) -> Dict[str, torch.Tensor]:
        """Convert sample to tensor dict."""
        result = {}

        # Ego history: [T_hist, 4]
        result["ego_history"] = torch.tensor(sample.ego_history, dtype=torch.float32)

        # Ego future GT: [H, 4]
        result["ego_future_gt"] = torch.tensor(
            sample.ego_future_gt, dtype=torch.float32
        )

        # Agent states: pad to [max_agents, T_total, 4]
        n_agents = min(sample.num_agents, self.max_agents)
        t_total = self.history_length + self.future_length

        agent_states = np.zeros((self.max_agents, t_total, 4), dtype=np.float32)
        if n_agents > 0 and sample.agent_states.shape[0] > 0:
            agent_states[:n_agents] = sample.agent_states[:n_agents, :t_total]
        result["agent_states"] = torch.tensor(agent_states)

        # Agent metadata: [max_agents, 3]
        agent_meta = np.zeros((self.max_agents, 3), dtype=np.float32)
        if n_agents > 0 and sample.agent_metadata.shape[0] > 0:
            agent_meta[:n_agents] = sample.agent_metadata[:n_agents]
        result["agent_metadata"] = torch.tensor(agent_meta)

        # Agent valid: [max_agents, T_total]
        agent_valid = np.zeros((self.max_agents, t_total), dtype=bool)
        if n_agents > 0 and sample.agent_valid.shape[0] > 0:
            agent_valid[:n_agents] = sample.agent_valid[:n_agents, :t_total]
        result["agent_valid"] = torch.tensor(agent_valid)

        # Agent mask (which agents exist)
        agent_mask = np.zeros(self.max_agents, dtype=bool)
        agent_mask[:n_agents] = True
        result["agent_mask"] = torch.tensor(agent_mask)

        # Lane centers: pad to [max_lanes, lane_points, 2]
        n_lanes = min(sample.num_lanes, self.max_lanes)

        lane_centers = np.zeros((self.max_lanes, self.lane_points, 2), dtype=np.float32)
        if n_lanes > 0 and sample.lane_centers.shape[0] > 0:
            lc = sample.lane_centers[:n_lanes, : self.lane_points]
            lane_centers[:n_lanes, : lc.shape[1]] = lc
        result["lane_centers"] = torch.tensor(lane_centers)

        # Lane headings: [max_lanes, lane_points]
        lane_headings = np.zeros((self.max_lanes, self.lane_points), dtype=np.float32)
        if n_lanes > 0 and sample.lane_headings.shape[0] > 0:
            lh = sample.lane_headings[:n_lanes, : self.lane_points]
            lane_headings[:n_lanes, : lh.shape[1]] = lh
        result["lane_headings"] = torch.tensor(lane_headings)

        # Lane mask
        lane_mask = np.zeros(self.max_lanes, dtype=bool)
        lane_mask[:n_lanes] = True
        result["lane_mask"] = torch.tensor(lane_mask)

        # Rule labels
        result["rule_applicability"] = torch.tensor(
            sample.rule_applicability, dtype=torch.float32
        )
        result["rule_violations"] = torch.tensor(
            sample.rule_violations_gt, dtype=torch.float32
        )
        result["rule_severity"] = torch.tensor(
            sample.rule_severity_gt, dtype=torch.float32
        )

        return result

    def get_rule_statistics(self) -> Dict[str, np.ndarray]:
        """
        Compute rule applicability/violation statistics over dataset.

        Returns:
            Dict with 'applicability_counts', 'violation_counts', 'total'
        """
        app_counts = np.zeros(self.num_rules)
        vio_counts = np.zeros(self.num_rules)

        for i in range(len(self)):
            if self._samples:
                sample = self._samples[i]
            else:
                sample = self._load_sample(i)

            app_counts += sample.rule_applicability.astype(float)
            vio_counts += sample.rule_violations_gt.astype(float)

        return {
            "applicability_counts": app_counts,
            "violation_counts": vio_counts,
            "total": len(self),
        }


class WeightedRuleSampler(Sampler):
    """
    Weighted sampler that upweights samples with rare rule applicability.

    This helps balance training when some rules are rarely applicable.
    Uses adaptive floor to ensure all samples have non-zero weight.

    Features:
    - Inverse frequency weighting for rare rules
    - Tier-aware weighting (safety rules get higher priority)
    - Adaptive floor: eps = 1/len(dataset)
    - Mixed sampling: (1-uniform_fraction)*weighted + uniform_fraction*uniform
    """

    def __init__(
        self,
        dataset: AugmentedDataset,
        num_samples: Optional[int] = None,
        replacement: bool = True,
        uniform_fraction: float = 0.2,
        tier_boost: Optional[Dict[str, float]] = None,
    ):
        """
        Initialize sampler.

        Args:
            dataset: AugmentedDataset to sample from
            num_samples: Number of samples per epoch (default: len(dataset))
            replacement: Whether to sample with replacement
            uniform_fraction: Fraction of sampling that's uniform
            tier_boost: Extra weight multiplier per tier (e.g., {'safety': 2.0})
        """
        self.dataset = dataset
        self.num_samples = num_samples or len(dataset)
        self.replacement = replacement
        self.uniform_fraction = uniform_fraction

        # Default tier boost: safety rules get 2x weight
        self.tier_boost = tier_boost or {
            "safety": 2.0,
            "legal": 1.5,
            "road": 1.0,
            "comfort": 0.5,
        }

        # Compute sample weights
        self.weights = self._compute_weights()

    def _compute_weights(self) -> np.ndarray:
        """Compute per-sample weights based on rule applicability."""
        import sys

        sys.path.insert(0, "/workspace/data/WOMD")
        sys.path.insert(0, "/workspace/data")
        from waymo_rule_eval.rules.rule_constants import (
            TIERS,
            TIER_BY_NAME,
            RULE_INDEX_MAP,
            NUM_RULES,
        )

        n = len(self.dataset)

        # Get rule statistics
        stats = self.dataset.get_rule_statistics()
        app_counts = stats["applicability_counts"]
        total = stats["total"]

        # Inverse frequency weights for rules
        # Add 1 to avoid division by zero
        rule_weights = total / (app_counts + 1)

        # Apply tier boost
        tier_multiplier = np.ones(NUM_RULES)
        for tier_name in TIERS:
            boost = self.tier_boost.get(tier_name, 1.0)
            for rule_id in TIER_BY_NAME.get(tier_name, []):
                if rule_id in RULE_INDEX_MAP:
                    idx = RULE_INDEX_MAP[rule_id]
                    tier_multiplier[idx] = boost

        rule_weights = rule_weights * tier_multiplier
        rule_weights = rule_weights / rule_weights.sum()  # Normalize

        # Compute per-sample weight as sum of applicable rule weights
        sample_weights = np.zeros(n)

        for i in range(n):
            if self.dataset._samples:
                sample = self.dataset._samples[i]
            else:
                sample = self.dataset._load_sample(i)

            applicable = sample.rule_applicability
            sample_weights[i] = (applicable * rule_weights).sum()

        # Add adaptive floor: minimum weight = 1 / n
        floor = 1.0 / n
        sample_weights = np.maximum(sample_weights, floor)

        # Mix with uniform distribution
        uniform_weights = np.ones(n) / n
        final_weights = (
            1 - self.uniform_fraction
        ) * sample_weights / sample_weights.sum() + self.uniform_fraction * uniform_weights

        return final_weights

    def __iter__(self):
        indices = np.random.choice(
            len(self.dataset),
            size=self.num_samples,
            replace=self.replacement,
            p=self.weights,
        )
        return iter(indices.tolist())

    def __len__(self) -> int:
        return self.num_samples


# Keep old name as alias for backward compatibility
WeightedSampler = WeightedRuleSampler


def collate_fn(samples: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """
    Collate function for DataLoader.

    Stacks samples into batched tensors.

    Args:
        samples: List of tensorized samples from AugmentedDataset

    Returns:
        Dict with batched tensors
    """
    batch = {}

    for key in samples[0].keys():
        tensors = [s[key] for s in samples]
        batch[key] = torch.stack(tensors, dim=0)

    return batch


def create_dataloader(
    dataset: AugmentedDataset,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 4,
    use_weighted_sampler: bool = False,
    **kwargs,
) -> torch.utils.data.DataLoader:
    """
    Create DataLoader for training.

    Args:
        dataset: AugmentedDataset
        batch_size: Batch size
        shuffle: Whether to shuffle (ignored if using weighted sampler)
        num_workers: Number of worker processes
        use_weighted_sampler: Use WeightedSampler for rule balance
        **kwargs: Additional DataLoader arguments

    Returns:
        DataLoader
    """
    if use_weighted_sampler:
        sampler = WeightedSampler(dataset)
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=num_workers,
            collate_fn=collate_fn,
            **kwargs,
        )
    else:
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=collate_fn,
            **kwargs,
        )

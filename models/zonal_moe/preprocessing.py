"""
Preprocessing utilities for Zonal MoE.

Handles:
- k-NN graph construction and caching
- Wall distance computation and caching
- Data loading and normalization
- Geometry-aware train/val splitting
"""

import os
import hashlib
import numpy as np
import torch
from torch_geometric.nn import knn_graph
from pathlib import Path
from typing import Tuple, List, Dict, Optional


def compute_geometry_fingerprint_fast(filepath: str) -> str:
    """Compute fingerprint from file without full load.

    Only reads the first and last 10 position values needed for fingerprinting,
    avoiding a full np.load of the entire file.
    """
    with np.load(filepath, mmap_mode="r") as data:
        pos = data["pos"]
        # Read only what we need
        first = pos[:10].flatten()
        last = pos[-10:].flatten()
        shape = np.array(pos.shape)
    data_bytes = np.concatenate([first, last, shape]).astype(np.float32).tobytes()
    return hashlib.md5(data_bytes).hexdigest()[:16]


def compute_geometry_fingerprint(pos: np.ndarray) -> str:
    """Compute a hash fingerprint for geometry-based caching.

    Args:
        pos: (N, 3) point positions
    Returns:
        Hex string fingerprint
    """
    data = (
        np.concatenate([pos[:10].flatten(), pos[-10:].flatten(), np.array(pos.shape)])
        .astype(np.float32)
        .tobytes()
    )
    return hashlib.md5(data).hexdigest()[:16]


def compute_wall_distance(
    pos: torch.Tensor, idcs_airfoil: torch.Tensor
) -> torch.Tensor:
    """Compute minimum distance from each point to the airfoil surface.

    Args:
        pos: (N, 3) all point positions
        idcs_airfoil: (M,) indices of airfoil surface points
    Returns:
        (N,) wall distance per point
    """
    airfoil_pos = pos[idcs_airfoil]  # (M, 3)
    N = pos.shape[0]

    batch_size = 10000
    wall_dist = torch.zeros(N, device=pos.device)

    for i in range(0, N, batch_size):
        end = min(i + batch_size, N)
        diff = pos[i:end].unsqueeze(1) - airfoil_pos.unsqueeze(0)
        dists = diff.norm(dim=-1)
        wall_dist[i:end] = dists.min(dim=1).values

    return wall_dist


def build_knn_graph(
    pos: torch.Tensor, k: int = 16, batch: torch.Tensor = None
) -> torch.Tensor:
    """Build k-NN graph from point positions."""
    return knn_graph(pos, k=k, batch=batch, loop=False)


class GeometryCache:
    """Caches k-NN graphs and wall distances per geometry."""

    def __init__(self, cache_dir: str = None):
        if cache_dir is None:
            cache_dir = os.path.expanduser("~/.cache/zonal_moe")
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _get_cache_path(self, fingerprint: str, suffix: str) -> Path:
        return self.cache_dir / f"{fingerprint}_{suffix}.pt"

    def get_or_compute_wall_dist(
        self, pos: torch.Tensor, idcs_airfoil: torch.Tensor, fingerprint: str = None
    ) -> torch.Tensor:
        if fingerprint is None:
            fingerprint = compute_geometry_fingerprint(pos.cpu().numpy())

        cache_path = self._get_cache_path(fingerprint, "wall_dist")

        if cache_path.exists():
            return torch.load(cache_path, map_location=pos.device, weights_only=True)

        wall_dist = compute_wall_distance(pos, idcs_airfoil)
        torch.save(wall_dist.cpu(), cache_path)
        return wall_dist

    def get_or_compute_knn(
        self, pos: torch.Tensor, k: int, fingerprint: str = None
    ) -> torch.Tensor:
        if fingerprint is None:
            fingerprint = compute_geometry_fingerprint(pos.cpu().numpy())

        cache_path = self._get_cache_path(fingerprint, f"knn_k{k}")

        if cache_path.exists():
            return torch.load(cache_path, map_location=pos.device, weights_only=True)

        edge_index = build_knn_graph(pos, k=k)
        torch.save(edge_index.cpu(), cache_path)
        return edge_index


class AirfoilDataset(torch.utils.data.Dataset):
    """Dataset for airfoil flow prediction.

    Each sample contains:
    - velocity_in: (5, N, 3) input velocity snapshots
    - velocity_out: (5, N, 3) target velocity snapshots
    - pos: (N, 3) point positions
    - idcs_airfoil: (M,) airfoil surface point indices

    Normalization stats are computed from a random subset of files (not just the first).
    """

    def __init__(
        self,
        data_dir: str,
        k_standard: int = 16,
        k_dense: int = 32,
        use_cache: bool = True,
        normalize: bool = True,
        device: str = "cpu",
        stats_sample_size: int = 30,
    ):
        self.data_dir = Path(data_dir)
        self.k_standard = k_standard
        self.k_dense = k_dense
        self.device = device
        self.normalize = normalize

        # Find all data files
        self.files = sorted(self.data_dir.glob("*.npz"))
        if len(self.files) == 0:
            raise ValueError(f"No .npz files found in {data_dir}")

        # Setup caching
        self.cache = GeometryCache() if use_cache else None

        # Precompute fingerprints (fast, avoids double-loading in geometry_aware_split)
        self._fingerprints = None

        # Compute normalization stats from random subset
        if normalize:
            self._compute_norm_stats(sample_size=stats_sample_size)

    def _compute_fingerprints(self):
        """Compute and cache all fingerprints (fast, uses mmap)."""
        if self._fingerprints is None:
            self._fingerprints = {}
            for idx, fpath in enumerate(self.files):
                fp = compute_geometry_fingerprint_fast(str(fpath))
                self._fingerprints[idx] = fp
        return self._fingerprints

    def get_fingerprint(self, idx: int) -> str:
        """Get fingerprint for a file index."""
        fps = self._compute_fingerprints()
        return fps[idx]

    def _compute_norm_stats(self, sample_size: int = 30):
        """Compute velocity/position normalization statistics from random file subset."""
        np.random.seed(42)
        n_files = min(sample_size, len(self.files))
        sample_indices = np.random.choice(len(self.files), n_files, replace=False)

        all_vel = []
        all_pos = []
        all_wall_dist = []
        all_vorticity = []

        for idx in sample_indices:
            data = np.load(self.files[idx])
            vel_in = data["velocity_in"]  # (5, N, 3)
            pos = data["pos"].astype(np.float32)
            idcs_airfoil = data["idcs_airfoil"]

            all_vel.append(vel_in.reshape(-1, 3))
            all_pos.append(pos)

            # Compute wall distance for stats
            pos_t = torch.tensor(pos)
            idcs_t = torch.tensor(idcs_airfoil, dtype=torch.long)
            wall_dist = compute_wall_distance(pos_t, idcs_t).numpy()
            all_wall_dist.append(wall_dist)

            # Compute vorticity proxy
            vorticity = np.linalg.norm(vel_in[-1] - vel_in[0], axis=-1)
            all_vorticity.append(vorticity)

        all_vel = np.concatenate(all_vel, axis=0)
        all_pos = np.concatenate(all_pos, axis=0)
        all_wall_dist = np.concatenate(all_wall_dist, axis=0)
        all_vorticity = np.concatenate(all_vorticity, axis=0)

        # Velocity stats
        self.vel_mean = torch.tensor(all_vel.mean(axis=0), dtype=torch.float32)
        self.vel_std = torch.tensor(all_vel.std(axis=0) + 1e-8, dtype=torch.float32)

        # Position stats
        self.pos_mean = torch.tensor(all_pos.mean(axis=0), dtype=torch.float32)
        self.pos_std = torch.tensor(all_pos.std(axis=0) + 1e-8, dtype=torch.float32)

        # Wall distance stats - filter out surface points (wall_dist=0) before percentiles
        # Surface points always have wall_dist=0 and would skew the threshold too low
        nonzero_wall = all_wall_dist[all_wall_dist > 1e-8]
        self.wall_dist_p95 = float(np.percentile(nonzero_wall, 95))
        self.wall_dist_p10 = (
            float(np.percentile(nonzero_wall, 10)) if len(nonzero_wall) > 0 else 0.01
        )

        # Vorticity stats
        self.vorticity_p95 = float(np.percentile(all_vorticity, 95))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        data = np.load(self.files[idx])

        vel_in = torch.tensor(data["velocity_in"], dtype=torch.float32)
        vel_out = torch.tensor(data["velocity_out"], dtype=torch.float32)
        pos = torch.tensor(data["pos"], dtype=torch.float32)
        idcs_airfoil = torch.tensor(data["idcs_airfoil"], dtype=torch.long)

        if self.normalize:
            vel_in = (vel_in - self.vel_mean) / self.vel_std
            vel_out = (vel_out - self.vel_mean) / self.vel_std
            pos_norm = (pos - self.pos_mean) / self.pos_std
        else:
            pos_norm = pos

        N = pos.shape[0]
        is_airfoil = torch.zeros(N, dtype=torch.bool)
        is_airfoil[idcs_airfoil] = True

        fingerprint = self.get_fingerprint(idx)

        if self.cache is not None:
            wall_dist = self.cache.get_or_compute_wall_dist(
                pos, idcs_airfoil, fingerprint
            )
            edge_index = self.cache.get_or_compute_knn(
                pos_norm, self.k_standard, fingerprint
            )
            edge_index_dense = self.cache.get_or_compute_knn(
                pos_norm, self.k_dense, fingerprint
            )
        else:
            wall_dist = compute_wall_distance(pos, idcs_airfoil)
            edge_index = build_knn_graph(pos_norm, k=self.k_standard)
            edge_index_dense = build_knn_graph(pos_norm, k=self.k_dense)

        return {
            "vel_in": vel_in,
            "vel_out": vel_out,
            "pos": pos_norm,
            "pos_raw": pos,
            "wall_dist": wall_dist,
            "is_airfoil": is_airfoil,
            "edge_index": edge_index,
            "edge_index_dense": edge_index_dense,
            "fingerprint": fingerprint,
            "file": str(self.files[idx]),
        }


def geometry_aware_split(
    dataset: AirfoilDataset, val_ratio: float = 0.1, seed: int = 42
) -> Tuple[List[int], List[int]]:
    """Split dataset by geometry, not by sample.

    Uses precomputed fingerprints from dataset (no double file loading).

    Args:
        dataset: AirfoilDataset instance
        val_ratio: fraction of geometries to hold out
        seed: random seed
    Returns:
        (train_indices, val_indices)
    """
    # Use dataset's precomputed fingerprints
    geometry_to_indices: Dict[str, List[int]] = {}
    for idx in range(len(dataset)):
        fp = dataset.get_fingerprint(idx)
        if fp not in geometry_to_indices:
            geometry_to_indices[fp] = []
        geometry_to_indices[fp].append(idx)

    geometries = list(geometry_to_indices.keys())
    np.random.seed(seed)
    np.random.shuffle(geometries)

    n_val = max(1, int(len(geometries) * val_ratio))
    val_geometries = set(geometries[:n_val])

    train_indices = []
    val_indices = []

    for fp, indices in geometry_to_indices.items():
        if fp in val_geometries:
            val_indices.extend(indices)
        else:
            train_indices.extend(indices)

    return train_indices, val_indices


def compute_polynomial_baseline(vel_in: torch.Tensor, degree: int = 2) -> torch.Tensor:
    """Compute quadratic extrapolation baseline from 5 input timesteps.

    Fits a degree-2 polynomial independently per point per velocity component
    and extrapolates to 5 future timesteps.

    Args:
        vel_in: (T=5, N, 3) input velocity snapshots
        degree: polynomial degree (default 2 = quadratic)
    Returns:
        poly_out: (5, N, 3) extrapolated velocities (the new baseline)
    """
    T, N, C = vel_in.shape
    device = vel_in.device

    # Input times (normalized 0 to 1)
    t_in = torch.linspace(0.0, 1.0, T, device=device)
    V_in = torch.vander(t_in, N=degree + 1, increasing=True)

    # Flatten velocities: (5, N*3)
    vel_flat = vel_in.reshape(T, -1)

    # Solve for coefficients using least squares
    coeffs = torch.linalg.lstsq(V_in, vel_flat).solution

    # Output times (extrapolate beyond t=1.0)
    t_out = torch.linspace(1.2, 2.0, T, device=device)
    V_out = torch.vander(t_out, N=degree + 1, increasing=True)

    # Extrapolate
    poly_flat = V_out @ coeffs
    poly_out = poly_flat.reshape(T, N, C)

    return poly_out


def compute_dataset_statistics(dataset: AirfoilDataset) -> Dict[str, float]:
    """Compute statistics needed for model initialization.

    Returns dict with wall_dist_scale and vorticity_scale for RoutingGate.
    """
    return {
        "wall_dist_scale": dataset.wall_dist_p95,
        "vorticity_scale": dataset.vorticity_p95,
        "wall_dist_threshold": dataset.wall_dist_p10,  # For WeightedMSELoss
    }

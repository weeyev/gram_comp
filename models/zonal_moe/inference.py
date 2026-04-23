"""
Inference utilities for Zonal MoE.

Provides functions for:
- Loading trained models
- Running predictions on new data
- Analyzing routing decisions

IMPORTANT: model.forward() returns RESIDUAL. predict() adds polynomial baseline for absolute velocity.
"""

import torch
import numpy as np
from pathlib import Path

from .model import ZonalMoE
from .preprocessing import (
    compute_wall_distance,
    build_knn_graph,
    compute_geometry_fingerprint,
    GeometryCache,
    compute_polynomial_baseline,
)


def load_model(
    checkpoint_path: str,
    device: str = None,
    wall_dist_scale: float = 1.0,
    vorticity_scale: float = 1.0,
    **model_kwargs,
) -> ZonalMoE:
    """Load a trained ZonalMoE model from checkpoint."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    defaults = dict(
        d_temporal=64,
        backbone_hidden=128,
        laminar_hidden=64,
        turbulent_hidden=256,
        heads=4,
        dropout=0.1,
        output_timesteps=5,
        output_channels=3,
        wall_dist_scale=wall_dist_scale,
        vorticity_scale=vorticity_scale,
    )
    defaults.update(model_kwargs)

    model = ZonalMoE(**defaults).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint)
    model.eval()

    return model


def predict(
    model: ZonalMoE,
    vel_in: np.ndarray,
    pos: np.ndarray,
    idcs_airfoil: np.ndarray,
    device: str = None,
    k_standard: int = 16,
    k_dense: int = 32,
    use_cache: bool = True,
    vel_mean: np.ndarray = None,
    vel_std: np.ndarray = None,
) -> np.ndarray:
    """Run prediction on a single sample.

    Args:
        model: Trained ZonalMoE model
        vel_in: (5, N, 3) input velocity snapshots
        pos: (N, 3) point positions
        idcs_airfoil: (M,) airfoil surface point indices
        device: Device to run on
        k_standard: k for standard graph (default 16)
        k_dense: k for dense graph (default 32)
        use_cache: Whether to use geometry caching
        vel_mean: velocity mean for denormalization (if normalized input)
        vel_std: velocity std for denormalization

    Returns:
        (5, N, 3) predicted ABSOLUTE velocity
    """
    if device is None:
        device = next(model.parameters()).device

    vel_in_t = torch.tensor(vel_in, dtype=torch.float32, device=device)
    pos_t = torch.tensor(pos, dtype=torch.float32, device=device)
    idcs_airfoil_t = torch.tensor(idcs_airfoil, dtype=torch.long, device=device)

    N = pos.shape[0]

    # Compute graph and wall distance
    if use_cache:
        cache = GeometryCache()
        fp = compute_geometry_fingerprint(pos)
        wall_dist = cache.get_or_compute_wall_dist(pos_t, idcs_airfoil_t, fp).to(device)
        edge_index = cache.get_or_compute_knn(pos_t, k_standard, fp).to(device)
        edge_index_dense = cache.get_or_compute_knn(pos_t, k_dense, fp).to(device)
    else:
        wall_dist = compute_wall_distance(pos_t, idcs_airfoil_t)
        edge_index = build_knn_graph(pos_t, k=k_standard)
        edge_index_dense = build_knn_graph(pos_t, k=k_dense)

    is_airfoil = torch.zeros(N, dtype=torch.bool, device=device)
    is_airfoil[idcs_airfoil_t] = True

    # Model returns RESIDUAL
    with torch.no_grad():
        residual = model(
            vel_in=vel_in_t,
            edge_index=edge_index,
            wall_dist=wall_dist,
            is_airfoil=is_airfoil,
            edge_index_dense=edge_index_dense,
        )

    # Add polynomial baseline to get ABSOLUTE velocity
    poly_baseline = compute_polynomial_baseline(vel_in_t)
    vel_pred = poly_baseline + residual  # (5, N, 3)

    vel_pred = vel_pred.cpu().numpy()

    # Denormalize if needed
    if vel_mean is not None and vel_std is not None:
        vel_pred = vel_pred * vel_std + vel_mean
        vel_in = vel_in * vel_std + vel_mean

    return vel_pred


def analyze_routing(
    model: ZonalMoE,
    vel_in: np.ndarray,
    pos: np.ndarray,
    idcs_airfoil: np.ndarray,
    device: str = None,
) -> dict:
    """Analyze routing gate decisions for a sample."""
    if device is None:
        device = next(model.parameters()).device

    vel_in_t = torch.tensor(vel_in, dtype=torch.float32, device=device)
    pos_t = torch.tensor(pos, dtype=torch.float32, device=device)
    idcs_airfoil_t = torch.tensor(idcs_airfoil, dtype=torch.long, device=device)

    wall_dist = compute_wall_distance(pos_t, idcs_airfoil_t)
    vorticity = model.compute_vorticity_proxy(vel_in_t)

    with torch.no_grad():
        gate_values = model.routing_gate(wall_dist, vorticity)

    gate_np = gate_values.cpu().numpy()
    wall_dist_np = wall_dist.cpu().numpy()
    vorticity_np = vorticity.cpu().numpy()

    return {
        "gate_values": gate_np,
        "wall_dist": wall_dist_np,
        "vorticity": vorticity_np,
        "turbulent_mask": gate_np > 0.5,
        "laminar_mask": gate_np <= 0.5,
        "stats": {
            "gate_mean": float(gate_np.mean()),
            "gate_std": float(gate_np.std()),
            "turbulent_fraction": float((gate_np > 0.5).mean()),
            "laminar_fraction": float((gate_np <= 0.5).mean()),
            "gate_min": float(gate_np.min()),
            "gate_max": float(gate_np.max()),
        },
    }


def batch_predict(
    model: ZonalMoE,
    data_files: list,
    output_dir: str,
    device: str = None,
    vel_mean: np.ndarray = None,
    vel_std: np.ndarray = None,
):
    """Run predictions on multiple files and save results."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for fpath in data_files:
        fpath = Path(fpath)
        data = np.load(fpath)

        vel_pred = predict(
            model,
            vel_in=data["velocity_in"],
            pos=data["pos"],
            idcs_airfoil=data["idcs_airfoil"],
            device=device,
            vel_mean=vel_mean,
            vel_std=vel_std,
        )

        out_path = output_dir / f"{fpath.stem}_pred.npz"
        np.savez(
            out_path,
            velocity_pred=vel_pred,
            velocity_in=data["velocity_in"],
            velocity_out=data["velocity_out"],
            pos=data["pos"],
            idcs_airfoil=data["idcs_airfoil"],
        )
        print(f"Saved {out_path}")

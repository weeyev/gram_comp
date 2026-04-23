"""
diagnose.py — post-training analysis for ZonalMoE
Run from the project root:
    python diagnose.py --checkpoint checkpoints/best_model.pt --data_dir /path/to/data

Outputs:
 1. Checkpoint integrity check
 2. Gate value distribution (laminar vs turbulent fraction per sample)
 3. Per-zone loss breakdown (did the turbulent expert actually help?)
 4. Epoch loss curve (from checkpoint if available)
 5. Worst-N samples by val loss
"""

import argparse
import sys
import numpy as np
import torch
from pathlib import Path
from collections import defaultdict

# ── import your package ──────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))
from models import ZonalMoE, AirfoilDataset, geometry_aware_split
from models.preprocessing import compute_wall_distance, build_knn_graph, compute_geometry_fingerprint, GeometryCache

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ─────────────────────────────────────────────────────────────────────────────
def load(ckpt_path):
    print(f"\n{'='*60}")
    print(f"  1. CHECKPOINT INSPECTION")
    print(f"{'='*60}")
    ckpt = torch.load(ckpt_path, map_location="cpu")
    print(f"  Keys in checkpoint : {list(ckpt.keys())}")
    print(f"  Saved at epoch     : {ckpt.get('epoch', 'N/A')}")
    print(f"  Best val loss      : {ckpt.get('loss', 'N/A'):.6f}")

    sd = ckpt["model_state_dict"]
    print(f"  State-dict entries : {len(sd)}")
    total = sum(v.numel() for v in sd.values())
    print(f"  Total parameters   : {total:,}")

    model = ZonalMoE(
        d_temporal=64,
        backbone_hidden=128,
        laminar_hidden=64,
        turbulent_hidden=256,
        heads=4,
        dropout=0.1,
        output_timesteps=5,
        output_channels=3,
    ).to(DEVICE)
    model.load_state_dict(sd)
    model.eval()
    print(f"  Weights loaded OK  ✓")
    return model


# ─────────────────────────────────────────────────────────────────────────────
def diagnose_gate(model, dataset, indices, n_samples=20):
    print(f"\n{'='*60}")
    print(f"  2. ROUTING GATE DISTRIBUTION  (first {n_samples} val samples)")
    print(f"{'='*60}")

    turb_fracs, lam_fracs, gate_means, gate_stds = [], [], [], []
    cache = GeometryCache()

    for i in indices[:n_samples]:
        sample = dataset[i]
        vel_in    = sample["vel_in"].to(DEVICE)
        wall_dist = sample["wall_dist"].to(DEVICE)
        is_airfoil= sample["is_airfoil"].to(DEVICE)

        vorticity = model.compute_vorticity_proxy(vel_in)

        with torch.no_grad():
            g = model.routing_gate(wall_dist, vorticity).cpu().numpy()

        turb_fracs.append((g > 0.5).mean())
        lam_fracs.append((g <= 0.5).mean())
        gate_means.append(g.mean())
        gate_stds.append(g.std())

    print(f"  {'Metric':<30} {'Mean':>10} {'Std':>10} {'Min':>10} {'Max':>10}")
    print(f"  {'-'*62}")
    for name, arr in [
        ("Gate mean (per sample)", gate_means),
        ("Gate std  (per sample)", gate_stds),
        ("Turbulent fraction (g>0.5)", turb_fracs),
        ("Laminar fraction  (g≤0.5)", lam_fracs),
    ]:
        a = np.array(arr)
        print(f"  {name:<30} {a.mean():>10.4f} {a.std():>10.4f} {a.min():>10.4f} {a.max():>10.4f}")

    # Flag: if turbulent fraction is near 0 or 1 the gate has collapsed
    avg_turb = np.mean(turb_fracs)
    if avg_turb < 0.05:
        print(f"\n  ⚠  Gate collapse → LAMINAR  (avg turbulent fraction {avg_turb:.3f})")
        print(f"     All points routed to laminar expert. Turbulent expert unused.")
    elif avg_turb > 0.95:
        print(f"\n  ⚠  Gate collapse → TURBULENT (avg turbulent fraction {avg_turb:.3f})")
        print(f"     All points routed to turbulent expert. Laminar expert unused.")
    else:
        print(f"\n  ✓  Gate appears healthy — {avg_turb*100:.1f}% turbulent, {(1-avg_turb)*100:.1f}% laminar")


# ─────────────────────────────────────────────────────────────────────────────
def diagnose_zone_loss(model, dataset, indices, n_samples=30):
    print(f"\n{'='*60}")
    print(f"  3. PER-ZONE LOSS BREAKDOWN  (first {n_samples} val samples)")
    print(f"{'='*60}")

    near_losses, far_losses, all_losses = [], [], []
    WALL_PERCENTILE = 20  # bottom 20% of wall dist = "near wall"

    for i in indices[:n_samples]:
        sample   = dataset[i]
        vel_in   = sample["vel_in"].to(DEVICE)
        vel_out  = sample["vel_out"].to(DEVICE)
        wall_dist= sample["wall_dist"].to(DEVICE)
        is_airfoil = sample["is_airfoil"].to(DEVICE)
        edge_index      = sample["edge_index"].to(DEVICE)
        edge_index_dense= sample["edge_index_dense"].to(DEVICE)

        with torch.no_grad():
            vel_pred = model(
                vel_in=vel_in,
                pos=sample["pos"].to(DEVICE),
                edge_index=edge_index,
                wall_dist=wall_dist,
                is_airfoil=is_airfoil,
                edge_index_dense=edge_index_dense,
            )

        # Point-wise MSE: (T, N, 3) -> (N,) mean over T and components
        mse = (vel_pred - vel_out).pow(2).mean(dim=[0, 2])  # (N,)

        # Zone split by wall distance percentile
        threshold = torch.quantile(wall_dist, WALL_PERCENTILE / 100.0)
        near_mask = wall_dist < threshold

        near_losses.append(mse[near_mask].mean().item())
        far_losses.append(mse[~near_mask].mean().item())
        all_losses.append(mse.mean().item())

    near = np.array(near_losses)
    far  = np.array(far_losses)
    tot  = np.array(all_losses)

    print(f"  {'Zone':<25} {'Mean MSE':>12} {'Std':>10} {'Ratio vs far':>14}")
    print(f"  {'-'*63}")
    print(f"  {'Near-wall (bot 20%)':<25} {near.mean():>12.6f} {near.std():>10.6f} {near.mean()/far.mean():>14.3f}x")
    print(f"  {'Far-field (top 80%)':<25} {far.mean():>12.6f} {far.std():>10.6f} {'1.000':>14}")
    print(f"  {'All points':<25} {tot.mean():>12.6f} {tot.std():>10.6f}")

    ratio = near.mean() / far.mean()
    if ratio > 3.0:
        print(f"\n  ⚠  Near-wall loss is {ratio:.1f}x higher than far-field.")
        print(f"     Turbulent expert is not fully closing the gap.")
    elif ratio < 1.5:
        print(f"\n  ✓  Near-wall and far-field losses are close — model handles both well.")
    else:
        print(f"\n  ~  Near-wall loss {ratio:.1f}x higher — some turbulent residual remains (expected).")


# ─────────────────────────────────────────────────────────────────────────────
def worst_samples(model, dataset, indices, n=10):
    print(f"\n{'='*60}")
    print(f"  4. WORST {n} SAMPLES BY VALIDATION LOSS")
    print(f"{'='*60}")

    sample_losses = []

    for i in indices:
        sample   = dataset[i]
        vel_in   = sample["vel_in"].to(DEVICE)
        vel_out  = sample["vel_out"].to(DEVICE)
        wall_dist= sample["wall_dist"].to(DEVICE)
        is_airfoil = sample["is_airfoil"].to(DEVICE)
        edge_index      = sample["edge_index"].to(DEVICE)
        edge_index_dense= sample["edge_index_dense"].to(DEVICE)

        with torch.no_grad():
            vel_pred = model(
                vel_in=vel_in,
                pos=sample["pos"].to(DEVICE),
                edge_index=edge_index,
                wall_dist=wall_dist,
                is_airfoil=is_airfoil,
                edge_index_dense=edge_index_dense,
            )

        loss = (vel_pred - vel_out).pow(2).mean().item()
        sample_losses.append((loss, sample["file"]))

    sample_losses.sort(reverse=True)
    print(f"\n  {'Rank':<6} {'MSE Loss':>12}  File")
    print(f"  {'-'*70}")
    for rank, (loss, fpath) in enumerate(sample_losses[:n], 1):
        print(f"  {rank:<6} {loss:>12.6f}  {Path(fpath).name}")


# ─────────────────────────────────────────────────────────────────────────────
def expert_weight_norms(model):
    print(f"\n{'='*60}")
    print(f"  5. EXPERT WEIGHT NORM COMPARISON")
    print(f"{'='*60}")
    print(f"  (checks if one expert has collapsed to near-zero weights)")
    print()

    for name, expert in [("Laminar", model.laminar_expert), ("Turbulent", model.turbulent_expert)]:
        norms = []
        for pname, p in expert.named_parameters():
            if p.requires_grad:
                norms.append(p.data.norm().item())
        print(f"  {name} expert — {len(norms)} param tensors")
        print(f"    mean weight norm : {np.mean(norms):.4f}")
        print(f"    max  weight norm : {np.max(norms):.4f}")
        print(f"    min  weight norm : {np.min(norms):.6f}")

    print()
    for pname, p in model.routing_gate.named_parameters():
        print(f"  Gate param '{pname}': norm={p.data.norm():.4f}, "
              f"mean={p.data.mean():.4f}, std={p.data.std():.4f}")


# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Diagnose trained ZonalMoE")
    parser.add_argument("--checkpoint", required=True, help="Path to best_model.pt")
    parser.add_argument("--data_dir",   required=True, help="Data directory with .npz files")
    parser.add_argument("--n_gate",     type=int, default=20,  help="Samples for gate analysis")
    parser.add_argument("--n_zone",     type=int, default=30,  help="Samples for zone loss")
    parser.add_argument("--n_worst",    type=int, default=10,  help="Worst N samples to report")
    args = parser.parse_args()

    # Load model
    model = load(args.checkpoint)

    # Load dataset (no re-training, just for data access)
    print(f"\nLoading dataset from {args.data_dir} ...")
    dataset = AirfoilDataset(
        data_dir=args.data_dir,
        k_standard=16, k_dense=32,
        use_cache=True, normalize=True, device="cpu"
    )
    _, val_indices = geometry_aware_split(dataset, val_ratio=0.1, seed=42)
    print(f"Val samples: {len(val_indices)}")

    # Run all diagnostics
    diagnose_gate(model, dataset, val_indices, n_samples=args.n_gate)
    diagnose_zone_loss(model, dataset, val_indices, n_samples=args.n_zone)
    worst_samples(model, dataset, val_indices, n=args.n_worst)
    expert_weight_norms(model)

    print(f"\n{'='*60}")
    print(f"  Diagnosis complete.")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
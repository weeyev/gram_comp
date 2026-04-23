"""
diagnose.py — answers the question: is the model actually learning anything
               beyond trivial persistence, and specifically on turbulent nodes?

Computes:
  1. Persistence baseline error (just copy last input step)
  2. Model error
  3. Both broken out by region: near-wall (turbulent) vs far-field (laminar)
  4. Per-component error (Vx, Vy, Vz separately) — turbulence shows up in Vy/Vz
"""

import glob
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.spatial import cKDTree

# ── config (must match training) ──────────────────────────────
DATA_FILE = "/Users/vihantiwari/Documents/projects/gram_comp/data/1023_24-4.npz"
WEIGHTS_PATH = "/Users/vihantiwari/gat_trained.pt"
K = 16
HIDDEN_DIM = 128
NUM_FEATURES = 19
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NEAR_WALL_THRESH = 0.05  # distance in original units to be "near wall"

# ── normalization ─────────────────────────────────────────────
all_files = sorted(
    [
        f
        for f in glob.glob("/Users/vihantiwari/Documents/projects/gram_comp/data/*.npz")
        if "_edges" not in f
    ]
)
s = np.load(all_files[0])
VEL_MEAN = torch.tensor(
    s["velocity_in"].transpose(1, 0, 2).reshape(-1, 15).astype(np.float32).mean(0),
    device=device,
)
VEL_STD = torch.tensor(
    s["velocity_in"].transpose(1, 0, 2).reshape(-1, 15).astype(np.float32).std(0)
    + 1e-8,
    device=device,
)
POS_MEAN = torch.tensor(s["pos"].astype(np.float32).mean(0), device=device)
POS_STD = torch.tensor(s["pos"].astype(np.float32).std(0) + 1e-8, device=device)


# ── model architecture (must match training) ──────────────────────────────
class GATLayer(nn.Module):
    def __init__(self, in_features, out_features, num_heads, k, dropout=0.1):
        super().__init__()
        self.k = k
        self.num_heads = num_heads
        self.head_dim = out_features // num_heads
        self.scale = self.head_dim**-0.5

        self.W_q = nn.Linear(in_features, out_features, bias=False)
        self.W_k = nn.Linear(in_features, out_features, bias=False)
        self.W_v = nn.Linear(in_features, out_features, bias=False)
        self.W_o = nn.Linear(out_features, out_features)
        self.norm = nn.LayerNorm(out_features)
        self.dropout = nn.Dropout(dropout)

    def forward(self, h, neighbors_gpu):
        N = h.shape[0]
        H, D = self.num_heads, self.head_dim

        Q = self.W_q(h).view(N, H, D)
        K = self.W_k(h)
        V = self.W_v(h)

        K_neighbors = K[neighbors_gpu].view(N, self.k, H, D)
        V_neighbors = V[neighbors_gpu].view(N, self.k, H, D)

        Q_exp = Q.unsqueeze(1)
        attn = (Q_exp * K_neighbors).sum(-1) * self.scale
        attn = F.softmax(attn, dim=1)
        attn = self.dropout(attn)

        out = (attn.unsqueeze(-1) * V_neighbors).sum(1)
        out = out.reshape(N, H * D)
        out = self.W_o(out)

        return self.norm(h + out) if h.shape[-1] == out.shape[-1] else self.norm(out)


class FeedForward(nn.Module):
    def __init__(self, dim, mult=2, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * mult, dim),
            nn.Dropout(dropout),
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        return self.norm(x + self.net(x))


class GATModel(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_dim,
        out_features,
        k,
        num_layers=4,
        heads=8,
        dropout=0.1,
    ):
        super().__init__()
        self.k = k

        self.input_proj = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(
                nn.ModuleList(
                    [
                        GATLayer(hidden_dim, hidden_dim, heads, k, dropout),
                        FeedForward(hidden_dim, mult=2, dropout=dropout),
                    ]
                )
            )

        self.output_proj = nn.Linear(hidden_dim, out_features)

    def forward(self, x, neighbors_gpu):
        h = self.input_proj(x)
        for gat, ff in self.layers:
            h = gat(h, neighbors_gpu)
            h = ff(h)
        return self.output_proj(h)


# Initialize and load weights
print("Loading model...")
model = GATModel(
    in_features=NUM_FEATURES,
    hidden_dim=HIDDEN_DIM,
    out_features=15,
    k=K,
    num_layers=4,
    heads=8,
    dropout=0.1,
).to(device)

state_dict = torch.load(WEIGHTS_PATH, map_location=device)
model.load_state_dict(state_dict)
model.eval()
print(f"✓ Loaded trained weights from {WEIGHTS_PATH}")

# ── data ──────────────────────────────────────────────────────
d = np.load(DATA_FILE)
pos = d["pos"].astype(np.float32)  # (N, 3) physical
idcs_airfoil = d["idcs_airfoil"]
vel_in_phys = d["velocity_in"].transpose(1, 0, 2)  # (N, 5, 3) physical
vel_out_phys = d["velocity_out"].transpose(1, 0, 2).astype(np.float32)  # (N, 5, 3)
N = pos.shape[0]

# normalized features for model
vel_in_norm = (
    vel_in_phys.reshape(-1, 15).astype(np.float32) - VEL_MEAN.cpu().numpy()
) / VEL_STD.cpu().numpy()
pos_norm = (pos - POS_MEAN.cpu().numpy()) / POS_STD.cpu().numpy()
is_surface = np.zeros((N, 1), dtype=np.float32)
is_surface[idcs_airfoil] = 1.0
nf = np.concatenate([vel_in_norm, pos_norm, is_surface], axis=1)

# KNN using cKDTree with k=16
tree = cKDTree(pos)
_, neighbors = tree.query(pos, k=K + 1, workers=-1)
neighbors = neighbors[:, 1:].astype(np.int64)  # (N, k)

# ── model prediction ──────────────────────────────────────────
nf_gpu = torch.tensor(nf, device=device, dtype=torch.float32)
neighbors_gpu = torch.tensor(neighbors, device=device, dtype=torch.long)

with torch.no_grad():
    pred_norm = model(nf_gpu, neighbors_gpu).cpu().numpy()

pred_phys = (pred_norm * VEL_STD.cpu().numpy() + VEL_MEAN.cpu().numpy()).reshape(
    N, 5, 3
)

# ── persistence baseline: repeat last input step ──────────────
vel_in_last = vel_in_phys[:, -1, :].astype(np.float32)  # (N, 3)
persistence = np.tile(vel_in_last[:, None, :], (1, 5, 1))  # (N, 5, 3)

# ── wall distances ────────────────────────────────────────────
tree = cKDTree(pos[idcs_airfoil])
wall_dist, _ = tree.query(pos, k=1)
near_wall = wall_dist < NEAR_WALL_THRESH
far_field = ~near_wall
print(f"Near-wall nodes: {near_wall.sum()} / {N} ({100 * near_wall.mean():.1f}%)")


# ── error computation (physical units) ───────────────────────
def l2_err(pred, gt):
    """(N, 5, 3) → (N, 5)"""
    return np.linalg.norm(pred - gt, axis=2)


model_err = l2_err(pred_phys, vel_out_phys)  # (N, 5)
persist_err = l2_err(persistence, vel_out_phys)  # (N, 5)

# ── summary table ─────────────────────────────────────────────
print("\n" + "=" * 60)
print(f"{'Region':<20} {'Model err':>12} {'Persist err':>12} {'Improvement':>12}")
print("=" * 60)

regions = [
    ("ALL nodes", np.ones(N, bool)),
    ("Far-field", far_field),
    ("Near-wall", near_wall),
]

for name, mask in regions:
    me = model_err[mask].mean()
    pe = persist_err[mask].mean()
    imp = (pe - me) / pe * 100
    print(f"{name:<20} {me:>12.4f} {pe:>12.4f} {imp:>+11.1f}%")

print("=" * 60)

# ── per-output-timestep ───────────────────────────────────────
print("\nModel vs Persistence error per output timestep (near-wall only):")
print(f"{'t':>4} {'Model':>10} {'Persist':>10} {'Δ%':>8}")
for t in range(5):
    me = model_err[near_wall, t].mean()
    pe = persist_err[near_wall, t].mean()
    print(f"{t:>4} {me:>10.4f} {pe:>10.4f} {(pe - me) / pe * 100:>+7.1f}%")

# ── per velocity component (near-wall) ───────────────────────
print("\nComponent-wise error near wall (physical m/s):")
comp_names = ["Vx (stream)", "Vy (span)", "Vz (normal)"]
for ci, cname in enumerate(comp_names):
    m_comp = np.abs(pred_phys[near_wall, :, ci] - vel_out_phys[near_wall, :, ci]).mean()
    p_comp = np.abs(
        persistence[near_wall, :, ci] - vel_out_phys[near_wall, :, ci]
    ).mean()
    print(
        f"  {cname}: model={m_comp:.4f}  persist={p_comp:.4f}  Δ={p_comp - m_comp:+.4f}"
    )

# ── quick verdict ─────────────────────────────────────────────
overall_imp = (persist_err.mean() - model_err.mean()) / persist_err.mean() * 100
nw_imp = (
    (persist_err[near_wall].mean() - model_err[near_wall].mean())
    / persist_err[near_wall].mean()
    * 100
)
print(f"\n{'=' * 60}")
print(f"VERDICT:")
print(f"  Overall improvement over persistence: {overall_imp:+.1f}%")
print(f"  Near-wall improvement over persistence: {nw_imp:+.1f}%")
if nw_imp < 5:
    print("  ⚠ Near-wall: model barely beats copying the last input step.")
    print("    The turbulent component is NOT being learned meaningfully.")
elif nw_imp < 20:
    print("  ~ Near-wall: modest improvement. Model learns some dynamics but")
    print("    likely smoothing over the high-frequency turbulent fluctuations.")
else:
    print("  ✓ Near-wall: meaningful improvement — model is learning dynamics.")
print(f"{'=' * 60}")

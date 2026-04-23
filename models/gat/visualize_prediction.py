import glob
import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.spatial import cKDTree
from tqdm import tqdm

# Config
k = 16
hidden_dim = 128
num_features = 19
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_FILE = "/Users/vihantiwari/Documents/projects/gram_comp/data/1023_24-4.npz"
WEIGHTS_FILE = "/Users/vihantiwari/gat_trained.pt"

# Normalization stats (need to match training)
all_files = sorted(
    [
        f
        for f in glob.glob("/Users/vihantiwari/Documents/projects/gram_comp/data/*.npz")
        if "_edges" not in f
    ]
)
sample = np.load(all_files[0])
vel_sample = sample["velocity_in"].transpose(1, 0, 2).reshape(-1, 15).astype(np.float32)
vel_mean = torch.tensor(vel_sample.mean(axis=0), device=device)
vel_std = torch.tensor(vel_sample.std(axis=0) + 1e-8, device=device)
pos_sample = sample["pos"].astype(np.float32)
pos_mean = torch.tensor(pos_sample.mean(axis=0), device=device)
pos_std = torch.tensor(pos_sample.std(axis=0) + 1e-8, device=device)


def build_graph_and_features(fpath):
    d = np.load(fpath)
    pos = d["pos"].astype(np.float32)
    idcs_airfoil = d["idcs_airfoil"]

    # KNN using scipy cKDTree with k=16 (same as training)
    tree = cKDTree(pos)
    _, neighbors = tree.query(pos, k=k + 1, workers=-1)
    neighbors = neighbors[:, 1:].astype(np.int64)  # (N, k)

    vel_in = d["velocity_in"].transpose(1, 0, 2).reshape(-1, 15).astype(np.float32)
    vel_out = d["velocity_out"].transpose(1, 0, 2).reshape(-1, 15).astype(np.float32)
    is_surface = np.zeros((pos.shape[0], 1), dtype=np.float32)
    is_surface[idcs_airfoil] = 1.0
    vel_in_norm = (vel_in - vel_mean.cpu().numpy()) / vel_std.cpu().numpy()
    vel_out_norm = (vel_out - vel_mean.cpu().numpy()) / vel_std.cpu().numpy()
    pos_norm = (pos - pos_mean.cpu().numpy()) / pos_std.cpu().numpy()
    nf = np.concatenate([vel_in_norm, pos_norm, is_surface], axis=1)
    return nf, vel_out_norm, neighbors, pos.shape[0], pos, idcs_airfoil


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


print("Loading model...")
model = GATModel(
    in_features=num_features,
    hidden_dim=hidden_dim,
    out_features=15,
    k=k,
    num_layers=4,
    heads=8,
    dropout=0.1,
).to(device)

# Load trained weights
state_dict = torch.load(WEIGHTS_FILE, map_location=device)
model.load_state_dict(state_dict)
model.eval()
print(f"Loaded trained weights from {WEIGHTS_FILE}")

print(f"Loading test data from {DATA_FILE}...")
nf, tf, neighbors, num_nodes, pos, idcs_airfoil = build_graph_and_features(DATA_FILE)

# Convert to tensors
nf_gpu = torch.tensor(nf, device=device, dtype=torch.float32)
neighbors_gpu = torch.tensor(neighbors, device=device, dtype=torch.long)

print("Running prediction...")
with torch.no_grad():
    pred = model(nf_gpu, neighbors_gpu)

# Move to CPU and denormalize
pred_np = pred.cpu().numpy()  # (N, 15) = (100000, 15)
pred_denorm = pred_np * vel_std.cpu().numpy() + vel_mean.cpu().numpy()  # (N, 15)
pred_denorm = pred_denorm.reshape(5, 100000, 3)  # (5, N, 3)
truth_denorm = tf * vel_std.cpu().numpy() + vel_mean.cpu().numpy()
truth_denorm = truth_denorm.reshape(5, 100000, 3)

# Calculate error
error = np.abs(pred_denorm - truth_denorm)
mean_error = error.mean()
print(f"Mean error: {mean_error:.4f}")

# Visualizations
print("Creating visualizations...")

# 1. XY slice comparison
fig, axes = plt.subplots(2, 5, figsize=(20, 8))
for t in range(5):
    mask = (pos[:, 2] > -0.05) & (pos[:, 2] < 0.05)
    axes[0, t].scatter(
        pos[mask, 0], pos[mask, 1], c=pred_denorm[t, mask, 0], s=0.5, cmap="coolwarm"
    )
    axes[0, t].set_title(f"Pred t={t}")
    axes[1, t].scatter(
        pos[mask, 0], pos[mask, 1], c=truth_denorm[t, mask, 0], s=0.5, cmap="coolwarm"
    )
    axes[1, t].set_title(f"Truth t={t}")
plt.suptitle(f"XY Slice - Velocity X (Mean Error: {mean_error:.4f})")
plt.tight_layout()
plt.savefig("visualization_xy.png", dpi=150)
plt.close()
print("Saved: visualization_xy.png")

# 2. Surface check
fig, ax = plt.subplots(1, 1, figsize=(8, 6))
surf_mask = np.zeros(num_nodes, dtype=bool)
surf_mask[idcs_airfoil] = True
pred_surf = pred_denorm[:, surf_mask, :].mean()
truth_surf = truth_denorm[:, surf_mask, :].mean()
ax.bar(["Predicted", "Truth"], [pred_surf, truth_surf])
ax.set_title("Surface Velocity (should be ~0)")
plt.savefig("visualization_surface.png", dpi=150)
plt.close()
print("Saved: visualization_surface.png")

# 3. Error histogram
fig, ax = plt.subplots(1, 1, figsize=(10, 6))
flat_error = error.flatten()
ax.hist(flat_error, bins=50, alpha=0.7)
ax.set_xlabel("Absolute Error")
ax.set_ylabel("Count")
ax.set_title(f"Error Distribution (Mean: {mean_error:.4f})")
plt.savefig("visualization_error_hist.png", dpi=150)
plt.close()
print("Saved: visualization_error_hist.png")

# 4. 3D error map
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection="3d")
t_idx = 2  # middle timestep
mask = np.random.choice(num_nodes, size=min(5000, num_nodes), replace=False)
error_t = error[t_idx, mask]
# Normalize error for color mapping
error_t_normalized = (error_t - error_t.min()) / (error_t.max() - error_t.min() + 1e-8)
sc = ax.scatter(
    pos[mask, 0], pos[mask, 1], pos[mask, 2], c=error_t_normalized, s=1, cmap="hot"
)
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
plt.colorbar(sc, label="Normalized Error")
ax.set_title(f"3D Error Map (t={t_idx})")
plt.savefig("visualization_3d_error.png", dpi=150)
plt.close()
print("Saved: visualization_3d_error.png")

print("""
=== VISUALIZATION COMPLETE ===
Check these files:
  - visualization_xy.png: Velocity comparison at cross-section
  - visualization_surface.png: Boundary condition check
  - visualization_error_hist.png: Error distribution
  - visualization_3d_error.png: 3D error map
""")

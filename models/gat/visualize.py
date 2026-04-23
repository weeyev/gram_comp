"""
Visualization script for GAT model predictions.
Run this AFTER training completes with saved weights.
"""

import glob
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.neighbors import kneighbors_graph
import tinygrad
from tinygrad import Tensor, nn

# Config
NUM_FEATURES = 19
HIDDEN_DIM = 64
K = 5
DATA_FILE = "/Users/vihantiwari/Documents/projects/gram_comp/data/1023_24-4.npz"  # Example test file
WEIGHTS_FILE = (
    "/Users/vihantiwari/Documents/projects/gram_comp/models/gat/gat_trained.npy"
)

# Normalization stats (must match training)
all_files = sorted(
    glob.glob("/Users/vihantiwari/Documents/projects/gram_comp/data/*.npz")
)
sample = np.load(all_files[0])
vel_sample = sample["velocity_in"].transpose(1, 0, 2).reshape(-1, 15).astype(np.float32)
vel_mean = vel_sample.mean(axis=0)
vel_std = vel_sample.std(axis=0) + 1e-8
pos_sample = sample["pos"].astype(np.float32)
pos_mean = pos_sample.mean(axis=0)
pos_std = pos_sample.std(axis=0) + 1e-8


def build_graph(fpath):
    d = np.load(fpath)
    pos = d["pos"].astype(np.float32)
    idcs_airfoil = d["idcs_airfoil"]
    vel_in = d["velocity_in"].transpose(1, 0, 2).reshape(-1, 15).astype(np.float32)
    vel_out = d["velocity_out"].transpose(1, 0, 2).reshape(-1, 15).astype(np.float32)
    is_surface = np.zeros((pos.shape[0], 1), dtype=np.float32)
    is_surface[idcs_airfoil] = 1.0
    vel_in_norm = (vel_in - vel_mean) / vel_std
    vel_out_norm = (vel_out - vel_mean) / vel_std
    pos_norm = (pos - pos_mean) / pos_std
    nf = np.concatenate([vel_in_norm, pos_norm, is_surface], axis=1)
    knn = kneighbors_graph(pos, n_neighbors=K, mode="connectivity", metric="euclidean")
    rows, cols = knn.nonzero()
    rows = rows.astype(np.int64)
    cols = cols.astype(np.int64)
    return nf, vel_out_norm, rows, cols, pos.shape[0], pos, idcs_airfoil


# Model architecture (same as training)
class GATLayers:
    def __init__(
        self, node_features, target_features, num_heads, concat, k, dropout=0.2
    ):
        self.k = k
        self.concat = concat
        self.num_heads = num_heads
        if concat:
            self.num_hidden = target_features // num_heads
        else:
            self.num_hidden = target_features
        self.linear = nn.Linear(
            node_features, self.num_heads * self.num_hidden, bias=False
        )
        self.attention = nn.Linear(self.num_hidden * 2, 1, bias=False)

    def __call__(self, h, rows, cols):
        num_nodes = h.shape[0]
        h = h.dropout(p=0.2)
        transform = self.linear(h).view(num_nodes, self.num_heads, self.num_hidden)
        rows_t = Tensor(rows) if isinstance(rows, np.ndarray) else rows
        cols_t = Tensor(cols) if isinstance(cols, np.ndarray) else cols
        h_source = transform[rows_t]
        h_target = transform[cols_t]
        h_source_reshaped = h_source.reshape(
            num_nodes, self.k, self.num_heads, self.num_hidden
        )
        h_target_reshaped = h_target.reshape(
            num_nodes, self.k, self.num_heads, self.num_hidden
        )
        edge_feat = h_source_reshaped.cat(h_target_reshaped, dim=-1)
        e_ij = self.attention(edge_feat).leaky_relu(0.2).squeeze(-1)
        alpha = e_ij.softmax(axis=1).dropout(0.1)
        out = (h_target_reshaped * alpha.unsqueeze(-1)).sum(axis=1)
        if self.concat:
            return out.reshape(num_nodes, self.num_heads * self.num_hidden)
        else:
            return out.mean(axis=1)


class GATModel:
    def __init__(
        self, node_features, hidden_features, target_features, k, heads=8, dropout=0.2
    ):
        self.k = k
        self.layer1 = GATLayers(
            node_features, hidden_features, num_heads=heads, concat=True, k=k
        )
        self.layer2 = GATLayers(
            hidden_features, target_features, num_heads=1, concat=False, k=k
        )

    def __call__(self, x, rows, cols):
        x = self.layer1(x, rows, cols).elu()
        return self.layer2(x, rows, cols)


# Load model and weights
print("Loading model...")
model = GATModel(NUM_FEATURES, HIDDEN_DIM, 15, k=5, heads=8, dropout=0.2)

try:
    saved = np.load(WEIGHTS_FILE, allow_pickle=True).item()
    model.layer1.linear.weight = Tensor(saved["layer1.linear.weight"])
    model.layer1.attention.weight = Tensor(saved["layer1.attention.weight"])
    model.layer2.linear.weight = Tensor(saved["layer2.linear.weight"])
    model.layer2.attention.weight = Tensor(saved["layer2.attention.weight"])
    print(f"Loaded trained weights from {WEIGHTS_FILE}")
except Exception as e:
    print(f"WARNING: Could not load weights: {e}")
    print("Using random initialization!")

# Load test data
print(f"Loading {DATA_FILE}...")
nf, tf, rows, cols, num_nodes, pos, idcs_airfoil = build_graph(DATA_FILE)

# Run prediction
print("Running forward pass...")
nf_tensor = Tensor(nf)
pred = model(nf_tensor, rows, cols)
pred_np = pred.numpy()

# Reshape: (100k, 15) -> (5, 100k, 3)
pred_velocity = pred_np.reshape(5, num_nodes, 3)
gt_velocity = tf.reshape(5, num_nodes, 3)

# Error
error = np.linalg.norm(pred_velocity - gt_velocity, axis=2)
print(f"Mean error: {error.mean():.4f}")

# ===== VISUALIZATIONS =====

# 1. Cross-section velocity comparison
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
x_mid = pos[:, 0].mean()
mask = np.abs(pos[:, 0] - x_mid) < 5

for row, t in enumerate([0, 4]):
    pm = np.linalg.norm(pred_velocity[t], axis=1)
    gm = np.linalg.norm(gt_velocity[t], axis=1)
    axes[row, 0].scatter(pos[mask, 1], pos[mask, 2], c=gm[mask], s=1, cmap="viridis")
    axes[row, 0].set_title(f"GT velocity (t={t})")
    axes[row, 1].scatter(pos[mask, 1], pos[mask, 2], c=pm[mask], s=1, cmav="viridis")
    axes[row, 1].set_title(f"Pred velocity (t={t})")
    axes[row, 2].scatter(pos[mask, 1], pos[mask, 2], c=error[t, mask], s=1, cmap="Reds")
    axes[row, 2].set_title(f"Error (t={t})")
plt.tight_layout()
plt.savefig(
    "/Users/vihantiwari/Documents/projects/gram_comp/models/gat/vis_xy.png", dpi=150
)
print("Saved: vis_xy.png")
plt.close()

# 2. Boundary condition check
fig, ax = plt.subplots(1, 1, figsize=(8, 6))
pred_surf = np.linalg.norm(pred_velocity[:, idcs_airfoil], axis=2).mean(axis=1)
gt_surf = np.linalg.norm(gt_velocity[:, idcs_airfoil], axis=2).mean(axis=1)
ax.plot(range(5), gt_surf, "b-o", label="GT at surface")
ax.plot(range(5), pred_surf, "r-o", label="Pred at surface")
ax.set_xlabel("Time step")
ax.set_ylabel("Mean velocity")
ax.set_title("Boundary: velocity at airfoil surface (should be ~0)")
ax.legend()
ax.grid(True)
plt.savefig(
    "/Users/vihantiwari/Documents/projects/gram_comp/models/gat/vis_surface.png",
    dpi=150,
)
print("Saved: vis_surface.png")
plt.close()

# 3. Error histogram
fig, ax = plt.subplots(1, 1, figsize=(8, 6))
flat = error.flatten()
ax.hist(flat, bins=50, alpha=0.7, edgecolor="black")
ax.axvline(flat.mean(), color="r", linestyle="--", label=f"Mean: {flat.mean():.3f}")
ax.axvline(
    np.median(flat),
    color="orange",
    linestyle="--",
    label=f"Median: {np.median(flat):.3f}",
)
ax.set_xlabel("Velocity error")
ax.set_ylabel("Count")
ax.set_title("Error distribution")
ax.legend()
plt.savefig(
    "/Users/vihantiwari/Documents/projects/gram_comp/models/gat/vis_hist.png", dpi=150
)
print("Saved: vis_hist.png")
plt.close()

# 4. 3D error
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
scatter = ax.scatter(
    pos[:, 0], pos[:, 1], pos[:, 2], c=error.mean(axis=0), s=0.5, cmap="Reds"
)
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.set_title("3D error distribution")
plt.colorbar(scatter, ax=ax, label="Mean error")
plt.savefig(
    "/Users/vihantiwari/Documents/projects/gram_comp/models/gat/vis_3d.png", dpi=150
)
print("Saved: vis_3d.png")
plt.close()

print("\n=== DONE ===")
print("Check: vis_xy.png, vis_surface.png, vis_hist.png, vis_3d.png")

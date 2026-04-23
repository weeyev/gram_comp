import os
import glob
import time
import threading
import queue
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.spatial import cKDTree
from tqdm import tqdm

# Use CUDA with mixed precision
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Enable TF32 for faster matmul on Ampere+
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# Hyperparameters
hidden_dim = 128
num_features = 19
k = 16
num_epochs = 15
lr = 1e-4

total_files = sorted([f for f in glob.glob("/root/data/*.npz") if "_edges" not in f])

# Normalization stats
sample = np.load(total_files[0])
np.random.seed(42)
np.random.shuffle(total_files)
val_split = int(0.1 * len(total_files))
train_files = total_files[val_split:]
val_files = total_files[:val_split]
print(f"Split: {len(train_files)} train, {len(val_files)} val files")

vel_sample = sample["velocity_in"].transpose(1, 0, 2).reshape(-1, 15).astype(np.float32)
vel_mean = torch.tensor(vel_sample.mean(axis=0), device=device)
vel_std = torch.tensor(vel_sample.std(axis=0) + 1e-8, device=device)
pos_sample = sample["pos"].astype(np.float32)
pos_mean = torch.tensor(pos_sample.mean(axis=0), device=device)
pos_std = torch.tensor(pos_sample.std(axis=0) + 1e-8, device=device)


class DataPrefetcher:
    """Background thread prefetcher to hide disk I/O."""

    def __init__(self, file_list, k, vel_mean, vel_std, pos_mean, pos_std):
        self.file_list = file_list
        self.k = k
        self.vel_mean = vel_mean.cpu().numpy()
        self.vel_std = vel_std.cpu().numpy()
        self.pos_mean = pos_mean.cpu().numpy()
        self.pos_std = pos_std.cpu().numpy()
        self.queue = queue.Queue(maxsize=4)
        self.stop_event = threading.Event()
        self.thread = None

    def _worker(self):
        for fpath in self.file_list:
            if self.stop_event.is_set():
                break

            d = np.load(fpath)
            if "pos" not in d or "velocity_in" not in d:
                continue

            pos = d["pos"].astype(np.float32)
            edge_path = fpath.replace(".npz", f"_edges_k{self.k}.npz")

            # KNN with scipy parallel
            if os.path.exists(edge_path):
                edge_data = np.load(edge_path)
                neighbors = edge_data["neighbors"]
            else:
                tree = cKDTree(pos)
                _, neighbors = tree.query(pos, k=self.k + 1, workers=-1)
                neighbors = neighbors[:, 1:].astype(np.int64)
                np.savez(edge_path, neighbors=neighbors)

            # Prepare features
            vel_in = (
                d["velocity_in"].transpose(1, 0, 2).reshape(-1, 15).astype(np.float32)
            )
            vel_out = (
                d["velocity_out"].transpose(1, 0, 2).reshape(-1, 15).astype(np.float32)
            )
            vel_in_norm = (vel_in - self.vel_mean) / self.vel_std
            vel_out_norm = (vel_out - self.vel_mean) / self.vel_std
            pos_norm = (pos - self.pos_mean) / self.pos_std
            is_surface = np.zeros((pos.shape[0], 1), dtype=np.float32)
            is_surface[d["idcs_airfoil"]] = 1.0
            nf = np.concatenate([vel_in_norm, pos_norm, is_surface], axis=1)

            self.queue.put((nf, vel_out_norm, neighbors))

        self.queue.put(None)  # Signal end

    def start(self):
        self.stop_event.clear()
        self.thread = threading.Thread(target=self._worker, daemon=True)
        self.thread.start()

    def get(self):
        return self.queue.get()

    def stop(self):
        self.stop_event.set()
        while not self.queue.empty():
            try:
                self.queue.get_nowait()
            except:
                pass


class GATLayer(nn.Module):
    """GAT layer with local attention over k neighbors."""

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
        """
        h: (N, in_features)
        neighbors_gpu: (N, k) - PRE-CONVERTED to GPU tensor, passed in
        """
        N = h.shape[0]
        H, D = self.num_heads, self.head_dim

        Q = self.W_q(h).view(N, H, D)
        K = self.W_k(h)
        V = self.W_v(h)

        # Gather neighbors - dense (N, k) format = single contiguous read
        K_neighbors = K[neighbors_gpu].view(N, self.k, H, D)
        V_neighbors = V[neighbors_gpu].view(N, self.k, H, D)

        # Local attention
        Q_exp = Q.unsqueeze(1)  # (N, 1, H, D)
        attn = (Q_exp * K_neighbors).sum(-1) * self.scale  # (N, k, H)
        attn = F.softmax(attn, dim=1)
        attn = self.dropout(attn)

        # Aggregate
        out = (attn.unsqueeze(-1) * V_neighbors).sum(1)  # (N, H, D)
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
        """neighbors_gpu is already on GPU - no allocation/transfer here."""
        h = self.input_proj(x)

        for gat, ff in self.layers:
            h = gat(h, neighbors_gpu)
            h = ff(h)

        return self.output_proj(h)


# Initialize model
model = GATModel(
    in_features=num_features,
    hidden_dim=hidden_dim,
    out_features=15,
    k=k,
    num_layers=4,
    heads=8,
    dropout=0.1,
).to(device)

print(f"Model: 4 layers, hidden={hidden_dim}, k={k}, heads=8")
print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

# Mixed precision scaler
scaler = torch.cuda.amp.GradScaler()
optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

# Save initial weights
torch.save(model.state_dict(), "/root/gram_comp/models/gat/gat_trained.pt")
print("Saved initial weights")

print(
    f"Training on {len(train_files)} files (PyTorch, k={k}, mixed precision, prefetching)"
)

for epoch in range(num_epochs):
    model.train()
    total_loss = 0.0
    step_count = 0
    np.random.shuffle(train_files)
    epoch_start = time.time()

    # Start prefetcher
    prefetcher = DataPrefetcher(train_files, k, vel_mean, vel_std, pos_mean, pos_std)
    prefetcher.start()

    pbar = tqdm(total=len(train_files), desc=f"Epoch {epoch}")

    while True:
        data = prefetcher.get()
        if data is None:
            break

        nf, tf, neighbors = data

        # Transfer to GPU ONCE outside model
        nf_gpu = torch.tensor(nf, device=device, dtype=torch.float32)
        tf_gpu = torch.tensor(tf, device=device, dtype=torch.float32)
        neighbors_gpu = torch.tensor(neighbors, device=device, dtype=torch.long)

        optimizer.zero_grad(set_to_none=True)

        # Mixed precision forward/backward
        with torch.cuda.amp.autocast():
            pred = model(nf_gpu, neighbors_gpu)
            loss = F.mse_loss(pred, tf_gpu)

        scaler.scale(loss).backward()

        # Gradient clipping
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        scaler.step(optimizer)
        scaler.update()

        # Accumulate loss WITHOUT sync (stays on GPU)
        total_loss += loss.detach()
        step_count += 1
        pbar.update(1)

        # Only sync and print periodically
        if step_count % 100 == 0:
            avg_loss = total_loss.item() / step_count
            pbar.set_postfix({"loss": f"{avg_loss:.4f}"})

    pbar.close()
    prefetcher.stop()

    epoch_time = time.time() - epoch_start
    avg_train_loss = total_loss.item() / step_count if step_count > 0 else 0
    print(f"Epoch {epoch} | train loss: {avg_train_loss:.6f} | time: {epoch_time:.1f}s")

    # Validation
    model.eval()
    val_loss = 0.0
    val_steps = 0

    with torch.no_grad():
        for vfpath in tqdm(val_files, desc=f"Epoch {epoch} val"):
            vd = np.load(vfpath)
            if "pos" not in vd or "velocity_in" not in vd:
                continue

            pos = vd["pos"].astype(np.float32)
            edge_path = vfpath.replace(".npz", f"_edges_k{k}.npz")

            if os.path.exists(edge_path):
                edge_data = np.load(edge_path)
                vneighbors = edge_data["neighbors"]
            else:
                tree = cKDTree(pos)
                _, vneighbors = tree.query(pos, k=k + 1, workers=-1)
                vneighbors = vneighbors[:, 1:].astype(np.int64)
                np.savez(edge_path, neighbors=vneighbors)

            vel_in = (
                vd["velocity_in"].transpose(1, 0, 2).reshape(-1, 15).astype(np.float32)
            )
            vel_out = (
                vd["velocity_out"].transpose(1, 0, 2).reshape(-1, 15).astype(np.float32)
            )

            vnf = np.concatenate(
                [
                    (vel_in - vel_mean.cpu().numpy()) / vel_std.cpu().numpy(),
                    (pos - pos_mean.cpu().numpy()) / pos_std.cpu().numpy(),
                    np.where(
                        np.isin(np.arange(pos.shape[0]), vd["idcs_airfoil"]), 1.0, 0.0
                    )
                    .reshape(-1, 1)
                    .astype(np.float32),
                ],
                axis=1,
            )
            vtf = (vel_out - vel_mean.cpu().numpy()) / vel_std.cpu().numpy()

            vnf_gpu = torch.tensor(vnf, device=device, dtype=torch.float32)
            vtf_gpu = torch.tensor(vtf, device=device, dtype=torch.float32)
            vneighbors_gpu = torch.tensor(vneighbors, device=device, dtype=torch.long)

            with torch.cuda.amp.autocast():
                vpred = model(vnf_gpu, vneighbors_gpu)
                vloss = F.mse_loss(vpred, vtf_gpu)

            val_loss += vloss.item()
            val_steps += 1

    avg_val_loss = val_loss / val_steps if val_steps > 0 else 0
    print(f"Epoch {epoch} | train: {avg_train_loss:.6f} | val: {avg_val_loss:.6f}")

    # Save checkpoint
    torch.save(model.state_dict(), "/root/gram_comp/models/gat/gat_trained.pt")
    print(f"Saved checkpoint after epoch {epoch}")

print("Training complete!")

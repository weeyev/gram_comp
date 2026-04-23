import os
import glob
import tinygrad
from tinygrad import Tensor, nn
import numpy as np
from sklearn.neighbors import kneighbors_graph
from tinygrad import TinyJit
from tinygrad.nn.optim import AdamW

# node features: 15 velocity + 3 pos + 1 is_surface = 19
num_features = 19
hidden_dim = 64
k = 5
BATCH_NODES = 4096
STEPS_PER_FILE = 2
all_files = sorted(
    glob.glob("/Users/vihantiwari/Documents/projects/gram_comp/data/*.npz")
)

# normalization stats from first file
sample = np.load(all_files[0])
vel_sample = sample["velocity_in"].transpose(1, 0, 2).reshape(-1, 15).astype(np.float32)
vel_mean = vel_sample.mean(axis=0)
vel_std = vel_sample.std(axis=0) + 1e-8
pos_sample = sample["pos"].astype(np.float32)
pos_mean = pos_sample.mean(axis=0)
pos_std = pos_sample.std(axis=0) + 1e-8


def build_graph_and_features(fpath):
    d = np.load(fpath)
    pos = d["pos"].astype(np.float32)
    idcs_airfoil = d["idcs_airfoil"]
    vel_in = d["velocity_in"].transpose(1, 0, 2).reshape(-1, 15).astype(np.float32)
    vel_out = d["velocity_out"].transpose(1, 0, 2).reshape(-1, 15).astype(np.float32)
    # is_surface flag
    is_surface = np.zeros((pos.shape[0], 1), dtype=np.float32)
    is_surface[idcs_airfoil] = 1.0
    # normalize and concat: vel(15) + pos(3) + is_surface(1) = 19
    vel_in_norm = (vel_in - vel_mean) / vel_std
    vel_out_norm = (vel_out - vel_mean) / vel_std
    pos_norm = (pos - pos_mean) / pos_std
    nf = np.concatenate([vel_in_norm, pos_norm, is_surface], axis=1)
    tf = vel_out_norm
    # build knn graph per file
    knn = kneighbors_graph(pos, n_neighbors=k, mode="connectivity", metric="euclidean")
    rows, cols = knn.nonzero()
    rows = rows.astype(np.int64)
    cols = cols.astype(np.int64)
    sort_idx = np.argsort(rows)
    rows, cols = rows[sort_idx], cols[sort_idx]
    return nf, tf, rows, cols, pos.shape[0]

def sample_subgraph(rows, cols, k, num_nodes, batch_size, nf_np, tf_np):
    sampled = np.random.choice(num_nodes, size=batch_size, replace=False)
    remap = {old: new for new, old in enumerate(sampled)}
    sub_rows, sub_cols = [], []
    for new_i, old_i in enumerate(sampled):
        neighbors = [remap[n] for n in cols[rows == old_i] if n in remap]
        while len(neighbors) < k:
            neighbors.append(new_i)
        sub_rows.extend([new_i] * k)
        sub_cols.extend(neighbors[:k])
    return (
        Tensor(nf_np[sampled]),
        Tensor(tf_np[sampled]),
        np.array(sub_rows, dtype=np.int64),
        np.array(sub_cols, dtype=np.int64),
    )

class GATLayers:
    def __init__(
        self, node_features, target_features, num_heads, concat, k, dropout=0.6
    ):
        self.k = k
        self.concat = concat
        self.num_heads = num_heads
        if concat:
            assert target_features % num_heads == 0
            self.num_hidden = target_features // num_heads
        else:
            self.num_hidden = target_features
        self.linear = nn.Linear(
            node_features, self.num_heads * self.num_hidden, bias=False
        )
        self.attention = nn.Linear(self.num_hidden * 2, 1, bias=False)

    def __call__(self, h, rows, cols):
        num_nodes = h.shape[0]
        h = h.dropout(p=0.1)
        transform = self.linear(h).view(num_nodes, self.num_heads, self.num_hidden)
        rows_t = Tensor(rows) if isinstance(rows, np.ndarray) else rows
        cols_t = Tensor(cols) if isinstance(cols, np.ndarray) else cols
        h_source = transform[rows_t]
        h_target = transform[cols_t]
        h_sourced_reshaped = h_source.reshape(
            num_nodes, self.k, self.num_heads, self.num_hidden
        )
        h_target_reshaped = h_target.reshape(
            num_nodes, self.k, self.num_heads, self.num_hidden
        )
        edge_feat = h_sourced_reshaped.cat(h_target_reshaped, dim=-1)
        e_ij = self.attention(edge_feat).leaky_relu(0.2)
        e_ij = e_ij.squeeze(-1)
        alpha = e_ij.softmax(axis=1)
        alpha = alpha.dropout(0.1)
        alpha_exp = alpha.unsqueeze(-1)
        out = (h_target_reshaped * alpha_exp).sum(axis=1)
        if self.concat:
            return out.reshape(num_nodes, self.num_heads * self.num_hidden)
        else:
            return out.mean(axis=1)


class GATModel:
    def __init__(
        self, node_features, hidden_features, target_features, k, heads=8, dropout=0.6
    ):
        self.dropout = dropout
        self.k = k
        self.layer1 = GATLayers(
            node_features, hidden_features, num_heads=heads, concat=True, k=k
        )
        self.layer2 = GATLayers(
            hidden_features, target_features, num_heads=1, concat=False, k=k
        )

    def __call__(self, x, rows, cols):
        x = self.layer1(x, rows, cols)
        x = x.elu()
        return self.layer2(x, rows, cols)


# output is 15 (velocity only, not pos/is_surface)
model = GATModel(num_features, hidden_dim, 15, k=5, heads=8, dropout=0.6)

Tensor.training = True
optimizer = AdamW(nn.state.get_parameters(model), lr=1e-4)

print(f"Training on {len(all_files)} files")
print(
    f"Batch: {BATCH_NODES} nodes, {STEPS_PER_FILE} steps/file, features: {num_features} in -> 15 out"
)

for epoch in range(50):
    total = 0.0
    step_count = 0
    np.random.shuffle(all_files)
    for fi, fpath in enumerate(all_files):
        nf, tf, rows, cols, num_nodes = build_graph_and_features(fpath)
        for _ in range(STEPS_PER_FILE):
            nf_b, tf_b, r_b, c_b = sample_subgraph(
                rows, cols, k, num_nodes, BATCH_NODES, nf, tf
            )
            optimizer.zero_grad()
            pred = model(nf_b, r_b, c_b)
            loss = (pred - tf_b).pow(2).mean()
            loss.backward()
            optimizer.step()
            total += loss.numpy()
            step_count += 1
        if (fi + 1) % 50 == 0:
            print(
                f"  epoch {epoch} | file {fi + 1}/{len(all_files)} | avg loss: {total / step_count:.6f}"
            )
    print(f"epoch {epoch} | loss: {total / step_count:.6f}")
import os
import torch
import numpy as np
from sklearn.neighbors import kneighbors_graph
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric.data import Data, DataLoader
from glob import glob

DATA_DIR = "/Users/vihantiwari/Documents/projects/gram_comp/data"
data_files = sorted(glob(os.path.join(DATA_DIR, "*.npz")))[:50]  # Use first 50 files
print(f"Using {len(data_files)} data files")


def load_sample(path):
    data = np.load(path)
    pos = data["pos"]
    velocity_in = data["velocity_in"]
    velocity_out = data["velocity_out"]

    num_nodes = pos.shape[0]
    node_features = np.concatenate(
        [pos, velocity_in.transpose(1, 0, 2).reshape(num_nodes, 15)], axis=1
    )
    target_features = (
        (velocity_out - velocity_in).transpose(1, 0, 2).reshape(num_nodes, 15)
    )

    knn = kneighbors_graph(pos, n_neighbors=5, mode="connectivity", metric="euclidean")
    rows, cols = knn.nonzero()
    edge_index = torch.tensor(np.array([rows, cols]), dtype=torch.long)

    x = torch.tensor(node_features, dtype=torch.float)
    y = torch.tensor(target_features, dtype=torch.float)

    return Data(x=x, edge_index=edge_index, y=y)


print("Loading all graphs...")
dataset = []
for i, f in enumerate(data_files):
    try:
        data = load_sample(f)
        dataset.append(data)
        if (i + 1) % 10 == 0:
            print(f"Loaded {i + 1}/{len(data_files)}")
    except Exception as e:
        print(f"Failed on {f}: {e}")

print(f"Total samples: {len(dataset)}")

train_loader = DataLoader(dataset, batch_size=1, shuffle=True)


class GATModel(nn.Module):
    def __init__(
        self, in_channels=18, hidden_channels=64, out_channels=15, heads=8, dropout=0.6
    ):
        super().__init__()
        self.dropout = dropout
        self.conv1 = GATConv(
            in_channels, hidden_channels, heads=heads, concat=True, dropout=dropout
        )
        self.conv2 = GATConv(
            hidden_channels * heads,
            hidden_channels,
            heads=heads,
            concat=True,
            dropout=dropout,
        )
        self.conv3 = GATConv(
            hidden_channels * heads,
            hidden_channels,
            heads=heads,
            concat=False,
            dropout=dropout,
        )
        self.out_proj = nn.Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x1 = self.conv1(x, edge_index)
        x1 = F.elu(x1)
        x2 = self.conv2(x1, edge_index)
        x2 = F.elu(x2)
        x3 = self.conv3(x2, edge_index)
        out = self.out_proj(x3)
        return out


model = GATModel(
    in_channels=18, hidden_channels=64, out_channels=15, heads=8, dropout=0.6
)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

model.train()
for epoch in range(20):
    total_loss = 0
    for batch in train_loader:
        optimizer.zero_grad()
        out = model(batch.x, batch.edge_index)
        loss = (out - batch.y).pow(2).mean()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch}, Avg Loss: {total_loss / len(dataset):.6f}")

print("Training complete!")
torch.save(model.state_dict(), "gat_model.pt")

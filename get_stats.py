import sys
from models.zonal_moe.preprocessing import AirfoilDataset
import torch

if __name__ == "__main__":
    dataset = AirfoilDataset(
        "/Users/vihantiwari/Documents/projects/gram_comp/data", normalize=True
    )
    print("vel_mean =", dataset.vel_mean.tolist())
    print("vel_std =", dataset.vel_std.tolist())
    print("pos_mean =", dataset.pos_mean.tolist())
    print("pos_std =", dataset.pos_std.tolist())

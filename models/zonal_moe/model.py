"""
Zonal Expert Graph Transformer for Transient Airflow Prediction
GRaM Competition @ ICLR 2026

Architecture:
- Temporal Encoder: per-point transformer over 5 input timesteps
- Shared GAT Backbone: 2-layer GAT with global readout
- Zone Routing Gate: physics-informed routing based on wall_dist + vorticity
- Expert Branches: Laminar (lightweight) and Turbulent (deep) GAT experts
- Residual Prediction: predicts change from last known timestep

Convention: Model returns RESIDUAL (not absolute velocity).
            Add vel_in[-1] at inference time to get absolute prediction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_scatter import scatter_mean


class TemporalEncoder(nn.Module):
    """Per-point transformer that processes 5 input velocity timesteps.

    Each point's velocity trajectory is encoded as 5 tokens (vel + time),
    processed by a small transformer, and pooled to a single embedding.

    Input: (T, N, 3) where T=5 timesteps, N=num_points (unbatched only)
    Output: (N, d_temporal)
    """

    def __init__(self, d_temporal=64, nhead=4, num_layers=2, dim_feedforward=128):
        super().__init__()
        self.d_temporal = d_temporal
        self.proj = nn.Linear(4, d_temporal)  # 3 velocity + 1 time
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_temporal,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            batch_first=True,
            dropout=0.1,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, vel_in, t_vals=None, chunk_size: int = 4096):
        """
        Args:
            vel_in: (T, N, 3) - T=5 input velocity snapshots
            t_vals: (T,) normalized time values [0, 1], defaults to linspace
            chunk_size: max points per transformer call (avoids CUDA grid overflow)
        Returns:
            (N, d_temporal) temporal embeddings per point
        """
        T, N, _ = vel_in.shape

        if t_vals is None:
            t_vals = torch.linspace(0, 1, T, device=vel_in.device)

        # Expand time: (T,) -> (T, N, 1)
        t = t_vals.view(T, 1, 1).expand(T, N, 1)

        # Concat velocity and time: (T, N, 4) -> (N, T, 4)
        x = torch.cat([vel_in, t], dim=-1).permute(1, 0, 2)

        # Project first (cheap, no grid limit)
        x = self.proj(x)  # (N, T, d_temporal)

        # Chunked transformer to avoid CUDA grid size overflow
        # With N=50k+ and nhead=4, attention kernels launch 200k+ blocks
        outputs = []
        for start in range(0, N, chunk_size):
            chunk = x[start : start + chunk_size]  # (C, T, d_temporal)
            # Use math-only SDPA (more stable than flash on some GPUs)
            with torch.backends.cuda.sdp_kernel(
                enable_flash=False, enable_mem_efficient=False, enable_math=True
            ):
                chunk = self.transformer(chunk)  # (C, T, d_temporal)
            outputs.append(chunk.mean(dim=1))  # (C, d_temporal)

        return torch.cat(outputs, dim=0)  # (N, d_temporal)


class GATLayer(nn.Module):
    """Graph Attention Layer using PyG's GATConv."""

    def __init__(self, in_channels, out_channels, heads=4, dropout=0.1, concat=True):
        super().__init__()
        self.conv = GATConv(
            in_channels,
            out_channels // heads if concat else out_channels,
            heads=heads,
            dropout=dropout,
            concat=concat,
        )
        self.norm = nn.LayerNorm(out_channels)

    def forward(self, x, edge_index):
        x = self.conv(x, edge_index)
        x = self.norm(x)
        return F.elu(x)


class SharedBackbone(nn.Module):
    """Shared 2-layer GAT backbone with global readout broadcast."""

    def __init__(self, in_channels, hidden_channels=128, heads=4, dropout=0.1):
        super().__init__()
        self.gat1 = GATLayer(in_channels, hidden_channels, heads=heads, dropout=dropout)
        self.gat2 = GATLayer(
            hidden_channels, hidden_channels, heads=heads, dropout=dropout
        )
        self.out_channels = hidden_channels * 2  # local + global

    def forward(self, x, edge_index, batch=None):
        """
        Args:
            x: (N, in_channels) node features
            edge_index: (2, E) graph connectivity
            batch: (N,) batch assignment (None = single graph)
        Returns:
            (N, hidden_channels * 2) node features with global context
        """
        x = self.gat1(x, edge_index)
        x = self.gat2(x, edge_index)

        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        # Global mean per graph
        g_global = scatter_mean(x, batch, dim=0)
        g_broadcast = g_global[batch]

        # Concatenate local and global
        x = torch.cat([x, g_broadcast], dim=-1)
        return x


class RoutingGate(nn.Module):
    """Pure physics-based routing gate (no learned parameters to collapse)."""

    def __init__(
        self, in_features=None, hidden_dim=32, wall_dist_scale=1.0, vorticity_scale=1.0
    ):
        super().__init__()
        self.register_buffer("wall_dist_scale", torch.tensor(wall_dist_scale))
        self.register_buffer("vorticity_scale", torch.tensor(vorticity_scale))

    def set_normalization_stats(self, wall_dist_scale: float, vorticity_scale: float):
        """Set normalization scales from training data (e.g., 95th percentile)."""
        self.wall_dist_scale.fill_(wall_dist_scale)
        self.vorticity_scale.fill_(vorticity_scale)

    def forward(self, H, wall_dist, vorticity):
        # H is ignored; routing is completely immune to collapse
        wall_dist_norm = wall_dist / (self.wall_dist_scale + 1e-8)
        vorticity_norm = vorticity / (self.vorticity_scale + 1e-8)

        # Threshold at ~10% of the 95th percentile wall distance.
        # Near-wall points -> positive raw score -> turbulent
        # Far-field points -> strongly negative raw score -> laminar
        # High vorticity -> boost to turbulent
        g_raw = (0.1 - wall_dist_norm) * 20.0 + (vorticity_norm * 5.0)

        g = torch.sigmoid(g_raw)
        return g


class LaminarExpert(nn.Module):
    """Lightweight expert for laminar (far-field) regions.
    2-layer GAT with hidden dim 64.
    """

    def __init__(
        self, in_channels, out_channels, hidden_channels=64, heads=4, dropout=0.1
    ):
        super().__init__()
        self.gat1 = GATLayer(in_channels, hidden_channels, heads=heads, dropout=dropout)
        self.gat2 = GATLayer(
            hidden_channels, out_channels, heads=1, dropout=dropout, concat=False
        )

    def forward(self, x, edge_index):
        x = self.gat1(x, edge_index)
        x = self.gat2(x, edge_index)
        return x


class TurbulentExpert(nn.Module):
    """Deep expert for turbulent (near-wall, wake) regions.
    4-layer GAT with hidden dim 256, residual connections every 2 layers.
    """

    def __init__(
        self, in_channels, out_channels, hidden_channels=256, heads=4, dropout=0.1
    ):
        super().__init__()
        self.input_proj = nn.Linear(in_channels, hidden_channels)

        self.gat1 = GATLayer(
            hidden_channels, hidden_channels, heads=heads, dropout=dropout
        )
        self.gat2 = GATLayer(
            hidden_channels, hidden_channels, heads=heads, dropout=dropout
        )
        self.gat3 = GATLayer(
            hidden_channels, hidden_channels, heads=heads, dropout=dropout
        )
        self.gat4 = GATLayer(
            hidden_channels, hidden_channels, heads=heads, dropout=dropout
        )

        self.output_proj = nn.Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index, edge_index_dense=None):
        """Uses denser k=32 graph if provided."""
        ei = edge_index_dense if edge_index_dense is not None else edge_index

        x = self.input_proj(x)

        # Block 1 with residual
        h = self.gat1(x, ei)
        h = self.gat2(h, ei)
        x = x + h

        # Block 2 with residual
        h = self.gat3(x, ei)
        h = self.gat4(h, ei)
        x = x + h

        x = self.output_proj(x)
        return x


class ZonalMoE(nn.Module):
    """Zonal Expert Graph Transformer for Transient Airflow Prediction.

    IMPORTANT: forward() returns RESIDUAL prediction, not absolute velocity.
    To get absolute velocity: vel_pred = vel_in[-1] + model(...)

    This convention ensures consistent behavior in train/val/inference.
    """

    def __init__(
        self,
        d_temporal=64,
        backbone_hidden=128,
        laminar_hidden=64,
        turbulent_hidden=256,
        heads=4,
        dropout=0.1,
        output_timesteps=5,
        output_channels=3,
        wall_dist_scale=1.0,
        vorticity_scale=1.0,
    ):
        super().__init__()

        self.output_timesteps = output_timesteps
        self.output_channels = output_channels

        self.temporal_encoder = TemporalEncoder(
            d_temporal=d_temporal, nhead=heads, num_layers=2, dim_feedforward=128
        )

        backbone_in = d_temporal + 2  # temporal + wall_dist + is_airfoil
        self.backbone = SharedBackbone(
            in_channels=backbone_in,
            hidden_channels=backbone_hidden,
            heads=heads,
            dropout=dropout,
        )

        expert_in = self.backbone.out_channels
        expert_out = output_timesteps * output_channels  # 15

        self.routing_gate = RoutingGate(
            in_features=expert_in,
            hidden_dim=32,
            wall_dist_scale=wall_dist_scale,
            vorticity_scale=vorticity_scale,
        )

        self.laminar_expert = LaminarExpert(
            in_channels=expert_in,
            out_channels=expert_out,
            hidden_channels=laminar_hidden,
            heads=heads,
            dropout=dropout,
        )

        self.turbulent_expert = TurbulentExpert(
            in_channels=expert_in,
            out_channels=expert_out,
            hidden_channels=turbulent_hidden,
            heads=heads,
            dropout=dropout,
        )

    def compute_vorticity_proxy(self, vel_in):
        """Compute vorticity proxy: |v_T - v_0| per point.

        Args:
            vel_in: (T, N, 3) input velocity snapshots
        Returns:
            (N,) vorticity proxy per point
        """
        return (vel_in[-1] - vel_in[0]).norm(dim=-1)

    def forward(
        self,
        vel_in,
        edge_index,
        wall_dist,
        is_airfoil,
        batch=None,
        edge_index_dense=None,
        debug_gate=False,
    ):
        """
        Args:
            vel_in: (T, N, 3) - T=5 input velocity snapshots (unbatched)
            edge_index: (2, E) - k-NN graph connectivity (k=16)
            wall_dist: (N,) - distance to airfoil surface
            is_airfoil: (N,) - binary flag for surface points
            batch: (N,) - batch assignment (None for single graph)
            edge_index_dense: (2, E') - optional denser k=32 graph
            debug_gate: if True, print gate routing stats for debugging

        Returns:
            residual: (T_out, N, 3) - predicted RESIDUAL (add to vel_in[-1] for absolute)
        """
        # Store debug flag for use in gate computation
        self._debug_gate = debug_gate
        T, N, C = vel_in.shape

        # 1. Temporal encoding
        temporal_emb = self.temporal_encoder(vel_in)  # (N, d_temporal)

        # 2. Compute vorticity proxy for routing
        vorticity = self.compute_vorticity_proxy(vel_in)  # (N,)

        # 3. Build node features
        wall_dist_feat = wall_dist.unsqueeze(-1)
        is_airfoil_feat = is_airfoil.float().unsqueeze(-1)
        node_features = torch.cat(
            [temporal_emb, wall_dist_feat, is_airfoil_feat], dim=-1
        )

        # 4. Shared backbone
        H = self.backbone(node_features, edge_index, batch)

        # 5. Physics-informed routing gate (now purely physical)
        g = self.routing_gate(H, wall_dist, vorticity)
        self._last_g = g.detach()

        # Load balancing auxiliary loss is removed since gate is not learned
        self.balance_loss = 0.0

        # DEBUG: print gate stats
        if self._debug_gate:
            turb_frac = (g > 0.5).float().mean().item()
            print(
                f"  [gate] mean={g.mean():.4f} std={g.std():.4f} turbulent={turb_frac * 100:.1f}%"
            )

        # 6. Expert branches
        laminar_out = self.laminar_expert(H, edge_index)
        turbulent_out = self.turbulent_expert(H, edge_index, edge_index_dense)

        # 7. Soft mixing
        g_expanded = g.unsqueeze(-1)
        mixed = g_expanded * turbulent_out + (1 - g_expanded) * laminar_out

        # 8. Reshape to (T_out, N, 3)
        residual = mixed.view(N, self.output_timesteps, self.output_channels)
        residual = residual.permute(1, 0, 2)  # (T_out, N, 3)

        return residual

    def predict(
        self,
        vel_in,
        edge_index,
        wall_dist,
        is_airfoil,
        batch=None,
        edge_index_dense=None,
    ):
        """Convenience method that returns absolute velocity prediction.

        Returns:
            vel_pred: (T_out, N, 3) - absolute predicted velocity
        """
        residual = self.forward(
            vel_in, edge_index, wall_dist, is_airfoil, batch, edge_index_dense
        )
        last_frame = vel_in[-1]  # (N, 3)
        return last_frame.unsqueeze(0) + residual

    def set_normalization_stats(self, wall_dist_scale: float, vorticity_scale: float):
        """Set normalization scales for routing gate (call after computing from training data)."""
        self.routing_gate.set_normalization_stats(wall_dist_scale, vorticity_scale)

    def get_routing_stats(self):
        """Get routing statistics from the last forward pass."""
        if not hasattr(self, "_last_g") or self._last_g is None:
            return {
                "gate_mean": 0.0,
                "gate_std": 0.0,
                "turbulent_fraction": 0.0,
                "laminar_fraction": 0.0,
            }

        g = self._last_g
        return {
            "gate_mean": g.mean().item(),
            "gate_std": g.std().item(),
            "turbulent_fraction": (g > 0.5).float().mean().item(),
            "laminar_fraction": (g <= 0.5).float().mean().item(),
        }

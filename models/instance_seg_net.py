import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict
import numpy as np

from models.pointnet_seg import (
    _SetAbstractionWithAttention,
    _FeaturePropagationWithAttention,
    ChannelAttention,
    PositionAttention,
)


EMBED_DIM = 32


class InstanceSegNet(nn.Module):
    def __init__(self, input_channels: int = 6, embed_dim: int = EMBED_DIM):
        super().__init__()
        self.input_channels = input_channels
        self.embed_dim = embed_dim

        in_ch = input_channels
        self.sa1 = _SetAbstractionWithAttention(512, 0.2, 32, 3 + in_ch, [64, 64, 128], use_attention=True)
        self.sa2 = _SetAbstractionWithAttention(128, 0.4, 64, 3 + 128, [128, 128, 256], use_attention=True)
        self.sa3 = _SetAbstractionWithAttention(None, None, None, 3 + 256, [256, 512, 1024], use_attention=True)

        self.fp3 = _FeaturePropagationWithAttention(256 + 1024, [256, 256], use_attention=True)
        self.fp2 = _FeaturePropagationWithAttention(128 + 256, [256, 256, 128], use_attention=True)
        self.fp1 = _FeaturePropagationWithAttention(in_ch + 128, [128, 128, 128], use_attention=True)

        self.position_attn = PositionAttention(128, 128)
        self.se_global = ChannelAttention(1024, reduction=16)
        self.se_mid = ChannelAttention(256, reduction=8)
        self.fuse_conv = nn.Conv1d(128 + 1024 + 256, 128, 1)
        self.fuse_bn = nn.BatchNorm1d(128)

        shared_dim = 256
        self.shared_conv = nn.Conv1d(128, shared_dim, 1)
        self.shared_bn = nn.BatchNorm1d(shared_dim)
        self.shared_drop = nn.Dropout(0.3)

        self.seg_head = nn.Sequential(
            nn.Conv1d(shared_dim, 128, 1), nn.BatchNorm1d(128), nn.ReLU(inplace=True), nn.Dropout(0.2),
            nn.Conv1d(128, 2, 1),
        )

        self.inst_embed_head = nn.Sequential(
            nn.Conv1d(shared_dim, 128, 1), nn.BatchNorm1d(128), nn.ReLU(inplace=True), nn.Dropout(0.2),
            nn.Conv1d(128, embed_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        B, N, C_in = x.shape
        l0_xyz = x[:, :, :3].contiguous()
        l0_points = x.contiguous()

        l1_xyz, l1_points, _ = self.sa1(l0_xyz, l0_points)
        l2_xyz, l2_points, _ = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points, _ = self.sa3(l2_xyz, l2_points)

        l3_feat_bcn = l3_points.permute(0, 2, 1).contiguous()
        l3_feat_bcn = self.se_global(l3_feat_bcn)
        l3_points = l3_feat_bcn.permute(0, 2, 1).contiguous()

        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
        l2_feat_bcn = l2_points.permute(0, 2, 1).contiguous()
        l2_feat_bcn = self.se_mid(l2_feat_bcn)
        l2_points = l2_feat_bcn.permute(0, 2, 1).contiguous()

        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
        l0_points = self.fp1(l0_xyz, l1_xyz, l0_points, l1_points)

        l0_feat_bcn = l0_points.permute(0, 2, 1).contiguous()
        feat = self.position_attn(l0_feat_bcn, l0_xyz)

        global_feat = F.adaptive_avg_pool1d(l3_feat_bcn, 1).expand(-1, -1, N)
        mid_feat = F.interpolate(l2_feat_bcn, size=N, mode='linear', align_corners=True)
        multi_scale = torch.cat([feat, global_feat, mid_feat], dim=1)
        fused = F.relu(self.fuse_bn(self.fuse_conv(multi_scale)))

        shared = F.relu(self.shared_bn(self.shared_conv(fused)))
        shared = self.shared_drop(shared)

        seg_logits = self.seg_head(shared).permute(0, 2, 1).contiguous()
        inst_embed = self.inst_embed_head(shared).permute(0, 2, 1).contiguous()

        return {"seg_logits": seg_logits, "inst_embed": inst_embed}


def instance_disc_loss(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    delta_d: float = 3.0,
    alpha: float = 1.0,
    beta: float = 1.0,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    B, N, D = embeddings.shape
    total_loss = torch.tensor(0.0, device=embeddings.device)
    metrics = {"var": 0.0, "dist": 0.0, "ortho": 0.0}
    n_valid = 0

    for b in range(B):
        emb_b = embeddings[b]
        lbl_b = labels[b]
        unique_labels = torch.unique(lbl_b)
        unique_labels = unique_labels[unique_labels >= 0]
        if len(unique_labels) < 2:
            continue

        centers = []
        for lbl in unique_labels:
            mask = lbl_b == lbl
            if mask.sum() == 0:
                continue
            center = emb_b[mask].mean(dim=0)
            centers.append(center)

        centers = torch.stack(centers)
        K = len(centers)

        var_loss = torch.tensor(0.0, device=embeddings.device)
        for i, lbl in enumerate(unique_labels):
            mask = lbl_b == lbl
            if mask.sum() == 0:
                continue
            dists_sq = ((emb_b[mask] - centers[i]) ** 2).sum(dim=-1)
            var_loss = var_loss + dists_sq.mean()
        var_loss = var_loss / max(K, 1)

        dist_loss = torch.tensor(0.0, device=embeddings.device)
        if K > 1:
            for i in range(K):
                for j in range(i + 1, K):
                    d = torch.norm(centers[i] - centers[j])
                    dist_loss = dist_loss + torch.clamp(2 * delta_d - d, min=0.0) ** 2
            dist_loss = dist_loss / (K * (K - 1) / 2)

        ortho_loss = torch.tensor(0.0, device=embeddings.device)
        if K > 1:
            centers_norm = F.normalize(centers, p=2, dim=-1)
            sim_matrix = torch.mm(centers_norm, centers_norm.t())
            sim_matrix.fill_diagonal_(0.0)
            ortho_loss = (sim_matrix ** 2).sum() / (K * (K - 1))

        total_loss = total_loss + alpha * var_loss + beta * dist_loss + 0.5 * ortho_loss
        metrics["var"] += var_loss.item()
        metrics["dist"] += dist_loss.item()
        metrics["ortho"] += ortho_loss.item()
        n_valid += 1

    if n_valid > 0:
        for k in metrics:
            metrics[k] /= n_valid
        total_loss = total_loss / n_valid

    return total_loss, metrics


def instance_seg_loss(
    outputs: Dict[str, torch.Tensor],
    targets: Dict[str, torch.Tensor],
    seg_weight: float = 1.0,
    inst_weight: float = 5.0,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    seg_logits = outputs["seg_logits"]
    inst_embed = outputs["inst_embed"]
    seg_target = targets["seg_labels"]
    fg_mask = (seg_target == 1)

    B, N, _ = seg_logits.shape
    seg_flat = seg_logits.reshape(-1, 2)
    seg_tgt_flat = seg_target.reshape(-1)

    class_counts = torch.bincount(seg_tgt_flat, minlength=2).float()
    class_counts = torch.clamp(class_counts, min=1.0)
    class_weights = (N * B) / (2 * class_counts)
    seg_loss = F.cross_entropy(seg_flat, seg_tgt_flat, weight=class_weights)

    pred_probs = F.softmax(seg_flat, dim=-1)
    fg_prob = pred_probs[:, 1]
    fg_tgt = (seg_tgt_flat == 1).float()
    intersection = (fg_prob * fg_tgt).sum()
    dice = (2.0 * intersection + 1.0) / (fg_prob.sum() + fg_tgt.sum() + 1.0)
    seg_loss = seg_loss + (1.0 - dice)

    loss_dict = {"seg": seg_loss.item()}
    total_loss = seg_weight * seg_loss

    if fg_mask.sum() > 0:
        inst_fg_embed = inst_embed[fg_mask].unsqueeze(0)

        if "inst_labels" in targets:
            raw_inst_lbl = targets["inst_labels"]
            inst_fg_lbl = raw_inst_lbl[fg_mask].unsqueeze(0)
            unique_ids = torch.unique(inst_fg_lbl)
            remap = {v.item(): i for i, v in enumerate(unique_ids)}
            inst_fg_lbl_mapped = torch.zeros_like(inst_fg_lbl)
            for orig, new in remap.items():
                inst_fg_lbl_mapped[inst_fg_lbl == orig] = new
        else:
            B_fg, N_fg = inst_fg_embed.shape[0], inst_fg_embed.shape[1]
            inst_fg_lbl_mapped = torch.zeros(B_fg, N_fg, dtype=torch.long, device=inst_fg_embed.device)

        inst_loss, disc_metrics = instance_disc_loss(
            inst_fg_embed, inst_fg_lbl_mapped, delta_d=3.0
        )
        total_loss = total_loss + inst_weight * inst_loss
        loss_dict["inst"] = inst_loss.item()
        loss_dict["inst_var"] = disc_metrics["var"]
        loss_dict["inst_dist"] = disc_metrics["dist"]
        loss_dict["inst_ortho"] = disc_metrics["ortho"]
    else:
        loss_dict["inst"] = 0.0
        loss_dict["inst_var"] = 0.0
        loss_dict["inst_dist"] = 0.0
        loss_dict["inst_ortho"] = 0.0

    return total_loss, loss_dict


def build_instance_seg_model(config: dict) -> InstanceSegNet:
    model_cfg = config.get("model", {})
    return InstanceSegNet(
        input_channels=model_cfg.get("input_channels", 6),
        embed_dim=model_cfg.get("embed_dim", EMBED_DIM),
    )

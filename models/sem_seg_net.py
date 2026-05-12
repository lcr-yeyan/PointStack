import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict

from models.pointnet_seg import (
    _SetAbstractionWithAttention,
    _FeaturePropagationWithAttention,
    ChannelAttention,
    PositionAttention,
)

NUM_CLASSES = 11


class SemSegNet(nn.Module):
    def __init__(self, input_channels: int = 6, num_classes: int = NUM_CLASSES):
        super().__init__()
        self.input_channels = input_channels
        self.num_classes = num_classes

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
        self.fuse_conv = nn.Conv1d(128 + 1024 + 256, 256, 1)
        self.fuse_bn = nn.BatchNorm1d(256)

        self.seg_head = nn.Sequential(
            nn.Conv1d(256, 256, 1), nn.BatchNorm1d(256), nn.ReLU(inplace=True), nn.Dropout(0.3),
            nn.Conv1d(256, 128, 1), nn.BatchNorm1d(128), nn.ReLU(inplace=True), nn.Dropout(0.2),
            nn.Conv1d(128, 64, 1), nn.BatchNorm1d(64), nn.ReLU(inplace=True),
            nn.Conv1d(64, num_classes, 1),
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

        seg_logits = self.seg_head(fused).permute(0, 2, 1).contiguous()

        return {"seg_logits": seg_logits}


def _lovasz_grad(gt_sorted):
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)
    jaccard = 1.0 - intersection / union
    if p > 1:
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
    return jaccard


def _lovasz_softmax_flat(probas, labels, classes=None):
    if classes is None:
        classes = list(range(probas.shape[1]))
    losses = []
    for c in classes:
        fg = (labels == c).float()
        if fg.sum() == 0:
            continue
        prob_c = probas[:, c]
        errors = (fg - prob_c).abs()
        errors_sorted, perm = torch.sort(errors, descending=True)
        fg_sorted = fg[perm]
        grad = _lovasz_grad(fg_sorted)
        losses.append(torch.dot(errors_sorted, grad))
    if len(losses) == 0:
        return probas[:, 0].sum() * 0
    return torch.stack(losses).mean()


def lovasz_softmax_loss(seg_logits, seg_target):
    B, N, C = seg_logits.shape
    seg_flat = seg_logits.reshape(-1, C)
    tgt_flat = seg_target.reshape(-1)
    probas = F.softmax(seg_flat, dim=-1)
    return _lovasz_softmax_flat(probas, tgt_flat)


def dice_loss(seg_logits, seg_target, smooth=1.0):
    B, N, C = seg_logits.shape
    seg_flat = seg_logits.reshape(-1, C)
    tgt_flat = seg_target.reshape(-1)
    probas = F.softmax(seg_flat, dim=-1)

    tgt_one_hot = F.one_hot(tgt_flat, C).float()
    intersection = (probas * tgt_one_hot).sum(0)
    union = probas.sum(0) + tgt_one_hot.sum(0)
    dice = (2.0 * intersection + smooth) / (union + smooth)
    return 1.0 - dice.mean()


def sem_seg_loss(outputs, targets, seg_weight=1.0):
    seg_logits = outputs["seg_logits"]
    seg_target = targets["seg_labels"]

    B, N, C = seg_logits.shape
    seg_flat = seg_logits.reshape(-1, C)
    seg_tgt_flat = seg_target.reshape(-1)

    class_counts = torch.bincount(seg_tgt_flat, minlength=C).float()
    class_counts = torch.clamp(class_counts, min=1.0)
    class_weights = (N * B) / (C * class_counts)
    class_weights = torch.clamp(class_weights, max=5.0)

    ce_loss = F.cross_entropy(seg_flat, seg_tgt_flat, weight=class_weights)

    fg_mask = seg_tgt_flat > 0
    if fg_mask.sum() > 0:
        fg_logits = seg_flat[fg_mask]
        fg_tgt = seg_tgt_flat[fg_mask]
        fg_ce = F.cross_entropy(fg_logits, fg_tgt)
    else:
        fg_ce = torch.tensor(0.0, device=seg_logits.device)

    pred_probs = F.softmax(seg_flat, dim=-1)
    pt = pred_probs.gather(1, seg_tgt_flat.unsqueeze(1)).squeeze(1)
    focal_gamma = 2.0
    focal_loss = F.nll_loss((1.0 - pt) ** focal_gamma * torch.log(pt + 1e-12), seg_tgt_flat, reduction='mean')

    lov_loss = lovasz_softmax_loss(seg_logits, seg_target)
    dic_loss = dice_loss(seg_logits, seg_target)

    total_loss = seg_weight * (ce_loss + 0.5 * fg_ce + 0.5 * focal_loss + lov_loss + 0.5 * dic_loss)
    loss_dict = {
        "total": total_loss.item(),
        "ce": ce_loss.item(),
        "fg_ce": fg_ce.item(),
        "focal": focal_loss.item(),
        "lovasz": lov_loss.item(),
        "dice": dic_loss.item(),
    }
    return total_loss, loss_dict


def build_sem_seg_model(config: dict) -> SemSegNet:
    model_cfg = config.get("model", {})
    return SemSegNet(
        input_channels=model_cfg.get("input_channels", 6),
        num_classes=model_cfg.get("num_classes", NUM_CLASSES),
    )

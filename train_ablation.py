"""消融实验训练脚本 - 依次训练所有模型变体并评估"""

import os, json, time, sys
import numpy as np
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from loguru import logger
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models.pointnet_seg import (
    PointNetPlusPlusSeg,
    _SetAbstraction, _SetAbstractionWithAttention,
    _FeaturePropagation, _FeaturePropagationWithAttention,
    ChannelAttention, PositionAttention,
)

BATCH_SIZE = 8
NUM_POINTS = 2048
NUM_EPOCHS = 100
LR = 0.001
WEIGHT_DECAY = 1e-4
EARLY_STOP_PATIENCE = 20
NUM_CLASSES = 2

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

FX, FY = 506.35, 506.27
CX, CY = 334.19, 335.43
NEAR = 0.05

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "training_data")
ABLATION_LOG_DIR = os.path.join(DATA_DIR, "ablation_logs")
os.makedirs(ABLATION_LOG_DIR, exist_ok=True)


def normalize_points_6ch(points):
    z_raw = points[:, 2].copy()
    z_min, z_max = z_raw.min(), z_raw.max()
    z_range = max(z_max - z_min, 1e-8)
    z_mean = z_raw.mean()
    centroid = np.mean(points, axis=0, keepdims=True)
    points_centered = points - centroid
    max_dist = np.max(np.linalg.norm(points_centered, axis=1))
    max_dist = max(max_dist, 1e-8)
    xy_norm = points_centered / max_dist
    z_abs = (z_raw - z_min) / z_range
    z_rel = (z_raw - z_mean) / max(z_range, max_dist)
    z_h = (z_raw - z_min + 1e-8) / (max_dist + 1e-8)
    return np.concatenate([xy_norm, z_abs[:, None], z_rel[:, None], z_h[:, None]], axis=-1).astype(np.float32)


def normalize_points_3ch(points):
    centroid = np.mean(points, axis=0, keepdims=True)
    points_centered = points - centroid
    max_dist = np.max(np.linalg.norm(points_centered, axis=1))
    max_dist = max(max_dist, 1e-8)
    return (points_centered / max_dist).astype(np.float32)


class StackingDataset(Dataset):
    def __init__(self, root_dir, num_points=NUM_POINTS, augment=False, use_6ch=True):
        self.scene_dirs = sorted([
            os.path.join(root_dir, d) for d in os.listdir(root_dir)
            if os.path.isdir(os.path.join(root_dir, d))
        ])
        self.num_points = num_points
        self.augment = augment
        self.use_6ch = use_6ch

    def __len__(self):
        return len(self.scene_dirs)

    def __getitem__(self, idx):
        d = self.scene_dirs[idx]
        depth = np.load(os.path.join(d, "depth_noisy.npy"))
        labels = np.load(os.path.join(d, "semantic_labels.npy"))

        h, w = depth.shape
        valid = depth > NEAR
        uu, vv = np.meshgrid(np.arange(w, dtype=np.float32),
                             np.arange(h, dtype=np.float32))
        Z = depth[valid]
        X = (uu[valid] - CX) * Z / FX
        Y = (vv[valid] - CY) * Z / FY
        pts = np.stack([X, Y, Z], axis=-1)
        lbs = labels[valid]

        n = len(pts)
        if n >= self.num_points:
            idxs = np.random.choice(n, self.num_points, replace=False)
        else:
            idxs = np.random.choice(n, self.num_points, replace=True)
        pts_sampled = pts[idxs]
        lbs_sampled = lbs[idxs]

        if self.augment:
            ang = np.random.uniform(0, 2 * np.pi)
            cos_a, sin_a = np.cos(ang), np.sin(ang)
            rot = np.array([[cos_a, -sin_a, 0], [sin_a, cos_a, 0], [0, 0, 1]], dtype=np.float32)
            pts_sampled = pts_sampled @ rot.T
            s = np.random.uniform(0.8, 1.2)
            pts_sampled *= s
            pts_sampled += np.random.normal(0, 0.005, pts_sampled.shape).astype(np.float32)

        if self.use_6ch:
            feat = normalize_points_6ch(pts_sampled)
        else:
            feat = normalize_points_3ch(pts_sampled)

        return (torch.from_numpy(feat).float(),
                torch.from_numpy(lbs_sampled).long())


def compute_iou(pred, target, num_classes):
    ious = []
    for c in range(num_classes):
        intersection = ((pred == c) & (target == c)).sum().float()
        union = ((pred == c) | (target == c)).sum().float()
        ious.append((intersection + 1) / (union + 1))
    miou = sum(ious) / len(ious)
    return miou, ious


class AblationModel(nn.Module):
    """可配置的消融实验模型

    通过 flags 控制各模块的开关:
      - use_channel_attn: SA/FP 层内嵌 SE 通道注意力
      - use_position_attn: FP1 后空间自注意力
      - use_multiscale_fusion: 全局+中层特征融合
      - use_residual_head: 残差分割头
      - use_global_se: 全局 SE (SA3 后)
      - use_mid_se: 中层 SE (FP3 后)
    """

    def __init__(
        self,
        input_channels: int = 6,
        num_classes: int = 2,
        use_channel_attn: bool = False,
        use_position_attn: bool = False,
        use_multiscale_fusion: bool = False,
        use_residual_head: bool = False,
        use_global_se: bool = False,
        use_mid_se: bool = False,
    ):
        super().__init__()
        in_ch = input_channels

        sa_cls = _SetAbstractionWithAttention if use_channel_attn else _SetAbstraction
        fp_cls = _FeaturePropagationWithAttention if use_channel_attn else _FeaturePropagation

        self.sa1 = sa_cls(512, 0.2, 32, 3 + in_ch, [64, 64, 128],
                          use_attention=use_channel_attn)
        self.sa2 = sa_cls(128, 0.4, 64, 3 + 128, [128, 128, 256],
                          use_attention=use_channel_attn)
        self.sa3 = sa_cls(None, None, None, 3 + 256, [256, 512, 1024],
                          use_attention=use_channel_attn)
        self.fp3 = fp_cls(256 + 1024, [256, 256],
                          use_attention=use_channel_attn)
        self.fp2 = fp_cls(128 + 256, [256, 256, 128],
                          use_attention=use_channel_attn)
        self.fp1 = fp_cls(in_ch + 128, [128, 128, 128],
                          use_attention=use_channel_attn)

        self.use_position_attn = use_position_attn
        if use_position_attn:
            self.position_attn = PositionAttention(128, 128)

        self.use_multiscale_fusion = use_multiscale_fusion
        self.use_global_se = use_global_se
        self.use_mid_se = use_mid_se
        self.use_residual_head = use_residual_head

        if use_global_se:
            self.se_global = ChannelAttention(1024, reduction=16)
        if use_mid_se:
            self.se_mid = ChannelAttention(256, reduction=8)

        if use_multiscale_fusion:
            self.fuse_conv = nn.Conv1d(128 + 1024 + 256, 128, 1)
            self.fuse_bn = nn.BatchNorm1d(128)
            head_in_ch = 128
        else:
            head_in_ch = 128

        if use_residual_head:
            self.conv1 = nn.Conv1d(head_in_ch, 256, 1)
            self.bn1 = nn.BatchNorm1d(256)
            self.drop1 = nn.Dropout(0.3)
            self.conv2 = nn.Conv1d(256, 128, 1)
            self.bn2 = nn.BatchNorm1d(128)
            self.drop2 = nn.Dropout(0.2)
            self.conv3 = nn.Conv1d(128, num_classes, 1)
        else:
            self.conv1 = nn.Conv1d(head_in_ch, 128, 1)
            self.bn1 = nn.BatchNorm1d(128)
            self.drop1 = nn.Dropout(0.5)
            self.conv2 = nn.Conv1d(128, num_classes, 1)

    def forward(self, x):
        B, N, C_in = x.shape
        identity_matrix = torch.eye(3, dtype=x.dtype, device=x.device).unsqueeze(0).expand(B, -1, -1)

        l0_xyz = x[:, :, :3].contiguous()
        l0_points = x.contiguous()

        l1_xyz, l1_points, _ = self.sa1(l0_xyz, l0_points)
        l2_xyz, l2_points, _ = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points, _ = self.sa3(l2_xyz, l2_points)

        l3_feat_bcn = l3_points.permute(0, 2, 1).contiguous()
        if self.use_global_se:
            l3_feat_bcn = self.se_global(l3_feat_bcn)
            l3_points = l3_feat_bcn.permute(0, 2, 1).contiguous()

        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
        l2_feat_bcn = l2_points.permute(0, 2, 1).contiguous()
        if self.use_mid_se:
            l2_feat_bcn = self.se_mid(l2_feat_bcn)
            l2_points = l2_feat_bcn.permute(0, 2, 1).contiguous()

        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
        l0_points = self.fp1(l0_xyz, l1_xyz, l0_points, l1_points)

        l0_feat_bcn = l0_points.permute(0, 2, 1).contiguous()

        if self.use_position_attn:
            feat = self.position_attn(l0_feat_bcn, l0_xyz)
        else:
            feat = l0_feat_bcn

        if self.use_multiscale_fusion:
            global_feat = F.adaptive_avg_pool1d(l3_feat_bcn, 1).expand(-1, -1, N)
            mid_feat = F.interpolate(l2_feat_bcn, size=N, mode='linear', align_corners=True)
            multi_scale = torch.cat([feat, global_feat, mid_feat], dim=1)
            feat = F.relu(self.fuse_bn(self.fuse_conv(multi_scale)))

        if self.use_residual_head:
            feat = F.relu(self.bn1(self.conv1(feat)))
            feat = self.drop1(feat)
            residual = feat
            feat = F.relu(self.bn2(self.conv2(feat)))
            feat = feat + F.relu(self.bn2(self.conv2(residual)))
            feat = self.drop2(feat)
            out = self.conv3(feat)
        else:
            feat = F.relu(self.bn1(self.conv1(feat)))
            feat = self.drop1(feat)
            out = self.conv2(feat)

        out = out.permute(0, 2, 1).contiguous()
        return out, identity_matrix, identity_matrix


ABLATION_CONFIGS = {
    "baseline_3ch": {
        "name": "Baseline (3ch XYZ)",
        "model_type": "PointNetPlusPlusSeg",
        "input_channels": 3,
        "use_6ch": False,
        "ablation_flags": None,
    },
    "plus_6ch_input": {
        "name": "+6ch Input",
        "model_type": "PointNetPlusPlusSeg",
        "input_channels": 6,
        "use_6ch": True,
        "ablation_flags": None,
    },
    "plus_channel_attn": {
        "name": "+Channel Attention",
        "model_type": "AblationModel",
        "input_channels": 6,
        "use_6ch": True,
        "ablation_flags": {
            "use_channel_attn": True,
            "use_position_attn": False,
            "use_multiscale_fusion": False,
            "use_residual_head": False,
            "use_global_se": False,
            "use_mid_se": False,
        },
    },
    "plus_position_attn": {
        "name": "+Position Attention",
        "model_type": "AblationModel",
        "input_channels": 6,
        "use_6ch": True,
        "ablation_flags": {
            "use_channel_attn": True,
            "use_position_attn": True,
            "use_multiscale_fusion": False,
            "use_residual_head": False,
            "use_global_se": False,
            "use_mid_se": False,
        },
    },
    "plus_multiscale": {
        "name": "+MultiScale Fusion",
        "model_type": "AblationModel",
        "input_channels": 6,
        "use_6ch": True,
        "ablation_flags": {
            "use_channel_attn": True,
            "use_position_attn": True,
            "use_multiscale_fusion": True,
            "use_residual_head": False,
            "use_global_se": False,
            "use_mid_se": False,
        },
    },
}


def create_model(config):
    if config["model_type"] == "PointNetPlusPlusSeg":
        return PointNetPlusPlusSeg(
            input_channels=config["input_channels"],
            num_classes=NUM_CLASSES,
        )
    elif config["model_type"] == "AblationModel":
        return AblationModel(
            input_channels=config["input_channels"],
            num_classes=NUM_CLASSES,
            **config["ablation_flags"],
        )
    else:
        raise ValueError(f"Unknown model type: {config['model_type']}")


def train_one_config(config_key, config):
    log_dir = os.path.join(ABLATION_LOG_DIR, config_key)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(os.path.join(log_dir, "sample_predictions"), exist_ok=True)

    logger.info("=" * 60)
    logger.info(f"Training: {config['name']} ({config_key})")
    logger.info(f"Log dir: {log_dir}")
    logger.info("=" * 60)

    train_set = StackingDataset(
        os.path.join(DATA_DIR, "train"),
        augment=True,
        use_6ch=config["use_6ch"],
    )
    val_set = StackingDataset(
        os.path.join(DATA_DIR, "val"),
        augment=False,
        use_6ch=config["use_6ch"],
    )
    logger.info(f"Train: {len(train_set)} scenes, Val: {len(val_set)} scenes")

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=0, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=0, pin_memory=True)

    model = create_model(config).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model params: {n_params:,}")

    class_weights = torch.tensor([1.0, 3.0], device=DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

    history = defaultdict(list)
    best_miou = 0.0
    best_epoch = 0
    patience_counter = 0

    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        train_loss_sum = 0.0
        train_miou_sum = 0.0
        train_batches = 0

        for feat, target in train_loader:
            feat, target = feat.to(DEVICE), target.to(DEVICE)
            optimizer.zero_grad()

            logits, _, _ = model(feat)
            loss = F.cross_entropy(
                logits.reshape(-1, NUM_CLASSES),
                target.reshape(-1),
                weight=class_weights,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss_sum += loss.item()
            pred = logits.argmax(dim=-1)
            miou, _ = compute_iou(pred, target, NUM_CLASSES)
            train_miou_sum += miou.item()
            train_batches += 1

        avg_train_loss = train_loss_sum / max(train_batches, 1)
        avg_train_miou = train_miou_sum / max(train_batches, 1)

        model.eval()
        val_loss_sum = 0.0
        val_miou_sum = 0.0
        val_ious = np.zeros(NUM_CLASSES)
        val_batches = 0

        with torch.no_grad():
            for feat, target in val_loader:
                feat, target = feat.to(DEVICE), target.to(DEVICE)
                logits, _, _ = model(feat)
                loss = F.cross_entropy(
                    logits.reshape(-1, NUM_CLASSES),
                    target.reshape(-1),
                    weight=class_weights,
                )
                val_loss_sum += loss.item()
                pred = logits.argmax(dim=-1)
                miou, ious = compute_iou(pred, target, NUM_CLASSES)
                val_miou_sum += miou.item()
                for c in range(NUM_CLASSES):
                    val_ious[c] += ious[c].item()
                val_batches += 1

        avg_val_loss = val_loss_sum / max(val_batches, 1)
        avg_val_miou = val_miou_sum / max(val_batches, 1)
        val_ious /= max(val_batches, 1)

        scheduler.step()
        lr_now = scheduler.get_last_lr()[0]

        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)
        history["train_miou"].append(avg_train_miou)
        history["val_miou"].append(avg_val_miou)
        history["val_iou_table"].append(val_ious.tolist())
        history["lr"].append(lr_now)

        logger.info(
            f"Epoch {epoch:3d}/{NUM_EPOCHS} | "
            f"LR {lr_now:.5f} | "
            f"Train Loss {avg_train_loss:.4f} mIoU {avg_train_miou:.4f} | "
            f"Val Loss {avg_val_loss:.4f} mIoU {avg_val_miou:.4f} | "
            f"IoU[table={val_ious[0]:.3f} obj={val_ious[1]:.3f}]"
        )

        if avg_val_miou > best_miou:
            best_miou = avg_val_miou
            best_epoch = epoch
            patience_counter = 0
            torch.save({
                "epoch": epoch,
                "miou": best_miou,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "config": config,
            }, os.path.join(log_dir, "best_model.pth"))
            logger.info(f"  >>> Best model saved (mIoU={best_miou:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOP_PATIENCE:
                logger.info(f"Early stopping at epoch {epoch}")
                break

    metrics = {
        "config_key": config_key,
        "config_name": config["name"],
        "best_epoch": best_epoch,
        "best_val_miou": float(best_miou),
        "best_val_iou_table": val_ious.tolist(),
        "num_params": n_params,
        "history": {k: v for k, v in history.items() if k != "val_iou_table"},
    }
    with open(os.path.join(log_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    axes[0].plot(history["train_loss"], label="Train")
    axes[0].plot(history["val_loss"], label="Val")
    axes[0].set_title(f"{config['name']} - Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(history["train_miou"], label="Train")
    axes[1].plot(history["val_miou"], label="Val")
    axes[1].set_title(f"{config['name']} - mIoU")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    ious_arr = np.array(history["val_iou_table"])
    axes[2].plot(ious_arr[:, 0], label="Table")
    axes[2].plot(ious_arr[:, 1], label="Object")
    axes[2].set_title(f"{config['name']} - Per-Class IoU")
    axes[2].set_xlabel("Epoch")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(os.path.join(log_dir, "training_curves.png"), dpi=150,
                facecolor="white", edgecolor="none")
    plt.close()

    logger.info(f"Training complete: {config['name']} | Best mIoU={best_miou:.4f} @ epoch {best_epoch}")
    return metrics


def main():
    logger.info(f"Device: {DEVICE}")
    logger.info(f"Ablation log dir: {ABLATION_LOG_DIR}")

    all_metrics = {}

    for config_key, config in ABLATION_CONFIGS.items():
        best_path = os.path.join(ABLATION_LOG_DIR, config_key, "best_model.pth")
        if os.path.exists(best_path):
            logger.info(f"Skipping {config['name']} - already trained ({best_path})")
            metrics_path = os.path.join(ABLATION_LOG_DIR, config_key, "metrics.json")
            if os.path.exists(metrics_path):
                with open(metrics_path) as f:
                    all_metrics[config_key] = json.load(f)
            continue

        metrics = train_one_config(config_key, config)
        all_metrics[config_key] = metrics

    logger.info("\n" + "=" * 60)
    logger.info("Ablation Study Summary")
    logger.info("=" * 60)
    for config_key, config in ABLATION_CONFIGS.items():
        if config_key in all_metrics:
            m = all_metrics[config_key]
            logger.info(f"  {config['name']:25s} | mIoU={m['best_val_miou']:.4f} | "
                        f"Params={m['num_params']:,} | Epoch={m['best_epoch']}")

    with open(os.path.join(ABLATION_LOG_DIR, "ablation_summary.json"), "w") as f:
        json.dump(all_metrics, f, indent=2)

    logger.info(f"\nSummary saved to {os.path.join(ABLATION_LOG_DIR, 'ablation_summary.json')}")


if __name__ == "__main__":
    main()
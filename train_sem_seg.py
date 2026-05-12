"""
PointNet++Attention 训练: 2类语义分割 (桌面=0, 物体=1)
=====================================================
输入: 6ch 归一化点云 → 输出: 逐点 2 类 logits

训练产出保存至 camera_operate/training_data/train_logs/
  - loss_curve.png         损失曲线
  - metrics.json           每 epoch 详细指标
  - best_model.pth         最佳模型
  - sample_predictions/    验证集预测样本
  - training_report.txt    训练报告
"""

import os, sys, json, time
import numpy as np
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from loguru import logger
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import open3d as o3d

from models.pointnet_seg import PointNetPlusPlusAttentionSeg

# ── 配置 ───────────────────────────────────────────────────────────
BATCH_SIZE = 8
NUM_POINTS = 2048
NUM_EPOCHS = 100
LR = 0.001
WEIGHT_DECAY = 1e-4
EARLY_STOP_PATIENCE = 20
NUM_CLASSES = 2
INPUT_CHANNELS = 6

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

FX, FY = 506.35, 506.27
CX, CY = 334.19, 335.43
NEAR = 0.05

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "training_data")
LOG_DIR = os.path.join(DATA_DIR, "train_logs")
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(os.path.join(LOG_DIR, "sample_predictions"), exist_ok=True)

# ── 6ch 归一化 ─────────────────────────────────────────────────────
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


# ── Dataset ────────────────────────────────────────────────────────
class StackingDataset(Dataset):
    def __init__(self, root_dir, num_points=NUM_POINTS, augment=False):
        self.scene_dirs = sorted([
            os.path.join(root_dir, d) for d in os.listdir(root_dir)
            if os.path.isdir(os.path.join(root_dir, d))
        ])
        self.num_points = num_points
        self.augment = augment

    def __len__(self):
        return len(self.scene_dirs)

    def __getitem__(self, idx):
        d = self.scene_dirs[idx]
        depth = np.load(os.path.join(d, "depth_noisy.npy"))
        labels = np.load(os.path.join(d, "semantic_labels.npy"))

        # depth → point cloud
        h, w = depth.shape
        valid = depth > NEAR
        uu, vv = np.meshgrid(np.arange(w, dtype=np.float32),
                             np.arange(h, dtype=np.float32))
        Z = depth[valid]
        X = (uu[valid] - CX) * Z / FX
        Y = (vv[valid] - CY) * Z / FY
        pts = np.stack([X, Y, Z], axis=-1)
        lbs = labels[valid]

        # sample
        n = len(pts)
        if n >= self.num_points:
            idxs = np.random.choice(n, self.num_points, replace=False)
        else:
            idxs = np.random.choice(n, self.num_points, replace=True)
        pts_sampled = pts[idxs]
        lbs_sampled = lbs[idxs]

        # augment
        if self.augment:
            # random rotation around Z
            ang = np.random.uniform(0, 2 * np.pi)
            cos_a, sin_a = np.cos(ang), np.sin(ang)
            rot = np.array([[cos_a, -sin_a, 0], [sin_a, cos_a, 0], [0, 0, 1]], dtype=np.float32)
            pts_sampled = pts_sampled @ rot.T
            # random scale
            s = np.random.uniform(0.8, 1.2)
            pts_sampled *= s
            # random jitter
            pts_sampled += np.random.normal(0, 0.005, pts_sampled.shape).astype(np.float32)

        # 6ch normalize
        feat = normalize_points_6ch(pts_sampled)

        return (torch.from_numpy(feat).float(),
                torch.from_numpy(lbs_sampled).long())


# ── Metrics ─────────────────────────────────────────────────────────
def compute_iou(pred, target, num_classes):
    ious = []
    for c in range(num_classes):
        intersection = ((pred == c) & (target == c)).sum().float()
        union = ((pred == c) | (target == c)).sum().float()
        ious.append((intersection + 1) / (union + 1))  # smooth
    miou = sum(ious) / len(ious)
    return miou, ious


# ── Training ────────────────────────────────────────────────────────
def train():
    logger.info(f"Device: {DEVICE}")
    logger.info(f"Data dir: {DATA_DIR}")
    logger.info(f"Log dir: {LOG_DIR}")

    # dataset
    train_set = StackingDataset(os.path.join(DATA_DIR, "train"), augment=True)
    val_set = StackingDataset(os.path.join(DATA_DIR, "val"), augment=False)
    logger.info(f"Train: {len(train_set)} scenes, Val: {len(val_set)} scenes")

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=0, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=0, pin_memory=True)

    # model
    model = PointNetPlusPlusAttentionSeg(
        input_channels=INPUT_CHANNELS, num_classes=NUM_CLASSES).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model params: {n_params:,}")

    # loss weights (object class weighted higher to counter table dominance)
    class_weights = torch.tensor([1.0, 3.0], device=DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

    history = defaultdict(list)
    best_miou = 0.0
    best_epoch = 0
    patience_counter = 0

    logger.info("=" * 50)
    logger.info("Training started")
    logger.info("=" * 50)

    for epoch in range(1, NUM_EPOCHS + 1):
        # ── train ──
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

        # ── val ──
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

        # record
        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)
        history["train_miou"].append(avg_train_miou)
        history["val_miou"].append(avg_val_miou)
        history["val_iou_table"].append(val_ious.tolist())
        history["lr"].append(lr_now)

        # print
        logger.info(
            f"Epoch {epoch:3d}/{NUM_EPOCHS} | "
            f"LR {lr_now:.5f} | "
            f"Train Loss {avg_train_loss:.4f} mIoU {avg_train_miou:.4f} | "
            f"Val Loss {avg_val_loss:.4f} mIoU {avg_val_miou:.4f} | "
            f"IoU[table={val_ious[0]:.3f} obj={val_ious[1]:.3f}]"
        )

        # best model
        if avg_val_miou > best_miou:
            best_miou = avg_val_miou
            best_epoch = epoch
            patience_counter = 0
            torch.save({
                "epoch": epoch,
                "miou": best_miou,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            }, os.path.join(LOG_DIR, "best_model.pth"))
            logger.info(f"  >>> Best model saved (mIoU={best_miou:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOP_PATIENCE:
                logger.info(f"Early stopping at epoch {epoch}")
                break

    # ── save artifacts ──────────────────────────────────────────
    # loss curves
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    axes[0].plot(history["train_loss"], label="Train")
    axes[0].plot(history["val_loss"], label="Val")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Cross Entropy")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(history["train_miou"], label="Train")
    axes[1].plot(history["val_miou"], label="Val")
    axes[1].set_title("Mean IoU")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("mIoU")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # per-class IoU over time
    ious_arr = np.array(history["val_iou_table"])
    axes[2].plot(ious_arr[:, 0], label="Table")
    axes[2].plot(ious_arr[:, 1], label="Object")
    axes[2].set_title("Per-Class IoU (Val)")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("IoU")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(LOG_DIR, "loss_curves.png"), dpi=150)
    plt.close()
    logger.info("Loss curves saved")

    # training report
    report = f"""Training Report
===============
Model: PointNet++ with Attention (PP-Attention)
Input: 6ch normalized point cloud (XYZ + 3 Z normalizations)
Output: 2-class semantic segmentation (0=table, 1=object)
Parameters: {n_params:,}

Training setup:
  - Dataset: {len(train_set)} train / {len(val_set)} val scenes
  - Batch size: {BATCH_SIZE}
  - Points per sample: {NUM_POINTS}
  - Optimizer: AdamW (lr={LR}, weight_decay={WEIGHT_DECAY})
  - Scheduler: CosineAnnealing ({NUM_EPOCHS} epochs)
  - Loss: Weighted CrossEntropy (table=1.0, object=3.0)
  - Early stopping: patience={EARLY_STOP_PATIENCE}

Results:
  - Best epoch: {best_epoch}
  - Best mIoU: {best_miou:.4f}
  - Best IoU (table): {ious_arr[best_epoch-1][0]:.4f}
  - Best IoU (object): {ious_arr[best_epoch-1][1]:.4f}
  - Final Train Loss: {history['train_loss'][-1]:.4f}
  - Final Val Loss: {history['val_loss'][-1]:.4f}

Model saved to: {os.path.join(LOG_DIR, 'best_model.pth')}
"""
    with open(os.path.join(LOG_DIR, "training_report.txt"), "w") as f:
        f.write(report)
    logger.info("Training report saved")

    # metrics JSON
    with open(os.path.join(LOG_DIR, "metrics.json"), "w") as f:
        json.dump({
            "train_loss": history["train_loss"],
            "val_loss": history["val_loss"],
            "train_miou": history["train_miou"],
            "val_miou": history["val_miou"],
            "val_iou_per_class": history["val_iou_table"],
            "lr": history["lr"],
            "best_epoch": best_epoch,
            "best_miou": best_miou,
        }, f, indent=2)

    # sample predictions on val set
    logger.info("Generating sample predictions...")
    model.eval()
    val_set_np = StackingDataset(os.path.join(DATA_DIR, "val"), augment=False)
    sample_idxs = np.random.choice(len(val_set_np), min(8, len(val_set_np)), replace=False)

    with torch.no_grad():
        for si, idx in enumerate(sample_idxs):
            feat, target = val_set_np[idx]
            feat_b = feat.unsqueeze(0).to(DEVICE)
            logits, _, _ = model(feat_b)
            pred = logits.argmax(dim=-1).squeeze(0).cpu().numpy()
            target_np = target.numpy()
            feat_np = feat.numpy()

            # reconstruct XYZ from features
            # feat channels: xy_norm, z_abs, z_rel, z_h
            # We can't perfectly reconstruct XYZ, but we can viz projection
            # Use the xy normalized coords as a rough 2D scatter
            fig, axes2 = plt.subplots(1, 2, figsize=(12, 5))
            xy = feat_np[:, :2]
            axes2[0].scatter(xy[:, 0], xy[:, 1], c=target_np, s=1,
                             cmap="RdYlGn", vmin=0, vmax=1)
            axes2[0].set_title(f"Ground Truth  (sample {si})")
            axes2[0].set_aspect("equal")
            axes2[1].scatter(xy[:, 0], xy[:, 1], c=pred, s=1,
                             cmap="RdYlGn", vmin=0, vmax=1)
            axes2[1].set_title(f"Prediction  (sample {si})")
            axes2[1].set_aspect("equal")
            plt.tight_layout()
            plt.savefig(os.path.join(LOG_DIR, "sample_predictions",
                                     f"sample_{si:02d}.png"), dpi=120)
            plt.close()

    logger.info("Sample predictions saved")

    # model architecture summary
    arch = f"""Model Architecture
==================
{model}

Input: (B, {NUM_POINTS}, {INPUT_CHANNELS})
  Channels 0-1: XY centered + scaled
  Channel 2: Z absolute normalized
  Channel 3: Z relative to mean
  Channel 4: Z height ratio
  Channel 5: (unused placeholder)
Total parameters: {n_params:,}
Device: {DEVICE}
"""
    with open(os.path.join(LOG_DIR, "architecture.txt"), "w") as f:
        f.write(arch)

    logger.info(f"\nAll artifacts saved to: {LOG_DIR}")
    return model, history


if __name__ == "__main__":
    train()

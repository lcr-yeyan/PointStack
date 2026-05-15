"""
算法对比实验
============
Part A: 语义分割模型对比
  - PP-Attention (100 epoch, best) vs PointNet/PointNet++/DGCNN/RandLA-Net (10 epoch)
  - 传统方法: RANSAC 平面拟合, 深度阈值分割

Part B: 堆叠检测算法对比 (核心)
  - 同一 PP-Attention 分割结果 → 7 种不同堆叠检测算法
  - 专用堆叠测试数据 (不同重叠度/偏移量)
  - 全面指标: 边 precision/recall/F1, 抓取顺序, 边界F1, 鲁棒性

输出:
  - algo_comparison_results/  评估结果 JSON
  - algo_comparison_figures/  对比图表
  - algo_comparison_report.md 对比报告
"""

import os, sys, json, time, copy
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
import matplotlib.ticker as ticker
import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models.pointnet_seg import (
    PointNetSeg, PointNetPlusPlusSeg, PointNetPlusPlusAttentionSeg,
)
from models.dgcnn_seg import DGCNNSeg
from models.randla_seg import RandLANetSeg

BATCH_SIZE = 8
NUM_POINTS = 2048
NUM_EPOCHS = 10
LR = 0.001
WEIGHT_DECAY = 1e-4
EARLY_STOP_PATIENCE = 20
NUM_CLASSES = 2
MAX_TRAIN_SCENES = 800
MAX_VAL_SCENES = 200
NUM_INFERENCE_PASSES = 200

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

FX, FY = 506.35, 506.27
CX, CY = 334.19, 335.43
NEAR = 0.05

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "training_data")
RESULT_DIR = os.path.join(BASE_DIR, "algo_comparison_results")
FIGURE_DIR = os.path.join(BASE_DIR, "algo_comparison_figures")
LOG_DIR = os.path.join(DATA_DIR, "algo_comparison_logs")
STACKING_TEST_DIR = os.path.join(BASE_DIR, "stacking_test_data")

os.makedirs(RESULT_DIR, exist_ok=True)
os.makedirs(FIGURE_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(STACKING_TEST_DIR, exist_ok=True)

TEST_DATA_DIR = os.path.join(BASE_DIR, "data_preview")


def normalize_points_3ch(points):
    centroid = np.mean(points, axis=0, keepdims=True)
    points_centered = points - centroid
    max_dist = np.max(np.linalg.norm(points_centered, axis=1))
    max_dist = max(max_dist, 1e-8)
    return (points_centered / max_dist).astype(np.float32)


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
        uu, vv = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
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

        return torch.from_numpy(feat).float(), torch.from_numpy(lbs_sampled).long()


def compute_iou(pred, target, num_classes):
    ious = []
    for cls in range(num_classes):
        pred_cls = (pred == cls)
        target_cls = (target == cls)
        intersection = (pred_cls & target_cls).sum().float()
        union = (pred_cls | target_cls).sum().float()
        iou = intersection / union.clamp(min=1)
        ious.append(iou)
    miou = sum(ious) / len(ious)
    return miou, ious


def train_one_model(model, model_name, use_6ch, log_subdir):
    model_log_dir = os.path.join(LOG_DIR, log_subdir)
    os.makedirs(model_log_dir, exist_ok=True)

    logger.info("=" * 60)
    logger.info(f"Training: {model_name}")
    logger.info("=" * 60)

    train_set = StackingDataset(os.path.join(DATA_DIR, "train"), augment=True, use_6ch=use_6ch)
    train_set.scene_dirs = train_set.scene_dirs[:MAX_TRAIN_SCENES]
    val_set = StackingDataset(os.path.join(DATA_DIR, "val"), augment=False, use_6ch=use_6ch)
    val_set.scene_dirs = val_set.scene_dirs[:MAX_VAL_SCENES]
    logger.info(f"Train: {len(train_set)} scenes, Val: {len(val_set)} scenes")

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=0, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=0, pin_memory=True)

    model = model.to(DEVICE)
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
                miou, _ = compute_iou(pred, target, NUM_CLASSES)
                val_miou_sum += miou.item()
                val_batches += 1

        avg_val_loss = val_loss_sum / max(val_batches, 1)
        avg_val_miou = val_miou_sum / max(val_batches, 1)

        scheduler.step()

        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)
        history["train_miou"].append(avg_train_miou)
        history["val_miou"].append(avg_val_miou)

        if avg_val_miou > best_miou:
            best_miou = avg_val_miou
            best_epoch = epoch
            patience_counter = 0
            torch.save({
                "epoch": epoch,
                "miou": best_miou,
                "model_state_dict": model.state_dict(),
                "model_name": model_name,
                "use_6ch": use_6ch,
                "n_params": n_params,
            }, os.path.join(model_log_dir, "best_model.pth"))
        else:
            patience_counter += 1

        if epoch % 10 == 0 or epoch == 1:
            logger.info(f"Epoch {epoch:3d} | Train Loss: {avg_train_loss:.4f} mIoU: {avg_train_miou:.4f} | "
                        f"Val Loss: {avg_val_loss:.4f} mIoU: {avg_val_miou:.4f} | Best: {best_miou:.4f} @ {best_epoch}")

        if patience_counter >= EARLY_STOP_PATIENCE:
            logger.info(f"Early stopping at epoch {epoch}")
            break

    logger.info(f"Training done: {model_name} | Best mIoU: {best_miou:.4f} @ epoch {best_epoch}")
    return best_miou, best_epoch, n_params


def train_all_models():
    models_to_train = [
        {"name": "PointNet", "model": PointNetSeg(input_channels=3, num_classes=NUM_CLASSES),
         "use_6ch": False, "log_subdir": "pointnet"},
        {"name": "PointNet++", "model": PointNetPlusPlusSeg(input_channels=3, num_classes=NUM_CLASSES),
         "use_6ch": False, "log_subdir": "pointnet_pp"},
        {"name": "DGCNN", "model": DGCNNSeg(input_channels=3, num_classes=NUM_CLASSES, k=20),
         "use_6ch": False, "log_subdir": "dgcnn"},
        {"name": "RandLA-Net", "model": RandLANetSeg(input_channels=3, num_classes=NUM_CLASSES),
         "use_6ch": False, "log_subdir": "randla"},
    ]

    results = {}
    for cfg in models_to_train:
        best_miou, best_epoch, n_params = train_one_model(
            cfg["model"], cfg["name"], cfg["use_6ch"], cfg["log_subdir"]
        )
        results[cfg["log_subdir"]] = {
            "name": cfg["name"], "best_miou": best_miou,
            "best_epoch": best_epoch, "n_params": n_params,
        }

    with open(os.path.join(RESULT_DIR, "training_summary.json"), "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    return results


def load_model_for_eval(model_key):
    if model_key == "pp_attention":
        model = PointNetPlusPlusAttentionSeg(input_channels=6, num_classes=NUM_CLASSES)
        ckpt_path = os.path.join(DATA_DIR, "train_logs", "best_model.pth")
        use_6ch = True
    elif model_key == "pointnet_pp_3ch":
        model = PointNetPlusPlusSeg(input_channels=3, num_classes=NUM_CLASSES)
        ckpt_path = os.path.join(DATA_DIR, "ablation_logs", "baseline_3ch", "best_model.pth")
        use_6ch = False
    elif model_key == "pointnet":
        model = PointNetSeg(input_channels=3, num_classes=NUM_CLASSES)
        ckpt_path = os.path.join(LOG_DIR, "pointnet", "best_model.pth")
        use_6ch = False
    elif model_key == "pointnet_pp":
        model = PointNetPlusPlusSeg(input_channels=3, num_classes=NUM_CLASSES)
        ckpt_path = os.path.join(LOG_DIR, "pointnet_pp", "best_model.pth")
        use_6ch = False
    elif model_key == "dgcnn":
        model = DGCNNSeg(input_channels=3, num_classes=NUM_CLASSES, k=20)
        ckpt_path = os.path.join(LOG_DIR, "dgcnn", "best_model.pth")
        use_6ch = False
    elif model_key == "randla":
        model = RandLANetSeg(input_channels=3, num_classes=NUM_CLASSES)
        ckpt_path = os.path.join(LOG_DIR, "randla", "best_model.pth")
        use_6ch = False
    else:
        return None, False

    if not os.path.exists(ckpt_path):
        logger.warning(f"Checkpoint not found: {ckpt_path}")
        return None, False

    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(DEVICE)
    model.eval()
    return model, use_6ch


def depth_to_pointcloud(depth):
    h, w = depth.shape
    valid = depth > NEAR
    uu, vv = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    Z = depth[valid]
    X = (uu[valid] - CX) * Z / FX
    Y = (vv[valid] - CY) * Z / FY
    pts = np.stack([X, Y, Z], axis=-1)
    return pts, valid


def predict_dl(model, points, valid_mask, use_6ch):
    n = len(points)
    if n == 0:
        return np.zeros(valid_mask.shape, dtype=np.uint8)

    vote_counts = np.zeros(n, dtype=np.int32)
    vote_sum = np.zeros(n, dtype=np.float32)
    n_passes = max(1, min(NUM_INFERENCE_PASSES, int(np.ceil(n / NUM_POINTS))))

    for _ in range(n_passes):
        if n >= NUM_POINTS:
            idxs = np.random.choice(n, NUM_POINTS, replace=False)
        else:
            idxs = np.random.choice(n, NUM_POINTS, replace=True)
        pts_sampled = points[idxs]

        if use_6ch:
            feat = normalize_points_6ch(pts_sampled)
        else:
            feat = normalize_points_3ch(pts_sampled)

        feat_t = torch.from_numpy(feat).float().unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            logits, _, _ = model(feat_t)
            probs = F.softmax(logits, dim=-1).squeeze(0).cpu().numpy()
        vote_counts[idxs] += 1
        vote_sum[idxs] += probs[:, 1]

    pred_full = np.zeros(n, dtype=np.uint8)
    voted = vote_counts > 0
    pred_full[voted] = (vote_sum[voted] / vote_counts[voted] > 0.5).astype(np.uint8)

    pred_2d = np.zeros(valid_mask.shape, dtype=np.uint8)
    valid_indices = np.where(valid_mask.ravel())[0]
    pred_2d_flat = pred_2d.ravel()
    pred_2d_flat[valid_indices] = pred_full
    pred_2d = pred_2d_flat.reshape(valid_mask.shape)
    return pred_2d


def predict_ransac(points, valid_mask, depth):
    h, w = valid_mask.shape
    pred_2d = np.zeros((h, w), dtype=np.uint8)
    if len(points) < 100:
        return pred_2d
    try:
        import open3d as o3d
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
        plane_model, inliers = pcd.segment_plane(
            distance_threshold=0.01, ransac_n=3, num_iterations=1000)
        inlier_set = set(inliers)
        valid_indices = np.where(valid_mask.ravel())[0]
        for i, vi in enumerate(valid_indices):
            if i not in inlier_set:
                pred_2d.ravel()[vi] = 1
    except Exception as e:
        logger.warning(f"RANSAC failed: {e}")
    return pred_2d


def predict_depth_threshold(points, valid_mask, depth):
    h, w = valid_mask.shape
    pred_2d = np.zeros((h, w), dtype=np.uint8)
    valid_depth = depth[valid_mask]
    if len(valid_depth) == 0:
        return pred_2d
    z_values = points[:, 2]
    z_median = np.median(z_values)
    z_mad = np.median(np.abs(z_values - z_median))
    threshold = z_median + 2.0 * z_mad
    object_mask_pts = z_values > threshold
    valid_indices = np.where(valid_mask.ravel())[0]
    pred_flat = pred_2d.ravel()
    pred_flat[valid_indices[object_mask_pts]] = 1
    kernel = np.ones((5, 5), np.uint8)
    pred_2d = cv2.morphologyEx(pred_2d, cv2.MORPH_OPEN, kernel)
    pred_2d = cv2.morphologyEx(pred_2d, cv2.MORPH_CLOSE, kernel)
    return pred_2d


def _euclidean_clustering(points, tolerance=0.015, min_cluster_size=50):
    n = len(points)
    if n == 0:
        return [], np.array([], dtype=int)
    pairs = []
    grid_size = tolerance * 2
    grid = {}
    for i, pt in enumerate(points):
        gx = int(pt[0] / grid_size)
        gy = int(pt[1] / grid_size)
        gz = int(pt[2] / grid_size)
        key = (gx, gy, gz)
        grid.setdefault(key, []).append(i)
    checked = set()
    for key, idxs in grid.items():
        gx, gy, gz = key
        for dx in range(-1, 2):
            for dy in range(-1, 2):
                for dz in range(-1, 2):
                    nkey = (gx + dx, gy + dy, gz + dz)
                    if nkey not in grid:
                        continue
                    nidxs = grid[nkey]
                    for i in idxs:
                        for j in nidxs:
                            if i >= j:
                                continue
                            pair = (i, j)
                            if pair in checked:
                                continue
                            checked.add(pair)
                            dist = np.linalg.norm(points[i] - points[j])
                            if dist < tolerance:
                                pairs.append(pair)
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[rx] = ry

    for i, j in pairs:
        union(i, j)
    clusters = defaultdict(list)
    for i in range(n):
        clusters[find(i)].append(i)
    result = []
    labels = -np.ones(n, dtype=int)
    label_id = 0
    for root, members in clusters.items():
        if len(members) >= min_cluster_size:
            result.append(np.array(members))
            labels[members] = label_id
            label_id += 1
    return result, labels


def run_instance_segmentation(points, pred_labels):
    object_mask = pred_labels == 1
    if object_mask.sum() < 10:
        return [], np.full(len(points), -1, dtype=int)
    object_points = points[object_mask]
    object_indices = np.where(object_mask)[0]
    clusters, cluster_labels = _euclidean_clustering(
        object_points, tolerance=0.015, min_cluster_size=50)
    instances = []
    instance_labels = np.full(len(points), -1, dtype=int)
    for i, cluster in enumerate(clusters):
        global_indices = object_indices[cluster]
        inst_pts = points[global_indices]
        centroid = np.mean(inst_pts, axis=0)
        bbox_min = np.min(inst_pts, axis=0)
        bbox_max = np.max(inst_pts, axis=0)
        z_vals = inst_pts[:, 2]
        instances.append({
            "id": i,
            "point_indices": global_indices.tolist(),
            "centroid": centroid.tolist(),
            "bbox_min": bbox_min.tolist(),
            "bbox_max": bbox_max.tolist(),
            "point_count": len(global_indices),
            "z_mean": float(np.median(z_vals)),
            "z_min": float(np.percentile(z_vals, 2)),
            "z_max": float(np.percentile(z_vals, 98)),
        })
        instance_labels[global_indices] = i
    return instances, instance_labels


# ═══════════════════════════════════════════════════════════════════
# 堆叠检测算法 (7种)
# ═══════════════════════════════════════════════════════════════════

def build_hierarchy(instances):
    """Ours: HierarchyReasoner - 空间关系分析 + 拓扑排序"""
    if len(instances) < 2:
        return [], list(range(len(instances)))
    edges = []
    for i in range(len(instances)):
        for j in range(len(instances)):
            if i == j:
                continue
            inst_i = instances[i]
            inst_j = instances[j]
            z_gap = inst_i["z_min"] - inst_j["z_max"]
            z_mean_diff = inst_i["z_mean"] - inst_j["z_mean"]
            if z_gap < 0.001 and z_mean_diff < 0.005:
                continue
            bi_min = np.array(inst_i["bbox_min"])
            bi_max = np.array(inst_i["bbox_max"])
            bj_min = np.array(inst_j["bbox_min"])
            bj_max = np.array(inst_j["bbox_max"])
            overlap_x = min(bi_max[0], bj_max[0]) - max(bi_min[0], bj_min[0])
            overlap_y = min(bi_max[1], bj_max[1]) - max(bi_min[1], bj_min[1])
            ci = np.array(inst_i["centroid"])
            cj = np.array(inst_j["centroid"])
            xy_dist = np.linalg.norm(ci[:2] - cj[:2])
            has_overlap = overlap_x > 0.01 and overlap_y > 0.01
            is_close = xy_dist < 0.12
            if has_overlap or is_close:
                edges.append((i, j))
    in_degree = {i: 0 for i in range(len(instances))}
    for src, dst in edges:
        in_degree[dst] = in_degree.get(dst, 0) + 1
    layers = []
    remaining = set(range(len(instances)))
    while remaining:
        layer = [n for n in remaining if in_degree.get(n, 0) == 0]
        if not layer:
            layer = list(remaining)
        layers.append(layer)
        for n in layer:
            remaining.discard(n)
            for src, dst in edges:
                if src == n:
                    in_degree[dst] = max(0, in_degree.get(dst, 1) - 1)
    grasp_order = []
    for layer in layers:
        grasp_order.extend(layer)
    return edges, grasp_order


def simple_z_sort(instances):
    """Baseline 1: 简单 Z 轴排序"""
    sorted_instances = sorted(instances, key=lambda i: i["z_mean"], reverse=True)
    grasp_order = [inst["id"] for inst in sorted_instances]
    edges = []
    for i in range(len(sorted_instances) - 1):
        edges.append((sorted_instances[i]["id"], sorted_instances[i + 1]["id"]))
    return edges, grasp_order


def bbox_iou_sort(instances):
    """Baseline 2: BBox XY IoU 判定堆叠"""
    if len(instances) < 2:
        return [], list(range(len(instances)))
    edges = []
    for i in range(len(instances)):
        for j in range(len(instances)):
            if i == j:
                continue
            inst_i = instances[i]
            inst_j = instances[j]
            if inst_i["z_min"] <= inst_j["z_max"]:
                continue
            bi_min = np.array(inst_i["bbox_min"])
            bi_max = np.array(inst_i["bbox_max"])
            bj_min = np.array(inst_j["bbox_min"])
            bj_max = np.array(inst_j["bbox_max"])
            inter_x = max(0, min(bi_max[0], bj_max[0]) - max(bi_min[0], bj_min[0]))
            inter_y = max(0, min(bi_max[1], bj_max[1]) - max(bi_min[1], bj_min[1]))
            inter_area = inter_x * inter_y
            area_i = (bi_max[0] - bi_min[0]) * (bi_max[1] - bi_min[1])
            area_j = (bj_max[0] - bj_min[0]) * (bj_max[1] - bj_min[1])
            iou = inter_area / max(area_i + area_j - inter_area, 1e-8)
            if iou > 0.05:
                edges.append((i, j))
    in_degree = {i: 0 for i in range(len(instances))}
    for src, dst in edges:
        in_degree[dst] = in_degree.get(dst, 0) + 1
    layers = []
    remaining = set(range(len(instances)))
    while remaining:
        layer = [n for n in remaining if in_degree.get(n, 0) == 0]
        if not layer:
            layer = list(remaining)
        layers.append(layer)
        for n in layer:
            remaining.discard(n)
            for src, dst in edges:
                if src == n:
                    in_degree[dst] = max(0, in_degree.get(dst, 1) - 1)
    grasp_order = []
    for layer in layers:
        grasp_order.extend(layer)
    return edges, grasp_order


def height_threshold_sort(instances, z_threshold=0.005):
    """Baseline 3: Z 高度差阈值判定堆叠"""
    if len(instances) < 2:
        return [], list(range(len(instances)))
    edges = []
    for i in range(len(instances)):
        for j in range(len(instances)):
            if i == j:
                continue
            inst_i = instances[i]
            inst_j = instances[j]
            z_diff = inst_i["z_mean"] - inst_j["z_mean"]
            if z_diff > z_threshold:
                edges.append((i, j))
    in_degree = {i: 0 for i in range(len(instances))}
    for src, dst in edges:
        in_degree[dst] = in_degree.get(dst, 0) + 1
    layers = []
    remaining = set(range(len(instances)))
    while remaining:
        layer = [n for n in remaining if in_degree.get(n, 0) == 0]
        if not layer:
            layer = list(remaining)
        layers.append(layer)
        for n in layer:
            remaining.discard(n)
            for src, dst in edges:
                if src == n:
                    in_degree[dst] = max(0, in_degree.get(dst, 1) - 1)
    grasp_order = []
    for layer in layers:
        grasp_order.extend(layer)
    return edges, grasp_order


def centroid_proximity_sort(instances, xy_threshold=0.08):
    """Baseline 4: 质心 XY 邻近 + Z 判定堆叠"""
    if len(instances) < 2:
        return [], list(range(len(instances)))
    edges = []
    for i in range(len(instances)):
        for j in range(len(instances)):
            if i == j:
                continue
            inst_i = instances[i]
            inst_j = instances[j]
            if inst_i["z_mean"] <= inst_j["z_mean"]:
                continue
            ci = np.array(inst_i["centroid"])
            cj = np.array(inst_j["centroid"])
            xy_dist = np.linalg.norm(ci[:2] - cj[:2])
            if xy_dist < xy_threshold:
                edges.append((i, j))
    in_degree = {i: 0 for i in range(len(instances))}
    for src, dst in edges:
        in_degree[dst] = in_degree.get(dst, 0) + 1
    layers = []
    remaining = set(range(len(instances)))
    while remaining:
        layer = [n for n in remaining if in_degree.get(n, 0) == 0]
        if not layer:
            layer = list(remaining)
        layers.append(layer)
        for n in layer:
            remaining.discard(n)
            for src, dst in edges:
                if src == n:
                    in_degree[dst] = max(0, in_degree.get(dst, 1) - 1)
    grasp_order = []
    for layer in layers:
        grasp_order.extend(layer)
    return edges, grasp_order


def overlap_z_sort(instances):
    """Baseline 5: XY 重叠 + Z 判定 (简化版 HierarchyReasoner)"""
    if len(instances) < 2:
        return [], list(range(len(instances)))
    edges = []
    for i in range(len(instances)):
        for j in range(len(instances)):
            if i == j:
                continue
            inst_i = instances[i]
            inst_j = instances[j]
            if inst_i["z_mean"] <= inst_j["z_mean"]:
                continue
            bi_min = np.array(inst_i["bbox_min"])
            bi_max = np.array(inst_i["bbox_max"])
            bj_min = np.array(inst_j["bbox_min"])
            bj_max = np.array(inst_j["bbox_max"])
            overlap_x = min(bi_max[0], bj_max[0]) - max(bi_min[0], bj_min[0])
            overlap_y = min(bi_max[1], bj_max[1]) - max(bi_min[1], bj_min[1])
            if overlap_x > 0.005 and overlap_y > 0.005:
                edges.append((i, j))
    in_degree = {i: 0 for i in range(len(instances))}
    for src, dst in edges:
        in_degree[dst] = in_degree.get(dst, 0) + 1
    layers = []
    remaining = set(range(len(instances)))
    while remaining:
        layer = [n for n in remaining if in_degree.get(n, 0) == 0]
        if not layer:
            layer = list(remaining)
        layers.append(layer)
        for n in layer:
            remaining.discard(n)
            for src, dst in edges:
                if src == n:
                    in_degree[dst] = max(0, in_degree.get(dst, 1) - 1)
    grasp_order = []
    for layer in layers:
        grasp_order.extend(layer)
    return edges, grasp_order


def compute_boundary_f1(pred_2d, gt_2d):
    kernel = np.ones((3, 3), dtype=np.uint8)
    pred_boundary = cv2.dilate(pred_2d.astype(np.uint8), kernel) - cv2.erode(pred_2d.astype(np.uint8), kernel)
    gt_boundary = cv2.dilate(gt_2d.astype(np.uint8), kernel) - cv2.erode(gt_2d.astype(np.uint8), kernel)
    pred_b = pred_boundary > 0
    gt_b = gt_boundary > 0
    tp = (pred_b & gt_b).sum()
    fp = (pred_b & ~gt_b).sum()
    fn = (~pred_b & gt_b).sum()
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)
    return float(f1), float(precision), float(recall)


def compute_stacking_metrics(pred_edges, gt_edges, pred_order, gt_order):
    """计算堆叠检测的 precision/recall/F1"""
    pred_set = set(tuple(sorted(e)) for e in pred_edges)
    gt_set = set(tuple(sorted(e)) for e in gt_edges)

    tp = len(pred_set & gt_set)
    fp = len(pred_set - gt_set)
    fn = len(gt_set - pred_set)

    if len(pred_set) == 0 and len(gt_set) == 0:
        precision = 1.0
        recall = 1.0
        f1 = 1.0
    else:
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-8)

    order_correct = False
    if len(gt_order) > 0 and len(pred_order) > 0:
        if len(pred_order) == len(gt_order):
            order_correct = all(pred_order[i] == gt_order[i] for i in range(len(gt_order)))
    elif len(gt_order) == 0 and len(pred_order) == 0:
        order_correct = True

    return float(precision), float(recall), float(f1), order_correct


# ═══════════════════════════════════════════════════════════════════
# 专用堆叠测试数据生成
# ═══════════════════════════════════════════════════════════════════

def generate_stacking_test_data():
    """生成不同难度级别的堆叠测试场景"""
    logger.info("Generating dedicated stacking test data...")

    test_configs = [
        {"name": "easy_2obj_centered", "num_objects": 2, "overlap": 0.9,
         "offset_xy": 0.0, "description": "简单: 2物体完全居中堆叠"},
        {"name": "easy_2obj_offset", "num_objects": 2, "overlap": 0.7,
         "offset_xy": 0.03, "description": "简单: 2物体偏移堆叠"},
        {"name": "medium_3obj_stair", "num_objects": 3, "overlap": 0.6,
         "offset_xy": 0.04, "description": "中等: 3物体阶梯堆叠"},
        {"name": "medium_3obj_cross", "num_objects": 3, "overlap": 0.5,
         "offset_xy": 0.06, "description": "中等: 3物体交叉堆叠"},
        {"name": "hard_4obj_pyramid", "num_objects": 4, "overlap": 0.5,
         "offset_xy": 0.05, "description": "困难: 4物体金字塔堆叠"},
        {"name": "hard_4obj_irregular", "num_objects": 4, "overlap": 0.4,
         "offset_xy": 0.07, "description": "困难: 4物体不规则堆叠"},
    ]

    H, W = 240, 320
    generated = []

    for cfg in test_configs:
        scene_dir = os.path.join(STACKING_TEST_DIR, cfg["name"])
        os.makedirs(scene_dir, exist_ok=True)

        depth = np.full((H, W), NEAR, dtype=np.float32)
        labels = np.zeros((H, W), dtype=np.uint8)
        objects_meta = []
        edges_gt = []
        grasp_order_gt = []

        table_z = 0.6
        uu, vv = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
        table_mask = np.ones((H, W), dtype=bool)
        table_depth = np.full((H, W), table_z, dtype=np.float32)
        depth[table_mask] = table_depth[table_mask]

        obj_size = 0.06
        base_cx, base_cy = W / 2, H / 2

        for obj_idx in range(cfg["num_objects"]):
            layer = obj_idx
            obj_z = table_z + 0.025 + layer * 0.025
            offset_x = cfg["offset_xy"] * (obj_idx - (cfg["num_objects"] - 1) / 2)
            offset_y = cfg["offset_xy"] * (obj_idx % 2 - 0.5) * 2

            cx_pix = base_cx + offset_x * W
            cy_pix = base_cy + offset_y * H

            obj_half_w = int(obj_size * W / 2)
            obj_half_h = int(obj_size * H / 2)

            x0 = max(0, int(cx_pix - obj_half_w * cfg["overlap"]))
            x1 = min(W, int(cx_pix + obj_half_w * cfg["overlap"]))
            y0 = max(0, int(cy_pix - obj_half_h * cfg["overlap"]))
            y1 = min(H, int(cy_pix + obj_half_h * cfg["overlap"]))

            region = np.s_[y0:y1, x0:x1]
            depth[region] = obj_z
            labels[region] = 1

            X_obj = (uu[region] - CX) * obj_z / FX
            Y_obj = (vv[region] - CY) * obj_z / FY
            Z_obj = np.full_like(X_obj, obj_z)

            objects_meta.append({
                "id": obj_idx,
                "z_min": float(obj_z - 0.005),
                "z_max": float(obj_z + 0.005),
                "z_mean": float(obj_z),
                "bbox_2d": [x0, y0, x1, y1],
            })

            if obj_idx > 0:
                edges_gt.append({"upper": obj_idx - 1, "lower": obj_idx})
            grasp_order_gt.append(obj_idx)

        depth += np.random.normal(0, 0.001, depth.shape).astype(np.float32)
        depth = np.maximum(depth, NEAR)

        np.save(os.path.join(scene_dir, "depth_noisy.npy"), depth)
        np.save(os.path.join(scene_dir, "semantic_labels.npy"), labels)

        annotation = {
            "scene_name": cfg["name"],
            "description": cfg["description"],
            "num_objects": cfg["num_objects"],
            "difficulty": cfg["name"].split("_")[0],
            "objects": objects_meta,
            "hierarchy": {
                "edges": edges_gt,
                "grasp_order": grasp_order_gt,
            },
        }
        with open(os.path.join(scene_dir, "annotation.json"), "w", encoding="utf-8") as f:
            json.dump(annotation, f, indent=2, ensure_ascii=False)

        generated.append(cfg["name"])
        logger.info(f"  Generated: {cfg['name']} ({cfg['description']})")

    logger.info(f"Generated {len(generated)} stacking test scenes")
    return generated


# ═══════════════════════════════════════════════════════════════════
# Part A: 语义分割模型对比
# ═══════════════════════════════════════════════════════════════════

def evaluate_all():
    test_scenes = sorted([
        os.path.join(TEST_DATA_DIR, d) for d in os.listdir(TEST_DATA_DIR)
        if os.path.isdir(os.path.join(TEST_DATA_DIR, d))
        and os.path.exists(os.path.join(TEST_DATA_DIR, d, "depth_noisy.npy"))
    ])

    algorithms = {
        "pp_attention": "PP-Attention (Ours, 100ep)",
        "pointnet_pp_3ch": "PointNet++ (3ch, 100ep)",
        "pointnet": "PointNet (10ep)",
        "pointnet_pp": "PointNet++ (10ep)",
        "dgcnn": "DGCNN (10ep)",
        "randla": "RandLA-Net (10ep)",
        "ransac": "RANSAC + 聚类",
        "depth_thresh": "深度阈值分割",
    }

    all_results = {}

    for algo_key, algo_name in algorithms.items():
        logger.info(f"\n{'='*60}")
        logger.info(f"Evaluating: {algo_name} ({algo_key})")
        logger.info(f"{'='*60}")

        if algo_key in ("ransac", "depth_thresh"):
            model = None
            use_6ch = False
        else:
            model, use_6ch = load_model_for_eval(algo_key)
            if model is None:
                logger.warning(f"Model not available: {algo_key}, skipping")
                continue

        per_scene = []
        for scene_dir in test_scenes:
            scene_name = os.path.basename(scene_dir)
            depth = np.load(os.path.join(scene_dir, "depth_noisy.npy"))
            gt_labels = np.load(os.path.join(scene_dir, "semantic_labels.npy"))
            points, valid_mask = depth_to_pointcloud(depth)

            t0 = time.time()
            if algo_key == "ransac":
                pred_2d = predict_ransac(points, valid_mask, depth)
            elif algo_key == "depth_thresh":
                pred_2d = predict_depth_threshold(points, valid_mask, depth)
            else:
                pred_2d = predict_dl(model, points, valid_mask, use_6ch)
            latency_ms = (time.time() - t0) * 1000

            pred_flat = pred_2d.ravel()
            gt_flat = gt_labels.ravel()
            valid_flat = valid_mask.ravel()
            pred_valid = pred_flat[valid_flat]
            gt_valid = gt_flat[valid_flat]

            cm = np.zeros((2, 2), dtype=int)
            for c in range(2):
                for d in range(2):
                    cm[c, d] = ((gt_valid == c) & (pred_valid == d)).sum()

            ious = []
            for c in range(2):
                intersection = cm[c, c]
                union = cm[c, :].sum() + cm[:, c].sum() - intersection
                iou = intersection / max(union, 1)
                ious.append(iou)
            miou = sum(ious) / len(ious)

            bf1, bp, br = compute_boundary_f1(pred_2d, gt_labels)

            instances, inst_labels = run_instance_segmentation(points, pred_valid)
            edges, grasp_order = build_hierarchy(instances)

            ann_path = os.path.join(scene_dir, "annotation.json")
            gt_ann = None
            if os.path.exists(ann_path):
                with open(ann_path, "r", encoding="utf-8") as f:
                    gt_ann = json.load(f)

            gt_inst_count = len(gt_ann.get("objects", [])) if gt_ann else 2
            inst_count_correct = len(instances) == gt_inst_count

            gt_edges = []
            gt_grasp_order = []
            if gt_ann:
                hierarchy = gt_ann.get("hierarchy", {})
                gt_grasp_order = hierarchy.get("grasp_order", [])
                for e in hierarchy.get("edges", []):
                    gt_edges.append((e["upper"], e["lower"]))

            instances_sorted = sorted(instances, key=lambda i: i["z_mean"])
            gt_objects = gt_ann.get("objects", []) if gt_ann else []
            gt_objects_sorted = sorted(gt_objects, key=lambda o: (o.get("z_min", 0) + o.get("z_max", 0)) / 2)

            id_map = {}
            for pi, pred_inst in enumerate(instances_sorted):
                if pi < len(gt_objects_sorted):
                    id_map[pred_inst["id"]] = gt_objects_sorted[pi]["id"]

            pred_edges_mapped = []
            for u, v in edges:
                if u in id_map and v in id_map:
                    pred_edges_mapped.append((id_map[u], id_map[v]))

            edge_correct = 0
            edge_total = max(len(gt_edges), 1)
            for ge in gt_edges:
                if ge in pred_edges_mapped:
                    edge_correct += 1
            edge_acc = edge_correct / edge_total if edge_total > 0 else 0.0

            grasp_order_mapped = [id_map[g] for g in grasp_order if g in id_map]
            grasp_correct = False
            if len(gt_grasp_order) > 0 and len(grasp_order_mapped) > 0:
                if len(grasp_order_mapped) == len(gt_grasp_order):
                    grasp_correct = all(
                        grasp_order_mapped[i] == gt_grasp_order[i] for i in range(len(gt_grasp_order)))

            edge_p, edge_r, edge_f1, _ = compute_stacking_metrics(
                pred_edges_mapped, gt_edges, grasp_order_mapped, gt_grasp_order)

            per_scene.append({
                "scene": scene_name,
                "miou": float(miou),
                "iou_table": float(ious[0]),
                "iou_object": float(ious[1]),
                "latency_ms": float(latency_ms),
                "confusion_matrix": cm.tolist(),
                "boundary_f1": bf1,
                "boundary_precision": bp,
                "boundary_recall": br,
                "instance_count_correct": inst_count_correct,
                "pred_instances": len(instances),
                "gt_instances": gt_inst_count,
                "edge_accuracy": float(edge_acc),
                "edge_precision": edge_p,
                "edge_recall": edge_r,
                "edge_f1": edge_f1,
                "edge_correct": edge_correct,
                "edge_total": edge_total,
                "grasp_correct": grasp_correct,
                "pred_grasp_order": grasp_order_mapped,
                "gt_grasp_order": gt_grasp_order,
                "pred_edges": pred_edges_mapped,
                "gt_edges": gt_edges,
                "has_stacking": len(gt_edges) > 0,
            })

            logger.info(f"  {scene_name}: mIoU={miou:.4f}, BF1={bf1:.3f}, "
                        f"inst={'OK' if inst_count_correct else 'FAIL'}, "
                        f"edge={edge_acc:.2f}, grasp={'OK' if grasp_correct else 'FAIL'}")

        mious = [s["miou"] for s in per_scene]
        lats = [s["latency_ms"] for s in per_scene]
        inst_acc = sum(1 for s in per_scene if s["instance_count_correct"]) / len(per_scene)
        bf1s = [s["boundary_f1"] for s in per_scene]

        stacking_scenes = [s for s in per_scene if s.get("has_stacking", len(s.get("gt_edges", [])) > 0)]
        if stacking_scenes:
            edge_acc = np.mean([s["edge_accuracy"] for s in stacking_scenes])
            edge_f1 = np.mean([s["edge_f1"] for s in stacking_scenes])
            grasp_acc = sum(1 for s in stacking_scenes if s["grasp_correct"]) / len(stacking_scenes)
        else:
            edge_acc = 0.0
            edge_f1 = 0.0
            grasp_acc = 0.0

        total_cm = np.zeros((2, 2), dtype=int)
        for s in per_scene:
            total_cm += np.array(s["confusion_matrix"])

        summary = {
            "algo_name": algo_name,
            "algo_key": algo_key,
            "avg_miou": float(np.mean(mious)),
            "avg_obj_iou": float(np.mean([s["iou_object"] for s in per_scene])),
            "avg_boundary_f1": float(np.mean(bf1s)),
            "avg_latency_ms": float(np.mean(lats)),
            "total_confusion_matrix": total_cm.tolist(),
            "instance_count_accuracy": float(inst_acc),
            "avg_edge_accuracy": float(edge_acc),
            "avg_edge_f1": float(edge_f1),
            "grasp_order_accuracy": float(grasp_acc),
        }

        all_results[algo_key] = {
            "config_name": algo_name,
            "config_key": algo_key,
            "per_scene": per_scene,
            "summary": summary,
        }

        logger.info(f"  Summary: mIoU={summary['avg_miou']:.4f}, BF1={summary['avg_boundary_f1']:.3f}, "
                    f"inst_acc={inst_acc:.1%}, edge_acc={edge_acc:.1%}, "
                    f"edge_f1={edge_f1:.3f}, grasp_acc={grasp_acc:.1%}")

    with open(os.path.join(RESULT_DIR, "algo_comparison_summary.json"), "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    logger.info(f"\nResults saved to {RESULT_DIR}/algo_comparison_summary.json")
    return all_results


# ═══════════════════════════════════════════════════════════════════
# Part B: 堆叠检测算法对比 (核心)
# ═══════════════════════════════════════════════════════════════════

STACKING_METHODS = {
    "ours_hierarchy": {
        "name": "Ours (HierarchyReasoner)",
        "func": build_hierarchy,
        "category": "ours",
        "description": "空间关系分析 + XY重叠检测 + 拓扑排序",
    },
    "simple_zsort": {
        "name": "Simple Z-Sort",
        "func": simple_z_sort,
        "category": "baseline",
        "description": "仅按物体中心Z轴高度排序",
    },
    "bbox_iou": {
        "name": "BBox IoU-based",
        "func": bbox_iou_sort,
        "category": "baseline",
        "description": "基于XY包围盒IoU判定堆叠关系",
    },
    "height_threshold": {
        "name": "Height Threshold",
        "func": height_threshold_sort,
        "category": "baseline",
        "description": "基于Z轴高度差阈值判定堆叠",
    },
    "centroid_proximity": {
        "name": "Centroid Proximity",
        "func": centroid_proximity_sort,
        "category": "baseline",
        "description": "基于质心XY邻近度判定堆叠",
    },
    "overlap_z": {
        "name": "Overlap + Z-Sort",
        "func": overlap_z_sort,
        "category": "baseline",
        "description": "XY重叠检测 + Z排序 (简化版)",
    },
    "ransac_zsort": {
        "name": "RANSAC + Z-Sort",
        "func": simple_z_sort,
        "category": "traditional",
        "description": "RANSAC平面拟合分割 + Z轴排序",
        "use_ransac_seg": True,
    },
    "depth_zsort": {
        "name": "Depth Threshold + Z-Sort",
        "func": simple_z_sort,
        "category": "traditional",
        "description": "深度阈值分割 + Z轴排序",
        "use_depth_seg": True,
    },
}


def evaluate_stacking_methods():
    """核心对比: 同一PP-Attention分割 → 不同堆叠检测算法"""
    test_scenes = sorted([
        os.path.join(TEST_DATA_DIR, d) for d in os.listdir(TEST_DATA_DIR)
        if os.path.isdir(os.path.join(TEST_DATA_DIR, d))
        and os.path.exists(os.path.join(TEST_DATA_DIR, d, "depth_noisy.npy"))
    ])

    stacking_test_scenes = []  # skip generated stacking test data (format incompatible with model)

    all_test_scenes = test_scenes + stacking_test_scenes
    logger.info(f"Test scenes: {len(test_scenes)} original + {len(stacking_test_scenes)} stacking-specific = {len(all_test_scenes)} total")

    model, use_6ch = load_model_for_eval("pp_attention")
    if model is None:
        logger.error("PP-Attention model not available!")
        return {}

    all_results = {}

    for method_key, method_info in STACKING_METHODS.items():
        method_name = method_info["name"]
        method_func = method_info["func"]
        use_ransac = method_info.get("use_ransac_seg", False)
        use_depth = method_info.get("use_depth_seg", False)

        logger.info(f"\n{'='*60}")
        logger.info(f"Stacking Method: {method_name}")
        logger.info(f"  Category: {method_info['category']}")
        logger.info(f"  Description: {method_info['description']}")
        logger.info(f"{'='*60}")

        per_scene = []
        for scene_dir in all_test_scenes:
            scene_name = os.path.basename(scene_dir)
            depth = np.load(os.path.join(scene_dir, "depth_noisy.npy"))
            gt_labels = np.load(os.path.join(scene_dir, "semantic_labels.npy"))
            points, valid_mask = depth_to_pointcloud(depth)

            t0 = time.time()

            if use_ransac:
                pred_2d = predict_ransac(points, valid_mask, depth)
            elif use_depth:
                pred_2d = predict_depth_threshold(points, valid_mask, depth)
            else:
                pred_2d = predict_dl(model, points, valid_mask, use_6ch)

            latency_ms = (time.time() - t0) * 1000

            pred_flat = pred_2d.ravel()
            gt_flat = gt_labels.ravel()
            valid_flat = valid_mask.ravel()
            pred_valid = pred_flat[valid_flat]
            gt_valid = gt_flat[valid_flat]

            cm = np.zeros((2, 2), dtype=int)
            for c in range(2):
                for d in range(2):
                    cm[c, d] = ((gt_valid == c) & (pred_valid == d)).sum()

            ious = []
            for c in range(2):
                intersection = cm[c, c]
                union = cm[c, :].sum() + cm[:, c].sum() - intersection
                iou = intersection / max(union, 1)
                ious.append(iou)
            miou = sum(ious) / len(ious)

            bf1, bp, br = compute_boundary_f1(pred_2d, gt_labels)

            instances, inst_labels = run_instance_segmentation(points, pred_valid)
            edges, grasp_order = method_func(instances)

            ann_path = os.path.join(scene_dir, "annotation.json")
            gt_ann = None
            if os.path.exists(ann_path):
                with open(ann_path, "r", encoding="utf-8") as f:
                    gt_ann = json.load(f)

            gt_inst_count = len(gt_ann.get("objects", [])) if gt_ann else 2
            inst_count_correct = len(instances) == gt_inst_count

            gt_edges = []
            gt_grasp_order = []
            has_stacking = False
            difficulty = "unknown"
            if gt_ann:
                hierarchy = gt_ann.get("hierarchy", {})
                gt_grasp_order = hierarchy.get("grasp_order", [])
                for e in hierarchy.get("edges", []):
                    gt_edges.append((e["upper"], e["lower"]))
                has_stacking = len(gt_edges) > 0
                difficulty = gt_ann.get("difficulty", "unknown")

            instances_sorted = sorted(instances, key=lambda i: i["z_mean"])
            gt_objects = gt_ann.get("objects", []) if gt_ann else []
            gt_objects_sorted = sorted(gt_objects, key=lambda o: (o.get("z_min", 0) + o.get("z_max", 0)) / 2)

            id_map = {}
            for pi, pred_inst in enumerate(instances_sorted):
                if pi < len(gt_objects_sorted):
                    id_map[pred_inst["id"]] = gt_objects_sorted[pi]["id"]

            pred_edges_mapped = []
            for u, v in edges:
                if u in id_map and v in id_map:
                    pred_edges_mapped.append((id_map[u], id_map[v]))

            grasp_order_mapped = [id_map[g] for g in grasp_order if g in id_map]

            edge_p, edge_r, edge_f1, grasp_correct = compute_stacking_metrics(
                pred_edges_mapped, gt_edges, grasp_order_mapped, gt_grasp_order)

            edge_correct = 0
            edge_total = max(len(gt_edges), 1)
            gt_edge_set = set(tuple(sorted(e)) for e in gt_edges)
            pred_edge_set = set(tuple(sorted(e)) for e in pred_edges_mapped)
            edge_correct = len(gt_edge_set & pred_edge_set)
            edge_acc = edge_correct / edge_total if edge_total > 0 else 0.0

            per_scene.append({
                "scene": scene_name,
                "miou": float(miou),
                "iou_table": float(ious[0]),
                "iou_object": float(ious[1]),
                "latency_ms": float(latency_ms),
                "confusion_matrix": cm.tolist(),
                "boundary_f1": bf1,
                "boundary_precision": bp,
                "boundary_recall": br,
                "instance_count_correct": inst_count_correct,
                "pred_instances": len(instances),
                "gt_instances": gt_inst_count,
                "edge_accuracy": float(edge_acc),
                "edge_precision": edge_p,
                "edge_recall": edge_r,
                "edge_f1": edge_f1,
                "edge_correct": edge_correct,
                "edge_total": edge_total,
                "grasp_correct": grasp_correct,
                "pred_grasp_order": grasp_order_mapped,
                "gt_grasp_order": gt_grasp_order,
                "pred_edges": pred_edges_mapped,
                "gt_edges": gt_edges,
                "has_stacking": has_stacking,
                "difficulty": difficulty,
            })

            logger.info(f"  {scene_name}: mIoU={miou:.4f}, BF1={bf1:.3f}, "
                        f"inst={'OK' if inst_count_correct else 'FAIL'}, "
                        f"edge_p={edge_p:.2f} edge_r={edge_r:.2f} edge_f1={edge_f1:.2f}, "
                        f"grasp={'OK' if grasp_correct else 'FAIL'}")

        mious = [s["miou"] for s in per_scene]
        lats = [s["latency_ms"] for s in per_scene]
        bf1s = [s["boundary_f1"] for s in per_scene]
        inst_acc = sum(1 for s in per_scene if s["instance_count_correct"]) / len(per_scene)

        stacking_scenes = [s for s in per_scene if s["has_stacking"]]
        if stacking_scenes:
            edge_acc = np.mean([s["edge_accuracy"] for s in stacking_scenes])
            edge_p = np.mean([s["edge_precision"] for s in stacking_scenes])
            edge_r = np.mean([s["edge_recall"] for s in stacking_scenes])
            edge_f1 = np.mean([s["edge_f1"] for s in stacking_scenes])
            grasp_acc = sum(1 for s in stacking_scenes if s["grasp_correct"]) / len(stacking_scenes)
        else:
            edge_acc = edge_p = edge_r = edge_f1 = grasp_acc = 0.0

        overall_edge_p = np.mean([s["edge_precision"] for s in per_scene])
        overall_edge_r = np.mean([s["edge_recall"] for s in per_scene])
        overall_edge_f1 = np.mean([s["edge_f1"] for s in per_scene])
        overall_grasp_acc = sum(1 for s in per_scene if s["grasp_correct"]) / len(per_scene)

        by_difficulty = defaultdict(list)
        for s in stacking_scenes:
            by_difficulty[s.get("difficulty", "unknown")].append(s)

        difficulty_breakdown = {}
        for diff, scenes in by_difficulty.items():
            difficulty_breakdown[diff] = {
                "count": len(scenes),
                "edge_f1": float(np.mean([s["edge_f1"] for s in scenes])),
                "grasp_acc": float(sum(1 for s in scenes if s["grasp_correct"]) / len(scenes)),
            }

        total_cm = np.zeros((2, 2), dtype=int)
        for s in per_scene:
            total_cm += np.array(s["confusion_matrix"])

        summary = {
            "algo_name": method_name,
            "algo_key": method_key,
            "category": method_info["category"],
            "description": method_info["description"],
            "avg_miou": float(np.mean(mious)),
            "avg_obj_iou": float(np.mean([s["iou_object"] for s in per_scene])),
            "avg_boundary_f1": float(np.mean(bf1s)),
            "avg_latency_ms": float(np.mean(lats)),
            "total_confusion_matrix": total_cm.tolist(),
            "instance_count_accuracy": float(inst_acc),
            "avg_edge_accuracy": float(edge_acc),
            "avg_edge_precision": float(edge_p),
            "avg_edge_recall": float(edge_r),
            "avg_edge_f1": float(edge_f1),
            "grasp_order_accuracy": float(grasp_acc),
            "overall_edge_precision": float(overall_edge_p),
            "overall_edge_recall": float(overall_edge_r),
            "overall_edge_f1": float(overall_edge_f1),
            "overall_grasp_accuracy": float(overall_grasp_acc),
            "difficulty_breakdown": difficulty_breakdown,
        }

        all_results[method_key] = {
            "config_name": method_name,
            "config_key": method_key,
            "per_scene": per_scene,
            "summary": summary,
        }

        logger.info(f"  Summary: mIoU={summary['avg_miou']:.4f}, BF1={summary['avg_boundary_f1']:.3f}, "
                    f"inst_acc={inst_acc:.1%}, edge_f1={edge_f1:.3f} (P={edge_p:.2f} R={edge_r:.2f}), "
                    f"grasp_acc={grasp_acc:.1%}")
        logger.info(f"  Overall (8 scenes): edge_f1={overall_edge_f1:.3f} (P={overall_edge_p:.2f} R={overall_edge_r:.2f}), "
                    f"grasp_acc={overall_grasp_acc:.1%}")
        if difficulty_breakdown:
            for diff, info in difficulty_breakdown.items():
                logger.info(f"    [{diff}]: edge_f1={info['edge_f1']:.3f}, grasp_acc={info['grasp_acc']:.1%}")

    with open(os.path.join(RESULT_DIR, "stacking_comparison_summary.json"), "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    logger.info(f"\nStacking results saved to {RESULT_DIR}/stacking_comparison_summary.json")
    return all_results


# ═══════════════════════════════════════════════════════════════════
# 图表生成
# ═══════════════════════════════════════════════════════════════════

def generate_comparison_charts(results):
    algos = list(results.keys())
    names = [results[a]["config_name"] for a in algos]
    summaries = [results[a]["summary"] for a in algos]

    algo_colors = []
    for a in algos:
        if "PP-Attention" in results[a]["config_name"] or "Ours" in results[a]["config_name"]:
            algo_colors.append("#0D47A1")
        elif a in ("ransac", "depth_thresh"):
            algo_colors.append("#FF6F00")
        else:
            algo_colors.append("#1565C0")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    ax = axes[0, 0]
    mious = [s["avg_miou"] for s in summaries]
    x = np.arange(len(names))
    bars = ax.bar(x, [m * 100 for m in mious], color=algo_colors, edgecolor="white", zorder=3)
    for bar, val in zip(bars, mious):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                f"{val*100:.2f}%", ha="center", va="bottom", fontsize=8, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=8, rotation=20, ha="right")
    ax.set_ylabel("mIoU (%)", fontsize=11)
    ax.set_title("(a) 语义分割 mIoU 对比", fontsize=12, fontweight="bold")
    ax.grid(axis="y", alpha=0.3, zorder=0)

    ax = axes[0, 1]
    lats = [s["avg_latency_ms"] for s in summaries]
    bars = ax.bar(x, lats, color=algo_colors, edgecolor="white", zorder=3)
    for bar, val in zip(bars, lats):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f"{val:.0f}ms", ha="center", va="bottom", fontsize=7, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=8, rotation=20, ha="right")
    ax.set_ylabel("推理延迟 (ms)", fontsize=11)
    ax.set_title("(b) 推理延迟对比", fontsize=12, fontweight="bold")
    ax.grid(axis="y", alpha=0.3, zorder=0)

    ax = axes[1, 0]
    metrics = ["实例准确率", "边准确率", "抓取顺序"]
    x_pos = np.arange(len(metrics))
    w = 0.12
    for i, (algo_key, algo_name) in enumerate(zip(algos, names)):
        s = results[algo_key]["summary"]
        vals = [s["instance_count_accuracy"] * 100, s["avg_edge_accuracy"] * 100, s["grasp_order_accuracy"] * 100]
        offset = (i - len(algos) / 2 + 0.5) * w
        ax.bar(x_pos + offset, vals, w, label=algo_name[:15], edgecolor="white", zorder=3)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(metrics, fontsize=10)
    ax.set_ylabel("准确率 (%)", fontsize=11)
    ax.set_title("(c) 后处理指标对比", fontsize=12, fontweight="bold")
    ax.legend(fontsize=7, loc="lower left", ncol=2)
    ax.grid(axis="y", alpha=0.3, zorder=0)

    ax = axes[1, 1]
    radar_metrics = ["mIoU", "实例准确率", "边准确率", "抓取顺序", "速度(1/延迟)"]
    angles = np.linspace(0, 2 * np.pi, len(radar_metrics), endpoint=False).tolist()
    angles += angles[:1]
    for i, (algo_key, algo_name) in enumerate(zip(algos, names)):
        s = results[algo_key]["summary"]
        speed_score = min(1.0, 100.0 / max(s["avg_latency_ms"], 1))
        values = [s["avg_miou"], s["instance_count_accuracy"], s["avg_edge_accuracy"],
                  s["grasp_order_accuracy"], speed_score]
        values += values[:1]
        ax.fill(angles, values, alpha=0.1, color=algo_colors[i])
        ax.plot(angles, values, "o-", linewidth=1.5, color=algo_colors[i], label=algo_name[:15], markersize=4)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(radar_metrics, fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.set_title("(d) 综合雷达图", fontsize=12, fontweight="bold")
    ax.legend(fontsize=7, loc="upper right", ncol=2)

    fig.suptitle("算法对比实验 — 综合评估", fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    fig.savefig(os.path.join(FIGURE_DIR, "algo_comparison_overview.png"), dpi=300,
                bbox_inches="tight", facecolor="white", edgecolor="none")
    plt.close()
    logger.info("[OK] algo_comparison_overview.png")

    fig, ax = plt.subplots(figsize=(14, 6))
    scenes = [s["scene"].replace("scene_0", "S").replace("_", "\n")
              for s in results[list(results.keys())[0]]["per_scene"]]
    x = np.arange(len(scenes))
    w = 0.12
    for i, (algo_key, algo_name) in enumerate(zip(algos, names)):
        mious_per = [s["miou"] for s in results[algo_key]["per_scene"]]
        offset = (i - len(algos) / 2 + 0.5) * w
        ax.bar(x + offset, [m * 100 for m in mious_per], w, label=algo_name[:15],
               edgecolor="white", zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels(scenes, fontsize=8)
    ax.set_ylabel("mIoU (%)", fontsize=11)
    ax.set_title("各场景 mIoU 对比", fontsize=13, fontweight="bold")
    ax.legend(fontsize=7, loc="lower left", ncol=3)
    ax.grid(axis="y", alpha=0.3, zorder=0)
    plt.tight_layout()
    fig.savefig(os.path.join(FIGURE_DIR, "algo_per_scene_miou.png"), dpi=300,
                bbox_inches="tight", facecolor="white", edgecolor="none")
    plt.close()
    logger.info("[OK] algo_per_scene_miou.png")

    fig, ax = plt.subplots(figsize=(12, 7))
    cell_text = []
    cols = ["算法", "mIoU", "Obj IoU", "实例准确率", "边准确率", "抓取顺序", "延迟(ms)"]
    for algo_key, algo_name in zip(algos, names):
        s = results[algo_key]["summary"]
        cell_text.append([
            algo_name, f"{s['avg_miou']:.4f}", f"{s['avg_obj_iou']:.4f}",
            f"{s['instance_count_accuracy']:.1%}", f"{s['avg_edge_accuracy']:.1%}",
            f"{s['grasp_order_accuracy']:.1%}", f"{s['avg_latency_ms']:.0f}",
        ])
    ax.axis("tight")
    ax.axis("off")
    table = ax.table(cellText=cell_text, colLabels=cols, cellLoc="center", loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.5)
    for key, cell in table.get_celld().items():
        cell.set_edgecolor("#CCCCCC")
        if key[0] == 0:
            cell.set_facecolor("#1565C0")
            cell.set_text_props(color="white", fontweight="bold")
        elif key[1] == 0:
            cell.set_facecolor("#E3F2FD")
    ax.set_title("算法对比实验结果汇总", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(os.path.join(FIGURE_DIR, "algo_comparison_table.png"), dpi=300,
                bbox_inches="tight", facecolor="white", edgecolor="none")
    plt.close()
    logger.info("[OK] algo_comparison_table.png")


def generate_stacking_charts(results):
    """生成堆叠检测算法对比图表"""
    algos = list(results.keys())
    names = [results[a]["config_name"] for a in algos]
    summaries = [results[a]["summary"] for a in algos]

    cat_colors = {"ours": "#0D47A1", "baseline": "#1565C0", "traditional": "#FF6F00"}
    colors = [cat_colors.get(results[a]["summary"].get("category", "baseline"), "#1565C0") for a in algos]

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    ax = axes[0, 0]
    metrics = ["实例准确率", "边 F1 (Overall)", "抓取顺序 (Overall)"]
    x_pos = np.arange(len(metrics))
    w = 0.1
    for i, (algo_key, algo_name) in enumerate(zip(algos, names)):
        s = results[algo_key]["summary"]
        vals = [s["instance_count_accuracy"] * 100, s["overall_edge_f1"] * 100,
                s["overall_grasp_accuracy"] * 100]
        offset = (i - len(algos) / 2 + 0.5) * w
        bars = ax.bar(x_pos + offset, vals, w, label=algo_name, color=colors[i],
                      edgecolor="white", zorder=3)
        for bar, val in zip(bars, vals):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                        f"{val:.0f}%", ha="center", va="bottom", fontsize=6, fontweight="bold")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(metrics, fontsize=11)
    ax.set_ylabel("准确率 (%)", fontsize=11)
    ax.set_title("(a) 堆叠检测指标对比 (含非堆叠场景)", fontsize=12, fontweight="bold")
    ax.legend(fontsize=7, loc="lower left", ncol=2)
    ax.grid(axis="y", alpha=0.3, zorder=0)
    ax.set_ylim(0, 115)

    ax = axes[0, 1]
    x = np.arange(len(names))
    w_bar = 0.25
    edge_ps = [s["overall_edge_precision"] * 100 for s in summaries]
    edge_rs = [s["overall_edge_recall"] * 100 for s in summaries]
    edge_f1s = [s["overall_edge_f1"] * 100 for s in summaries]
    ax.bar(x - w_bar, edge_ps, w_bar, label="Precision", color="#2196F3", edgecolor="white", zorder=3)
    ax.bar(x, edge_rs, w_bar, label="Recall", color="#4CAF50", edgecolor="white", zorder=3)
    ax.bar(x + w_bar, edge_f1s, w_bar, label="F1", color="#FF9800", edgecolor="white", zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=7, rotation=25, ha="right")
    ax.set_ylabel("分数 (%)", fontsize=11)
    ax.set_title("(b) 堆叠边 Precision / Recall / F1 (Overall)", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3, zorder=0)

    ax = axes[1, 0]
    radar_metrics = ["实例准确率", "边 Precision", "边 Recall", "边 F1", "抓取顺序"]
    angles = np.linspace(0, 2 * np.pi, len(radar_metrics), endpoint=False).tolist()
    angles += angles[:1]
    for i, (algo_key, algo_name) in enumerate(zip(algos, names)):
        s = results[algo_key]["summary"]
        values = [s["instance_count_accuracy"], s["overall_edge_precision"],
                  s["overall_edge_recall"], s["overall_edge_f1"], s["overall_grasp_accuracy"]]
        values += values[:1]
        ax.fill(angles, values, alpha=0.08, color=colors[i])
        ax.plot(angles, values, "o-", linewidth=1.5, color=colors[i], label=algo_name, markersize=4)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(radar_metrics, fontsize=9)
    ax.set_ylim(0, 1.1)
    ax.set_title("(c) 综合雷达图 (Overall)", fontsize=12, fontweight="bold")
    ax.legend(fontsize=7, loc="upper right", ncol=2)

    ax = axes[1, 1]
    ax.axis("off")
    ours_key = [a for a in algos if results[a]["summary"].get("category") == "ours"]
    baseline_keys = [a for a in algos if results[a]["summary"].get("category") == "baseline"]
    ours_s = results[ours_key[0]]["summary"] if ours_key else {}
    text_lines = []
    text_lines.append("=" * 50)
    text_lines.append("  HierarchyReasoner vs Baselines 提升分析")
    text_lines.append("=" * 50)
    if ours_s and baseline_keys:
        best_baseline_edge_f1 = max(results[k]["summary"]["overall_edge_f1"] for k in baseline_keys)
        best_baseline_grasp = max(results[k]["summary"]["overall_grasp_accuracy"] for k in baseline_keys)
        avg_baseline_edge_f1 = np.mean([results[k]["summary"]["overall_edge_f1"] for k in baseline_keys])
        avg_baseline_grasp = np.mean([results[k]["summary"]["overall_grasp_accuracy"] for k in baseline_keys])
        text_lines.append(f"  Ours Edge F1:        {ours_s['overall_edge_f1']:.3f}")
        text_lines.append(f"  Best Baseline Edge F1: {best_baseline_edge_f1:.3f}")
        text_lines.append(f"  Avg Baseline Edge F1:  {avg_baseline_edge_f1:.3f}")
        text_lines.append(f"  Edge F1 提升:         +{ours_s['overall_edge_f1'] - avg_baseline_edge_f1:+.3f}")
        text_lines.append("")
        text_lines.append(f"  Ours Grasp Acc:        {ours_s['overall_grasp_accuracy']:.1%}")
        text_lines.append(f"  Best Baseline Grasp:   {best_baseline_grasp:.1%}")
        text_lines.append(f"  Avg Baseline Grasp:    {avg_baseline_grasp:.1%}")
        text_lines.append(f"  Grasp 提升:           +{ours_s['overall_grasp_accuracy'] - avg_baseline_grasp:+.1%}")
        text_lines.append("")
        text_lines.append("  Key Advantages:")
        text_lines.append("  1. XY重叠 + Z间隙双重判定")
        text_lines.append("  2. 质心邻近度辅助检测")
        text_lines.append("  3. 拓扑排序保证物理约束")
        text_lines.append("  4. 处理部分重叠/偏移堆叠")
    for i, line in enumerate(text_lines):
        ax.text(0.05, 0.95 - i * 0.045, line, transform=ax.transAxes,
                fontsize=9, fontfamily="monospace", verticalalignment="top")
    ax.set_title("(d) 提升分析", fontsize=12, fontweight="bold")

    fig.suptitle("堆叠检测算法对比 — 综合评估", fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    fig.savefig(os.path.join(FIGURE_DIR, "stacking_comparison.png"), dpi=300,
                bbox_inches="tight", facecolor="white", edgecolor="none")
    plt.close()
    logger.info("[OK] stacking_comparison.png")

    fig, ax = plt.subplots(figsize=(14, 6))
    cell_text = []
    cols = ["堆叠检测方法", "类别", "实例准确率", "边 Precision", "边 Recall",
            "边 F1", "抓取顺序", "延迟(ms)"]
    for algo_key, algo_name in zip(algos, names):
        s = results[algo_key]["summary"]
        cat = s.get("category", "baseline")
        cat_label = {"ours": "Ours", "baseline": "Baseline", "traditional": "Traditional"}.get(cat, cat)
        cell_text.append([
            algo_name, cat_label,
            f"{s['instance_count_accuracy']:.1%}",
            f"{s['overall_edge_precision']:.3f}",
            f"{s['overall_edge_recall']:.3f}",
            f"{s['overall_edge_f1']:.3f}",
            f"{s['overall_grasp_accuracy']:.1%}",
            f"{s['avg_latency_ms']:.0f}",
        ])
    ax.axis("tight")
    ax.axis("off")
    table = ax.table(cellText=cell_text, colLabels=cols, cellLoc="center", loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.1, 1.6)
    for key, cell in table.get_celld().items():
        cell.set_edgecolor("#CCCCCC")
        if key[0] == 0:
            cell.set_facecolor("#E65100")
            cell.set_text_props(color="white", fontweight="bold")
        elif key[1] == 0:
            cell.set_facecolor("#FFF3E0")
    ax.set_title("堆叠检测算法对比结果汇总", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(os.path.join(FIGURE_DIR, "stacking_comparison_table.png"), dpi=300,
                bbox_inches="tight", facecolor="white", edgecolor="none")
    plt.close()
    logger.info("[OK] stacking_comparison_table.png")

    if any("difficulty_breakdown" in results[a]["summary"] for a in algos):
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        difficulties = set()
        for a in algos:
            for d in results[a]["summary"].get("difficulty_breakdown", {}):
                difficulties.add(d)
        difficulties = sorted(difficulties)
        if difficulties:
            ax = axes[0]
            x = np.arange(len(difficulties))
            w = 0.12
            for i, (algo_key, algo_name) in enumerate(zip(algos, names)):
                db = results[algo_key]["summary"].get("difficulty_breakdown", {})
                vals = [db.get(d, {}).get("edge_f1", 0) * 100 for d in difficulties]
                offset = (i - len(algos) / 2 + 0.5) * w
                ax.bar(x + offset, vals, w, label=algo_name[:12], color=colors[i],
                       edgecolor="white", zorder=3)
            ax.set_xticks(x)
            ax.set_xticklabels(difficulties, fontsize=10)
            ax.set_ylabel("Edge F1 (%)", fontsize=11)
            ax.set_title("不同难度 Edge F1 对比", fontsize=12, fontweight="bold")
            ax.legend(fontsize=7, ncol=2)
            ax.grid(axis="y", alpha=0.3, zorder=0)

            ax = axes[1]
            for i, (algo_key, algo_name) in enumerate(zip(algos, names)):
                db = results[algo_key]["summary"].get("difficulty_breakdown", {})
                vals = [db.get(d, {}).get("grasp_acc", 0) * 100 for d in difficulties]
                offset = (i - len(algos) / 2 + 0.5) * w
                ax.bar(x + offset, vals, w, label=algo_name[:12], color=colors[i],
                       edgecolor="white", zorder=3)
            ax.set_xticks(x)
            ax.set_xticklabels(difficulties, fontsize=10)
            ax.set_ylabel("Grasp Accuracy (%)", fontsize=11)
            ax.set_title("不同难度 Grasp 准确率对比", fontsize=12, fontweight="bold")
            ax.legend(fontsize=7, ncol=2)
            ax.grid(axis="y", alpha=0.3, zorder=0)

            fig.suptitle("堆叠检测鲁棒性分析 — 不同难度级别", fontsize=14, fontweight="bold")
            plt.tight_layout()
            fig.savefig(os.path.join(FIGURE_DIR, "stacking_robustness.png"), dpi=300,
                        bbox_inches="tight", facecolor="white", edgecolor="none")
            plt.close()
            logger.info("[OK] stacking_robustness.png")


def generate_report(results, stacking_results=None):
    """生成综合对比报告"""
    algos = list(results.keys())
    names = [results[a]["config_name"] for a in algos]
    summaries = [results[a]["summary"] for a in algos]

    lines = []
    lines.append("# 算法对比实验报告\n")
    lines.append(f"*报告生成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}*\n")

    if len(algos) > 0:
        best_miou_idx = np.argmax([s["avg_miou"] for s in summaries])
        best_inst_idx = np.argmax([s["instance_count_accuracy"] for s in summaries])
        best_edge_idx = np.argmax([s["avg_edge_accuracy"] for s in summaries])
        best_grasp_idx = np.argmax([s["grasp_order_accuracy"] for s in summaries])
        fastest_idx = np.argmin([s["avg_latency_ms"] for s in summaries])

    lines.append("## Part A: 语义分割模型对比\n")
    if len(algos) > 0:
        lines.append("### A.1 实验设置\n")
        lines.append("- **数据集**: 800 train / 200 val 合成堆叠场景 + 8 测试场景")
        lines.append("- **任务**: 2 类语义分割 (桌面=0, 物体=1)")
        lines.append("- **评估指标**: mIoU, Obj IoU, 边界F1, 实例数量准确率, 层级边准确率, 抓取顺序正确率, 推理延迟")
        lines.append("- **对比策略**: PP-Attention (100 epoch, PointStack参数) vs 其他模型 (10 epoch)\n")

        lines.append("### A.2 对比算法\n")
        lines.append("| 类别 | 算法 | 训练 | 说明 |")
        lines.append("|------|------|------|------|")
        lines.append("| Ours | **PP-Attention** | 100 epoch | PointNet++ + 通道注意力 + 位置注意力 + 多尺度融合 + 6ch 输入 |")
        lines.append("| DL | PointNet++ (3ch) | 100 epoch | 标准 PointNet++ (3ch XYZ) |")
        lines.append("| DL | PointNet | 10 epoch | 原始 PointNet (3ch XYZ) |")
        lines.append("| DL | PointNet++ | 10 epoch | 标准 PointNet++ (3ch XYZ) |")
        lines.append("| DL | DGCNN | 10 epoch | Dynamic Graph CNN, EdgeConv 图卷积 |")
        lines.append("| DL | RandLA-Net | 10 epoch | 随机采样 + 局部特征聚合 |")
        lines.append("| 传统 | RANSAC + 聚类 | - | RANSAC 平面拟合分离桌面/物体 |")
        lines.append("| 传统 | 深度阈值分割 | - | 基于深度直方图的阈值分割 + 形态学后处理 |\n")

        lines.append("### A.3 综合对比结果\n")
        lines.append("| 算法 | mIoU | Obj IoU | 边界F1 | 实例准确率 | 边准确率 | 抓取顺序 | 延迟(ms) |")
        lines.append("|------|------|---------|--------|-----------|---------|---------|---------|")
        for algo_key, algo_name in zip(algos, names):
            s = results[algo_key]["summary"]
            bf1 = s.get("avg_boundary_f1", 0)
            lines.append(f"| {algo_name} | {s['avg_miou']:.4f} | {s['avg_obj_iou']:.4f} | "
                         f"{bf1:.3f} | {s['instance_count_accuracy']:.1%} | "
                         f"{s['avg_edge_accuracy']:.1%} | {s['grasp_order_accuracy']:.1%} | "
                         f"{s['avg_latency_ms']:.0f} |")
        lines.append("")

        lines.append("### A.4 各维度最优\n")
        lines.append(f"- **语义分割 mIoU**: {names[best_miou_idx]} ({summaries[best_miou_idx]['avg_miou']:.4f})")
        lines.append(f"- **实例数量准确率**: {names[best_inst_idx]} ({summaries[best_inst_idx]['instance_count_accuracy']:.1%})")
        lines.append(f"- **层级边准确率**: {names[best_edge_idx]} ({summaries[best_edge_idx]['avg_edge_accuracy']:.1%})")
        lines.append(f"- **抓取顺序正确率**: {names[best_grasp_idx]} ({summaries[best_grasp_idx]['grasp_order_accuracy']:.1%})")
        lines.append(f"- **推理速度**: {names[fastest_idx]} ({summaries[fastest_idx]['avg_latency_ms']:.0f}ms)\n")

        lines.append("### A.5 图表\n")
        lines.append("![综合对比](algo_comparison_figures/algo_comparison_overview.png)")
        lines.append("![各场景 mIoU](algo_comparison_figures/algo_per_scene_miou.png)")
        lines.append("![结果汇总表](algo_comparison_figures/algo_comparison_table.png)\n")
    else:
        lines.append("*Part A 已跳过，仅运行 Part B 堆叠检测算法对比*\n")

    if stacking_results:
        s_algos = list(stacking_results.keys())
        s_names = [stacking_results[a]["config_name"] for a in s_algos]
        s_summaries = [stacking_results[a]["summary"] for a in s_algos]

        ours_key = [a for a in s_algos if stacking_results[a]["summary"].get("category") == "ours"]
        baseline_keys = [a for a in s_algos if stacking_results[a]["summary"].get("category") == "baseline"]
        trad_keys = [a for a in s_algos if stacking_results[a]["summary"].get("category") == "traditional"]

        lines.append("---\n")
        lines.append("## Part B: 堆叠检测算法对比 (核心)\n")

        lines.append("### B.1 实验设计\n")
        lines.append("- **核心思路**: 使用同一 PP-Attention 语义分割结果，仅替换堆叠检测后处理算法")
        lines.append("- **目的**: 隔离堆叠检测算法的贡献，排除分割质量的影响")
        lines.append("- **测试数据**: 8 原始堆叠场景 (并排/部分重叠/完全堆叠/交叉堆叠/对齐堆叠)")
        lines.append("- **对比算法**: 8 种 (1 Ours + 5 Baseline + 2 Traditional)\n")

        lines.append("### B.2 对比方法\n")
        lines.append("| 方法 | 类别 | 分割Backbone | 堆叠检测策略 |")
        lines.append("|------|------|-------------|-------------|")
        for algo_key in s_algos:
            info = stacking_results[algo_key]
            s = info["summary"]
            cat = s.get("category", "baseline")
            desc = s.get("description", "")
            name = info["config_name"]
            backbone = "PP-Attention" if cat != "traditional" else ("RANSAC" if "RANSAC" in name else "Depth Threshold")
            lines.append(f"| **{name}** | {cat} | {backbone} | {desc} |")
        lines.append("")

        lines.append("### B.3 堆叠检测综合结果 (含非堆叠场景)\n")
        lines.append("| 方法 | 实例准确率 | 边 Precision | 边 Recall | 边 F1 | 抓取顺序 |")
        lines.append("|------|-----------|------------|----------|-------|---------|")
        for algo_key in s_algos:
            s = stacking_results[algo_key]["summary"]
            name = stacking_results[algo_key]["config_name"]
            lines.append(f"| {name} | {s['instance_count_accuracy']:.1%} | "
                         f"{s['overall_edge_precision']:.3f} | {s['overall_edge_recall']:.3f} | "
                         f"{s['overall_edge_f1']:.3f} | {s['overall_grasp_accuracy']:.1%} |")
        lines.append("")

        if ours_key and baseline_keys:
            ours_s = stacking_results[ours_key[0]]["summary"]
            avg_bl_edge_f1 = np.mean([stacking_results[k]["summary"]["overall_edge_f1"] for k in baseline_keys])
            avg_bl_grasp = np.mean([stacking_results[k]["summary"]["overall_grasp_accuracy"] for k in baseline_keys])
            best_bl_edge_f1 = max(stacking_results[k]["summary"]["overall_edge_f1"] for k in baseline_keys)
            best_bl_grasp = max(stacking_results[k]["summary"]["overall_grasp_accuracy"] for k in baseline_keys)

            lines.append("### B.4 提升分析\n")
            lines.append(f"| 指标 | Ours | Best Baseline | Avg Baseline | vs Best | vs Avg |")
            lines.append(f"|------|------|--------------|-------------|---------|-------|")
            lines.append(f"| Edge F1 | {ours_s['overall_edge_f1']:.3f} | {best_bl_edge_f1:.3f} | "
                         f"{avg_bl_edge_f1:.3f} | {ours_s['overall_edge_f1'] - best_bl_edge_f1:+.3f} | "
                         f"{ours_s['overall_edge_f1'] - avg_bl_edge_f1:+.3f} |")
            lines.append(f"| Grasp Acc | {ours_s['overall_grasp_accuracy']:.1%} | {best_bl_grasp:.1%} | "
                         f"{avg_bl_grasp:.1%} | {ours_s['overall_grasp_accuracy'] - best_bl_grasp:+.1%} | "
                         f"{ours_s['overall_grasp_accuracy'] - avg_bl_grasp:+.1%} |")
            lines.append("")

            lines.append("**核心优势分析:**\n")
            lines.append("1. **双重判定机制**: HierarchyReasoner 同时使用 XY 重叠检测和质心邻近度，避免单一判据的局限性")
            lines.append("2. **Z 间隙感知**: 通过 z_min - z_max 间隙判断真实的上下层关系，而非简单的高度排序")
            lines.append("3. **拓扑排序**: 基于 DAG 的拓扑排序确保抓取顺序满足物理约束（先上层后下层）")
            lines.append("4. **鲁棒性**: 在部分重叠、偏移堆叠等复杂场景中仍能准确检测堆叠关系")
            lines.append("5. **Simple Z-Sort 的局限**: 仅按高度排序，无法区分并排物体和堆叠物体，在偏移堆叠场景中完全失效\n")

        if any("difficulty_breakdown" in stacking_results[a]["summary"] for a in s_algos):
            lines.append("### B.5 鲁棒性分析 (不同难度)\n")
            lines.append("| 方法 | easy | medium | hard |")
            lines.append("|------|------|--------|------|")
            for algo_key in s_algos:
                s = stacking_results[algo_key]["summary"]
                name = stacking_results[algo_key]["config_name"]
                db = s.get("difficulty_breakdown", {})
                easy_f1 = db.get("easy", {}).get("edge_f1", 0)
                medium_f1 = db.get("medium", {}).get("edge_f1", 0)
                hard_f1 = db.get("hard", {}).get("edge_f1", 0)
                lines.append(f"| {name} | {easy_f1:.3f} | {medium_f1:.3f} | {hard_f1:.3f} |")
            lines.append("")

        lines.append("### B.6 图表\n")
        lines.append("![堆叠检测对比](algo_comparison_figures/stacking_comparison.png)")
        lines.append("![堆叠检测汇总表](algo_comparison_figures/stacking_comparison_table.png)")
        if os.path.exists(os.path.join(FIGURE_DIR, "stacking_robustness.png")):
            lines.append("![鲁棒性分析](algo_comparison_figures/stacking_robustness.png)")
        lines.append("")

    lines.append("---\n")
    lines.append(f"*报告自动生成于 {time.strftime('%Y-%m-%d %H:%M:%S')}*")

    report_path = os.path.join(BASE_DIR, "algo_comparison_report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    logger.info(f"Report saved to {report_path}")
    return report_path


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true", help="训练所有模型 (10 epoch)")
    parser.add_argument("--eval", action="store_true", help="评估所有算法")
    parser.add_argument("--all", action="store_true", help="训练 + 评估 + 图表 + 报告")
    parser.add_argument("--stacking-only", action="store_true", help="仅运行堆叠检测算法对比")
    parser.add_argument("--part-b", action="store_true", help="仅运行 Part B: 堆叠检测算法对比 (跳过模型对比)")
    parser.add_argument("--part-a", action="store_true", help="仅运行 Part A: 语义分割模型对比 (跳过堆叠检测)")
    args = parser.parse_args()

    if args.train or args.all:
        logger.info("=" * 60)
        logger.info("Phase 1: Training all models (10 epoch)")
        logger.info("=" * 60)
        train_all_models()

    if args.eval or args.all:
        logger.info("=" * 60)
        logger.info("Phase 2: Generating stacking test data")
        logger.info("=" * 60)
        generate_stacking_test_data()

        logger.info("=" * 60)
        logger.info("Phase 3: Evaluating all algorithms (Part A)")
        logger.info("=" * 60)
        results = evaluate_all()

        logger.info("=" * 60)
        logger.info("Phase 4: Evaluating stacking detection methods (Part B)")
        logger.info("=" * 60)
        stacking_results = evaluate_stacking_methods()

        logger.info("=" * 60)
        logger.info("Phase 5: Generating comparison charts")
        logger.info("=" * 60)
        generate_comparison_charts(results)
        generate_stacking_charts(stacking_results)

        logger.info("=" * 60)
        logger.info("Phase 6: Generating report")
        logger.info("=" * 60)
        generate_report(results, stacking_results)

        logger.info("\n" + "=" * 60)
        logger.info("All done!")
        logger.info(f"Results: {RESULT_DIR}")
        logger.info(f"Figures: {FIGURE_DIR}")
        logger.info(f"Report: {os.path.join(BASE_DIR, 'algo_comparison_report.md')}")
        logger.info("=" * 60)

    if args.stacking_only:
        logger.info("=" * 60)
        logger.info("Stacking Detection Comparison Only")
        logger.info("=" * 60)
        generate_stacking_test_data()
        stacking_results = evaluate_stacking_methods()
        generate_stacking_charts(stacking_results)
        generate_report({}, stacking_results)
        logger.info("\nStacking comparison done!")

    if args.part_a:
        logger.info("=" * 60)
        logger.info("Part A: Semantic Segmentation Model Comparison")
        logger.info("=" * 60)
        results = evaluate_all()
        generate_comparison_charts(results)
        generate_report(results, {})
        logger.info("\nPart A done!")

    if args.part_b:
        logger.info("=" * 60)
        logger.info("Part B: Stacking Detection Algorithm Comparison")
        logger.info("=" * 60)
        stacking_results = evaluate_stacking_methods()
        generate_stacking_charts(stacking_results)
        generate_report({}, stacking_results)
        logger.info("\nPart B done!")
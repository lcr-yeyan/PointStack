"""
模型测试: 加载 best_model.pth，在 data_preview 场景上推理
========================================================
输出: camera_operate/test_results/
  - 每个场景: GT vs 预测对比图 + 预测点云 + 语义标签 + metrics
  - 总览: summary.json + 全部场景的对比拼图
"""

import os, json, time
import numpy as np
import cv2
import open3d as o3d
import torch
from collections import defaultdict

from loguru import logger
from models.pointnet_seg import PointNetPlusPlusAttentionSeg

# ── 参数 ──
FX, FY = 506.35, 506.27
CX, CY = 334.19, 335.43
NEAR = 0.05
NUM_POINTS = 2048
NUM_CLASSES = 2
INPUT_CHANNELS = 6
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "training_data", "train_logs", "best_model.pth")
DATA_DIR = os.path.join(BASE_DIR, "data_preview")
OUT_DIR = os.path.join(BASE_DIR, "test_results")
os.makedirs(OUT_DIR, exist_ok=True)

COLORS_CLASS = {
    0: [0.35, 0.35, 0.40],  # table gray
    1: [0.0, 0.9, 0.0],     # object green
}
COLORS_PRED_MATCH = {
    "TP": [0.0, 1.0, 0.0],     # green: 预测对
    "FP": [1.0, 0.0, 0.0],     # red: 误检
    "FN": [0.0, 0.0, 1.0],     # blue: 漏检
    "TN": [0.35, 0.35, 0.40],  # gray: 正确背景
}


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


def load_model():
    model = PointNetPlusPlusAttentionSeg(input_channels=INPUT_CHANNELS, num_classes=NUM_CLASSES)
    ckpt = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(DEVICE)
    model.eval()
    logger.info(f"Model loaded: epoch={ckpt['epoch']}, mIoU={ckpt['miou']:.4f}")
    return model


def depth_to_points(depth_m):
    h, w = depth_m.shape
    valid = depth_m > NEAR
    uu, vv = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    Z = depth_m[valid]
    X = (uu[valid] - CX) * Z / FX
    Y = (vv[valid] - CY) * Z / FY
    return np.stack([X, Y, Z], axis=-1).astype(np.float64), valid, uu[valid], vv[valid]


def predict_scene(model, depth_m, seg_gt):
    """对单帧推理，返回逐点预测标签"""
    points, valid_mask, u_arr, v_arr = depth_to_points(depth_m)
    n_all = len(points)

    # 6ch normalize + sample
    idxs = np.random.choice(n_all, NUM_POINTS, replace=(n_all < NUM_POINTS))
    pts_sample = points[idxs]
    feat_sample = normalize_points_6ch(pts_sample)

    with torch.no_grad():
        feat_t = torch.from_numpy(feat_sample).unsqueeze(0).float().to(DEVICE)
        logits, _, _ = model(feat_t)
        pred_sample = logits.argmax(dim=-1).squeeze(0).cpu().numpy()

    # NN mapping back to all points
    from scipy.spatial import cKDTree
    tree = cKDTree(pts_sample[:, :3])
    _, nn_idx = tree.query(points[:, :3], k=1)
    pred_all = pred_sample[nn_idx]

    return pred_all, points


def compute_metrics(pred, gt):
    """计算逐类 IoU + 混淆矩阵"""
    ious = {}
    for c in range(NUM_CLASSES):
        inter = ((pred == c) & (gt == c)).sum()
        union = ((pred == c) | (gt == c)).sum()
        ious[c] = float((inter + 1) / (union + 1))
    miou = np.mean(list(ious.values()))

    # confusion matrix
    cm = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=int)
    for c_gt in range(NUM_CLASSES):
        for c_pred in range(NUM_CLASSES):
            cm[c_gt, c_pred] = ((gt == c_gt) & (pred == c_pred)).sum()

    return miou, ious, cm


def visualize_comparison(rgb, depth_m, pred, gt):
    """对比图: RGB | GT labels | Pred labels | Error map"""
    h, w = depth_m.shape

    if rgb.shape[:2] != (h, w):
        rgb = cv2.resize(rgb, (w, h))

    # GT viz
    gt_vis = np.zeros((h, w, 3), dtype=np.uint8)
    gt_vis[gt == 0] = [60, 60, 60]
    gt_vis[gt == 1] = [0, 255, 0]

    # Pred viz
    pred_vis = np.zeros((h, w, 3), dtype=np.uint8)
    pred_vis[pred == 0] = [60, 60, 60]
    pred_vis[pred == 1] = [0, 255, 0]

    # Error map: TP=green, FP=red, FN=blue
    err_vis = np.zeros((h, w, 3), dtype=np.uint8)
    err_vis[(pred == 1) & (gt == 1)] = [0, 255, 0]    # TP
    err_vis[(pred == 1) & (gt == 0)] = [0, 0, 255]    # FP
    err_vis[(pred == 0) & (gt == 1)] = [255, 0, 0]    # FN
    err_vis[(pred == 0) & (gt == 0)] = [60, 60, 60]   # TN

    row1 = np.hstack([rgb, gt_vis])
    row2 = np.hstack([pred_vis, err_vis])
    canvas = np.vstack([row1, row2])

    labels = ["RGB", "Ground Truth", "Prediction", "Error (blue=FP, red=FN)"]
    for i, label in enumerate(labels):
        cx = (i % 2) * w + 5
        cy = (i // 2) * h + 20
        cv2.putText(canvas, label, (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    return canvas


def run_tests():
    model = load_model()
    scenes = sorted([d for d in os.listdir(DATA_DIR) if d.startswith("scene_")])

    all_metrics = []
    summary_rows = []

    for scene_name in scenes:
        scene_dir = os.path.join(DATA_DIR, scene_name)
        logger.info(f"\n{'='*40}\n{scene_name}\n{'='*40}")

        depth = np.load(os.path.join(scene_dir, "depth_clean.npy"))
        gt = np.load(os.path.join(scene_dir, "semantic_labels.npy"))
        rgb = cv2.imread(os.path.join(scene_dir, "rgb.png"))
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)

        # inference
        t0 = time.perf_counter()
        pred, points = predict_scene(model, depth, gt)
        t_ms = (time.perf_counter() - t0) * 1000

        # flatten GT to match pred (both are valid points only)
        valid_mask = depth > NEAR
        gt_flat = gt[valid_mask]

        miou, ious, cm = compute_metrics(pred, gt_flat)

        logger.info(f"  mIoU={miou:.4f} | table={ious[0]:.4f} obj={ious[1]:.4f} | {t_ms:.0f}ms")
        logger.info(f"  CM: TP={cm[1,1]} FP={cm[0,1]} FN={cm[1,0]} TN={cm[0,0]}")

        all_metrics.append({
            "scene": scene_name,
            "miou": miou,
            "iou_table": ious[0],
            "iou_object": ious[1],
            "latency_ms": t_ms,
            "confusion_matrix": cm.tolist(),
        })

        # visualize (map pred back to 2D image)
        pred_2d = np.zeros_like(gt, dtype=np.uint8)
        pred_2d[valid_mask] = pred
        vis = visualize_comparison(rgb, depth, pred_2d, gt)
        cv2.imwrite(os.path.join(OUT_DIR, f"{scene_name}_compare.png"), cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))

        # labeled pointcloud (prediction colors)
        colors_pcd = np.zeros((len(points), 3), dtype=np.float64)
        for c in range(NUM_CLASSES):
            colors_pcd[pred == c] = COLORS_CLASS[c]
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors_pcd)
        o3d.io.write_point_cloud(os.path.join(OUT_DIR, f"{scene_name}_pred.ply"), pcd)

        # summary row
        with open(os.path.join(scene_dir, "annotation.json"), encoding="utf-8") as f:
            ann = json.load(f)
        summary_rows.append({
            "scene": scene_name,
            "description": ann.get("description_cn", ""),
            "has_stacking": ann["has_stacking"],
            "miou": round(miou, 4),
            "iou_object": round(ious[1], 4),
            "latency_ms": round(t_ms, 1),
            "correct_pixels": int(cm[1, 1] + cm[0, 0]),
            "total_pixels": int(cm.sum()),
        })

    # summary
    mious = [m["miou"] for m in all_metrics]
    lats = [m["latency_ms"] for m in all_metrics]
    summary = {
        "model": "PointNet++Attention",
        "checkpoint": MODEL_PATH,
        "num_scenes": len(scenes),
        "avg_miou": round(np.mean(mious), 4),
        "avg_latency_ms": round(np.mean(lats), 1),
        "per_scene": summary_rows,
        "detailed": all_metrics,
    }
    with open(os.path.join(OUT_DIR, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # print summary table
    logger.info(f"\n{'='*60}")
    logger.info("TEST RESULTS SUMMARY")
    logger.info(f"{'='*60}")
    logger.info(f"  Avg mIoU     : {summary['avg_miou']:.4f}")
    logger.info(f"  Avg Latency  : {summary['avg_latency_ms']:.0f} ms")
    for row in summary_rows:
        logger.info(f"  {row['scene']:30s} | mIoU={row['miou']:.4f} obj={row['iou_object']:.4f} | {row['latency_ms']:.0f}ms")
    logger.info(f"\nAll results saved to: {OUT_DIR}")


if __name__ == "__main__":
    run_tests()

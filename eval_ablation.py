"""消融实验评估脚本 - 语义分割 + 后处理层级推理评估

对每个消融模型变体:
  1. 在 8 个 data_preview 场景上推理语义分割
  2. 运行后处理 (实例分割 + 层级推理 + 抓取顺序)
  3. 与 annotation.json 真值对比
  4. 输出: mIoU, 实例数量准确率, 层级边准确率, 抓取顺序正确率, 推理延迟
"""

import os, json, time, sys
import numpy as np
import torch
from collections import defaultdict
from scipy.spatial import cKDTree
from loguru import logger

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models.pointnet_seg import PointNetPlusPlusAttentionSeg, PointNetPlusPlusSeg
from modules.postprocess import (
    Instance, InstanceSegmentationResult,
    _euclidean_clustering, _fit_support_plane,
)
from modules.hierarchy import HierarchyReasoner, build_hierarchy

FX, FY = 506.35, 506.27
CX, CY = 334.19, 335.43
NEAR = 0.05
NUM_POINTS = 2048
NUM_CLASSES = 2
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data_preview")
ABLATION_LOG_DIR = os.path.join(BASE_DIR, "training_data", "ablation_logs")
FULL_MODEL_PATH = os.path.join(BASE_DIR, "training_data", "train_logs", "best_model.pth")
OUT_DIR = os.path.join(BASE_DIR, "ablation_eval_results")
os.makedirs(OUT_DIR, exist_ok=True)


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


def depth_to_points(depth_m):
    h, w = depth_m.shape
    valid = depth_m > NEAR
    uu, vv = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    Z = depth_m[valid]
    X = (uu[valid] - CX) * Z / FX
    Y = (vv[valid] - CY) * Z / FY
    return np.stack([X, Y, Z], axis=-1).astype(np.float64), valid


def compute_iou(pred, target, num_classes):
    ious = []
    for c in range(num_classes):
        intersection = ((pred == c) & (target == c)).sum()
        union = ((pred == c) | (target == c)).sum()
        ious.append((intersection + 1) / (union + 1))
    miou = sum(ious) / len(ious)
    return miou, ious


def compute_confusion_matrix(pred, target, num_classes):
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for c in range(num_classes):
        for p in range(num_classes):
            cm[c, p] = ((target == c) & (pred == p)).sum()
    return cm


def predict_scene(model, depth_m, use_6ch=True):
    points, valid_mask = depth_to_points(depth_m)
    n_all = len(points)

    idxs = np.random.choice(n_all, NUM_POINTS, replace=(n_all < NUM_POINTS))
    pts_sample = points[idxs]

    if use_6ch:
        feat_sample = normalize_points_6ch(pts_sample)
    else:
        feat_sample = normalize_points_3ch(pts_sample)

    t0 = time.time()
    with torch.no_grad():
        feat_t = torch.from_numpy(feat_sample).unsqueeze(0).float().to(DEVICE)
        logits, _, _ = model(feat_t)
        pred_sample = logits.argmax(dim=-1).squeeze(0).cpu().numpy()
    latency = (time.time() - t0) * 1000

    tree = cKDTree(pts_sample[:, :3])
    _, nn_idx = tree.query(points[:, :3], k=1)
    pred_all = pred_sample[nn_idx]

    return pred_all, points, valid_mask, latency


def run_instance_segmentation(points, pred_labels):
    object_mask = pred_labels == 1
    if object_mask.sum() < 10:
        return [], np.full(len(points), -1, dtype=int)

    object_points = points[object_mask]
    object_indices = np.where(object_mask)[0]

    clusters, cluster_labels = _euclidean_clustering(
        object_points, tolerance=0.015, min_cluster_size=50
    )

    instances = []
    instance_labels = np.full(len(points), -1, dtype=int)

    for i, cluster in enumerate(clusters):
        global_indices = object_indices[cluster]
        inst_pts = points[global_indices]

        centroid = np.mean(inst_pts, axis=0)
        bbox_min = np.min(inst_pts, axis=0)
        bbox_max = np.max(inst_pts, axis=0)
        z_vals = inst_pts[:, 2]

        inst = Instance(
            id=i,
            point_indices=global_indices,
            centroid=centroid,
            bbox_min=bbox_min,
            bbox_max=bbox_max,
            point_count=len(global_indices),
            z_mean=float(np.median(z_vals)),
            z_min=float(np.percentile(z_vals, 2)),
            z_max=float(np.percentile(z_vals, 98)),
        )
        instances.append(inst)
        instance_labels[global_indices] = i

    return instances, instance_labels


def load_gt_from_annotation(scene_dir):
    ann_path = os.path.join(scene_dir, "annotation.json")
    if not os.path.exists(ann_path):
        return None
    with open(ann_path) as f:
        ann = json.load(f)
    return ann


def compute_gt_hierarchy(ann):
    """从 annotation.json 计算真值层级关系"""
    objects = ann.get("objects", [])
    num_objects = ann.get("num_objects", len(objects))
    has_stacking = ann.get("has_stacking", False)

    gt_edges = []
    if has_stacking and len(objects) >= 2:
        sorted_objs = sorted(objects, key=lambda o: o.get("z_min", 0))
        for i in range(len(sorted_objs) - 1):
            upper = sorted_objs[i + 1]
            lower = sorted_objs[i]
            gt_edges.append({
                "upper": upper["id"],
                "lower": lower["id"],
                "type": "direct_support",
            })

    gt_grasp_order = []
    if has_stacking:
        sorted_objs = sorted(objects, key=lambda o: o.get("z_max", 0), reverse=True)
        gt_grasp_order = [o["id"] for o in sorted_objs]
    else:
        gt_grasp_order = [o["id"] for o in objects]

    return {
        "num_objects": num_objects,
        "has_stacking": has_stacking,
        "gt_edges": gt_edges,
        "gt_grasp_order": gt_grasp_order,
    }


def evaluate_postprocess(instances, points, ann):
    """评估后处理结果"""
    gt = compute_gt_hierarchy(ann)
    gt_num_objects = gt["num_objects"]
    gt_edges = gt["gt_edges"]
    gt_grasp_order = gt["gt_grasp_order"]

    pred_num_instances = len(instances)
    instance_count_correct = (pred_num_instances == gt_num_objects)

    hierarchy_config = {
        "hierarchy": {
            "z_gap_threshold": 0.06,
            "xy_overlap_min": 0.02,
            "contact_z_tolerance": 0.035,
            "indirect_support_enabled": True,
            "max_indirect_depth": 3,
            "z_gap_factor_by_size": True,
            "stability_weight_centroid": 0.4,
            "stability_weight_contact": 0.3,
            "stability_weight_support": 0.3,
            "max_z_gap_for_stacking": 0.30,
            "xy_proximity_max": 0.10,
            "min_z_gap_for_stacking": 0.005,
            "min_z_mean_diff": 0.02,
            "min_xy_overlap_for_stacking": 0.03,
        },
        "_fx": FX,
        "_fy": FY,
        "_cx": CX,
        "_cy": CY,
    }

    hierarchy_result = build_hierarchy(instances, points, hierarchy_config)
    pred_edges = hierarchy_result.edges
    pred_grasp_order = hierarchy_result.grasp_order

    edge_correct = 0
    edge_total = max(len(gt_edges), 1)
    for gt_edge in gt_edges:
        for pred_edge in pred_edges:
            if (pred_edge.upper_id == gt_edge["upper"] and
                pred_edge.lower_id == gt_edge["lower"]):
                edge_correct += 1
                break
    edge_accuracy = edge_correct / edge_total

    grasp_correct = False
    if len(gt_grasp_order) > 0 and len(pred_grasp_order) > 0:
        if len(gt_grasp_order) == len(pred_grasp_order):
            grasp_correct = all(
                gt_grasp_order[i] == pred_grasp_order[i]
                for i in range(len(gt_grasp_order))
            )
        elif len(pred_grasp_order) >= len(gt_grasp_order):
            grasp_correct = all(
                g in pred_grasp_order[:len(gt_grasp_order)]
                for g in gt_grasp_order
            )

    return {
        "instance_count_correct": instance_count_correct,
        "pred_instances": pred_num_instances,
        "gt_instances": gt_num_objects,
        "edge_accuracy": edge_accuracy,
        "edge_correct": edge_correct,
        "edge_total": edge_total,
        "grasp_correct": grasp_correct,
        "pred_grasp_order": pred_grasp_order,
        "gt_grasp_order": gt_grasp_order,
        "pred_edges": [(e.upper_id, e.lower_id) for e in pred_edges],
        "gt_edges": [(e["upper"], e["lower"]) for e in gt_edges],
    }


def load_ablation_model(config_key):
    """加载消融模型"""
    if config_key == "full_pp_attention":
        model = PointNetPlusPlusAttentionSeg(input_channels=6, num_classes=NUM_CLASSES)
        ckpt = torch.load(FULL_MODEL_PATH, map_location=DEVICE, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        model.to(DEVICE)
        model.eval()
        return model, True

    model_dir = os.path.join(ABLATION_LOG_DIR, config_key)
    model_path = os.path.join(model_dir, "best_model.pth")
    if not os.path.exists(model_path):
        logger.warning(f"Model not found: {model_path}")
        return None, None

    ckpt = torch.load(model_path, map_location=DEVICE, weights_only=False)
    config = ckpt.get("config", {})

    if config.get("model_type") == "PointNetPlusPlusSeg" or config_key in ("baseline_3ch", "plus_6ch_input"):
        input_ch = config.get("input_channels", 6 if config_key == "plus_6ch_input" else 3)
        model = PointNetPlusPlusSeg(input_channels=input_ch, num_classes=NUM_CLASSES)
        use_6ch = (input_ch == 6)
    else:
        from train_ablation import AblationModel
        flags = config.get("ablation_flags", {})
        input_ch = config.get("input_channels", 6)
        model = AblationModel(input_channels=input_ch, num_classes=NUM_CLASSES, **flags)
        use_6ch = True

    model.load_state_dict(ckpt["model_state_dict"])
    model.to(DEVICE)
    model.eval()
    return model, use_6ch


def evaluate_all():
    scene_dirs = sorted([
        os.path.join(DATA_DIR, d) for d in os.listdir(DATA_DIR)
        if os.path.isdir(os.path.join(DATA_DIR, d))
    ])

    configs_to_eval = [
        ("baseline_3ch", "Baseline (3ch XYZ)"),
        ("plus_6ch_input", "+6ch Input"),
        ("plus_channel_attn", "+Channel Attention"),
        ("plus_position_attn", "+Position Attention"),
        ("plus_multiscale", "+MultiScale Fusion"),
        ("full_pp_attention", "Full PP-Attention"),
    ]

    all_results = {}

    for config_key, config_name in configs_to_eval:
        logger.info(f"\n{'='*60}")
        logger.info(f"Evaluating: {config_name} ({config_key})")
        logger.info(f"{'='*60}")

        model, use_6ch = load_ablation_model(config_key)
        if model is None:
            logger.warning(f"Skipping {config_name} - model not trained yet")
            continue

        config_results = {
            "config_name": config_name,
            "config_key": config_key,
            "per_scene": [],
            "summary": {},
        }

        total_miou = 0.0
        total_obj_iou = 0.0
        total_latency = 0.0
        total_cm = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=int)
        instance_count_correct = 0
        total_edge_accuracy = 0.0
        grasp_order_correct = 0
        num_scenes = 0

        for scene_dir in scene_dirs:
            scene_name = os.path.basename(scene_dir)
            depth = np.load(os.path.join(scene_dir, "depth_noisy.npy"))
            labels = np.load(os.path.join(scene_dir, "semantic_labels.npy"))
            ann = load_gt_from_annotation(scene_dir)

            pred_labels, points, valid_mask, latency = predict_scene(model, depth, use_6ch)

            gt_flat = labels[valid_mask]
            miou, ious = compute_iou(pred_labels, gt_flat, NUM_CLASSES)
            cm = compute_confusion_matrix(pred_labels, gt_flat, NUM_CLASSES)

            instances, instance_labels = run_instance_segmentation(points, pred_labels)

            post_eval = None
            if ann is not None:
                post_eval = evaluate_postprocess(instances, points, ann)

            scene_result = {
                "scene": scene_name,
                "miou": float(miou),
                "iou_table": float(ious[0]),
                "iou_object": float(ious[1]),
                "latency_ms": float(latency),
                "confusion_matrix": cm.tolist(),
            }

            if post_eval:
                scene_result["postprocess"] = post_eval
                instance_count_correct += int(post_eval["instance_count_correct"])
                total_edge_accuracy += post_eval["edge_accuracy"]
                grasp_order_correct += int(post_eval["grasp_correct"])

            config_results["per_scene"].append(scene_result)
            total_miou += miou
            total_obj_iou += ious[1]
            total_latency += latency
            total_cm += cm
            num_scenes += 1

            logger.info(f"  {scene_name}: mIoU={miou:.4f}, obj_IoU={ious[1]:.4f}, "
                       f"latency={latency:.1f}ms"
                       + (f", inst_acc={post_eval['instance_count_correct']}, "
                          f"edge_acc={post_eval['edge_accuracy']:.2f}, "
                          f"grasp={post_eval['grasp_correct']}" if post_eval else ""))

        config_results["summary"] = {
            "avg_miou": float(total_miou / num_scenes),
            "avg_obj_iou": float(total_obj_iou / num_scenes),
            "avg_latency_ms": float(total_latency / num_scenes),
            "total_confusion_matrix": total_cm.tolist(),
            "instance_count_accuracy": float(instance_count_correct / num_scenes) if num_scenes > 0 else 0.0,
            "avg_edge_accuracy": float(total_edge_accuracy / num_scenes) if num_scenes > 0 else 0.0,
            "grasp_order_accuracy": float(grasp_order_correct / num_scenes) if num_scenes > 0 else 0.0,
        }

        all_results[config_key] = config_results

        logger.info(f"  Summary: mIoU={config_results['summary']['avg_miou']:.4f}, "
                   f"obj_IoU={config_results['summary']['avg_obj_iou']:.4f}, "
                   f"latency={config_results['summary']['avg_latency_ms']:.1f}ms, "
                   f"inst_acc={config_results['summary']['instance_count_accuracy']:.2%}, "
                   f"edge_acc={config_results['summary']['avg_edge_accuracy']:.2%}, "
                   f"grasp_acc={config_results['summary']['grasp_order_accuracy']:.2%}")

    with open(os.path.join(OUT_DIR, "ablation_eval_summary.json"), "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    logger.info(f"\nResults saved to {os.path.join(OUT_DIR, 'ablation_eval_summary.json')}")

    logger.info("\n" + "=" * 70)
    logger.info("FINAL COMPARISON TABLE")
    logger.info("=" * 70)
    logger.info(f"{'Config':<25s} {'mIoU':>8s} {'ObjIoU':>8s} {'InstAcc':>8s} {'EdgeAcc':>8s} {'GraspAcc':>8s} {'Lat(ms)':>8s}")
    logger.info("-" * 70)
    for config_key, config_name in configs_to_eval:
        if config_key in all_results:
            s = all_results[config_key]["summary"]
            logger.info(f"{config_name:<25s} {s['avg_miou']:>8.4f} {s['avg_obj_iou']:>8.4f} "
                       f"{s['instance_count_accuracy']:>8.2%} {s['avg_edge_accuracy']:>8.2%} "
                       f"{s['grasp_order_accuracy']:>8.2%} {s['avg_latency_ms']:>8.1f}")

    return all_results


if __name__ == "__main__":
    evaluate_all()
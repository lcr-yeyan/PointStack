"""非交互式实时仿真测试 - 自动生成随机场景并收集数据"""
import os, sys, json, time, random
import numpy as np
import cv2
import torch
import pybullet as p
from scipy.spatial import cKDTree
from collections import defaultdict
from loguru import logger

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.pointnet_seg import PointNetPlusPlusAttentionSeg
from modules.hierarchy import build_hierarchy, HierarchyResult
from modules.postprocess import InstanceClustering, Instance
from utils.config import load_config
from omegaconf import OmegaConf
import open3d as o3d

NUM_POINTS = 2048
NUM_CLASSES = 2
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NEAR = 0.01
FAR = 2.0

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "realtime_results")
os.makedirs(OUT_DIR, exist_ok=True)

COLORS = [
    (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
    (255, 0, 255), (0, 255, 255), (128, 255, 0), (255, 128, 0),
    (0, 128, 255), (128, 0, 255),
]


def normalize_points_6ch(points):
    xyz = points[:, :3].copy()
    centroid = xyz.mean(axis=0)
    xyz_centered = xyz - centroid
    dist = np.sqrt((xyz_centered ** 2).sum(axis=1)).max()
    if dist < 1e-6:
        dist = 1.0
    xyz_norm = xyz_centered / dist
    z = points[:, 2:3].copy()
    z_centered = z - z.mean()
    z_norm1 = z_centered / (z_centered.std() + 1e-6)
    z_norm2 = (z - z.min()) / (z.max() - z.min() + 1e-6)
    z_norm3 = z / (z.max() + 1e-6)
    return np.concatenate([xyz_norm, z_norm1, z_norm2, z_norm3], axis=1).astype(np.float32)


def depth_to_points(depth_m):
    h, w = depth_m.shape
    fx = fy = w * 1.2
    cx, cy = w / 2, h / 2
    v, u = np.mgrid[0:h, 0:w]
    valid = (depth_m > NEAR) & (depth_m < FAR)
    z = depth_m[valid]
    x = (u[valid] - cx) * z / fx
    y = (v[valid] - cy) * z / fy
    points = np.stack([x, y, z], axis=-1).astype(np.float32)
    return points, valid, u[valid], v[valid]


def random_scene(rng):
    num_objects = rng.integers(1, 4)
    stacking = rng.random() < 0.6
    objects = []
    for i in range(num_objects):
        w = rng.uniform(0.03, 0.08)
        d = rng.uniform(0.03, 0.08)
        h = rng.uniform(0.02, 0.06)
        x = rng.uniform(-0.12, 0.12)
        y = rng.uniform(-0.10, 0.10)
        if stacking and i > 0:
            prev = objects[i - 1]
            x = prev["x"] + rng.uniform(-0.03, 0.03)
            y = prev["y"] + rng.uniform(-0.03, 0.03)
            z = prev["z"] + prev["h"]
        else:
            z = 0.43
        objects.append({"w": w, "d": d, "h": h, "x": x, "y": y, "z": z})
    return objects, stacking


class AutoRealtimeTester:
    def __init__(self):
        self.client_id = None
        self.model = None
        self.rng = np.random.default_rng(42)

    def connect(self):
        self.client_id = p.connect(p.DIRECT)
        p.setGravity(0, 0, -9.8, physicsClientId=self.client_id)
        p.setTimeStep(1 / 240, physicsClientId=self.client_id)

    def load_model(self):
        ckpt_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "training_data", "train_logs", "best_model.pth"
        )
        self.model = PointNetPlusPlusAttentionSeg(
            input_channels=6, num_classes=NUM_CLASSES, use_normal=False
        ).to(DEVICE)
        ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.eval()
        logger.info(f"Model loaded: epoch={ckpt.get('epoch', '?')}")

    def setup_scene(self, objects):
        p.resetSimulation(physicsClientId=self.client_id)
        p.setGravity(0, 0, -9.8, physicsClientId=self.client_id)

        table_vis = p.createVisualShape(
            p.GEOM_BOX, halfExtents=[0.2, 0.15, 0.005],
            rgbaColor=[0.5, 0.5, 0.5, 1], physicsClientId=self.client_id
        )
        table_col = p.createCollisionShape(
            p.GEOM_BOX, halfExtents=[0.2, 0.15, 0.005],
            physicsClientId=self.client_id
        )
        p.createMultiBody(0, table_col, table_vis, [0, 0, 0.425],
                          physicsClientId=self.client_id)

        for obj in objects:
            vis = p.createVisualShape(
                p.GEOM_BOX, halfExtents=[obj["w"] / 2, obj["d"] / 2, obj["h"] / 2],
                rgbaColor=[0.2, 0.6, 0.9, 1], physicsClientId=self.client_id
            )
            col = p.createCollisionShape(
                p.GEOM_BOX, halfExtents=[obj["w"] / 2, obj["d"] / 2, obj["h"] / 2],
                physicsClientId=self.client_id
            )
            p.createMultiBody(0.1, col, vis, [obj["x"], obj["y"], obj["z"]],
                              physicsClientId=self.client_id)

        for _ in range(100):
            p.stepSimulation(physicsClientId=self.client_id)

    def render(self):
        w, h = 640, 480
        fx = fy = w * 1.2
        cx, cy = w / 2, h / 2
        near, far = 0.01, 2.0

        view = p.computeViewMatrix(
            [0, 0, 0.8], [0, 0, 0.45], [0, -1, 0],
            physicsClientId=self.client_id
        )
        proj = p.computeProjectionMatrixFOV(
            60, w / h, near, far, physicsClientId=self.client_id
        )

        _, _, rgb_img, depth_img, _ = p.getCameraImage(
            w, h, view, proj, renderer=p.ER_BULLET_HARDWARE_OPENGL,
            physicsClientId=self.client_id
        )

        rgb = np.array(rgb_img, dtype=np.uint8).reshape(h, w, 4)[:, :, :3]
        depth_buffer = np.array(depth_img, dtype=np.float32).reshape(h, w)
        far_val = far
        near_val = near
        depth_m = far_val * near_val / (far_val - (far_val - near_val) * depth_buffer)

        return rgb, depth_m

    def infer(self, depth_m):
        points, valid, u_arr, v_arr = depth_to_points(depth_m)
        h, w = depth_m.shape
        n_all = len(points)
        idxs = np.random.choice(n_all, NUM_POINTS, replace=(n_all < NUM_POINTS))
        pts_sample = points[idxs]
        feat_sample = normalize_points_6ch(pts_sample)

        with torch.no_grad():
            feat_t = torch.from_numpy(feat_sample).unsqueeze(0).float().to(DEVICE)
            logits, _, _ = self.model(feat_t)
            pred_sample = logits.argmax(dim=-1).squeeze(0).cpu().numpy()

        tree = cKDTree(pts_sample[:, :3])
        _, nn_idx = tree.query(points[:, :3], k=1)
        pred_all = pred_sample[nn_idx]

        sem_2d = np.zeros((h, w), dtype=np.uint8)
        sem_2d[valid] = pred_all
        return sem_2d, points, valid

    def cluster_and_hierarchy(self, points, seg_labels):
        fg_mask = seg_labels == 1
        if fg_mask.sum() < 50:
            return [], HierarchyResult(edges=[], layers={}, grasp_order=[], dag_adj={},
                                       stability_scores={}, stacking_groups={}, group_layers={},
                                       group_grasp_orders={})

        config = load_config("configs/default_config.yaml")
        config_dict = OmegaConf.to_container(config, resolve=True)

        fg_indices_all = np.where(fg_mask)[0]
        fg_pts = points[fg_indices_all]
        pcd_fg = o3d.geometry.PointCloud()
        pcd_fg.points = o3d.utility.Vector3dVector(fg_pts.astype(np.float64))
        pcd_fg = pcd_fg.voxel_down_sample(voxel_size=0.003)
        pts_down = np.asarray(pcd_fg.points).astype(np.float64)
        tree = cKDTree(fg_pts[:, :3])
        _, map_idx = tree.query(pts_down[:, :3], k=1)
        fg_indices = fg_indices_all[map_idx]

        clusterer = InstanceClustering(config_dict)
        clusters, _ = clusterer._z_layered_cluster(points, fg_indices)

        instances = []
        for member_arr in clusters:
            if len(member_arr) < clusterer.min_point_count:
                continue
            pts = points[member_arr]
            instances.append(Instance(
                id=len(instances), point_indices=member_arr.astype(np.int64),
                centroid=np.mean(pts, axis=0), bbox_min=np.min(pts, axis=0),
                bbox_max=np.max(pts, axis=0), point_count=len(member_arr),
                z_mean=float(np.mean(pts[:, 2])), z_min=float(np.min(pts[:, 2])),
                z_max=float(np.max(pts[:, 2])),
            ))

        if len(instances) > 2:
            instances = self._merge_nearby(instances, points)

        if len(instances) > 1:
            pts_neg = points.copy()
            pts_neg[:, 2] = -pts_neg[:, 2]
            insts_neg = []
            for inst in instances:
                ni = Instance(
                    id=inst.id, point_indices=inst.point_indices.copy(),
                    centroid=np.array([inst.centroid[0], inst.centroid[1], -inst.centroid[2]]),
                    bbox_min=np.array([inst.bbox_min[0], inst.bbox_min[1], -inst.bbox_max[2]]),
                    bbox_max=np.array([inst.bbox_max[0], inst.bbox_max[1], -inst.bbox_min[2]]),
                    point_count=inst.point_count, z_mean=-inst.z_mean,
                    z_min=-inst.z_max, z_max=-inst.z_min,
                )
                ni._height = inst.z_max - inst.z_min
                insts_neg.append(ni)
            hierarchy = build_hierarchy(insts_neg, pts_neg, config_dict)
        else:
            hierarchy = HierarchyResult(
                edges=[], layers={0: [0]} if instances else {},
                grasp_order=list(range(len(instances))),
                dag_adj={}, stability_scores={}, stacking_groups={},
                group_layers={}, group_grasp_orders={})

        return instances, hierarchy

    def _merge_nearby(self, instances, points, z_thr=0.025, xy_gap=0.015):
        n = len(instances)
        parent = list(range(n))
        def find(x):
            while parent[x] != x: parent[x] = parent[parent[x]]; x = parent[x]
            return x
        def union(a, b):
            ra, rb = find(a), find(b)
            if ra != rb: parent[ra] = rb

        for i in range(n):
            for j in range(i + 1, n):
                a, b = instances[i], instances[j]
                if abs(a.z_mean - b.z_mean) > z_thr: continue
                x_gap = max(0, max(a.bbox_min[0], b.bbox_min[0]) - min(a.bbox_max[0], b.bbox_max[0]))
                y_gap = max(0, max(a.bbox_min[1], b.bbox_min[1]) - min(a.bbox_max[1], b.bbox_max[1]))
                if x_gap < xy_gap and y_gap < xy_gap: union(i, j)

        groups = defaultdict(list)
        for i in range(n): groups[find(i)].append(i)

        new_insts = []
        for members in groups.values():
            if len(members) == 1:
                inst = instances[members[0]]; inst.id = len(new_insts); new_insts.append(inst)
            else:
                all_idx = np.unique(np.concatenate([instances[m].point_indices for m in members]))
                pts = points[all_idx]
                new_insts.append(Instance(id=len(new_insts), point_indices=all_idx.astype(np.int64),
                                          centroid=np.mean(pts, axis=0), bbox_min=np.min(pts, axis=0),
                                          bbox_max=np.max(pts, axis=0), point_count=len(all_idx),
                                          z_mean=float(np.mean(pts[:, 2])), z_min=float(np.min(pts[:, 2])),
                                          z_max=float(np.max(pts[:, 2]))))
        return new_insts

    def run(self, num_scenes=30):
        self.connect()
        self.load_model()

        logger.info(f"Auto realtime test: {num_scenes} scenes")
        results_log = []

        for scene_idx in range(num_scenes):
            objects, is_stacking = random_scene(self.rng)
            self.setup_scene(objects)

            t0 = time.perf_counter()
            rgb, depth_m = self.render()
            sem_pred, points, valid = self.infer(depth_m)
            instances, hierarchy = self.cluster_and_hierarchy(points, sem_pred[valid])
            t_total = (time.perf_counter() - t0) * 1000

            n_inst = len(instances)
            n_edges = len(hierarchy.edges) if hierarchy else 0
            n_direct = sum(1 for e in hierarchy.edges if not e.indirect) if hierarchy else 0
            go = hierarchy.grasp_order if hierarchy else []
            n_layers = len(hierarchy.layers) if hierarchy else 0

            stacking_detected = n_direct > 0
            correct = stacking_detected == is_stacking

            results_log.append({
                "scene": scene_idx + 1,
                "num_objects": len(objects),
                "gt_stacking": is_stacking,
                "pred_stacking": stacking_detected,
                "correct": correct,
                "num_instances": n_inst,
                "num_edges": n_edges,
                "num_direct_edges": n_direct,
                "num_layers": n_layers,
                "grasp_order": go,
                "latency_ms": round(t_total, 1),
            })

            logger.info(f"Scene {scene_idx + 1:2d}: {len(objects)} objs, "
                        f"GT={'S' if is_stacking else 'N'} Pred={'S' if stacking_detected else 'N'} "
                        f"{'✓' if correct else '✗'} | {n_inst} insts, {n_direct} edges, "
                        f"{n_layers} layers | {t_total:.0f}ms")

        p.disconnect(physicsClientId=self.client_id)

        lats = [r["latency_ms"] for r in results_log]
        correct_count = sum(1 for r in results_log if r["correct"])
        stacking_scenes = [r for r in results_log if r["gt_stacking"]]
        non_stacking_scenes = [r for r in results_log if not r["gt_stacking"]]
        stacking_correct = sum(1 for r in stacking_scenes if r["correct"])
        non_stacking_correct = sum(1 for r in non_stacking_scenes if r["correct"])

        summary = {
            "total_scenes": len(results_log),
            "accuracy": round(correct_count / len(results_log), 4),
            "stacking_accuracy": round(stacking_correct / max(len(stacking_scenes), 1), 4),
            "non_stacking_accuracy": round(non_stacking_correct / max(len(non_stacking_scenes), 1), 4),
            "avg_latency_ms": round(np.mean(lats), 1),
            "min_latency_ms": round(np.min(lats), 1),
            "max_latency_ms": round(np.max(lats), 1),
            "std_latency_ms": round(np.std(lats), 1),
            "avg_instances": round(np.mean([r["num_instances"] for r in results_log]), 2),
            "avg_edges": round(np.mean([r["num_edges"] for r in results_log]), 2),
            "results": results_log,
        }

        with open(os.path.join(OUT_DIR, "auto_realtime_summary.json"), "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        logger.info(f"\n{'='*60}")
        logger.info(f"SUMMARY: {num_scenes} scenes")
        logger.info(f"  Accuracy: {summary['accuracy']:.2%} ({correct_count}/{len(results_log)})")
        logger.info(f"  Stacking acc: {summary['stacking_accuracy']:.2%} ({stacking_correct}/{len(stacking_scenes)})")
        logger.info(f"  Non-stacking acc: {summary['non_stacking_accuracy']:.2%} ({non_stacking_correct}/{len(non_stacking_scenes)})")
        logger.info(f"  Avg latency: {summary['avg_latency_ms']}ms (min={summary['min_latency_ms']}, max={summary['max_latency_ms']})")
        logger.info(f"  Results saved to: {OUT_DIR}")


if __name__ == "__main__":
    tester = AutoRealtimeTester()
    tester.run(num_scenes=30)
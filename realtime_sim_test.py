"""
实时仿真推理: PyBullet 随机场景 → PointNet++Attention → 聚类 → 层级 → 可视化
=========================================================================
输出: camera_operate/sim_test_results/
"""

import os, sys, time, json
import numpy as np
import cv2
import open3d as o3d
import torch
from collections import defaultdict

# suppress debug logs before module imports
from loguru import logger
logger.remove()
logger.add(sys.stderr, level="INFO", format="{time:HH:mm:ss} | {level:<7} | {message}")

from models.pointnet_seg import PointNetPlusPlusAttentionSeg

# PyBullet
try:
    import pybullet as p
    import pybullet_data
except ImportError:
    logger.error("pybullet not installed"); sys.exit(1)

from scipy.spatial import cKDTree
from scipy.ndimage import gaussian_filter1d

# ── 参数 ──
FX, FY = 506.35, 506.27
CX, CY = 334.19, 335.43
IMG_W, IMG_H = 640, 576
NEAR, FAR = 0.05, 2.0
CAM_HEIGHT = 0.50
CAM_TARGET = [0, 0, 0.04]
FOV_DEG = 2 * np.arctan(IMG_H / (2 * FY)) * 180 / np.pi

NUM_POINTS = 2048
NUM_CLASSES = 2
INPUT_CHANNELS = 6
DEVICE = torch.device("cuda")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "training_data", "train_logs", "best_model.pth")
OUT_DIR = os.path.join(BASE_DIR, "sim_test_results")
os.makedirs(OUT_DIR, exist_ok=True)

import logging
logging.getLogger().setLevel(logging.WARNING)

from modules.postprocess import Instance, InstanceSegmentationResult
from modules.hierarchy import build_hierarchy, HierarchyResult

COLORS = [
    (46, 204, 113), (52, 152, 219), (231, 76, 60), (241, 196, 15),
    (155, 89, 182), (26, 188, 156), (230, 126, 34),
]


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


def random_cuboid(rng):
    half = [rng.uniform(0.03, 0.07) for _ in range(3)]
    color = [rng.uniform(0.1, 0.9) for _ in range(3)]
    return half, color


def random_scene(rng):
    """随机场景: 70% 2物体, 20% 3物体, 10% 4物体"""
    n = rng.choice([2, 2, 2, 2, 2, 2, 2, 3, 3, 4])  # 7:2:1

    cuboids = [random_cuboid(rng) for _ in range(n)]
    is_stack = rng.random() < (0.75 if n <= 2 else 0.85)

    objects = []
    if not is_stack:
        # all on table, spread out
        positions = []
        for i, (half, color) in enumerate(cuboids):
            for _ in range(20):
                x = rng.uniform(-0.15, 0.15)
                y = rng.uniform(-0.12, 0.12)
                if not positions or all(np.sqrt((x - px)**2 + (y - py)**2) > 0.06
                                         for px, py in positions):
                    positions.append((x, y))
                    break
            rz = rng.uniform(-45, 45)
            objects.append({"half": half, "color": color,
                            "pos": (x, y, half[2], rz)})
        return objects, False

    # stacking: build a chain/tree
    # first object on table
    h0, c0 = cuboids[0]
    base_x, base_y = rng.uniform(-0.03, 0.03), rng.uniform(-0.03, 0.03)
    objects.append({"half": h0, "color": c0,
                    "pos": (base_x, base_y, h0[2], rng.uniform(-30, 30))})

    z_stack = 2 * h0[2]  # top of first object

    for half, color in cuboids[1:]:
        # overlap: 30-100% with the previous object
        overlap = rng.uniform(0.3, 1.0)
        off_x = (objects[-1]["half"][0] + half[0]) * (1 - overlap) * rng.choice([-1, 1])
        off_y = rng.uniform(-0.03, 0.03)
        px = objects[-1]["pos"][0] + off_x
        py = objects[-1]["pos"][1] + off_y
        rz = rng.uniform(-30, 30)
        objects.append({"half": half, "color": color,
                        "pos": (px, py, z_stack + half[2], rz)})
        z_stack += 2 * half[2]

    return objects, True


class RealtimeSimTester:
    def __init__(self):
        self.client_id = None
        self.model = None
        self.rng = np.random.RandomState()

    def connect(self):
        self.client_id = p.connect(p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=self.client_id)

    def load_model(self):
        self.model = PointNetPlusPlusAttentionSeg(input_channels=INPUT_CHANNELS, num_classes=NUM_CLASSES)
        ckpt = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.to(DEVICE)
        self.model.eval()
        logger.info(f"Model loaded: epoch={ckpt['epoch']}, mIoU={ckpt['miou']:.4f}")

    def setup_scene(self, objects):
        p.resetSimulation(physicsClientId=self.client_id)
        p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=self.client_id)
        p.setGravity(0, 0, -9.81, physicsClientId=self.client_id)
        col = p.createCollisionShape(p.GEOM_PLANE, physicsClientId=self.client_id)
        vis = p.createVisualShape(p.GEOM_PLANE, rgbaColor=[0.35, 0.35, 0.40, 1.0], physicsClientId=self.client_id)
        p.createMultiBody(0, col, vis, basePosition=[0, 0, 0], physicsClientId=self.client_id)
        for obj in objects:
            h, c, (x, y, z, rz) = obj["half"], obj["color"], obj["pos"]
            col = p.createCollisionShape(p.GEOM_BOX, halfExtents=h, physicsClientId=self.client_id)
            vis = p.createVisualShape(p.GEOM_BOX, halfExtents=h, rgbaColor=[*c, 1.0], physicsClientId=self.client_id)
            orn = p.getQuaternionFromEuler([0, 0, np.radians(rz)], physicsClientId=self.client_id)
            p.createMultiBody(0.05, col, vis, basePosition=[x, y, z], baseOrientation=orn, physicsClientId=self.client_id)
        for _ in range(50):
            p.stepSimulation(physicsClientId=self.client_id)

    def render(self):
        view = p.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=CAM_TARGET, distance=CAM_HEIGHT, yaw=0, pitch=-90, roll=0,
            upAxisIndex=2, physicsClientId=self.client_id)
        proj = p.computeProjectionMatrixFOV(fov=FOV_DEG, aspect=IMG_W / IMG_H, nearVal=NEAR, farVal=FAR,
                                             physicsClientId=self.client_id)
        _, _, rgb_raw, depth_raw, _ = p.getCameraImage(
            width=IMG_W, height=IMG_H, viewMatrix=view, projectionMatrix=proj,
            renderer=p.ER_TINY_RENDERER, physicsClientId=self.client_id)
        rgb = rgb_raw[:, :, :3].astype(np.uint8).copy()
        depth_norm = depth_raw.astype(np.float32)
        depth_m = FAR * NEAR / (FAR - (FAR - NEAR) * depth_norm)
        depth_m[depth_m > FAR * 0.99] = 0.0
        return rgb, depth_m

    def infer(self, depth_m):
        """模型推理 → 语义标签"""
        h, w = depth_m.shape
        valid = depth_m > NEAR
        uu, vv = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
        Z = depth_m[valid]
        X = (uu[valid] - CX) * Z / FX
        Y = (vv[valid] - CY) * Z / FY
        points = np.stack([X, Y, Z], axis=-1).astype(np.float64)
        n_all = len(points)

        if n_all == 0:
            return np.zeros((h, w), dtype=np.uint8), points

        idxs = np.random.choice(n_all, NUM_POINTS, replace=(n_all < NUM_POINTS))
        pts_sample = points[idxs]
        feat = normalize_points_6ch(pts_sample)

        with torch.no_grad():
            feat_t = torch.from_numpy(feat).unsqueeze(0).float().to(DEVICE)
            logits, _, _ = self.model(feat_t)
            pred_sample = logits.argmax(dim=-1).squeeze(0).cpu().numpy()

        tree = cKDTree(pts_sample[:, :3])
        _, nn_idx = tree.query(points[:, :3], k=1)
        pred_all = pred_sample[nn_idx]

        # to 2D
        sem_2d = np.zeros((h, w), dtype=np.uint8)
        sem_2d[valid] = pred_all
        return sem_2d, points, valid

    def cluster_and_hierarchy(self, points, seg_labels):
        """实例聚类 + 层级推理 (降采样加速)"""
        fg_mask = seg_labels == 1
        if fg_mask.sum() < 50:
            return [], HierarchyResult(edges=[], layers={}, grasp_order=[], dag_adj={},
                                       stability_scores={}, stacking_groups={}, group_layers={},
                                       group_grasp_orders={})

        from modules.postprocess import InstanceClustering
        from omegaconf import OmegaConf
        import open3d as o3d

        config = OmegaConf.load("configs/default_config.yaml")
        config_dict = OmegaConf.to_container(config, resolve=True)

        # voxel downsampling for speed
        fg_indices_all = np.where(fg_mask)[0]
        fg_pts = points[fg_indices_all]
        pcd_fg = o3d.geometry.PointCloud()
        pcd_fg.points = o3d.utility.Vector3dVector(fg_pts.astype(np.float64))
        pcd_fg = pcd_fg.voxel_down_sample(voxel_size=0.003)
        pts_down = np.asarray(pcd_fg.points).astype(np.float64)
        n_down = len(pts_down)
        # map downsampled points back to original indices
        tree = cKDTree(fg_pts[:, :3])
        _, map_idx = tree.query(pts_down[:, :3], k=1)
        fg_indices = fg_indices_all[map_idx]

        clusterer = InstanceClustering(config_dict)
        clusters, _ = clusterer._z_layered_cluster(points, fg_indices)

        instances = []
        for local_idx, member_arr in enumerate(clusters):
            if len(member_arr) < clusterer.min_point_count:
                continue
            global_idx = member_arr
            pts = points[global_idx]
            instances.append(Instance(
                id=len(instances), point_indices=global_idx.astype(np.int64),
                centroid=np.mean(pts, axis=0), bbox_min=np.min(pts, axis=0),
                bbox_max=np.max(pts, axis=0), point_count=len(global_idx),
                z_mean=float(np.mean(pts[:, 2])), z_min=float(np.min(pts[:, 2])),
                z_max=float(np.max(pts[:, 2])),
            ))

        # refine: merge nearby
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

    def visualize(self, rgb, depth_m, sem_pred, instances, hierarchy, t_ms):
        h, w = depth_m.shape
        if rgb.shape[:2] != (h, w): rgb = cv2.resize(rgb, (w, h))

        valid = depth_m > NEAR
        dmin, dmax = depth_m[valid].min(), depth_m[valid].max()
        dclip = np.clip(depth_m, dmin, dmax)
        depth_vis = cv2.applyColorMap(((dclip - dmin) / max(dmax - dmin, 0.001) * 255).astype(np.uint8), cv2.COLORMAP_JET)

        # sem pred viz
        sem_vis = np.zeros((h, w, 3), dtype=np.uint8)
        sem_vis[sem_pred == 0] = [60, 60, 60]
        sem_vis[sem_pred == 1] = [0, 255, 0]

        # --- simplified 2-panel: RGB | Depth with stacking badge ---
        row1 = np.hstack([rgb, depth_vis])

        # instance bboxes on both panels
        for idx, inst in enumerate(instances):
            color = COLORS[idx % len(COLORS)]
            pts = inst.point_indices
            valid_idx = np.where(valid.ravel())[0]
            rev_map = {vi: i for i, vi in enumerate(valid_idx)}
            img_pts = []
            for pi in pts[:5000]:
                if pi in rev_map:
                    r, c = divmod(rev_map[pi], w)
                    img_pts.append((c, r))
            if img_pts:
                img_pts = np.array(img_pts)
                x1, y1 = img_pts[:, 0].min(), img_pts[:, 1].min()
                x2, y2 = img_pts[:, 0].max(), img_pts[:, 1].max()
                # bbox on RGB
                cv2.rectangle(row1, (x1, y1), (x2, y2), color, 2)
                cv2.putText(row1, f"Obj{inst.id}", (x1, max(0, y1 - 5)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
                # bbox on Depth (offset by w)
                cv2.rectangle(row1, (x1 + w, y1), (x2 + w, y2), color, 2)
                cv2.putText(row1, f"Obj{inst.id}", (x1 + w, max(0, y1 - 5)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

        # stacking badge on BOTH panels
        n_direct = sum(1 for e in hierarchy.edges if not e.indirect) if hierarchy else 0
        badge_text = "STACKING" if n_direct > 0 else "NO STACKING"
        badge_color = (0, 100, 255) if n_direct > 0 else (0, 200, 100)
        badge_bg = (0, 60, 180) if n_direct > 0 else (0, 120, 60)
        font = cv2.FONT_HERSHEY_SIMPLEX
        (tw, th), _ = cv2.getTextSize(badge_text, font, 0.8, 2)
        # RGB panel (top-right of left half)
        bx_rgb, by_rgb = w - tw - 15, 40
        cv2.rectangle(row1, (bx_rgb - 8, by_rgb - th - 8), (bx_rgb + tw + 8, by_rgb + 8), badge_bg, -1)
        cv2.rectangle(row1, (bx_rgb - 8, by_rgb - th - 8), (bx_rgb + tw + 8, by_rgb + 8), badge_color, 2)
        cv2.putText(row1, badge_text, (bx_rgb, by_rgb), font, 0.8, (255, 255, 255), 2)
        # Depth panel (top-right of right half)
        bx_dep, by_dep = w * 2 - tw - 15, 40
        cv2.rectangle(row1, (bx_dep - 8, by_dep - th - 8), (bx_dep + tw + 8, by_dep + 8), badge_bg, -1)
        cv2.rectangle(row1, (bx_dep - 8, by_dep - th - 8), (bx_dep + tw + 8, by_dep + 8), badge_color, 2)
        cv2.putText(row1, badge_text, (bx_dep, by_dep), font, 0.8, (255, 255, 255), 2)

        # --- info panel below ---
        info_h = 160
        full_canvas = np.zeros((h + info_h, w * 2, 3), dtype=np.uint8)
        full_canvas[:h, :] = row1
        full_canvas[h:, :] = (25, 25, 30)

        n_inst = len(instances)
        n_edges = len(hierarchy.edges) if hierarchy else 0
        go = hierarchy.grasp_order if hierarchy else []
        n_layers = len(hierarchy.layers) if hierarchy else 0

        y = h + 18
        cv2.putText(full_canvas, f"Objects: {n_inst}  |  Layers: {n_layers}  |  "
                    f"Edges: {n_edges} ({n_direct} direct)  |  "
                    f"Grasp: {go}  |  {t_ms:.0f}ms",
                    (8, y), font, 0.5, (255, 255, 100), 1)
        y += 28

        if n_direct > 0 and hierarchy:
            children = defaultdict(list)
            for e in hierarchy.edges:
                if not e.indirect: children[e.upper_id].append(e)
            all_lower = {e.lower_id for e in hierarchy.edges if not e.indirect}
            roots = [inst.id for inst in instances if inst.id not in all_lower]
            visited = set()
            def draw_tree(nid, indent, yy):
                if nid in visited: return yy
                visited.add(nid)
                c = COLORS[nid % len(COLORS)]
                inst = instances[nid]
                ht = inst.z_max - inst.z_min
                pfx = "  " * indent + ("|-- " if indent else "")
                cv2.putText(full_canvas, f"{pfx}Obj{nid} h={ht:.3f}m", (8, yy),
                            font, 0.45, c, 1); yy += 17
                for e in children.get(nid, []):
                    tag = "CONTACT" if e.contact else "NEAR"
                    cv2.putText(full_canvas, f"{'  '*(indent+1)}[{tag}]", (8, yy),
                                font, 0.35, (140, 140, 140), 1); yy += 14
                    yy = draw_tree(e.lower_id, indent + 1, yy)
                return yy
            for rid in sorted(roots): y = draw_tree(rid, 0, y)

        # panel labels
        cv2.putText(full_canvas, "RGB", (w // 2 - 15, h - 8), font, 0.5, (0, 255, 0), 2)
        cv2.putText(full_canvas, "Depth", (w + w // 2 - 25, h - 8), font, 0.5, (0, 255, 0), 2)

        return full_canvas

    def run(self, max_frames=None):
        self.connect()
        self.load_model()

        logger.info("Real-time Sim Test")
        logger.info("  'n' = next scene  |  's' = save  |  'q' = quit")
        logger.info("  Scene stays until you press 'n'")
        logger.info("=" * 60)

        cv2.namedWindow("Real-time Sim Inference", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Real-time Sim Inference", 1280, 720)

        frame = 0
        results_log = []
        objects, is_stacking = random_scene(self.rng)
        self.setup_scene(objects)
        need_new_scene = True

        try:
            while True:
                if need_new_scene:
                    objects, is_stacking = random_scene(self.rng)
                    self.setup_scene(objects)
                    need_new_scene = False

                t0 = time.perf_counter()
                rgb, depth_m = self.render()
                sem_pred, points, valid = self.infer(depth_m)
                instances, hierarchy = self.cluster_and_hierarchy(points, sem_pred[valid])
                t_total = (time.perf_counter() - t0) * 1000

                frame += 1
                # Only log first render of each scene
                if frame == 1 or need_new_scene:
                    n_inst = len(instances)
                    n_edges = len(hierarchy.edges) if hierarchy else 0
                    go = hierarchy.grasp_order if hierarchy else []
                    results_log.append({
                        "frame": frame, "stacking": is_stacking,
                        "num_instances": n_inst, "num_edges": n_edges,
                        "grasp_order": go, "latency_ms": round(t_total, 1),
                    })

                vis = self.visualize(rgb, depth_m, sem_pred, instances, hierarchy, t_total)
                cv2.imshow("Real-time Sim Inference", vis)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    cv2.imwrite(os.path.join(OUT_DIR, f"frame_{frame:04d}.png"), vis)
                    logger.info(f"Frame {frame} saved to {OUT_DIR}")
                elif key == ord('n'):
                    need_new_scene = True

        except KeyboardInterrupt:
            print()
            logger.info("Interrupted")
        finally:
            cv2.destroyAllWindows()
            p.disconnect(physicsClientId=self.client_id)

            # save summary
            if results_log:
                lats = [r["latency_ms"] for r in results_log]
                insts = [r["num_instances"] for r in results_log]
                summary = {
                    "total_frames": len(results_log),
                    "avg_latency_ms": round(np.mean(lats), 1),
                    "avg_instances": round(np.mean(insts), 2),
                    "results": results_log,
                }
                with open(os.path.join(OUT_DIR, "realtime_summary.json"), "w", encoding="utf-8") as f:
                    json.dump(summary, f, indent=2)
                logger.info(f"Summary saved. Avg latency: {summary['avg_latency_ms']}ms")
                logger.info(f"Results in: {OUT_DIR}")


if __name__ == "__main__":
    tester = RealtimeSimTester()
    tester.run()

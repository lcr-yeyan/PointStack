"""
两长方体堆叠场景数据生成
========================
使用 PyBullet 仿真，Orbbec Femto Bolt 相机参数（垂直向下），
生成 8 种堆叠配置的 RGB / 深度图 / 点云 / 标注。

输出: camera_operate/data_preview/
"""

import os
import sys
import json
import time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import open3d as o3d

try:
    import pybullet as p
    import pybullet_data
    _PB_AVAILABLE = True
except ImportError:
    _PB_AVAILABLE = False
    print("pybullet not installed")

from loguru import logger

# ── Orbbec Femto Bolt 相机参数 ──────────────────────────────────────
FX, FY = 506.35, 506.27
CX, CY = 334.19, 335.43
IMG_W, IMG_H = 640, 576
NEAR, FAR = 0.05, 2.0

# 俯视相机高度 (高于桌面)
CAM_HEIGHT = 0.50
CAM_TARGET = [0, 0, 0.04]

# 长方体定义: (name, half_extents [x,y,z], color_rgb)
# 12×8×6 cm  红棕色
CUBOID_A = ("cuboid_A", [0.06, 0.04, 0.03], [0.85, 0.30, 0.15])
# 8×6×5 cm  蓝色
CUBOID_B = ("cuboid_B", [0.04, 0.03, 0.025], [0.15, 0.35, 0.85])

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data_preview")


# ── 8 种堆叠配置 ───────────────────────────────────────────────────
# 每个配置: (scene_id, 描述, [物体位置列表], has_stacking, 层级)
# 物体位置: (center_x, center_y, center_z, euler_z_deg)
# z 是物体中心到桌面的高度 = half_extents[2]

SCENES = []

hA, hB = CUBOID_A[1][2], CUBOID_B[1][2]  # 0.03, 0.025
sxA, syA = CUBOID_A[1][0], CUBOID_A[1][1]  # 0.06, 0.04
sxB, syB = CUBOID_B[1][0], CUBOID_B[1][1]  # 0.04, 0.03
z_bottom_A = hA      # A 底部物体中心Z
z_top_B = 2 * hA + hB  # B 堆叠在 A 上方的中心Z

# 1. 并排放置 (无堆叠)
# A中心(-sxA-0.02, 0, hA), B中心(sxB+0.02, 0, hB)
gap = 0.03
SCENES.append({
    "id": "side_by_side",
    "description_en": "two cuboids placed separately on table",
    "description_cn": "两长方体独立放置，无接触",
    "objects": [
        {"cuboid": CUBOID_A, "pos": (-(sxA + gap), 0.00, hA, 0)},
        {"cuboid": CUBOID_B, "pos": (sxB + gap, 0.00, hB, 0)},
    ],
    "has_stacking": False,
    "layers": {0: [0, 1]},
    "grasp_order": [0, 1],
})

# 2. 部分重叠 ~25% — B 偏移到 A 的 X 边缘，仅覆盖 25%
offset_25 = (sxA + sxB) * 0.75
SCENES.append({
    "id": "partial_overlap_25",
    "description_en": "top cuboid covers ~25% of bottom",
    "description_cn": "上方物体仅覆盖下方物体~25%",
    "objects": [
        {"cuboid": CUBOID_A, "pos": (0.00, 0.00, z_bottom_A, 0)},
        {"cuboid": CUBOID_B, "pos": (offset_25, 0.00, z_top_B, 0)},
    ],
    "has_stacking": True,
    "layers": {0: [0], 1: [1]},
    "grasp_order": [1, 0],
})

# 3. 部分重叠 ~50%
offset_50 = (sxA + sxB) * 0.50
SCENES.append({
    "id": "partial_overlap_50",
    "description_en": "top cuboid covers ~50% of bottom",
    "description_cn": "上方物体覆盖下方物体~50%",
    "objects": [
        {"cuboid": CUBOID_A, "pos": (0.00, 0.00, z_bottom_A, 0)},
        {"cuboid": CUBOID_B, "pos": (offset_50, 0.00, z_top_B, 0)},
    ],
    "has_stacking": True,
    "layers": {0: [0], 1: [1]},
    "grasp_order": [1, 0],
})

# 4. 部分重叠 ~75%
offset_75 = (sxA + sxB) * 0.25
SCENES.append({
    "id": "partial_overlap_75",
    "description_en": "top cuboid covers ~75% of bottom",
    "description_cn": "上方物体覆盖下方物体~75%",
    "objects": [
        {"cuboid": CUBOID_A, "pos": (0.00, 0.00, z_bottom_A, 0)},
        {"cuboid": CUBOID_B, "pos": (offset_75, 0.00, z_top_B, 0)},
    ],
    "has_stacking": True,
    "layers": {0: [0], 1: [1]},
    "grasp_order": [1, 0],
})

# 5. 完全居中堆叠
SCENES.append({
    "id": "full_stack_centered",
    "description_en": "top cuboid fully on top, centered",
    "description_cn": "上方物体完全在下方物体正上方居中",
    "objects": [
        {"cuboid": CUBOID_A, "pos": (0.00, 0.00, z_bottom_A, 0)},
        {"cuboid": CUBOID_B, "pos": (0.00, 0.00, z_top_B, 0)},
    ],
    "has_stacking": True,
    "layers": {0: [0], 1: [1]},
    "grasp_order": [1, 0],
})

# 6. 堆叠但偏移到 A 的边缘
edge_offset = sxA - sxB  # B 左边缘恰好贴着 A 左边缘，B 往右几乎悬空
SCENES.append({
    "id": "full_stack_offset",
    "description_en": "top cuboid on top but near edge of bottom",
    "description_cn": "上方物体堆叠但偏移到下方物体边缘",
    "objects": [
        {"cuboid": CUBOID_A, "pos": (0.00, 0.00, z_bottom_A, 0)},
        {"cuboid": CUBOID_B, "pos": (edge_offset, 0.00, z_top_B, 0)},
    ],
    "has_stacking": True,
    "layers": {0: [0], 1: [1]},
    "grasp_order": [1, 0],
})

# 7. 十字交叉堆叠 — B 绕 Z 轴旋转 90°
SCENES.append({
    "id": "cross_stack",
    "description_en": "top cuboid rotated 90 degrees across bottom",
    "description_cn": "上方物体旋转90°横跨下方物体",
    "objects": [
        {"cuboid": CUBOID_A, "pos": (0.00, 0.00, z_bottom_A, 0)},
        {"cuboid": CUBOID_B, "pos": (0.00, 0.00, z_top_B, 90)},
    ],
    "has_stacking": True,
    "layers": {0: [0], 1: [1]},
    "grasp_order": [1, 0],
})

# 8. 尺寸不同但对齐堆叠 (两者都带旋转)
SCENES.append({
    "id": "aligned_stack",
    "description_en": "different-size cuboids, both slightly rotated",
    "description_cn": "两物体尺寸不同，各略带旋转，中心对齐",
    "objects": [
        {"cuboid": CUBOID_A, "pos": (0.00, 0.00, z_bottom_A, 20)},
        {"cuboid": CUBOID_B, "pos": (0.00, 0.00, z_top_B, -15)},
    ],
    "has_stacking": True,
    "layers": {0: [0], 1: [1]},
    "grasp_order": [1, 0],
})


class StackingDataGenerator:
    def __init__(self):
        self.client_id = None
        self.body_ids = []
        self.scene_objects = []
        self.fov_deg = 2 * np.arctan(IMG_H / (2 * FY)) * 180 / np.pi  # ~59.3°

    def connect(self):
        self.client_id = p.connect(p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath(),
                                  physicsClientId=self.client_id)

    def disconnect(self):
        if self.client_id is not None:
            p.disconnect(physicsClientId=self.client_id)

    def _create_ground(self):
        col = p.createCollisionShape(p.GEOM_PLANE, physicsClientId=self.client_id)
        vis = p.createVisualShape(p.GEOM_PLANE,
                                  rgbaColor=[0.35, 0.35, 0.40, 1.0],
                                  physicsClientId=self.client_id)
        p.createMultiBody(0, col, vis, basePosition=[0, 0, 0],
                          physicsClientId=self.client_id)

    def _create_cuboid(self, name, half_extents, color_rgb, pos, euler_z_deg):
        """创建长方体"""
        col = p.createCollisionShape(p.GEOM_BOX, halfExtents=half_extents,
                                     physicsClientId=self.client_id)
        vis = p.createVisualShape(p.GEOM_BOX, halfExtents=half_extents,
                                  rgbaColor=[*color_rgb, 1.0],
                                  physicsClientId=self.client_id)
        orn = p.getQuaternionFromEuler([0, 0, np.radians(euler_z_deg)],
                                       physicsClientId=self.client_id)
        body_id = p.createMultiBody(
            baseMass=0.05,
            baseCollisionShapeIndex=col,
            baseVisualShapeIndex=vis,
            basePosition=list(pos),
            baseOrientation=orn,
            physicsClientId=self.client_id,
        )
        p.changeDynamics(body_id, -1, lateralFriction=0.8,
                         spinningFriction=0.002, restitution=0.02,
                         physicsClientId=self.client_id)
        return body_id

    def setup_scene(self, scene_config):
        """搭建场景"""
        p.resetSimulation(physicsClientId=self.client_id)
        p.setAdditionalSearchPath(pybullet_data.getDataPath(),
                                  physicsClientId=self.client_id)
        p.setGravity(0, 0, -9.81, physicsClientId=self.client_id)
        self._create_ground()
        self.body_ids = []
        self.scene_objects = []

        for i, obj_spec in enumerate(scene_config["objects"]):
            name, he, color = obj_spec["cuboid"]
            x, y, z, rz = obj_spec["pos"]
            bid = self._create_cuboid(name, he, color, (x, y, z), rz)
            self.body_ids.append(bid)
            self.scene_objects.append({
                "id": i,
                "name": name,
                "half_extents": he,
                "color": color,
                "position": (x, y, z),
                "euler_z_deg": rz,
                "z_min": z - he[2],
                "z_max": z + he[2],
            })

        # 稳定物理
        for _ in range(100):
            p.stepSimulation(physicsClientId=self.client_id)

    def render_depth_rgb_seg(self):
        """渲染 RGB + 深度 + 物体分割掩码 (俯视)"""
        view_matrix = p.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=CAM_TARGET,
            distance=CAM_HEIGHT,
            yaw=0, pitch=-90, roll=0,
            upAxisIndex=2,
            physicsClientId=self.client_id,
        )

        aspect = IMG_W / IMG_H
        proj_matrix = p.computeProjectionMatrixFOV(
            fov=self.fov_deg,
            aspect=aspect,
            nearVal=NEAR,
            farVal=FAR,
            physicsClientId=self.client_id,
        )

        _, _, rgb_raw, depth_raw, seg_raw = p.getCameraImage(
            width=IMG_W, height=IMG_H,
            viewMatrix=view_matrix,
            projectionMatrix=proj_matrix,
            renderer=p.ER_TINY_RENDERER,
            physicsClientId=self.client_id,
        )

        # RGB: (H, W, 4) RGBA → (H, W, 3) RGB
        rgb = rgb_raw[:, :, :3].astype(np.uint8).copy()

        # Depth: normalized [0,1] → meters
        depth_norm = depth_raw.astype(np.float32)
        depth_m = FAR * NEAR / (FAR - (FAR - NEAR) * depth_norm)
        depth_m[depth_m > FAR * 0.99] = 0.0

        # Segmentation: unique object ID per pixel (ground=0 or obj body id)
        seg = seg_raw.astype(np.int32).copy()

        return rgb, depth_m, seg

    def _build_semantic_labels(self, seg):
        """从 PyBullet 分割掩码 → 逐点语义标签
        0 = 桌面 (seg==0 或 seg==ground_id)
        1 = 物体 (前景区, 任何非地面 body id)
        """
        labels = np.zeros((IMG_H, IMG_W), dtype=np.uint8)
        # body_ids 是 PyBullet 分配的物体 ID, 不是 0 的就是物体
        labels[seg > 0] = 1
        return labels

    @staticmethod
    def _add_realistic_noise(depth_clean, seg=None):
        """仿真深度 → 加入真实 ToF 噪声

        返回 (depth_noisy, noise_info_dict)
        模拟:
        1. 高斯深度抖动 (σ=3mm)
        2. 中心区域随机缺失 (模拟 ToF 饱和度)
        3. 边缘飞点 / 边界模糊
        4. 随机散粒无效点
        """
        rng = np.random.RandomState()
        h, w = depth_clean.shape
        depth = depth_clean.copy()
        valid = depth > NEAR
        noise_info = {}

        # 1. 高斯抖动 σ≈3mm (ToF 典型值)
        gauss_noise = rng.normal(0, 0.003, (h, w)).astype(np.float32)
        depth[valid] += gauss_noise[valid]
        noise_info["gaussian_sigma_mm"] = 3.0

        # 2. 中心区域缺失 (模拟 ToF 传感器中心饱和)
        # 中心更近 → 更多缺失
        cy, cx = h / 2, w / 2
        uu = (np.arange(w, dtype=np.float32) - cx) / cx
        vv = (np.arange(h, dtype=np.float32) - cy) / cy
        uv_dist = np.sqrt(uu[None, :] ** 2 + vv[:, None] ** 2)
        # 中心权重高, 缺失概率 = 中心距离 × 基础概率
        center_drop_prob = np.exp(-uv_dist * 2.0) * 0.12
        drop_mask = rng.random((h, w)) < center_drop_prob
        depth[drop_mask & valid] = 0.0
        noise_info["center_drop_max_prob"] = 0.12
        n_dropped = (drop_mask & valid).sum()
        noise_info["center_dropped_pixels"] = int(n_dropped)

        # 3. 边界腐蚀 — 物体边缘随机向内丢失1-2像素
        if seg is not None:
            from scipy import ndimage
            obj_mask = seg > 0
            # 膨胀再腐蚀：膨胀的边界就是物体边缘
            dilated = ndimage.binary_dilation(obj_mask, structure=np.ones((3, 3)), iterations=2)
            eroded = ndimage.binary_erosion(obj_mask, structure=np.ones((3, 3)), iterations=2)
            boundary = dilated & (~eroded)
            # 边缘 30% 设为无效
            edge_drop = boundary & (rng.random((h, w)) < 0.30)
            depth[edge_drop & valid] = 0.0
            noise_info["edge_dropped_pixels"] = int((edge_drop & valid).sum())

        # 4. 随机散粒无效点
        speckle_prob = 0.005
        speckle = rng.random((h, w)) < speckle_prob
        depth[speckle & (depth > NEAR)] = 0.0
        noise_info["speckle_prob"] = speckle_prob

        # 统计
        valid_after = (depth > NEAR).sum()
        valid_before = valid.sum()
        noise_info["valid_before"] = int(valid_before)
        noise_info["valid_after"] = int(valid_after)
        noise_info["missing_pct"] = round((1 - valid_after / valid_before) * 100, 1)

        return depth, noise_info

    def depth_to_pointcloud(self, depth_m, rgb=None):
        """深度图 → 点云 (与实景 Orbbec 相同方法)"""
        h, w = depth_m.shape
        valid = depth_m > NEAR

        uu, vv = np.meshgrid(np.arange(w, dtype=np.float32),
                             np.arange(h, dtype=np.float32))
        Z = depth_m[valid]
        X = (uu[valid] - CX) * Z / FX
        Y = (vv[valid] - CY) * Z / FY

        points = np.stack([X, Y, Z], axis=-1)

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))

        if rgb is not None and rgb.shape[:2] == (h, w):
            colors = rgb[valid].astype(np.float64) / 255.0
            pcd.colors = o3d.utility.Vector3dVector(colors[:, ::-1])

        return pcd

    def generate_annotation(self, scene_config):
        """生成标注 JSON"""
        objects = []
        for obj in self.scene_objects:
            objects.append({
                "id": obj["id"],
                "type": "cuboid",
                "name": obj["name"],
                "half_extents": obj["half_extents"],
                "color_rgb": obj["color"],
                "position": list(obj["position"]),
                "z_min": obj["z_min"],
                "z_max": obj["z_max"],
            })

        hierarchy = {
            "layers": scene_config["layers"],
            "edges": [],
            "grasp_order": scene_config["grasp_order"],
        }

        # 自动生成 edges
        if scene_config["has_stacking"]:
            for layer_num in sorted(scene_config["layers"].keys()):
                if layer_num > 0:
                    uppers = scene_config["layers"][layer_num]
                    lowers = scene_config["layers"][layer_num - 1]
                    for uid in uppers:
                        for lid in lowers:
                            upper_obj = self.scene_objects[uid]
                            lower_obj = self.scene_objects[lid]
                            hierarchy["edges"].append({
                                "upper": uid,
                                "lower": lid,
                                "upper_name": upper_obj["name"],
                                "lower_name": lower_obj["name"],
                                "contact": True,
                            })

        return {
            "scene_id": scene_config["id"],
            "description_cn": scene_config["description_cn"],
            "description_en": scene_config["description_en"],
            "num_objects": len(objects),
            "has_stacking": scene_config["has_stacking"],
            "objects": objects,
            "hierarchy": hierarchy,
        }

    def create_overlay(self, rgb, depth_m, annotation):
        """创建 RGB + 深度 + 标注叠加图"""
        h, w = depth_m.shape

        # depth colormap
        valid = depth_m > NEAR
        dmin = depth_m[valid].min() if valid.any() else 0.2
        dmax = depth_m[valid].max() if valid.any() else 1.0
        depth_clip = np.clip(depth_m, dmin, dmax)
        depth_vis = cv2.applyColorMap(
            ((depth_clip - dmin) / max(dmax - dmin, 0.001) * 255).astype(np.uint8),
            cv2.COLORMAP_JET)

        # RGB
        rgb_vis = rgb.copy()

        # 绘制物体标注
        for obj in annotation["objects"]:
            oid = obj["id"]
            color = tuple(int(c * 255) for c in obj["color_rgb"][::-1])  # RGB→BGR
            pos = obj["position"]
            he = obj["half_extents"]

            # 在物体大致位置画框
            z = pos[2]
            if z > NEAR:
                px = int((pos[0] / z) * FX + CX)
                py = int((pos[1] / z) * FY + CY)
                rx = int(he[0] / z * FX)
                ry = int(he[2] / z * FY)
                x1 = max(0, px - rx)
                y1 = max(0, py - ry)
                x2 = min(w - 1, px + rx)
                y2 = min(h - 1, py + ry)
                cv2.rectangle(rgb_vis, (x1, y1), (x2, y2), color, 2)
                cv2.putText(rgb_vis, f"Obj{oid} L{annotation['hierarchy']['layers']}",
                            (x1, max(0, y1 - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

        # 拼接
        display = np.hstack([rgb_vis, depth_vis])

        # 文字信息
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(display, f"Scene: {annotation['scene_id']}",
                    (5, 20), font, 0.5, (255, 255, 255), 2)
        cv2.putText(display, annotation["description_cn"],
                    (5, 40), font, 0.4, (200, 200, 200), 1)
        cv2.putText(display, f"Stacking: {annotation['has_stacking']} | "
                    f"Grasp: {annotation['hierarchy']['grasp_order']}",
                    (5, 60), font, 0.4, (100, 255, 100), 1)
        cv2.putText(display, "RGB", (IMG_W // 2 - 15, IMG_H - 10),
                    font, 0.5, (0, 255, 0), 2)
        cv2.putText(display, "Depth", (IMG_W + IMG_W // 2 - 25, IMG_H - 10),
                    font, 0.5, (0, 255, 0), 2)

        return display

    def generate_all(self):
        """生成全部场景"""
        logger.info("=" * 60)
        logger.info("Generating Two-Cuboid Stacking Data")
        logger.info(f"Camera: {IMG_W}x{IMG_H}, fx={FX:.2f}, fy={FY:.2f}")
        logger.info(f"FOV: {self.fov_deg:.1f}deg, Height: {CAM_HEIGHT:.2f}m")
        logger.info("=" * 60)

        self.connect()

        for idx, scene_config in enumerate(SCENES):
            scene_id = scene_config["id"]
            scene_dir = os.path.join(OUT_DIR, f"scene_{idx + 1:02d}_{scene_id}")
            os.makedirs(scene_dir, exist_ok=True)

            logger.info(f"\n[{idx + 1}/8] {scene_id}: {scene_config['description_cn']}")

            # setup + render
            self.setup_scene(scene_config)
            rgb, depth_clean, seg = self.render_depth_rgb_seg()

            # semantic labels: 0=table, 1=object
            sem_labels = self._build_semantic_labels(seg)

            # noisy depth
            depth_noisy, noise_info = self._add_realistic_noise(depth_clean, seg)

            # stats
            valid_clean = (depth_clean > NEAR).sum()
            valid_noisy = (depth_noisy > NEAR).sum()
            logger.info(f"  Depth clean: {valid_clean} pts → noisy: {valid_noisy} pts "
                        f"({noise_info['missing_pct']}% dropped)")

            # pointcloud (clean for visualization, noisy for training)
            pcd_clean = self.depth_to_pointcloud(depth_clean, rgb)
            pcd_noisy = self.depth_to_pointcloud(depth_noisy, rgb)
            logger.info(f"  PointCloud: {len(pcd_clean.points)} (clean) / "
                        f"{len(pcd_noisy.points)} (noisy) points")

            # annotation
            annotation = self.generate_annotation(scene_config)
            annotation["semantic_labels"] = {
                "format": "per-pixel uint8, shape (H, W)",
                "classes": {"0": "background / table", "1": "foreground / object"},
                "file": "semantic_labels.npy",
            }
            annotation["depth"] = {
                "clean_file": "depth_clean.npy",
                "noisy_file": "depth_noisy.npy",
                "noise_info": noise_info,
            }

            # save
            cv2.imwrite(os.path.join(scene_dir, "rgb.png"),
                        cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))

            np.save(os.path.join(scene_dir, "depth_clean.npy"), depth_clean)
            np.save(os.path.join(scene_dir, "depth_noisy.npy"), depth_noisy)
            np.save(os.path.join(scene_dir, "semantic_labels.npy"), sem_labels)

            # depth visualizations
            for _name, _dep in [("depth_clean", depth_clean), ("depth_noisy", depth_noisy)]:
                v = _dep.copy()
                vv = v > NEAR
                dvis = cv2.applyColorMap(
                    cv2.normalize(v, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8),
                    cv2.COLORMAP_JET)
                cv2.imwrite(os.path.join(scene_dir, f"{_name}.png"), dvis)

            # semantic label viz
            label_viz = np.zeros((IMG_H, IMG_W, 3), dtype=np.uint8)
            label_viz[sem_labels == 0] = [60, 60, 60]    # 桌面 = 深灰
            label_viz[sem_labels == 1] = [0, 255, 0]      # 物体 = 绿色
            cv2.imwrite(os.path.join(scene_dir, "semantic_labels.png"), label_viz)

            o3d.io.write_point_cloud(os.path.join(scene_dir, "pointcloud_clean.ply"), pcd_clean)
            o3d.io.write_point_cloud(os.path.join(scene_dir, "pointcloud_noisy.ply"), pcd_noisy)

            with open(os.path.join(scene_dir, "annotation.json"), "w",
                      encoding="utf-8") as f:
                json.dump(annotation, f, indent=2, ensure_ascii=False)

            # overlay (clean depth + labels)
            overlay = self.create_overlay(rgb, depth_clean, annotation)
            cv2.imwrite(os.path.join(scene_dir, "overlay.png"), overlay)

            # noisy overlay
            overlay_noisy = self.create_overlay(rgb, depth_noisy, annotation)
            cv2.imwrite(os.path.join(scene_dir, "overlay_noisy.png"), overlay_noisy)

            logger.info(f"  Saved: {scene_dir}")

        self.disconnect()
        logger.info("\n" + "=" * 60)
        logger.info(f"All 8 scenes generated in: {OUT_DIR}")
        logger.info("=" * 60)


def main():
    gen = StackingDataGenerator()
    gen.generate_all()


if __name__ == "__main__":
    main()

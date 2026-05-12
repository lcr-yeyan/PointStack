"""
批量随机堆叠场景生成器
======================
生成 800 train + 200 val 随机两长方体堆叠场景，
输出 noisy depth + semantic labels。

输出: camera_operate/training_data/train/ 和 val/
"""

import os, json, time
import numpy as np

import cv2
import open3d as o3d

try:
    import pybullet as p
    import pybullet_data
except ImportError:
    print("pybullet not installed"); exit(1)

from loguru import logger
from scipy import ndimage

# ── Orbbec Femto Bolt 相机参数 ──
FX, FY = 506.35, 506.27
CX, CY = 334.19, 335.43
IMG_W, IMG_H = 640, 576
NEAR, FAR = 0.05, 2.0
CAM_HEIGHT = 0.50
CAM_TARGET = [0, 0, 0.04]
FOV_DEG = 2 * np.arctan(IMG_H / (2 * FY)) * 180 / np.pi

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "training_data")


def random_cuboid(rng):
    """随机长方体: 各维 5-15cm"""
    half = [rng.uniform(0.025, 0.075) for _ in range(3)]
    color = [rng.uniform(0.1, 0.9), rng.uniform(0.1, 0.9), rng.uniform(0.1, 0.9)]
    return half, color


def random_scene(rng):
    """随机生成一个两长方体场景配置"""
    hA, cA = random_cuboid(rng)
    hB, cB = random_cuboid(rng)
    ha_z, hb_z = hA[2], hB[2]

    is_stacking = rng.random() < 0.70

    if is_stacking:
        # 重叠度: 25%-100%
        overlap = rng.uniform(0.25, 1.0)
        offset_x = (hA[0] + hB[0]) * (1.0 - overlap) * rng.choice([-1, 1])
        offset_y = (hB[1] - hA[1] + rng.uniform(-0.02, 0.02))
        # 确保 B 在 A 上
        if abs(offset_x) > hA[0] + hB[0] - 0.01:
            offset_x *= 0.5
        if abs(offset_y) > hA[1] + hB[1] - 0.01:
            offset_y *= 0.5

        base_x = rng.uniform(-0.02, 0.02)
        base_y = rng.uniform(-0.02, 0.02)
        rz_A = rng.uniform(-30, 30)
        rz_B = rng.uniform(-30, 30)

        objects = [
            {"half": hA, "color": cA, "pos": (base_x, base_y, ha_z, rz_A)},
            {"half": hB, "color": cB,
             "pos": (base_x + offset_x, base_y + offset_y, 2 * ha_z + hb_z, rz_B)},
        ]
        has_stacking = True
        layers = {0: [0], 1: [1]}
        grasp_order = [1, 0]
    else:
        # 并排放置
        gap = rng.uniform(0.01, 0.06)
        sep = hA[0] + hB[0] + gap
        sign = rng.choice([-1, 1])
        off_y = rng.uniform(-0.03, 0.03)
        rz_A = rng.uniform(-45, 45)
        rz_B = rng.uniform(-45, 45)
        objects = [
            {"half": hA, "color": cA, "pos": (-sep * 0.5, off_y, ha_z, rz_A)},
            {"half": hB, "color": cB, "pos": (sep * 0.5, -off_y, hb_z, rz_B)},
        ]
        has_stacking = False
        layers = {0: [0, 1]}
        grasp_order = [0, 1]

    return {
        "objects": objects,
        "has_stacking": has_stacking,
        "layers": layers,
        "grasp_order": grasp_order,
    }


class BatchGenerator:
    def __init__(self):
        self.client_id = None
        self.body_ids = []

    def connect(self):
        self.client_id = p.connect(p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=self.client_id)

    def disconnect(self):
        if self.client_id is not None:
            p.disconnect(physicsClientId=self.client_id)

    def _create_ground(self):
        col = p.createCollisionShape(p.GEOM_PLANE, physicsClientId=self.client_id)
        vis = p.createVisualShape(p.GEOM_PLANE,
                                  rgbaColor=[0.35, 0.35, 0.40, 1.0],
                                  physicsClientId=self.client_id)
        p.createMultiBody(0, col, vis, basePosition=[0, 0, 0], physicsClientId=self.client_id)

    def _create_cuboid(self, half, color, pos, euler_z):
        col = p.createCollisionShape(p.GEOM_BOX, halfExtents=half,
                                     physicsClientId=self.client_id)
        vis = p.createVisualShape(p.GEOM_BOX, halfExtents=half,
                                  rgbaColor=[*color, 1.0],
                                  physicsClientId=self.client_id)
        orn = p.getQuaternionFromEuler([0, 0, np.radians(euler_z)],
                                       physicsClientId=self.client_id)
        bid = p.createMultiBody(baseMass=0.05, baseCollisionShapeIndex=col,
                                baseVisualShapeIndex=vis,
                                basePosition=list(pos), baseOrientation=orn,
                                physicsClientId=self.client_id)
        p.changeDynamics(bid, -1, lateralFriction=0.8, spinningFriction=0.002,
                         restitution=0.02, physicsClientId=self.client_id)
        return bid

    def setup(self, scene_config):
        p.resetSimulation(physicsClientId=self.client_id)
        p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=self.client_id)
        p.setGravity(0, 0, -9.81, physicsClientId=self.client_id)
        self._create_ground()
        self.body_ids = []
        for obj in scene_config["objects"]:
            h, c, (x, y, z, rz) = obj["half"], obj["color"], obj["pos"]
            bid = self._create_cuboid(h, c, (x, y, z), rz)
            self.body_ids.append(bid)
        for _ in range(60):
            p.stepSimulation(physicsClientId=self.client_id)

    def render(self):
        view = p.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=CAM_TARGET, distance=CAM_HEIGHT,
            yaw=0, pitch=-90, roll=0, upAxisIndex=2,
            physicsClientId=self.client_id)
        proj = p.computeProjectionMatrixFOV(
            fov=FOV_DEG, aspect=IMG_W / IMG_H, nearVal=NEAR, farVal=FAR,
            physicsClientId=self.client_id)
        _, _, rgb_raw, depth_raw, seg_raw = p.getCameraImage(
            width=IMG_W, height=IMG_H, viewMatrix=view, projectionMatrix=proj,
            renderer=p.ER_TINY_RENDERER, physicsClientId=self.client_id)

        rgb = rgb_raw[:, :, :3].astype(np.uint8).copy()
        depth_norm = depth_raw.astype(np.float32)
        depth_m = FAR * NEAR / (FAR - (FAR - NEAR) * depth_norm)
        depth_m[depth_m > FAR * 0.99] = 0.0
        seg = seg_raw.astype(np.int32).copy()
        return rgb, depth_m, seg

    @staticmethod
    def sem_labels(seg):
        labels = np.zeros((IMG_H, IMG_W), dtype=np.uint8)
        labels[seg > 0] = 1
        return labels

    @staticmethod
    def add_noise(depth_clean, seg, rng):
        """加入真实 ToF 噪声"""
        h, w = depth_clean.shape
        depth = depth_clean.copy()
        valid = depth > NEAR

        # 1. 高斯抖动 3mm
        depth[valid] += rng.normal(0, 0.003, (h, w)).astype(np.float32)[valid]

        # 2. 中心缺失 (ToF 饱和)
        cy, cx = h / 2, w / 2
        uu = (np.arange(w, dtype=np.float32) - cx) / cx
        vv = (np.arange(h, dtype=np.float32) - cy) / cy
        uv_dist = np.sqrt(uu[None, :] ** 2 + vv[:, None] ** 2)
        center_prob = np.exp(-uv_dist * 2.0) * rng.uniform(0.08, 0.15)
        drop = rng.random((h, w)) < center_prob
        depth[drop & valid] = 0.0

        # 3. 边缘飞点
        if seg is not None:
            obj = seg > 0
            dilated = ndimage.binary_dilation(obj, np.ones((3, 3)), iterations=2)
            eroded = ndimage.binary_erosion(obj, np.ones((3, 3)), iterations=2)
            boundary = dilated & (~eroded)
            edge_drop = boundary & (rng.random((h, w)) < rng.uniform(0.20, 0.40))
            depth[edge_drop & (depth > NEAR)] = 0.0

        # 4. 散粒
        speckle = rng.random((h, w)) < rng.uniform(0.003, 0.008)
        depth[speckle & (depth > NEAR)] = 0.0

        return depth

    def generate_scene(self, scene_config, scene_id, split_dir):
        """生成单个场景文件"""
        scene_dir = os.path.join(split_dir, f"scene_{scene_id:04d}")
        os.makedirs(scene_dir, exist_ok=True)

        self.setup(scene_config)
        rgb, depth_clean, seg = self.render()
        sem = self.sem_labels(seg)
        rng = np.random.RandomState()
        depth_noisy = self.add_noise(depth_clean, seg, rng)

        # 标注
        annotation = {
            "scene_id": scene_id,
            "has_stacking": scene_config["has_stacking"],
            "num_objects": 2,
            "objects": [
                {"id": i, "half_extents": obj["half"], "color_rgb": obj["color"],
                 "position": list(obj["pos"])}
                for i, obj in enumerate(scene_config["objects"])
            ],
            "hierarchy": {
                "layers": scene_config["layers"],
                "grasp_order": scene_config["grasp_order"],
            },
        }
        annotation["semantic_classes"] = {"0": "table", "1": "object"}

        # 保存
        np.save(os.path.join(scene_dir, "depth_noisy.npy"), depth_noisy)
        np.save(os.path.join(scene_dir, "semantic_labels.npy"), sem)
        with open(os.path.join(scene_dir, "annotation.json"), "w", encoding="utf-8") as f:
            json.dump(annotation, f, indent=2, ensure_ascii=False)

        n_valid = (depth_noisy > NEAR).sum()
        n_obj = (sem == 1).sum()
        return n_valid, n_obj


def main():
    rng = np.random.RandomState(42)

    train_dir = os.path.join(OUT_DIR, "train")
    val_dir = os.path.join(OUT_DIR, "val")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    gen = BatchGenerator()
    gen.connect()

    total = {"train": 800, "val": 200}

    for split_name, split_dir, n_scenes in [("train", train_dir, total["train"]),
                                              ("val", val_dir, total["val"])]:
        logger.info(f"\n{'=' * 50}")
        logger.info(f"Generating {split_name}: {n_scenes} scenes")
        logger.info(f"{'=' * 50}")

        n_stacking = 0
        n_nonstacking = 0

        for i in range(n_scenes):
            cfg = random_scene(rng)
            n_valid, n_obj = gen.generate_scene(cfg, i, split_dir)
            if cfg["has_stacking"]:
                n_stacking += 1
            else:
                n_nonstacking += 1

            if (i + 1) % 100 == 0:
                logger.info(f"  [{i + 1}/{n_scenes}] stack={n_stacking} "
                            f"nonstack={n_nonstacking}")

        logger.info(f"Done {split_name}: stacking={n_stacking}, non-stacking={n_nonstacking}")

    gen.disconnect()
    logger.info(f"\nAll {sum(total.values())} scenes saved to {OUT_DIR}")


if __name__ == "__main__":
    main()

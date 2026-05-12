import numpy as np
import open3d as o3d
from typing import Dict, Any, Tuple, Optional
from loguru import logger


class PointCloudPreprocessor:
    def __init__(self, config: Dict[str, Any]):
        self.config = config["preprocessing"]
        self.roi_cfg = self.config.get("roi", {})
        self.sor_cfg = self.config.get("statistical_outlier_removal", {})
        self.voxel_cfg = self.config.get("voxel_downsample", {})
        self.normal_cfg = self.config.get("normal_estimation", {})
        self.sample_cfg = self.config.get("fixed_sampling", {})
        self.norm_cfg = self.config.get("normalization", {})

    def process(self, pcd: o3d.geometry.PointCloud) -> Tuple[np.ndarray, o3d.geometry.PointCloud]:
        pcd = self._crop_roi(pcd)
        pcd = self._remove_statistical_outliers(pcd)
        pcd = self._voxel_downsample(pcd)
        if self.normal_cfg:
            pcd = self._estimate_normals(pcd)
        points_np = np.asarray(pcd.points)
        points_normalized, scale_info = self._normalize(points_np)
        points_fixed = self._fixed_sample(points_normalized)
        return points_fixed, pcd

    def _crop_roi(self, pcd: o3d.geometry.PointCloud) -> o3d.geometry.PointCloud:
        if not self.roi_cfg:
            return pcd
        points = np.asarray(pcd.points)
        depth_min = self.roi_cfg.get("depth_min", 0.0)
        depth_max = self.roi_cfg.get("depth_max", 10.0)
        x_min = self.roi_cfg.get("x_min", -10.0)
        x_max = self.roi_cfg.get("x_max", 10.0)
        y_min = self.roi_cfg.get("y_min", -10.0)
        y_max = self.roi_cfg.get("y_max", 10.0)
        mask = (
            (points[:, 2] >= depth_min) & (points[:, 2] <= depth_max)
            & (points[:, 0] >= x_min) & (points[:, 0] <= x_max)
            & (points[:, 1] >= y_min) & (points[:, 1] <= y_max)
        )
        cropped = pcd.select_by_index(np.where(mask)[0])
        logger.debug(f"ROI crop: {len(points)} -> {len(cropped.points)} points")
        return cropped

    def _remove_statistical_outliers(self, pcd: o3d.geometry.PointCloud) -> o3d.geometry.PointCloud:
        nb_neighbors = self.sor_cfg.get("nb_neighbors", 20)
        std_ratio = self.sor_cfg.get("std_ratio", 2.0)
        pcd_clean, ind = pcd.remove_statistical_outlier(
            nb_neighbors=nb_neighbors, std_ratio=std_ratio
        )
        logger.debug(
            f"Statistical outlier removal: {len(pcd.points)} -> {len(pcd_clean.points)} points"
        )
        return pcd_clean

    def _voxel_downsample(self, pcd: o3d.geometry.PointCloud) -> o3d.geometry.PointCloud:
        voxel_size = self.voxel_cfg.get("voxel_size", 0.005)
        pcd_down = pcd.voxel_down_sample(voxel_size=voxel_size)
        logger.debug(f"Voxel downsample ({voxel_size}m): {len(pcd.points)} -> {len(pcd_down.points)} points")
        return pcd_down

    def _estimate_normals(self, pcd: o3d.geometry.PointCloud) -> o3d.geometry.PointCloud:
        k_nn = self.normal_cfg.get("k_nn", 30)
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=k_nn))
        pcd.orient_normals_towards_camera_location(camera_location=np.array([0.0, 0.0, 0.0]))
        return pcd

    @staticmethod
    def _normalize(points: np.ndarray) -> Tuple[np.ndarray, Dict[str, float]]:
        centroid = np.mean(points, axis=0)
        centered = points - centroid
        max_dist = np.max(np.linalg.norm(centered, axis=1))
        max_dist = max(max_dist, 1e-8)
        normalized = centered / max_dist
        scale_info = {"centroid": centroid.tolist(), "scale": float(max_dist)}
        return normalized, scale_info

    def _fixed_sample(self, points: np.ndarray) -> np.ndarray:
        num_points = self.sample_cfg.get("num_points", 1024)
        n_current = len(points)
        if n_current == 0:
            raise ValueError("Point cloud is empty after preprocessing")
        if n_current >= num_points:
            indices = np.random.choice(n_current, size=num_points, replace=False)
            sampled = points[indices]
        else:
            repeat_times = (num_points // n_current) + 1
            repeated = np.tile(points, (repeat_times, 1))[:num_points]
            shuffle_idx = np.random.permutation(num_points)
            sampled = repeated[shuffle_idx]
        return sampled.astype(np.float32)


def preprocess_pointcloud(
    pcd: o3d.geometry.PointCloud,
    config: Dict[str, Any],
) -> Tuple[np.ndarray, o3d.geometry.PointCloud]:
    preprocessor = PointCloudPreprocessor(config)
    return preprocessor.process(pcd)

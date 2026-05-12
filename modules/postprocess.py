import numpy as np
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass, field
from scipy.spatial import cKDTree
from scipy import ndimage
from loguru import logger


@dataclass
class Instance:
    id: int
    point_indices: np.ndarray
    centroid: np.ndarray = field(default_factory=lambda: np.zeros(3))
    bbox_min: np.ndarray = field(default_factory=lambda: np.zeros(3))
    bbox_max: np.ndarray = field(default_factory=lambda: np.zeros(3))
    point_count: int = 0
    z_mean: float = 0.0
    z_min: float = 0.0
    z_max: float = 0.0
    support_normal: Optional[np.ndarray] = None

    def __post_init__(self):
        self.point_count = len(self.point_indices)


@dataclass
class InstanceSegmentationResult:
    instances: List[Instance]
    instance_labels: np.ndarray
    num_instances: int


def _euclidean_clustering(
    points: np.ndarray,
    tolerance: float,
    min_cluster_size: int,
    max_cluster_size: int = 100000,
) -> Tuple[List[np.ndarray], np.ndarray]:
    N = len(points)
    if N == 0:
        return [], np.array([], dtype=int)
    parent = list(range(N))
    rank = [0] * N

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        rx, ry = find(x), find(y)
        if rx == ry:
            return
        if rank[rx] < rank[ry]:
            rx, ry = ry, rx
        parent[ry] = rx
        if rank[rx] == rank[ry]:
            rank[rx] += 1

    tree = cKDTree(points)
    pairs = tree.query_pairs(r=tolerance)
    logger.debug(f"Euclidean clustering: N={N}, tol={tolerance}, pairs={len(pairs)}")
    for i, j in pairs:
        union(i, j)

    clusters_dict = {}
    for i in range(N):
        root = find(i)
        if root not in clusters_dict:
            clusters_dict[root] = []
        clusters_dict[root].append(i)

    logger.debug(f"Union-Find result: {len(clusters_dict)} components, sizes={sorted([len(v) for v in clusters_dict.values()], reverse=True)[:10]}")
    labels = -np.ones(N, dtype=int)
    valid_clusters = []
    for label_id, (root, members) in enumerate(sorted(clusters_dict.items())):
        if len(members) < min_cluster_size or len(members) > max_cluster_size:
            continue
        member_arr = np.array(members, dtype=np.int64)
        labels[member_arr] = label_id
        valid_clusters.append(member_arr)

    return valid_clusters, labels


def _fit_support_plane(points: np.ndarray, bottom_ratio: float = 0.15) -> Tuple[np.ndarray, float]:
    n_bottom = max(int(len(points) * bottom_ratio), 5)
    z_sorted_idx = np.argsort(points[:, 2])
    bottom_pts = points[z_sorted_idx[:n_bottom]]
    centroid = np.mean(bottom_pts, axis=0)
    centered = bottom_pts - centroid
    if len(centered) < 3:
        return np.array([0.0, 0.0, 1.0]), centroid[2]
    try:
        _, _, vh = np.linalg.svd(centered)
        normal = vh[-1]
        if normal[2] < 0:
            normal = -normal
        plane_z = centroid[2]
        return normal, plane_z
    except Exception:
        return np.array([0.0, 0.0, 1.0]), centroid[2]


def _connected_components_2d(
    points: np.ndarray,
    fg_indices: np.ndarray,
    img_shape: Tuple[int, int] = (480, 640),
    fx: float = 615.0,
    fy: float = 615.0,
    cx: float = 320.0,
    cy: float = 240.0,
    min_cluster_size: int = 10,
) -> Tuple[List[np.ndarray], np.ndarray]:
    h, w = img_shape
    fg_mask_img = np.zeros((h, w), dtype=np.uint8)
    pt_to_pixel = {}
    for idx in fg_indices:
        if idx >= len(points):
            continue
        p = points[idx]
        z = max(p[2], 1e-6)
        px = int((p[0] / z) * fx + cx)
        py = int((p[1] / z) * fy + cy)
        if 0 <= px < w and 0 <= py < h:
            fg_mask_img[py, px] = 1
            pt_to_pixel[idx] = (py, px)
    labeled, num_features = ndimage.label(fg_mask_img, structure=np.ones((3, 3)))
    logger.debug(f"2D CC: {num_features} components from FG mask")
    valid_clusters = []
    labels_full = -np.ones(len(points), dtype=int)
    for comp_id in range(1, num_features + 1):
        comp_mask = labeled == comp_id
        member_pixels = np.argwhere(comp_mask)
        pixel_set = set(map(tuple, member_pixels))
        member_pts = [idx for idx, pix in pt_to_pixel.items() if tuple(pix) in pixel_set]
        if len(member_pts) < min_cluster_size:
            continue
        member_arr = np.array(member_pts, dtype=np.int64)
        local_label = len(valid_clusters)
        labels_full[member_arr] = local_label
        valid_clusters.append(member_arr)
    logger.debug(f"2D CC result: {len(valid_clusters)} valid clusters (min={min_cluster_size}), sizes={[len(c) for c in valid_clusters]}")
    return valid_clusters, labels_full


class InstanceClustering:
    def __init__(self, config: Dict[str, Any]):
        self.config = config.get("postprocess", {})
        cluster_cfg = self.config.get("cluster", {})
        self.tolerance = cluster_cfg.get("tolerance", 0.008)
        self.min_samples = cluster_cfg.get("min_samples", 8)
        filter_cfg = self.config.get("instance_filter", {})
        self.min_point_count = filter_cfg.get("min_point_count", 15)
        self.max_point_count = filter_cfg.get("max_point_count", 100000)

    def _auto_select_tolerance(self, fg_points: np.ndarray) -> float:
        if len(fg_points) < 50:
            return 0.005
        sample_size = min(1000, len(fg_points))
        sample_idx = np.random.choice(len(fg_points), sample_size, replace=False)
        sample_pts = fg_points[sample_idx]
        tree = cKDTree(sample_pts)
        k = min(10, sample_size - 1)
        dists, _ = tree.query(sample_pts, k=k + 1)
        kth_dists = dists[:, -1]
        p25, p50, p75 = np.percentile(kth_dists, [25, 50, 75])
        tol_auto = p25 * 1.8
        tol_auto = max(tol_auto, p50 * 0.9)
        tol_auto = min(tol_auto, p75 * 3.0)
        tol_auto = max(tol_auto, 0.001)
        tol_auto = min(tol_auto, 0.03)
        logger.debug(f"Auto-tol: {tol_auto:.4f} (p25={p25:.4f}, p50={p50:.4f}, p75={p75:.4f})")
        return tol_auto

    def _z_layered_cluster(
        self,
        points: np.ndarray,
        fg_indices: np.ndarray,
    ) -> Tuple[List[np.ndarray], np.ndarray]:
        fg_points = points[fg_indices]
        z_vals = fg_points[:, 2]
        z_min_val, z_max_val = z_vals.min(), z_vals.max()
        z_range = max(z_max_val - z_min_val, 1e-6)
        n_layers = max(3, int(z_range / 0.015))
        layer_height = z_range / n_layers
        all_clusters = []
        labels_full = -np.ones(len(points), dtype=int)
        global_label_counter = 0
        for layer_idx in range(n_layers):
            z_low = z_min_val + layer_idx * layer_height
            z_high = z_low + layer_height
            if layer_idx == n_layers - 1:
                z_high = z_max_val + 1e-6
            in_layer_mask = (z_vals >= z_low) & (z_vals < z_high)
            layer_fg_indices_local = np.where(in_layer_mask)[0]
            if len(layer_fg_indices_local) < self.min_samples:
                for local_i in layer_fg_indices_local:
                    gi = fg_indices[local_i]
                    labels_full[gi] = -1
                continue
            layer_pts = fg_points[layer_fg_indices_local]
            tol = self._auto_select_tolerance(layer_pts)
            raw_clusters, raw_labels = _euclidean_clustering(
                layer_pts,
                tolerance=tol,
                min_cluster_size=self.min_samples,
                max_cluster_size=self.max_point_count,
            )
            for local_cidx, member_arr in enumerate(raw_clusters):
                if len(member_arr) < self.min_point_count:
                    continue
                global_member_indices = fg_indices[layer_fg_indices_local[member_arr]]
                all_clusters.append(global_member_indices)
                labels_full[global_member_indices] = global_label_counter
                global_label_counter += 1
            for local_i in range(len(layer_fg_indices_local)):
                if raw_labels[local_i] == -1:
                    gi = fg_indices[layer_fg_indices_local[local_i]]
                    labels_full[gi] = -1
        logger.info(
            f"Z-layered cluster: {n_layers} layers, "
            f"{len(fg_points)} FG pts -> {len(all_clusters)} instances"
        )
        return all_clusters, labels_full

    def _is_topdown_view(self, fg_points: np.ndarray) -> bool:
        if len(fg_points) < 20:
            return False
        x_range = fg_points[:, 0].max() - fg_points[:, 0].min()
        y_range = fg_points[:, 1].max() - fg_points[:, 1].min()
        z_range = fg_points[:, 2].max() - fg_points[:, 2].min()
        xy_spread = max(x_range, y_range, 1e-6)
        is_topdown = (z_range / xy_spread < 0.3) and (xy_spread > 0.02)
        if is_topdown:
            logger.debug(f"Top-down view detected: z_range={z_range:.4f}, xy_spread={xy_spread:.4f}, ratio={z_range/xy_spread:.3f}")
        return is_topdown

    def _xy_euclidean_cluster(
        self,
        points: np.ndarray,
        fg_indices: np.ndarray,
    ) -> Tuple[List[np.ndarray], np.ndarray]:
        fg_points = points[fg_indices]
        xy_points = fg_points[:, :2]
        tol = self._auto_select_tolerance(xy_points)
        tol = max(tol, 0.003)
        tol = min(tol, 0.012)
        logger.info(f"XY-plane clustering (top-down): tol={tol:.4f}, N={len(xy_points)}")
        raw_clusters, raw_labels = _euclidean_clustering(
            xy_points,
            tolerance=tol,
            min_cluster_size=self.min_samples,
            max_cluster_size=self.max_point_count,
        )
        all_clusters = []
        labels_full = -np.ones(len(points), dtype=int)
        global_label_counter = 0
        for member_arr in raw_clusters:
            if len(member_arr) < self.min_point_count:
                continue
            global_member_indices = fg_indices[member_arr]
            all_clusters.append(global_member_indices)
            labels_full[global_member_indices] = global_label_counter
            global_label_counter += 1
        return all_clusters, labels_full

    def cluster(
        self,
        points: np.ndarray,
        foreground_mask: np.ndarray,
    ) -> InstanceSegmentationResult:
        fg_points = points[foreground_mask]
        fg_indices = np.where(foreground_mask)[0]
        logger.debug(f"cluster() input: points.shape={points.shape}, fg_mask sum={foreground_mask.sum()}")
        if len(fg_points) > 10:
            logger.debug(f"  fg_points range: X=[{fg_points[:,0].min():.4f},{fg_points[:,0].max():.4f}] Z=[{fg_points[:,2].min():.4f},{fg_points[:,2].max():.4f}]")
        if len(fg_points) == 0:
            return InstanceSegmentationResult(
                instances=[], instance_labels=np.array([], dtype=int), num_instances=0
            )
        is_topdown = self._is_topdown_view(fg_points)
        if is_topdown:
            cluster_indices_list, full_labels = self._xy_euclidean_cluster(points, fg_indices)
            is_z_layered = True
        elif self.config.get("use_z_layered_clustering", True) and len(fg_points) > 50:
            cluster_indices_list, full_labels = self._z_layered_cluster(points, fg_indices)
            is_z_layered = True
        else:
            logger.info(f"Clustering: method=3d_euclidean, tol={self.tolerance:.4f}, min_pts={self.min_point_count}")
            cluster_indices_list, cluster_labels = _euclidean_clustering(
                fg_points,
                tolerance=self.tolerance,
                min_cluster_size=self.min_samples,
                max_cluster_size=self.max_point_count,
            )
            full_labels = -np.ones(len(points), dtype=int)
            full_labels[fg_indices] = cluster_labels
            is_z_layered = False
        instances = []
        for local_idx, member_arr in enumerate(cluster_indices_list):
            if len(member_arr) < self.min_point_count:
                continue
            # _z_layered_cluster 返回的已经是全局索引
            if is_z_layered:
                global_indices = member_arr
            else:
                global_indices = fg_indices[member_arr]
            cluster_pts = points[global_indices]
            centroid = np.mean(cluster_pts, axis=0)
            bbox_min = np.min(cluster_pts, axis=0)
            bbox_max = np.max(cluster_pts, axis=0)
            z_mean = float(np.mean(cluster_pts[:, 2]))
            z_min = float(np.min(cluster_pts[:, 2]))
            z_max = float(np.max(cluster_pts[:, 2]))
            normal, plane_z = _fit_support_plane(cluster_pts)
            instance = Instance(
                id=len(instances),
                point_indices=global_indices,
                centroid=centroid,
                bbox_min=bbox_min,
                bbox_max=bbox_max,
                point_count=len(global_indices),
                z_mean=z_mean,
                z_min=z_min,
                z_max=z_max,
                support_normal=normal,
            )
            instances.append(instance)
        logger.info(
            f"3D Euclidean (tol={self.tolerance:.4f}m): {len(fg_points)} FG pts -> "
            f"{len(cluster_indices_list)} raw -> {len(instances)} instances (min={self.min_point_count})"
        )
        return InstanceSegmentationResult(
            instances=instances,
            instance_labels=full_labels,
            num_instances=len(instances),
        )


def run_instance_clustering(
    points: np.ndarray,
    segmentation_labels: np.ndarray,
    config: Dict[str, Any],
) -> InstanceSegmentationResult:
    foreground_class = 1
    foreground_mask = segmentation_labels == foreground_class
    clusterer = InstanceClustering(config)
    return clusterer.cluster(points, foreground_mask)


def oracle_instance_segmentation(
    points: np.ndarray,
    gt_labels: np.ndarray,
) -> InstanceSegmentationResult:
    fg_mask = gt_labels > 0
    fg_indices = np.where(fg_mask)[0]
    unique_labels = np.unique(gt_labels[fg_mask])
    instances = []
    full_labels = -np.ones(len(points), dtype=int)
    for lid, obj_id in enumerate(sorted(unique_labels)):
        member_indices = np.where(gt_labels == obj_id)[0]
        if len(member_indices) < 5:
            continue
        cluster_pts = points[member_indices]
        centroid = np.mean(cluster_pts, axis=0)
        bbox_min = np.min(cluster_pts, axis=0)
        bbox_max = np.max(cluster_pts, axis=0)
        z_mean = float(np.mean(cluster_pts[:, 2]))
        z_min = float(np.min(cluster_pts[:, 2]))
        z_max = float(np.max(cluster_pts[:, 2]))
        normal, _ = _fit_support_plane(cluster_pts)
        instance = Instance(
            id=lid,
            point_indices=member_indices,
            centroid=centroid,
            bbox_min=bbox_min,
            bbox_max=bbox_max,
            point_count=len(member_indices),
            z_mean=z_mean,
            z_min=z_min,
            z_max=z_max,
            support_normal=normal,
        )
        instances.append(instance)
        full_labels[member_indices] = lid
    logger.info(f"Oracle instance segmentation: {len(unique_labels)} GT objects -> {len(instances)} instances")
    return InstanceSegmentationResult(
        instances=instances,
        instance_labels=full_labels,
        num_instances=len(instances),
    )

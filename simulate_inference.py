"""
模拟推理: data_preview 场景 → 语义标签 → 实例聚类 → 层级推理 → 可视化
=====================================================================
展示从"模型输出"到"层级关系"的完整后处理流程。
"""

import os, sys, json
import numpy as np
import cv2
import open3d as o3d
from scipy.ndimage import gaussian_filter1d

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from loguru import logger
from omegaconf import OmegaConf
from utils.config import load_config
from modules.postprocess import Instance, InstanceSegmentationResult
from modules.hierarchy import build_hierarchy, HierarchyResult

FX, FY = 506.35, 506.27
CX, CY = 334.19, 335.43
NEAR = 0.05

COLORS = [
    (46, 204, 113), (52, 152, 219), (231, 76, 60), (241, 196, 15),
    (155, 89, 182), (26, 188, 156), (230, 126, 34),
]


def depth_to_points(depth_m, sem_labels):
    """深度图 + 语义标签 → 3D点 + 标签"""
    h, w = depth_m.shape
    valid = depth_m > NEAR
    uu, vv = np.meshgrid(np.arange(w, dtype=np.float32),
                         np.arange(h, dtype=np.float32))
    Z = depth_m[valid]
    X = (uu[valid] - CX) * Z / FX
    Y = (vv[valid] - CY) * Z / FY
    return np.stack([X, Y, Z], axis=-1).astype(np.float64), sem_labels[valid].copy()


def cluster_instances(points, seg_labels, config_dict):
    """
    语义标签 → 实例聚类 (z-layered clustering)
    模拟 models.postprocess.run_instance_clustering
    """
    from modules.postprocess import InstanceClustering

    fg_mask = seg_labels == 1
    if fg_mask.sum() == 0:
        return InstanceSegmentationResult([], -np.ones(len(points), dtype=int), 0)

    fg_points = points[fg_mask]
    fg_indices = np.where(fg_mask)[0]
    clusterer = InstanceClustering(config_dict)

    use_z_layered = config_dict.get("postprocess", {}).get("use_z_layered_clustering", True)
    if use_z_layered and len(fg_points) > 50:
        clusters, _ = clusterer._z_layered_cluster(points, fg_indices)
        is_z_layered = True
    else:
        from modules.postprocess import _euclidean_clustering
        tol = clusterer._auto_select_tolerance(fg_points)
        clusters, _ = _euclidean_clustering(fg_points, tolerance=tol,
                                            min_cluster_size=clusterer.min_samples,
                                            max_cluster_size=clusterer.max_point_count)
        is_z_layered = False

    instances = []
    full_labels = -np.ones(len(points), dtype=int)

    for local_idx, member_arr in enumerate(clusters):
        if len(member_arr) < clusterer.min_point_count:
            continue
        if is_z_layered:
            global_idx = member_arr
        else:
            global_idx = fg_indices[member_arr]

        pts = points[global_idx]
        centroid = np.mean(pts, axis=0)
        instances.append(Instance(
            id=len(instances),
            point_indices=global_idx.astype(np.int64),
            centroid=centroid,
            bbox_min=np.min(pts, axis=0),
            bbox_max=np.max(pts, axis=0),
            point_count=len(global_idx),
            z_mean=float(np.mean(pts[:, 2])),
            z_min=float(np.min(pts[:, 2])),
            z_max=float(np.max(pts[:, 2])),
        ))
        full_labels[global_idx] = len(instances) - 1

    return InstanceSegmentationResult(instances, full_labels, len(instances))


# ── 实例后处理改进 ──────────────────────────────────────────────────
OBJ_HEIGHT_MIN = 0.025   # 物体最小高度 2.5cm
OBJ_HEIGHT_MAX = 0.075   # 物体最大高度 7.5cm
Z_GAP_MIN = 0.008        # 最小 Z 间隙判定堆叠


def refine_height_split(instances, points):
    """
    改进 1: 高度先验切分 (解决第三类 Blind 堆叠)

    如果聚类结果的 Z 范围 > OBJ_HEIGHT_MAX，说明多个物体被合并了。
    在 Z 方向找自然间隙切分。
    """
    new_instances = []
    for inst in instances:
        z_range = inst.z_max - inst.z_min
        if z_range <= OBJ_HEIGHT_MAX:
            new_instances.append(inst)
            continue

        # Z 范围异常大 → 多物体被压成一个
        pts = points[inst.point_indices]
        z_vals = pts[:, 2]

        # 在 Z 方向找分界点: Z 直方图谷底
        n_bins = max(10, int(z_range / 0.003))
        hist, bin_edges = np.histogram(z_vals, bins=n_bins)

        # 找最深谷 (两峰之间的最低点)
        from scipy.ndimage import gaussian_filter1d
        hist_s = gaussian_filter1d(hist.astype(float), sigma=2.0)

        # 找谷
        valleys = []
        for i in range(2, len(hist_s) - 2):
            if (hist_s[i] < hist_s[i - 1] and hist_s[i] < hist_s[i + 1] and
                    hist_s[i] < hist_s[i - 2] and hist_s[i] < hist_s[i + 2]):
                valleys.append((i, hist_s[i]))

        if not valleys:
            new_instances.append(inst)
            continue

        # 选最深谷
        best_v = min(valleys, key=lambda x: x[1])[0]
        split_z = (bin_edges[best_v] + bin_edges[best_v + 1]) / 2

        # 切分
        lower_mask = pts[:, 2] < split_z
        upper_mask = pts[:, 2] >= split_z

        for mask, suffix in [(lower_mask, "_lower"), (upper_mask, "_upper")]:
            if mask.sum() < 50:
                continue
            sub_idx = inst.point_indices[mask]
            sub_pts = points[sub_idx]
            new_instances.append(Instance(
                id=0,
                point_indices=sub_idx.astype(np.int64),
                centroid=np.mean(sub_pts, axis=0),
                bbox_min=np.min(sub_pts, axis=0),
                bbox_max=np.max(sub_pts, axis=0),
                point_count=len(sub_idx),
                z_mean=float(np.mean(sub_pts[:, 2])),
                z_min=float(np.min(sub_pts[:, 2])),
                z_max=float(np.max(sub_pts[:, 2])),
            ))

    # re-id
    for i, inst in enumerate(new_instances):
        inst.id = i
    return new_instances


def refine_merge_fragments(instances, points):
    """
    改进 2: 碎片重连 (解决第五类 Cross 堆叠)

    如果两碎片满足:
    1. Z 均值相近 (同层)
    2. XY 包围盒在某一维度共线 (同物体被切开)
    3. 中间被另一物体隔开
    → 合并
    """
    if len(instances) <= 2:
        return instances

    merged = list(range(len(instances)))
    parent = list(range(len(instances)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for i in range(len(instances)):
        for j in range(i + 1, len(instances)):
            a, b = instances[i], instances[j]
            # 条件1: Z 相近
            if abs(a.z_mean - b.z_mean) > 0.015:
                continue

            # 条件2: XY 共线检查
            # 两碎片在 X 或 Y 方向上有延伸关系
            ax1, ax2 = a.bbox_min[0], a.bbox_max[0]
            ay1, ay2 = a.bbox_min[1], a.bbox_max[1]
            bx1, bx2 = b.bbox_min[0], b.bbox_max[0]
            by1, by2 = b.bbox_min[1], b.bbox_max[1]

            x_overlap = max(0, min(ax2, bx2) - max(ax1, bx1))
            y_overlap = max(0, min(ay2, by2) - max(ay1, by1))

            a_w, a_h = ax2 - ax1, ay2 - ay1
            b_w, b_h = bx2 - bx1, by2 - by1

            # Y 共线 + X 分离: 左-右碎片 (被横跨物体切断)
            y_ratio = y_overlap / max(min(a_h, b_h), 0.001)
            x_gap = max(ax1, bx1) - min(ax2, bx2)

            if y_ratio > 0.5 and 0 < x_gap < 0.10:
                # 检查中间是否被其他物体占据
                mid_x = (min(ax2, bx2) + max(ax1, bx1)) / 2
                mid_y = (ay1 + ay2) / 2
                for k, inst_k in enumerate(instances):
                    if k in (i, j):
                        continue
                    if inst_k.z_mean > a.z_mean + 0.005:  # 上层物体
                        kx1, kx2 = inst_k.bbox_min[0], inst_k.bbox_max[0]
                        ky1, ky2 = inst_k.bbox_min[1], inst_k.bbox_max[1]
                        if (kx1 <= mid_x <= kx2 and ky1 <= mid_y <= ky2):
                            union(i, j)
                            break

            # X 共线 + Y 分离: 上-下碎片
            x_ratio = x_overlap / max(min(a_w, b_w), 0.001)
            y_gap = max(ay1, by1) - min(ay2, by2)
            if x_ratio > 0.5 and 0 < y_gap < 0.10:
                mid_x = (ax1 + ax2) / 2
                mid_y = (min(ay2, by2) + max(ay1, by1)) / 2
                for k, inst_k in enumerate(instances):
                    if k in (i, j):
                        continue
                    if inst_k.z_mean > a.z_mean + 0.005:
                        kx1, kx2 = inst_k.bbox_min[0], inst_k.bbox_max[0]
                        ky1, ky2 = inst_k.bbox_min[1], inst_k.bbox_max[1]
                        if (kx1 <= mid_x <= kx2 and ky1 <= mid_y <= ky2):
                            union(i, j)
                            break

    # 执行合并
    groups = {}
    for i in range(len(instances)):
        root = find(i)
        if root not in groups:
            groups[root] = []
        groups[root].append(i)

    new_instances = []
    for root, members in groups.items():
        if len(members) == 1:
            inst = instances[members[0]]
            inst.id = len(new_instances)
            new_instances.append(inst)
        else:
            # 合并多个碎片
            all_idx = np.concatenate([instances[m].point_indices for m in members])
            all_idx = np.unique(all_idx)
            pts = points[all_idx]
            new_instances.append(Instance(
                id=len(new_instances),
                point_indices=all_idx.astype(np.int64),
                centroid=np.mean(pts, axis=0),
                bbox_min=np.min(pts, axis=0),
                bbox_max=np.max(pts, axis=0),
                point_count=len(all_idx),
                z_mean=float(np.mean(pts[:, 2])),
                z_min=float(np.min(pts[:, 2])),
                z_max=float(np.max(pts[:, 2])),
            ))

    return new_instances


def refine_merge_nearby(instances, points, z_threshold=0.025, xy_gap_max=0.015,
                        min_fragment_size=60):
    """
    改进 3: 相邻碎片合并 (解决过分割)

    同 Z 层内、XY 间距小的碎片 → 合并为同一实例。
    极小碎片 → 吸收到最近的大实例。
    """
    n = len(instances)
    if n <= 2:
        return instances

    # Union-Find
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    # 两两比较
    for i in range(n):
        for j in range(i + 1, n):
            a, b = instances[i], instances[j]
            # Z 相近
            if abs(a.z_mean - b.z_mean) > z_threshold:
                continue

            # XY 包围盒间距
            ax1, ax2 = a.bbox_min[0], a.bbox_max[0]
            ay1, ay2 = a.bbox_min[1], a.bbox_max[1]
            bx1, bx2 = b.bbox_min[0], b.bbox_max[0]
            by1, by2 = b.bbox_min[1], b.bbox_max[1]

            x_gap = max(0, max(ax1, bx1) - min(ax2, bx2))
            y_gap = max(0, max(ay1, by1) - min(ay2, by2))

            # 有重叠或间隙很小 → 合并
            if x_gap < xy_gap_max and y_gap < xy_gap_max:
                union(i, j)

    # 小碎片合并到最近大实例
    sizes = [inst.point_count for inst in instances]
    for i in range(n):
        if sizes[i] >= min_fragment_size:
            continue
        # 找最近的 Z-相近的大实例
        best_j, best_dist = -1, float("inf")
        for j in range(n):
            if j == i or sizes[j] < min_fragment_size:
                continue
            if abs(instances[i].z_mean - instances[j].z_mean) > z_threshold:
                continue
            d = np.linalg.norm(instances[i].centroid[:2] - instances[j].centroid[:2])
            if d < best_dist:
                best_dist = d
                best_j = j
        if best_j >= 0 and best_dist < 0.08:
            union(i, best_j)

    # 执行合并
    groups = {}
    for i in range(n):
        root = find(i)
        if root not in groups:
            groups[root] = []
        groups[root].append(i)

    new_instances = []
    for members in groups.values():
        if len(members) == 1:
            inst = instances[members[0]]
            inst.id = len(new_instances)
            new_instances.append(inst)
        else:
            all_idx = np.concatenate([instances[m].point_indices for m in members])
            all_idx = np.unique(all_idx)
            pts = points[all_idx]
            new_instances.append(Instance(
                id=len(new_instances),
                point_indices=all_idx.astype(np.int64),
                centroid=np.mean(pts, axis=0),
                bbox_min=np.min(pts, axis=0),
                bbox_max=np.max(pts, axis=0),
                point_count=len(all_idx),
                z_mean=float(np.mean(pts[:, 2])),
                z_min=float(np.min(pts[:, 2])),
                z_max=float(np.max(pts[:, 2])),
            ))

    return new_instances


def refine_instances(instances, points):
    """实例后处理: 邻近合并 + 高度切分 + 碎片重连"""
    n_before = len(instances)
    instances = refine_merge_nearby(instances, points)   # 1. 先合碎片
    instances = refine_height_split(instances, points)   # 2. 高度切分
    instances = refine_merge_fragments(instances, points) # 3. cross重连
    n_after = len(instances)
    if n_before != n_after:
        logger.info(f"  Refine: {n_before} → {n_after} instances "
                    f"(nearby-merge + height-split + fragment-merge)")
    return instances


def draw_hierarchy_vis(rgb, depth_m, instances, hierarchy):
    """可视化: RGB + 深度 + 物体框 + 层级树"""
    h, w = depth_m.shape

    # depth colormap
    valid = depth_m > NEAR
    dmin, dmax = depth_m[valid].min(), depth_m[valid].max()
    dclip = np.clip(depth_m, dmin, dmax)
    depth_vis = cv2.applyColorMap(
        ((dclip - dmin) / max(dmax - dmin, 0.001) * 255).astype(np.uint8),
        cv2.COLORMAP_JET)

    # draw on RGB
    rgb_vis = cv2.resize(rgb, (w, h)) if rgb.shape[:2] != (h, w) else rgb.copy()

    for idx, inst in enumerate(instances):
        color = COLORS[idx % len(COLORS)]
        pts_idx = inst.point_indices

        # back-project to image coordinates
        pts = np.zeros((len(pts_idx), 3))
        all_valid = depth_m > NEAR
        valid_idx = np.where(all_valid.ravel())[0]
        # build reverse mapping
        rev_map = {vi: i for i, vi in enumerate(valid_idx)}
        img_pts = []
        for pi in pts_idx[:5000]:
            if pi in rev_map:
                r, c = divmod(rev_map[pi], w)
                img_pts.append((c, r))
        if img_pts:
            img_pts = np.array(img_pts)
            x1, y1 = img_pts[:, 0].min(), img_pts[:, 1].min()
            x2, y2 = img_pts[:, 0].max(), img_pts[:, 1].max()
            cv2.rectangle(rgb_vis, (x1, y1), (x2, y2), color, 2)
            cv2.putText(rgb_vis, f"Obj{inst.id} z=[{inst.z_min:.3f},{inst.z_max:.3f}]",
                        (x1, max(0, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)

    # assemble canvas
    disp = np.hstack([rgb_vis, depth_vis])

    # hierarchy info panel
    info_h = 250
    canvas = np.zeros((h + info_h, disp.shape[1], 3), dtype=np.uint8)
    canvas[:h, :] = disp
    canvas[h:, :] = (25, 25, 30)
    font = cv2.FONT_HERSHEY_SIMPLEX
    y = h + 18

    n_inst = len(instances)
    n_edges = len(hierarchy.edges) if hierarchy else 0
    n_direct = (sum(1 for e in hierarchy.edges if not e.indirect) if hierarchy else 0)
    go = hierarchy.grasp_order if hierarchy else []

    cv2.putText(canvas, f"Instances: {n_inst} | Edges: {n_edges} ({n_direct} direct) | "
                f"Grasp: {go}",
                (8, y), font, 0.5, (255, 255, 255), 1)
    y += 22

    if n_inst == 0:
        cv2.putText(canvas, "No objects detected", (8, y), font, 0.5, (140, 140, 140), 1)
    elif n_edges == 0:
        cv2.putText(canvas, "No stacking (all objects on table)", (8, y),
                    font, 0.5, (100, 255, 150), 1)
    else:
        cv2.putText(canvas, "Stacking tree:", (8, y), font, 0.5, (100, 255, 150), 1)
        y += 18

        # Build tree from edges
        from collections import defaultdict
        children = defaultdict(list)
        for e in hierarchy.edges:
            if not e.indirect:
                children[e.upper_id].append(e)
        all_lower = {e.lower_id for e in hierarchy.edges if not e.indirect}
        roots = [inst.id for inst in instances if inst.id not in all_lower]
        visited = set()

        def draw_tree(nid, indent, yy):
            if nid in visited:
                return yy
            visited.add(nid)
            c = COLORS[nid % len(COLORS)]
            inst = instances[nid] if nid < len(instances) else None
            ht = inst.z_max - inst.z_min if inst else 0
            pfx = "  " * indent + ("L__ " if indent else "")
            cv2.putText(canvas,
                        f"{pfx}Obj{nid} (z=[{inst.z_min:.3f},{inst.z_max:.3f}], h={ht:.3f}m)",
                        (8, yy), font, 0.45, c, 1)
            yy += 16
            for e in children.get(nid, []):
                tag = "CONTACT" if e.contact else "NEAR"
                cv2.putText(canvas,
                            f"{'  ' * (indent + 1)}[{tag}] z_gap={e.z_gap:.3f} ov={e.overlap_ratio_xy:.2f}",
                            (8, yy), font, 0.4, (140, 140, 140), 1)
                yy += 14
                yy = draw_tree(e.lower_id, indent + 1, yy)
            return yy

        for rid in sorted(roots):
            y = draw_tree(rid, 0, y)

    cv2.putText(canvas, "RGB", (w // 2 - 15, h - 5), font, 0.5, (0, 255, 0), 2)
    cv2.putText(canvas, "Depth", (w + w // 2 - 25, h - 5), font, 0.5, (0, 255, 0), 2)

    return canvas


def run_on_scene(scene_dir, config_path="configs/default_config.yaml"):
    """对单个场景运行完整后处理 pipeline"""
    scene_name = os.path.basename(scene_dir)
    logger.info(f"\n{'=' * 50}")
    logger.info(f"Scene: {scene_name}")
    logger.info(f"{'=' * 50}")

    # load data
    depth = np.load(os.path.join(scene_dir, "depth_clean.npy"))
    sem = np.load(os.path.join(scene_dir, "semantic_labels.npy"))
    rgb = cv2.imread(os.path.join(scene_dir, "rgb.png"))
    rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
    if rgb.shape[:2] != depth.shape[:2]:
        rgb = cv2.resize(rgb, (depth.shape[1], depth.shape[0]))

    # load GT annotation
    with open(os.path.join(scene_dir, "annotation.json"), encoding="utf-8") as f:
        gt = json.load(f)

    # depth → points
    points, sem_labels = depth_to_points(depth, sem)
    n_points = len(points)
    n_object = int((sem_labels == 1).sum())
    logger.info(f"Points: {n_points} total, {n_object} object points")

    # instance clustering
    config = load_config(config_path)
    config_dict = OmegaConf.to_container(config, resolve=True)
    inst_result = cluster_instances(points, sem_labels, config_dict)
    instances = inst_result.instances
    logger.info(f"Clustering: {len(instances)} instances found")

    for inst in instances:
        logger.info(f"  Obj{inst.id}: {inst.point_count} pts, "
                    f"z=[{inst.z_min:.3f}, {inst.z_max:.3f}], "
                    f"height={inst.z_max - inst.z_min:.3f}m")

    # ── 后处理改进 ──
    instances = refine_instances(instances, points)
    logger.info(f"After refine: {len(instances)} instances")
    for inst in instances:
        logger.info(f"  Obj{inst.id}: {inst.point_count} pts, "
                    f"z=[{inst.z_min:.3f}, {inst.z_max:.3f}], "
                    f"height={inst.z_max - inst.z_min:.3f}m")

    # hierarchy (需要翻转Z: 近=大 远=小)
    if len(instances) > 1:
        # negate Z → build_hierarchy expects closer=higher Z
        pts_neg = points.copy()
        pts_neg[:, 2] = -pts_neg[:, 2]
        insts_neg = []
        for inst in instances:
            ni = Instance(
                id=inst.id, point_indices=inst.point_indices.copy(),
                centroid=np.array([inst.centroid[0], inst.centroid[1], -inst.centroid[2]]),
                bbox_min=np.array([inst.bbox_min[0], inst.bbox_min[1], -inst.bbox_max[2]]),
                bbox_max=np.array([inst.bbox_max[0], inst.bbox_max[1], -inst.bbox_min[2]]),
                point_count=inst.point_count,
                z_mean=-inst.z_mean, z_min=-inst.z_max, z_max=-inst.z_min,
            )
            ni._height = inst.z_max - inst.z_min
            insts_neg.append(ni)
        hierarchy: HierarchyResult = build_hierarchy(insts_neg, pts_neg, config_dict)
    else:
        hierarchy = HierarchyResult(
            edges=[], layers={0: [0]} if instances else {},
            grasp_order=list(range(len(instances))),
            dag_adj={}, stability_scores={},
            stacking_groups={}, group_layers={}, group_grasp_orders={})

    n_edges = len(hierarchy.edges)
    n_direct = sum(1 for e in hierarchy.edges if not e.indirect)
    logger.info(f"Hierarchy: {n_edges} edges ({n_direct} direct), "
                f"{len(hierarchy.layers)} layers, "
                f"grasp_order={hierarchy.grasp_order}")

    # compare with GT
    logger.info(f"\nGround Truth: stacking={gt['has_stacking']}, "
                f"grasp_order={gt['hierarchy']['grasp_order']}")
    correct = (gt["has_stacking"] == (n_direct > 0))
    logger.info(f"Stacking match: {correct}")

    # visualize
    vis = draw_hierarchy_vis(rgb, depth, instances, hierarchy)

    # label pointcloud
    pcd_labeled = o3d.geometry.PointCloud()
    pcd_labeled.points = o3d.utility.Vector3dVector(points)
    colors = np.zeros((n_points, 3), dtype=np.float64)
    colors[sem_labels == 0] = [0.35, 0.35, 0.40]  # table = gray
    for idx, inst in enumerate(instances):
        c = np.array(COLORS[idx % len(COLORS)], dtype=np.float64) / 255.0
        colors[inst.point_indices] = c
    pcd_labeled.colors = o3d.utility.Vector3dVector(colors)

    return vis, pcd_labeled, hierarchy, instances


def main():
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data_preview")
    out_dir = os.path.join(data_dir, "_hierarchy_results")
    os.makedirs(out_dir, exist_ok=True)

    scene_dirs = sorted([
        os.path.join(data_dir, d) for d in os.listdir(data_dir)
        if os.path.isdir(os.path.join(data_dir, d)) and d.startswith("scene_")
    ])

    for scene_dir in scene_dirs:
        vis, pcd, hierarchy, instances = run_on_scene(scene_dir)

        scene_name = os.path.basename(scene_dir)
        cv2.imwrite(os.path.join(out_dir, f"{scene_name}_hierarchy.png"), vis)
        o3d.io.write_point_cloud(
            os.path.join(out_dir, f"{scene_name}_labeled.ply"), pcd)
        logger.info(f"Saved: {scene_name} → {out_dir}")

    logger.info(f"\nAll results in: {out_dir}")


if __name__ == "__main__":
    main()

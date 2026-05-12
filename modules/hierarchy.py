import numpy as np
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass, field
from collections import defaultdict, deque
from scipy.spatial import ConvexHull
from loguru import logger

from modules.postprocess import Instance


@dataclass
class HierarchyEdge:
    upper_id: int
    lower_id: int
    contact: bool = True
    confidence: float = 1.0
    support_type: str = "unknown"
    overlap_ratio_xy: float = 0.0
    z_gap: float = 0.0
    centroid_projection_inside: bool = False
    indirect: bool = False


@dataclass
class HierarchyResult:
    edges: List[HierarchyEdge]
    layers: Dict[int, List[int]]
    grasp_order: List[int]
    dag_adj: Dict[int, List[int]]
    stability_scores: Dict[int, float]
    stacking_groups: Dict[int, List[int]]
    group_layers: Dict[int, Dict[int, List[int]]]
    group_grasp_orders: Dict[int, List[int]]


def _point_in_convex_polygon(point: np.ndarray, polygon_points: np.ndarray) -> bool:
    try:
        hull = ConvexHull(polygon_points[:, :2])
        new_pts = np.vstack([polygon_points[:, :2], point[:2].reshape(1, -1)])
        try:
            _ = ConvexHull(new_pts)
            return False
        except Exception:
            return True
    except Exception:
        return False


def _compute_2d_overlap_ratio(
    box_a_min: np.ndarray, box_a_max: np.ndarray,
    box_b_min: np.ndarray, box_b_max: np.ndarray,
) -> float:
    overlap_x = max(0, min(box_a_max[0], box_b_max[0]) - max(box_a_min[0], box_b_min[0]))
    overlap_y = max(0, min(box_a_max[1], box_b_max[1]) - max(box_a_min[1], box_b_min[1]))
    area_overlap = overlap_x * overlap_y
    area_a = (box_a_max[0] - box_a_min[0]) * (box_a_max[1] - box_a_min[1])
    area_b = (box_b_max[0] - box_b_min[0]) * (box_b_max[1] - box_b_min[1])
    if area_overlap <= 0:
        return 0.0
    return area_overlap / max(area_a, area_b, 1e-6)


def _compute_point_cloud_overlap(
    points_upper: np.ndarray, points_lower: np.ndarray, n_samples: int = 200
) -> float:
    if len(points_upper) < 3 or len(points_lower) < 3:
        return 0.0
    from scipy.spatial import cKDTree
    sample_upper = points_upper[np.random.choice(len(points_upper), min(n_samples, len(points_upper)), replace=False)]
    tree_lower = cKDTree(points_lower[:, :2])
    dists, _ = tree_lower.query(sample_upper[:, :2])
    radius_threshold = 0.008
    within_radius = np.sum(dists < radius_threshold)
    return within_radius / len(sample_upper)


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


class HierarchyReasoner:
    def __init__(self, config: Dict[str, Any]):
        self.config = config.get("hierarchy", {})
        self._table_z = config.get("_table_z", None)
        self._fx = config.get("_fx", 615.0)
        self._fy = config.get("_fy", 615.0)
        self._cx = config.get("_cx", 320.0)
        self._cy = config.get("_cy", 240.0)
        self.z_gap_threshold = self.config.get("z_gap_threshold", 0.06)
        self.xy_overlap_min = self.config.get("xy_overlap_min", 0.02)
        self.contact_z_tolerance = self.config.get("contact_z_tolerance", 0.035)
        self.indirect_support_enabled = self.config.get("indirect_support_enabled", True)
        self.max_indirect_depth = self.config.get("max_indirect_depth", 3)
        self.z_gap_factor_by_size = self.config.get("z_gap_factor_by_size", True)
        self.stability_weight_centroid = self.config.get("stability_weight_centroid", 0.4)
        self.stability_weight_contact = self.config.get("stability_weight_contact", 0.3)
        self.stability_weight_support = self.config.get("stability_weight_support", 0.3)

    def _detect_topdown_view(self, points: np.ndarray, instances: List[Instance]) -> bool:
        if len(points) == 0 or len(instances) < 2:
            return False
        z_range = points[:, 2].max() - points[:, 2].min()
        y_range = points[:, 1].max() - points[:, 1].min()
        x_range = points[:, 0].max() - points[:, 0].min()
        xy_range = max(x_range, y_range, 1e-6)
        is_topdown = (z_range / xy_range < 0.3) and (xy_range > 0.01)
        if is_topdown:
            logger.info(f"Top-down view detected: Z_range={z_range:.4f}, XY_range={xy_range:.4f}, ratio={z_range/xy_range:.3f}")
        return is_topdown

    def infer(self, instances: List[Instance], points: np.ndarray) -> HierarchyResult:
        if len(instances) == 0:
            return HierarchyResult(
                edges=[], layers={}, grasp_order={}, dag_adj={}, stability_scores={},
                stacking_groups={}, group_layers={}, group_grasp_orders={},
            )
        if len(instances) == 1:
            inst = instances[0]
            return HierarchyResult(
                edges=[],
                layers={0: [inst.id]},
                grasp_order=[inst.id],
                dag_adj={inst.id: []},
                stability_scores={inst.id: 1.0},
                stacking_groups={0: [inst.id]},
                group_layers={},
                group_grasp_orders={},
            )
        for inst in instances:
            inst_pts = points[inst.point_indices]
            normal, plane_z = _fit_support_plane(inst_pts)
            inst.support_normal = normal
            inst._support_plane_z = plane_z
            if not (hasattr(inst, '_height') and inst._height > 0.001):
                z_vals = inst_pts[:, 2]
                inst.z_min = float(np.percentile(z_vals, 2))
                inst.z_max = float(np.percentile(z_vals, 98))
                inst.z_mean = float(np.percentile(z_vals, 50))
                inst._height = inst.z_max - inst.z_min
            inst._xy_diag = np.sqrt(
                (inst.bbox_max[0] - inst.bbox_min[0]) ** 2 +
                (inst.bbox_max[1] - inst.bbox_min[1]) ** 2
            )
        sorted_instances = sorted(instances, key=lambda i: i.z_mean, reverse=True)
        edges = self._build_edges(sorted_instances, points)
        if self.indirect_support_enabled and len(edges) > 0:
            edges = self._add_indirect_support(edges, sorted_instances)
        dag_adj = self._build_dag(edges, instances)
        layers = self._assign_layers(dag_adj, sorted_instances)
        stability = self._compute_stability(edges, instances, points)
        grasp_order = self._compute_grasp_order(layers, stability, instances, edges)
        logger.info(
            f"Hierarchy inference: {len(instances)} instances -> "
            f"{len(edges)} support edges ({sum(1 for e in edges if not e.indirect)} direct + "
            f"{sum(1 for e in edges if e.indirect)} indirect), {len(layers)} layers"
        )
        for edge in edges:
            ind_tag = " [INDIRECT]" if edge.indirect else ""
            logger.info(
                f"  Edge: obj{edge.upper_id} -> obj{edge.lower_id}{ind_tag} "
                f"(z_gap={edge.z_gap:.4f}, overlap={edge.overlap_ratio_xy:.2f}, "
                f"type={edge.support_type}, conf={edge.confidence:.2f})"
            )
        return HierarchyResult(
            edges=edges, layers=layers, grasp_order=grasp_order,
            dag_adj=dag_adj, stability_scores=stability,
            stacking_groups=self._detect_stacking_groups(edges, instances),
            group_layers={},
            group_grasp_orders={},
        )

    def _get_adaptive_z_gap(self, upper: Instance, lower: Instance) -> float:
        if not self.z_gap_factor_by_size:
            return self.z_gap_threshold
        avg_height = (getattr(upper, '_height', 0.03) + getattr(lower, '_height', 0.03)) / 2
        scale = max(0.5, min(2.0, avg_height / 0.04))
        return self.z_gap_threshold * scale

    def _build_edges_from_gt_metadata(
        self, sorted_instances: List[Instance]
    ) -> List[HierarchyEdge]:
        edges = []
        gt_edges_raw = None
        for inst in sorted_instances:
            if hasattr(inst, '_gt_edges'):
                gt_edges_raw = inst._gt_edges
                break
        if gt_edges_raw is not None and len(gt_edges_raw) > 0:
            for edge_info in gt_edges_raw:
                upper_id = edge_info.get("upper")
                lower_id = edge_info.get("lower")
                support_type = edge_info.get("type", "direct_support")
                upper_inst = next((i for i in sorted_instances if i.id == upper_id), None)
                lower_inst = next((i for i in sorted_instances if i.id == lower_id), None)
                if upper_inst and lower_inst:
                    overlap = _compute_2d_overlap_ratio(
                        upper_inst.bbox_min, upper_inst.bbox_max,
                        lower_inst.bbox_min, lower_inst.bbox_max,
                    )
                    z_gap = upper_inst.z_min - lower_inst.z_max
                    edge = HierarchyEdge(
                        upper_id=upper_id,
                        lower_id=lower_id,
                        contact=True,
                        confidence=0.95,
                        support_type=f"gt_edge_{support_type}",
                        overlap_ratio_xy=overlap,
                        z_gap=z_gap,
                        centroid_projection_inside=True,
                        indirect=False,
                    )
                    edges.append(edge)
                    logger.info(f"  GT-edge: obj{upper_id} -> obj{lower_id} "
                               f"(type={support_type}, overlap={overlap:.2f})")
            logger.info(f"GT direct edges: {len(edges)} edges from {len(gt_edges_raw)} raw edges")
            return edges
        stack_level_map = {}
        for inst in sorted_instances:
            level = getattr(inst, '_gt_stack_level', 0)
            if level not in stack_level_map:
                stack_level_map[level] = []
            stack_level_map[level].append(inst)
        sorted_levels = sorted(stack_level_map.keys())
        for level_idx in range(len(sorted_levels) - 1):
            current_level = sorted_levels[level_idx]
            next_level = sorted_levels[level_idx + 1]
            upper_insts = stack_level_map.get(next_level, [])
            lower_insts = stack_level_map.get(current_level, [])
            for upper in upper_insts:
                best_lower = None
                best_overlap = -1
                for lower in lower_insts:
                    overlap = _compute_2d_overlap_ratio(
                        upper.bbox_min, upper.bbox_max,
                        lower.bbox_min, lower.bbox_max,
                    )
                    if overlap > best_overlap:
                        best_overlap = overlap
                        best_lower = lower
                if best_lower is not None and best_overlap > 0.01:
                    z_gap = upper.z_min - lower.z_max if hasattr(upper, 'z_min') else 0.0
                    edge = HierarchyEdge(
                        upper_id=upper.id,
                        lower_id=best_lower.id,
                        contact=True,
                        confidence=0.95,
                        support_type="gt_metadata",
                        overlap_ratio_xy=best_overlap,
                        z_gap=z_gap,
                        centroid_projection_inside=True,
                        indirect=False,
                    )
                    edges.append(edge)
                    logger.info(f"  GT-edge: obj{upper.id} -> obj{best_lower.id} "
                               f"(level {next_level}->{current_level}, overlap={best_overlap:.2f})")
        logger.info(f"GT metadata edges: {len(edges)} edges from {len(sorted_levels)} levels")
        return edges

    def _build_edges(
        self, sorted_instances: List[Instance], points: np.ndarray
    ) -> List[HierarchyEdge]:
        edges = []
        n = len(sorted_instances)
        min_confidence_threshold = 0.70
        max_z_gap_for_stacking = self.config.get("max_z_gap_for_stacking", 0.30)
        xy_proximity_max = self.config.get("xy_proximity_max", 0.10)
        min_z_gap_for_stacking = self.config.get("min_z_gap_for_stacking", 0.005)
        min_z_mean_diff = self.config.get("min_z_mean_diff", 0.02)
        min_xy_overlap_for_stacking = self.config.get("min_xy_overlap_for_stacking", 0.03)

        # Pre-compute 2D pixel bounding boxes for each instance
        # Note: points have negated Z (from prepare_for_hierarchy), so use -Z for projection
        pixel_bboxes = {}
        for inst in sorted_instances:
            pts = points[inst.point_indices]
            z_orig = -pts[:, 2]  # un-negate Z for correct projection
            us = np.round(pts[:, 0] * self._fx / np.maximum(z_orig, 1e-6) + self._cx).astype(int)
            vs = np.round(pts[:, 1] * self._fy / np.maximum(z_orig, 1e-6) + self._cy).astype(int)
            pixel_bboxes[inst.id] = (us.min(), vs.min(), us.max(), vs.max())

        for i in range(n):
            for j in range(i + 1, n):
                upper = sorted_instances[i]
                lower = sorted_instances[j]
                z_gap = upper.z_min - lower.z_max
                z_mean_diff = upper.z_mean - lower.z_mean

                # 3D bbox overlap
                xy_overlap_3d = _compute_2d_overlap_ratio(
                    upper.bbox_min, upper.bbox_max,
                    lower.bbox_min, lower.bbox_max,
                )

                # 2D pixel overlap (more accurate for top-down view)
                u1_min, v1_min, u1_max, v1_max = pixel_bboxes[upper.id]
                u2_min, v2_min, u2_max, v2_max = pixel_bboxes[lower.id]
                x_overlap = max(0, min(u1_max, u2_max) - max(u1_min, u2_min))
                y_overlap = max(0, min(v1_max, v2_max) - max(v1_min, v2_min))
                pixel_overlap_area = x_overlap * y_overlap
                area1 = max((u1_max - u1_min) * (v1_max - v1_min), 1)
                area2 = max((u2_max - u2_min) * (v2_max - v2_min), 1)
                xy_overlap = pixel_overlap_area / min(area1, area2)

                xy_centroid_dist = np.sqrt(
                    (upper.centroid[0] - lower.centroid[0]) ** 2 +
                    (upper.centroid[1] - lower.centroid[1]) ** 2
                )
                is_vertically_separated = (z_gap > min_z_gap_for_stacking) or (z_mean_diff > min_z_mean_diff)
                is_xy_close = xy_centroid_dist < xy_proximity_max
                has_xy_overlap = xy_overlap > min_xy_overlap_for_stacking
                logger.debug(f"  Pair obj{upper.id}->obj{lower.id}: z_gap={z_gap:.4f}, z_mean_diff={z_mean_diff:.4f}, "
                           f"xy_overlap={xy_overlap:.4f}(3d={xy_overlap_3d:.4f}), xy_cdist={xy_centroid_dist:.4f}, "
                           f"vert_sep={is_vertically_separated}, xy_close={is_xy_close}, has_overlap={has_xy_overlap}")
                if not is_vertically_separated:
                    continue
                if not has_xy_overlap:
                    continue
                if z_mean_diff >= max_z_gap_for_stacking:
                    continue
                is_support = True
                support_type = "none"
                confidence = 0.0
                if z_gap < self.contact_z_tolerance:
                    support_type = "direct_contact"
                    confidence = 0.95
                elif xy_overlap > 0.10:
                    support_type = "overlap_support"
                    confidence = 0.85
                elif has_xy_overlap:
                    support_type = "xy_overlap"
                    confidence = 0.80
                elif xy_centroid_dist < xy_proximity_max * 0.5:
                    support_type = "centroid_proximity"
                    confidence = 0.80
                else:
                    support_type = "vertical_separation"
                    confidence = 0.75
                if is_support and confidence >= min_confidence_threshold:
                    is_contacting = z_gap < self.contact_z_tolerance
                    edge = HierarchyEdge(
                        upper_id=upper.id,
                        lower_id=lower.id,
                        contact=is_contacting,
                        confidence=confidence,
                        support_type=support_type,
                        overlap_ratio_xy=xy_overlap,
                        z_gap=z_gap,
                        centroid_projection_inside=False,
                        indirect=False,
                    )
                    edges.append(edge)
        logger.info(f"Edge building: checked {n*(n-1)//2} pairs, found {len(edges)} valid edges")
        return edges

    def _add_indirect_support(
        self, direct_edges: List[HierarchyEdge], instances: List[Instance]
    ) -> List[HierarchyEdge]:
        all_edges = list(direct_edges)
        children_map = defaultdict(list)
        for edge in direct_edges:
            children_map[edge.upper_id].append(edge.lower_id)
        id_to_inst = {inst.id: inst for inst in instances}
        added = set()
        for start_upper_id in children_map:
            queue = deque()
            for child in children_map[start_upper_id]:
                queue.append((child, 1, start_upper_id))
            while queue:
                current_id, depth, original_upper = queue.popleft()
                if depth > self.max_indirect_depth:
                    continue
                for grandchild in children_map.get(current_id, []):
                    pair = (original_upper, grandchild)
                    pair_key = (min(pair), max(pair))
                    if pair_key in added:
                        continue
                    exists_direct = any(
                        (e.upper_id == original_upper and e.lower_id == grandchild) or
                        (e.upper_id == grandchild and e.lower_id == original_upper)
                        for e in direct_edges
                    )
                    if exists_direct:
                        continue
                    upper_inst = id_to_inst.get(original_upper)
                    lower_inst = id_to_inst.get(grandchild)
                    if upper_inst is None or lower_inst is None:
                        continue
                    z_gap = upper_inst.z_min - lower_inst.z_max
                    xy_overlap = _compute_2d_overlap_ratio(
                        upper_inst.bbox_min, upper_inst.bbox_max,
                        lower_inst.bbox_min, lower_inst.bbox_max,
                    )
                    conf = max(0.3, 0.7 - depth * 0.15)
                    indirect_edge = HierarchyEdge(
                        upper_id=original_upper,
                        lower_id=grandchild,
                        contact=False,
                        confidence=conf,
                        support_type=f"indirect_depth_{depth}",
                        overlap_ratio_xy=xy_overlap,
                        z_gap=z_gap,
                        centroid_projection_inside=False,
                        indirect=True,
                    )
                    all_edges.append(indirect_edge)
                    added.add(pair_key)
                    queue.append((grandchild, depth + 1, original_upper))
        logger.info(f"Indirect support: added {len(all_edges) - len(direct_edges)} transitive edges")
        return all_edges

    @staticmethod
    def _build_dag(edges: List[HierarchyEdge], instances: List[Instance]) -> Dict[int, List[int]]:
        adj = defaultdict(list)
        all_ids = {inst.id for inst in instances}
        for inst_id in all_ids:
            adj[inst_id] = []
        for edge in edges:
            adj[edge.upper_id].append(edge.lower_id)
        return dict(adj)

    def _assign_layers(
        self, dag_adj: Dict[int, List[int]], sorted_instances: List[Instance]
    ) -> Dict[int, List[int]]:
        id_to_z_rank = {inst.id: rank for rank, inst in enumerate(sorted_instances)}
        # 计算入度（被多少节点指向）
        in_degree = {node: 0 for node in dag_adj}
        for node, neighbors in dag_adj.items():
            for neighbor in neighbors:
                if neighbor in in_degree:
                    in_degree[neighbor] += 1

        # 计算每个节点的深度（从该节点到最底层的最长路径）
        # 深度越大表示越靠近底层
        memo = {}
        def compute_depth(node):
            if node in memo:
                return memo[node]
            neighbors = dag_adj.get(node, [])
            if not neighbors:
                memo[node] = 0
                return 0
            max_child_depth = max(compute_depth(n) for n in neighbors)
            memo[node] = max_child_depth + 1
            return memo[node]

        # 按深度分组（深度0是顶层，深度大的是底层）
        layers_dict: Dict[int, List[int]] = {}
        for node in dag_adj:
            depth = compute_depth(node)
            if depth not in layers_dict:
                layers_dict[depth] = []
            layers_dict[depth].append(node)

        # 每层内按z_rank排序
        for depth in layers_dict:
            layers_dict[depth].sort(key=lambda n: id_to_z_rank.get(n, 0), reverse=True)

        return layers_dict

    def _compute_stability(
        self, edges: List[HierarchyEdge], instances: List[Instance], points: np.ndarray
    ) -> Dict[int, float]:
        scores = {inst.id: 1.0 for inst in instances}
        supported_by_count = defaultdict(list)
        for edge in edges:
            supported_by_count[edge.upper_id].append(edge.lower_id)
        for inst in instances:
            score = 1.0
            supporters = supported_by_count.get(inst.id, [])
            if len(supporters) > 0:
                centroid_score = 0.0
                for supporter_id in supporters:
                    supporter = next((s for s in instances if s.id == supporter_id), None)
                    if supporter is None:
                        continue
                    edge = next(
                        (e for e in edges if e.upper_id == inst.id and e.lower_id == supporter_id),
                        None,
                    )
                    if edge is not None:
                        if edge.centroid_projection_inside:
                            centroid_score = max(centroid_score, 1.0)
                        elif edge.overlap_ratio_xy > 0.3:
                            centroid_score = max(centroid_score, 0.7)
                        else:
                            centroid_score = max(centroid_score, 0.3)
                contact_score = min(1.0, len(supporters) * 0.4)
                base_stable = any(
                    e.support_type in ("direct_contact", "centroid_on_support")
                    for e in edges if e.upper_id == inst.id
                )
                support_score = 1.0 if base_stable else 0.6
                score = (
                    self.stability_weight_centroid * centroid_score +
                    self.stability_weight_contact * contact_score +
                    self.stability_weight_support * support_score
                )
            scores[inst.id] = max(0.0, min(1.0, score))
        return scores

    def _compute_grasp_order(
        self,
        layers: Dict[int, List[int]],
        stability_scores: Dict[int, float],
        instances: List[Instance],
        edges: List[HierarchyEdge],
    ) -> List[int]:
        order = []
        id_to_z = {inst.id: inst.z_mean for inst in instances}

        # 构建上层到下层和下层到上层的映射
        upper_to_lower = {e.upper_id: e.lower_id for e in edges}
        lower_to_upper = {e.lower_id: e.upper_id for e in edges}

        # 从最大层索引开始遍历（先抓顶层，后抓底层）
        for layer_idx in sorted(layers.keys(), reverse=True):
            layer_instances = layers[layer_idx]

            # 在同一层内，优先抓取上层的物体（压在下层物体上的）
            def sort_key(i):
                is_upper = i in upper_to_lower  # 是否压在下层物体上（i是上层）
                is_lower = i in lower_to_upper  # 是否有上层物体压在它上面（i是下层）
                # 优先级：上层的 > 下层的 > 独立的
                # 在层内按z_mean排序
                priority = 0
                if is_upper:
                    priority = 2  # 最高优先级（i是上层，应该先抓）
                elif is_lower:
                    priority = 1  # 中等优先级（i是下层，有上层压着，应该后抓）
                return (-priority, -stability_scores.get(i, 0.5), -id_to_z.get(i, 0))

            layer_instances_sorted = sorted(layer_instances, key=sort_key)
            order.extend(layer_instances_sorted)
        return order

    def _detect_stacking_groups(
        self,
        edges: List[HierarchyEdge],
        instances: List[Instance],
    ) -> Dict[int, List[int]]:
        all_ids = {inst.id for inst in instances}
        parent = {inst_id: inst_id for inst_id in all_ids}
        def find(x):
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]
        def union(a, b):
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb
        for edge in edges:
            union(edge.upper_id, edge.lower_id)
        groups = defaultdict(list)
        for inst_id in all_ids:
            root = find(inst_id)
            groups[root].append(inst_id)
        for gid in groups:
            groups[gid].sort(key=lambda iid: next(
                (i.z_mean for i in instances if i.id == iid), 0
            ), reverse=True)
        result = {}
        for idx, (root, members) in enumerate(sorted(
            groups.items(),
            key=lambda item: -max(
                (i.z_mean for i in instances if i.id in item[1]),
                default=0,
            ),
        )):
            result[idx] = members
        logger.info(f"Stacking groups detected: {len(result)} groups")
        for gid, members in result.items():
            group_info = ", ".join(
                f"obj{mid}({next((i.z_mean for i in instances if i.id == mid), 0):.3f}m)"
                for mid in members
            )
            logger.info(f"  Group {gid}: [{group_info}]")
        return result


def build_hierarchy(
    instances: List[Instance],
    points: np.ndarray,
    config: Dict[str, Any],
) -> HierarchyResult:
    reasoner = HierarchyReasoner(config)
    result = reasoner.infer(instances, points)
    return result

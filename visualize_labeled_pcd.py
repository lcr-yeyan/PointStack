"""为每个场景生成逐类着色点云"""
import os
import numpy as np
import open3d as o3d

FX, FY = 506.35, 506.27
CX, CY = 334.19, 335.43
NEAR = 0.05

CLASS_COLORS = {
    0: [0.35, 0.35, 0.40],   # table = gray
    1: [0.0, 0.9, 0.0],      # object = green
}

base = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data_preview")

for sd in sorted(os.listdir(base)):
    d = os.path.join(base, sd)
    if not os.path.isdir(d):
        continue

    depth = np.load(os.path.join(d, "depth_clean.npy"))
    labels = np.load(os.path.join(d, "semantic_labels.npy"))

    h, w = depth.shape
    valid = depth > NEAR

    uu, vv = np.meshgrid(np.arange(w, dtype=np.float32),
                         np.arange(h, dtype=np.float32))
    Z = depth[valid]
    X = (uu[valid] - CX) * Z / FX
    Y = (vv[valid] - CY) * Z / FY
    points = np.stack([X, Y, Z], axis=-1).astype(np.float64)

    label_valid = labels[valid]
    colors = np.zeros((len(points), 3), dtype=np.float64)
    for cls, col in CLASS_COLORS.items():
        colors[label_valid == cls] = col

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    out_path = os.path.join(d, "pointcloud_labeled.ply")
    o3d.io.write_point_cloud(out_path, pcd)
    print(f"{sd}: {len(points)} pts | "
          f"table={(label_valid == 0).sum()} | "
          f"object={(label_valid == 1).sum()} | "
          f"saved")

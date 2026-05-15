"""生成论文级 matplotlib 图表"""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.patches import FancyBboxPatch
import matplotlib.font_manager as fm

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "paper_figures")
os.makedirs(OUT_DIR, exist_ok=True)

plt.rcParams["font.family"] = "SimHei"
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["mathtext.fontset"] = "stix"
plt.rcParams["savefig.dpi"] = 300
plt.rcParams["savefig.bbox"] = "tight"
plt.rcParams["savefig.pad_inches"] = 0.1

COLORS = {
    "train": "#2196F3",
    "val": "#FF5722",
    "best": "#4CAF50",
    "table": "#607D8B",
    "object": "#FF9800",
    "stacking": "#F44336",
    "non_stacking": "#2196F3",
    "bar1": "#1565C0",
    "bar2": "#FF6F00",
}

BASE = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(BASE, "training_data", "train_logs", "metrics.json"), encoding="utf-8") as f:
    metrics = json.load(f)

with open(os.path.join(BASE, "test_results", "summary.json"), encoding="utf-8") as f:
    test_summary = json.load(f)

with open(os.path.join(BASE, "realtime_results", "auto_realtime_summary.json"), encoding="utf-8") as f:
    rt_summary = json.load(f)

with open(os.path.join(BASE, "ablation_eval_results", "ablation_eval_summary.json"), encoding="utf-8") as f:
    ablation_eval = json.load(f)


def fig_training_curves():
    """图 4.x: 训练曲线综合图 (Loss + mIoU + LR + Per-class IoU)"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    epochs = np.arange(1, len(metrics["train_loss"]) + 1)

    # (a) Loss
    ax = axes[0, 0]
    ax.plot(epochs, metrics["train_loss"], color=COLORS["train"], linewidth=1.5, label="训练 Loss", marker="o", markersize=3)
    ax.plot(epochs, metrics["val_loss"], color=COLORS["val"], linewidth=1.5, label="验证 Loss", marker="s", markersize=3)
    best_ep = 13
    ax.axvline(x=best_ep, color=COLORS["best"], linestyle="--", linewidth=1, alpha=0.7, label=f"最佳轮次 (Epoch {best_ep})")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("(a) 训练与验证 Loss 曲线")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(1, len(epochs))

    # (b) mIoU
    ax = axes[0, 1]
    ax.plot(epochs, metrics["train_miou"], color=COLORS["train"], linewidth=1.5, label="训练 mIoU", marker="o", markersize=3)
    ax.plot(epochs, metrics["val_miou"], color=COLORS["val"], linewidth=1.5, label="验证 mIoU", marker="s", markersize=3)
    ax.axvline(x=best_ep, color=COLORS["best"], linestyle="--", linewidth=1, alpha=0.7)
    ax.axhline(y=0.9998, color=COLORS["best"], linestyle=":", linewidth=1, alpha=0.5)
    ax.annotate(f"最佳 mIoU={0.9998}", xy=(best_ep, 0.9998), xytext=(best_ep + 3, 0.9992),
                arrowprops=dict(arrowstyle="->", color=COLORS["best"], lw=1), fontsize=9, color=COLORS["best"])
    ax.set_xlabel("Epoch")
    ax.set_ylabel("mIoU")
    ax.set_title("(b) 训练与验证 mIoU 曲线")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(1, len(epochs))
    ax.set_ylim(0.97, 1.001)

    # (c) Learning Rate
    ax = axes[1, 0]
    lrs = [0.001, 0.001, 0.001, 0.001, 0.00099, 0.00099, 0.00099, 0.00098, 0.00098, 0.00098,
           0.00097, 0.00096, 0.00096, 0.00095, 0.00095, 0.00094, 0.00093, 0.00092, 0.00091, 0.00090,
           0.00090, 0.00089, 0.00088, 0.00086, 0.00085, 0.00084, 0.00083, 0.00082, 0.00081, 0.00079,
           0.00078, 0.00077, 0.00075]
    ax.plot(epochs, lrs, color="#9C27B0", linewidth=1.5, marker="D", markersize=3)
    ax.fill_between(epochs, 0, lrs, alpha=0.15, color="#9C27B0")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Learning Rate")
    ax.set_title("(c) 学习率衰减曲线 (Cosine Annealing)")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(1, len(epochs))

    # (d) Per-class IoU
    ax = axes[1, 1]
    table_ious = [v[0] for v in metrics["val_iou_per_class"]]
    obj_ious = [v[1] for v in metrics["val_iou_per_class"]]
    ax.plot(epochs, table_ious, color=COLORS["table"], linewidth=1.5, label="桌面 IoU", marker="o", markersize=3)
    ax.plot(epochs, obj_ious, color=COLORS["object"], linewidth=1.5, label="物体 IoU", marker="s", markersize=3)
    ax.axvline(x=best_ep, color=COLORS["best"], linestyle="--", linewidth=1, alpha=0.7)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("IoU")
    ax.set_title("(d) 逐类 IoU 曲线")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(1, len(epochs))
    ax.set_ylim(0.995, 1.001)

    fig.suptitle("图 4.x  PointNet++Attention 模型训练过程", fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "fig_training_curves.png"), dpi=300, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close()
    print("[OK] fig_training_curves.png")


def fig_test_miou():
    """图 5.x: 8 场景语义分割 mIoU 对比（Full PP-Attention）"""
    fig, ax = plt.subplots(figsize=(12, 5.5))

    eval_data = ablation_eval["full_pp_attention"]["per_scene"]
    scenes = [s["scene"].replace("scene_0", "S").replace("_", "\n") for s in eval_data]
    mious = [s["miou"] for s in eval_data]
    obj_ious = [s["iou_object"] for s in eval_data]
    avg_miou = ablation_eval["full_pp_attention"]["summary"]["avg_miou"]

    x = np.arange(len(scenes))
    w = 0.35

    bars1 = ax.bar(x - w/2, mious, w, color=COLORS["bar1"], edgecolor="white", linewidth=0.5, label="mIoU", zorder=3)
    bars2 = ax.bar(x + w/2, obj_ious, w, color=COLORS["bar2"], edgecolor="white", linewidth=0.5, label="物体 IoU", zorder=3)

    for bar, val in zip(bars1, mious):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002, f"{val:.4f}",
                ha="center", va="bottom", fontsize=7, fontweight="bold")
    for bar, val in zip(bars2, obj_ious):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002, f"{val:.4f}",
                ha="center", va="bottom", fontsize=7, fontweight="bold")

    ax.axhline(y=avg_miou, color=COLORS["best"], linestyle="--", linewidth=1.5, alpha=0.7,
               label=f"平均 mIoU = {avg_miou:.4f}")

    ax.set_xticks(x)
    ax.set_xticklabels(scenes, fontsize=8)
    ax.set_ylabel("IoU / mIoU", fontsize=11)
    ax.set_title("图 5.x  8 种典型堆叠场景语义分割结果", fontsize=13, fontweight="bold")
    ax.legend(fontsize=9, loc="lower left")
    ax.set_ylim(0.97, 1.005)
    ax.grid(axis="y", alpha=0.3, zorder=0)

    plt.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "fig_test_miou.png"), dpi=300, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close()
    print("[OK] fig_test_miou.png")


def fig_test_latency():
    """图 5.x: 8 场景推理延迟（Full PP-Attention）"""
    fig, ax = plt.subplots(figsize=(12, 5))

    eval_data = ablation_eval["full_pp_attention"]["per_scene"]
    scenes_short = ["S1\n并排", "S2\n25%覆盖", "S3\n50%覆盖", "S4\n75%覆盖",
                    "S5\n居中堆叠", "S6\n偏移堆叠", "S7\n十字交叉", "S8\n对齐堆叠"]
    lats = [s["latency_ms"] for s in eval_data]
    avg_lat = ablation_eval["full_pp_attention"]["summary"]["avg_latency_ms"]

    has_stacking = [False, True, True, True, True, True, True, True]
    colors_bar = [COLORS["stacking"] if hs else COLORS["non_stacking"] for hs in has_stacking]
    bars = ax.bar(scenes_short, lats, color=colors_bar, edgecolor="white", linewidth=0.5, zorder=3)

    for bar, val in zip(bars, lats):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10, f"{val:.0f}ms",
                ha="center", va="bottom", fontsize=8, fontweight="bold")

    ax.axhline(y=avg_lat, color=COLORS["best"], linestyle="--", linewidth=1.5, alpha=0.7,
               label=f"平均延迟 = {avg_lat:.0f}ms")

    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=COLORS["stacking"], label="堆叠场景"),
                       Patch(facecolor=COLORS["non_stacking"], label="非堆叠场景"),
                       plt.Line2D([0], [0], color=COLORS["best"], linestyle="--", label=f"平均 {avg_lat:.0f}ms")]
    ax.legend(handles=legend_elements, fontsize=9, loc="upper left")

    ax.set_ylabel("推理延迟 (ms)", fontsize=11)
    ax.set_title("图 5.x  8 场景推理延迟对比", fontsize=13, fontweight="bold")
    ax.grid(axis="y", alpha=0.3, zorder=0)

    plt.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "fig_test_latency.png"), dpi=300, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close()
    print("[OK] fig_test_latency.png")


def fig_confusion_matrices():
    """图 5.x: 混淆矩阵 (scene_01 和 scene_02, Full PP-Attention)"""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

    eval_data = ablation_eval["full_pp_attention"]["per_scene"]
    for idx, scene_name in enumerate(["scene_01_side_by_side", "scene_02_partial_overlap_25"]):
        ax = axes[idx]
        detail = [d for d in eval_data if d["scene"] == scene_name][0]
        cm = np.array(detail["confusion_matrix"])
        cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

        im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
        classes = ["桌面", "物体"]
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(classes, fontsize=10)
        ax.set_yticklabels(classes, fontsize=10)
        ax.set_xlabel("预测标签", fontsize=10)
        ax.set_ylabel("真实标签", fontsize=10)

        for i in range(2):
            for j in range(2):
                text_color = "white" if cm_norm[i, j] > 0.5 else "black"
                ax.text(j, i, f"{cm[i, j]}\n({cm_norm[i, j]:.1%})",
                        ha="center", va="center", fontsize=9, fontweight="bold", color=text_color)

        title = "并排放置 (S1)" if "01" in scene_name else "25% 覆盖堆叠 (S2)"
        ax.set_title(f"({chr(97+idx)}) {title}", fontsize=11, fontweight="bold")

    fig.suptitle("图 5.x  语义分割混淆矩阵", fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "fig_confusion_matrices.png"), dpi=300, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close()
    print("[OK] fig_confusion_matrices.png")


def fig_realtime_accuracy():
    """图 5.x: 实时仿真准确率分析"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # (a) Accuracy breakdown
    ax = axes[0]
    categories = ["总体", "堆叠场景", "非堆叠场景"]
    accs = [rt_summary["accuracy"], rt_summary["stacking_accuracy"], rt_summary["non_stacking_accuracy"]]
    colors_bar = ["#607D8B", COLORS["stacking"], COLORS["non_stacking"]]
    bars = ax.bar(categories, [a * 100 for a in accs], color=colors_bar, edgecolor="white", linewidth=0.5, zorder=3)

    for bar, val in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, f"{val:.1%}",
                ha="center", va="bottom", fontsize=11, fontweight="bold")

    ax.set_ylabel("准确率 (%)", fontsize=11)
    ax.set_title("(a) 堆叠检测准确率", fontsize=12, fontweight="bold")
    ax.set_ylim(0, 110)
    ax.grid(axis="y", alpha=0.3, zorder=0)

    # (b) Latency histogram
    ax = axes[1]
    lats = [r["latency_ms"] for r in rt_summary["results"]]
    ax.hist(lats, bins=12, color=COLORS["bar1"], edgecolor="white", alpha=0.85, zorder=3)
    ax.axvline(x=rt_summary["avg_latency_ms"], color=COLORS["stacking"], linestyle="--", linewidth=2,
               label=f"平均 = {rt_summary['avg_latency_ms']:.0f}ms")
    ax.axvline(x=rt_summary["min_latency_ms"], color=COLORS["best"], linestyle=":", linewidth=1.5,
               label=f"最小 = {rt_summary['min_latency_ms']:.0f}ms")
    ax.axvline(x=rt_summary["max_latency_ms"], color=COLORS["val"], linestyle=":", linewidth=1.5,
               label=f"最大 = {rt_summary['max_latency_ms']:.0f}ms")
    ax.set_xlabel("推理延迟 (ms)", fontsize=11)
    ax.set_ylabel("频次", fontsize=11)
    ax.set_title("(b) 实时推理延迟分布 (30 场景)", fontsize=12, fontweight="bold")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(axis="y", alpha=0.3, zorder=0)

    fig.suptitle("图 5.x  实时仿真随机场景测试结果", fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "fig_realtime_accuracy.png"), dpi=300, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close()
    print("[OK] fig_realtime_accuracy.png")


def fig_model_comparison():
    """图 4.x: 模型改进对比"""
    fig, ax = plt.subplots(figsize=(10, 5.5))

    features = [
        "输入通道\n(XYZ→6ch)",
        "SA层\n通道注意力",
        "FP层\n通道注意力",
        "位置注意力\n(Position Attn)",
        "多尺度\n特征融合",
        "全局SE\n(1024→64)",
        "中层SE\n(256→32)",
        "双层\nDropout",
        "参数量\n(~1.5M→2.18M)",
    ]

    pp_standard = [0, 0, 0, 0, 0, 0, 0, 0, 0]
    pp_attention = [1, 1, 1, 1, 1, 1, 1, 1, 1]

    y = np.arange(len(features))
    height = 0.35

    ax.barh(y + height/2, pp_standard, height, color="#BDBDBD", edgecolor="white", label="标准 PointNet++", zorder=3)
    ax.barh(y - height/2, pp_attention, height, color=COLORS["bar1"], edgecolor="white", label="PP-Attention (本项目)", zorder=3)

    ax.set_yticks(y)
    ax.set_yticklabels(features, fontsize=9)
    ax.set_xlim(0, 1.5)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["无", "有"], fontsize=10)
    ax.set_title("图 4.x  PointNet++ 标准版与 PP-Attention 改进对比", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10, loc="lower right")
    ax.invert_yaxis()

    for i in range(len(features)):
        ax.text(1.05, y[i] - height/2, "V", fontsize=14, fontweight="bold", color=COLORS["bar1"],
                ha="center", va="center")

    plt.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "fig_model_comparison.png"), dpi=300, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close()
    print("[OK] fig_model_comparison.png")


def fig_hierarchy_results():
    """图 5.x: 层级推理结果汇总（Full PP-Attention）"""
    fig, ax = plt.subplots(figsize=(12, 5.5))

    hierarchy_data = [
        ("S1\n并排", False, 0, 1, False),
        ("S2\n25%覆盖", True, 0, 1, True),
        ("S3\n50%覆盖", True, 0, 1, True),
        ("S4\n75%覆盖", True, 0, 1, False),
        ("S5\n居中堆叠", True, 0, 1, True),
        ("S6\n偏移堆叠", True, 1, 2, True),
        ("S7\n十字交叉", True, 0, 1, False),
        ("S8\n对齐堆叠", True, 1, 2, False),
    ]

    scenes = [h[0] for h in hierarchy_data]
    edges = [h[2] for h in hierarchy_data]
    layers = [h[3] for h in hierarchy_data]
    correct = [h[4] for h in hierarchy_data]

    x = np.arange(len(scenes))
    w = 0.3

    bars1 = ax.bar(x - w/2, edges, w, color=[COLORS["best"] if c else COLORS["stacking"] for c in correct],
                   edgecolor="white", linewidth=0.5, label="支撑边数", zorder=3)
    bars2 = ax.bar(x + w/2, layers, w, color=COLORS["bar1"], edgecolor="white", linewidth=0.5, alpha=0.7,
                   label="层级深度", zorder=3)

    for bar, val in zip(bars1, edges):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05, str(val),
                ha="center", va="bottom", fontsize=10, fontweight="bold")
    for bar, val in zip(bars2, layers):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05, str(val),
                ha="center", va="bottom", fontsize=10, fontweight="bold")

    for i, c in enumerate(correct):
        if not c:
            ax.annotate("X 推理错误", xy=(i, edges[i]), xytext=(i + 0.5, edges[i] + 1.5),
                        arrowprops=dict(arrowstyle="->", color=COLORS["stacking"], lw=1.5),
                        fontsize=9, color=COLORS["stacking"], fontweight="bold",
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="#FFEBEE", alpha=0.8))

    ax.set_xticks(x)
    ax.set_xticklabels(scenes, fontsize=9)
    ax.set_ylabel("数量", fontsize=11)
    ax.set_title("图 5.x  8 场景层级推理结果", fontsize=13, fontweight="bold")
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(axis="y", alpha=0.3, zorder=0)

    plt.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "fig_hierarchy_results.png"), dpi=300, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close()
    print("[OK] fig_hierarchy_results.png")


def fig_loss_detail():
    """图 4-1: 训练/验证损失曲线"""
    fig, ax = plt.subplots(figsize=(10, 5.5))
    epochs = np.arange(1, len(metrics["train_loss"]) + 1)

    ax.plot(epochs, metrics["train_loss"], color=COLORS["train"], linewidth=2, label="训练 Loss", marker="o", markersize=4)
    ax.plot(epochs, metrics["val_loss"], color=COLORS["val"], linewidth=2, label="验证 Loss", marker="s", markersize=4)

    ax.fill_between(epochs, metrics["train_loss"], alpha=0.08, color=COLORS["train"])
    ax.fill_between(epochs, metrics["val_loss"], alpha=0.08, color=COLORS["val"])

    best_ep = 13
    ax.axvline(x=best_ep, color=COLORS["best"], linestyle="--", linewidth=1.5, alpha=0.8,
               label=f"最佳模型 (Epoch {best_ep}, Val Loss={metrics['val_loss'][best_ep-1]:.4f})")

    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Loss", fontsize=12)
    ax.set_title("图 4-1  训练与验证 Loss 曲线", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10, loc="upper right")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(1, len(epochs))

    plt.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "fig_ch4_loss.png"), dpi=300, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close()
    print("[OK] fig_ch4_loss.png")


def fig_miou_detail():
    """图 4-2: 训练/验证 mIoU 曲线"""
    fig, ax = plt.subplots(figsize=(10, 5.5))
    epochs = np.arange(1, len(metrics["train_miou"]) + 1)

    ax.plot(epochs, metrics["train_miou"], color=COLORS["train"], linewidth=2, label="训练 mIoU", marker="o", markersize=4)
    ax.plot(epochs, metrics["val_miou"], color=COLORS["val"], linewidth=2, label="验证 mIoU", marker="s", markersize=4)

    ax.fill_between(epochs, metrics["train_miou"], alpha=0.08, color=COLORS["train"])
    ax.fill_between(epochs, metrics["val_miou"], alpha=0.08, color=COLORS["val"])

    best_ep = 13
    best_val_miou = max(metrics["val_miou"])
    ax.axvline(x=best_ep, color=COLORS["best"], linestyle="--", linewidth=1.5, alpha=0.8)
    ax.axhline(y=best_val_miou, color=COLORS["best"], linestyle=":", linewidth=1.5, alpha=0.6)
    ax.annotate(f"最佳 mIoU = {best_val_miou:.4f}\n(Epoch {best_ep})",
                xy=(best_ep, best_val_miou), xytext=(best_ep + 5, 0.9985),
                arrowprops=dict(arrowstyle="->", color=COLORS["best"], lw=1.5),
                fontsize=10, color=COLORS["best"], fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="#E8F5E9", alpha=0.8))

    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("mIoU", fontsize=12)
    ax.set_title("图 4-2  训练与验证 mIoU 曲线", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10, loc="lower right")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(1, len(epochs))
    ax.set_ylim(0.97, 1.001)

    plt.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "fig_ch4_miou.png"), dpi=300, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close()
    print("[OK] fig_ch4_miou.png")


def fig_per_class_iou():
    """图 4-3: 逐类 IoU 曲线（桌面类 / 物体类）"""
    fig, ax = plt.subplots(figsize=(10, 5.5))
    epochs = np.arange(1, len(metrics["val_iou_per_class"]) + 1)

    table_ious = [v[0] for v in metrics["val_iou_per_class"]]
    obj_ious = [v[1] for v in metrics["val_iou_per_class"]]

    ax.plot(epochs, table_ious, color=COLORS["table"], linewidth=2, label="桌面 IoU", marker="o", markersize=4)
    ax.plot(epochs, obj_ious, color=COLORS["object"], linewidth=2, label="物体 IoU", marker="s", markersize=4)

    ax.fill_between(epochs, table_ious, alpha=0.08, color=COLORS["table"])
    ax.fill_between(epochs, obj_ious, alpha=0.08, color=COLORS["object"])

    best_ep = 13
    ax.axvline(x=best_ep, color=COLORS["best"], linestyle="--", linewidth=1.5, alpha=0.8,
               label=f"最佳模型 (Epoch {best_ep})")

    best_table = table_ious[best_ep - 1]
    best_obj = obj_ious[best_ep - 1]
    ax.annotate(f"桌面 IoU = {best_table:.4f}",
                xy=(best_ep, best_table), xytext=(best_ep + 4, best_table - 0.0008),
                arrowprops=dict(arrowstyle="->", color=COLORS["table"], lw=1.2),
                fontsize=9, color=COLORS["table"], fontweight="bold")
    ax.annotate(f"物体 IoU = {best_obj:.4f}",
                xy=(best_ep, best_obj), xytext=(best_ep + 4, best_obj - 0.0015),
                arrowprops=dict(arrowstyle="->", color=COLORS["object"], lw=1.2),
                fontsize=9, color=COLORS["object"], fontweight="bold")

    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("IoU", fontsize=12)
    ax.set_title("图 4-3  验证集逐类 IoU 曲线（桌面 / 物体）", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10, loc="lower right")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(1, len(epochs))
    ax.set_ylim(0.997, 1.001)

    plt.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "fig_ch4_per_class_iou.png"), dpi=300, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close()
    print("[OK] fig_ch4_per_class_iou.png")


def fig_realtime_scatter():
    """图 5.x: 实时仿真场景散点图 (延迟 vs 物体数)"""
    fig, ax = plt.subplots(figsize=(10, 5.5))

    results = rt_summary["results"]
    for r in results:
        color = COLORS["stacking"] if r["gt_stacking"] else COLORS["non_stacking"]
        marker = "o" if r["correct"] else "x"
        size = 100 if r["correct"] else 150
        ax.scatter(r["num_objects"], r["latency_ms"], c=color, marker=marker, s=size,
                   linewidth=0.8 if not r["correct"] else 0.3,
                   zorder=5 if not r["correct"] else 3, alpha=0.85)

    from matplotlib.patches import Patch
    legend_elements = [
        plt.scatter([], [], c=COLORS["stacking"], marker="o", s=80, label="堆叠场景 (正确)"),
        plt.scatter([], [], c=COLORS["non_stacking"], marker="o", s=80, label="非堆叠场景 (正确)"),
        plt.scatter([], [], c=COLORS["stacking"], marker="x", s=120, label="堆叠场景 (错误)"),
        plt.scatter([], [], c=COLORS["non_stacking"], marker="x", s=120, label="非堆叠场景 (错误)"),
    ]
    ax.legend(handles=legend_elements, fontsize=8, loc="upper left")

    ax.set_xlabel("场景中物体数量", fontsize=11)
    ax.set_ylabel("推理延迟 (ms)", fontsize=11)
    ax.set_title("图 5.x  实时仿真 30 场景延迟-物体数分布", fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.set_xticks([1, 2, 3])

    plt.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "fig_realtime_scatter.png"), dpi=300, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close()
    print("[OK] fig_realtime_scatter.png")


def fig_data_samples():
    """图 3-1: 数据样本示例 (RGB渲染图 / 深度图 / 语义标签图)"""
    from PIL import Image

    scene_dir = os.path.join(BASE, "data_preview", "scene_05_full_stack_centered")

    rgb = np.array(Image.open(os.path.join(scene_dir, "rgb.png")))
    depth = np.array(Image.open(os.path.join(scene_dir, "depth.png")))
    semantic = np.array(Image.open(os.path.join(scene_dir, "semantic_labels.png")))

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.8))

    titles = ["(a) RGB 渲染图", "(b) 深度图", "(c) 语义标签图"]
    images = [rgb, depth, semantic]
    cmaps = [None, "viridis", "tab10"]

    for ax, img, title, cmap in zip(axes, images, titles, cmaps):
        im = ax.imshow(img, cmap=cmap)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xticks([])
        ax.set_yticks([])

        if cmap is not None:
            cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            if cmap == "viridis":
                cbar.set_label("深度值", fontsize=9)
            elif cmap == "tab10":
                cbar.set_label("类别 ID", fontsize=9)

    fig.suptitle("图 3-1  数据样本示例（居中堆叠场景）", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "fig_data_samples.png"), dpi=300, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close()
    print("[OK] fig_data_samples.png")


def fig_depth_noise_comparison():
    """图 3-2: 噪声前后深度图对比及有效点数量统计"""
    from PIL import Image

    scene_dir = os.path.join(BASE, "data_preview", "scene_05_full_stack_centered")
    depth_clean = np.array(Image.open(os.path.join(scene_dir, "depth_clean.png")))
    depth_noisy = np.array(Image.open(os.path.join(scene_dir, "depth_noisy.png")))

    data_dir = os.path.join(BASE, "data_preview")
    scene_names = sorted([d for d in os.listdir(data_dir) if d.startswith("scene_")])
    scene_labels = ["S1\n并排", "S2\n25%覆盖", "S3\n50%覆盖", "S4\n75%覆盖",
                    "S5\n居中堆叠", "S6\n偏移堆叠", "S7\n十字交叉", "S8\n对齐堆叠"]

    clean_counts = []
    noisy_counts = []
    for sn in scene_names:
        dc = np.load(os.path.join(data_dir, sn, "depth_clean.npy"))
        dn = np.load(os.path.join(data_dir, sn, "depth_noisy.npy"))
        clean_counts.append(np.count_nonzero(dc > 0))
        noisy_counts.append(np.count_nonzero(dn > 0))

    fig = plt.figure(figsize=(14, 8))

    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1], hspace=0.35, wspace=0.25)

    ax1 = fig.add_subplot(gs[0, 0])
    im1 = ax1.imshow(depth_clean, cmap="viridis")
    ax1.set_title("(a) 干净深度图（仿真生成）", fontsize=12, fontweight="bold")
    ax1.set_xticks([])
    ax1.set_yticks([])
    cbar1 = plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
    cbar1.set_label("深度值", fontsize=9)

    ax2 = fig.add_subplot(gs[0, 1])
    im2 = ax2.imshow(depth_noisy, cmap="viridis")
    ax2.set_title("(b) 加噪深度图（模拟真实传感器）", fontsize=12, fontweight="bold")
    ax2.set_xticks([])
    ax2.set_yticks([])
    cbar2 = plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
    cbar2.set_label("深度值", fontsize=9)

    ax3 = fig.add_subplot(gs[1, :])
    x = np.arange(len(scene_labels))
    w = 0.35

    bars1 = ax3.bar(x - w/2, clean_counts, w, color=COLORS["bar1"], edgecolor="white",
                    linewidth=0.5, label="干净深度图有效点数", zorder=3)
    bars2 = ax3.bar(x + w/2, noisy_counts, w, color=COLORS["bar2"], edgecolor="white",
                    linewidth=0.5, label="加噪深度图有效点数", zorder=3)

    for bar, val in zip(bars1, clean_counts):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 500,
                 f"{val}", ha="center", va="bottom", fontsize=6.5, fontweight="bold", color=COLORS["bar1"])
    for bar, val in zip(bars2, noisy_counts):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 500,
                 f"{val}", ha="center", va="bottom", fontsize=6.5, fontweight="bold", color=COLORS["bar2"])

    ax3.set_xticks(x)
    ax3.set_xticklabels(scene_labels, fontsize=8)
    ax3.set_ylabel("有效深度点数", fontsize=11)
    ax3.set_title("(c) 8 种场景噪声前后有效深度点数量对比", fontsize=12, fontweight="bold")
    ax3.legend(fontsize=9, loc="lower right")
    ax3.grid(axis="y", alpha=0.3, zorder=0)
    ax3.set_ylim(350000, 375000)

    fig.suptitle("图 3-2  噪声前后深度图对比及有效点数量统计", fontsize=14, fontweight="bold", y=1.01)
    fig.savefig(os.path.join(OUT_DIR, "fig_depth_noise_comparison.png"), dpi=300, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close()
    print("[OK] fig_depth_noise_comparison.png")


def fig_train_val_samples():
    """图 3-3: 训练集与验证集场景示例展示（噪声前后对比）"""
    from PIL import Image

    train_dir = os.path.join(BASE, "training_data", "train")
    val_dir = os.path.join(BASE, "training_data", "val")

    train_scenes = sorted(os.listdir(train_dir))[:2]
    val_scenes = sorted(os.listdir(val_dir))[:2]

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))

    row_labels = ["训练集", "验证集"]
    col_headers = ["干净深度图", "加噪深度图", "干净深度图", "加噪深度图"]

    for row_idx, (split_dir, scenes, label) in enumerate(
        [(train_dir, train_scenes, "训练集"), (val_dir, val_scenes, "验证集")]
    ):
        for col_pair in range(2):
            scene_name = scenes[col_pair]
            scene_dir = os.path.join(split_dir, scene_name)

            dc = np.array(Image.open(os.path.join(scene_dir, "depth_clean.png")))
            dn = np.array(Image.open(os.path.join(scene_dir, "depth_noisy.png")))

            ax_clean = axes[row_idx, col_pair * 2]
            ax_noisy = axes[row_idx, col_pair * 2 + 1]

            im_c = ax_clean.imshow(dc, cmap="viridis")
            ax_clean.set_title(f"{label} 场景{col_pair+1}\n干净深度图", fontsize=10, fontweight="bold")
            ax_clean.set_xticks([])
            ax_clean.set_yticks([])

            im_n = ax_noisy.imshow(dn, cmap="viridis")
            ax_noisy.set_title(f"{label} 场景{col_pair+1}\n加噪深度图", fontsize=10, fontweight="bold")
            ax_noisy.set_xticks([])
            ax_noisy.set_yticks([])

    fig.suptitle("图 3-3  训练集与验证集场景示例展示（噪声前后对比）", fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "fig_train_val_samples.png"), dpi=300, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close()
    print("[OK] fig_train_val_samples.png")


def fig_ablation_study():
    """图 5-1: 消融实验各配置 mIoU 对比柱状图"""
    configs = [
        "Baseline\n(3ch XYZ)",
        "+6ch\nInput",
        "+Channel\nAttn",
        "+Position\nAttn",
        "+Multi\nScale",
        "Full\nPP-Attn",
    ]
    mious = [0.9979, 0.9989, 0.9987, 0.9991, 0.9988, 0.9982]
    latencies = [390.1, 352.6, 343.8, 337.2, 311.3, 340.8]

    fig, ax1 = plt.subplots(figsize=(11, 5.5))

    x = np.arange(len(configs))
    w = 0.35

    bars1 = ax1.bar(x - w/2, [m * 100 for m in mious], w,
                    color=["#BDBDBD", "#90CAF9", "#42A5F5", "#1E88E5", "#1565C0", "#0D47A1"],
                    edgecolor="white", linewidth=0.5, zorder=3)
    ax1.set_ylabel("mIoU (%)", fontsize=12, color="#1565C0")
    ax1.tick_params(axis="y", labelcolor="#1565C0")
    ax1.set_ylim(99.5, 100.2)

    for bar, val in zip(bars1, mious):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                 f"{val*100:.2f}%", ha="center", va="bottom", fontsize=8.5, fontweight="bold", color="#1565C0")

    ax2 = ax1.twinx()
    bars2 = ax2.bar(x + w/2, latencies, w,
                    color=["#E0E0E0", "#FFCC80", "#FFA726", "#FF9800", "#F57C00", "#E65100"],
                    edgecolor="white", linewidth=0.5, zorder=3)
    ax2.set_ylabel("推理延迟 (ms)", fontsize=12, color="#E65100")
    ax2.tick_params(axis="y", labelcolor="#E65100")

    for bar, val in zip(bars2, latencies):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 3,
                 f"{val}ms", ha="center", va="bottom", fontsize=8, fontweight="bold", color="#E65100")

    ax1.set_xticks(x)
    ax1.set_xticklabels(configs, fontsize=9)
    ax1.set_title("图 5-1  消融实验各配置 mIoU 与推理延迟对比", fontsize=14, fontweight="bold")

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#1565C0", label="mIoU (%)"),
        Patch(facecolor="#E65100", label="推理延迟 (ms)"),
    ]
    ax1.legend(handles=legend_elements, fontsize=10, loc="upper left")
    ax1.grid(axis="y", alpha=0.3, zorder=0)

    plt.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "fig_ablation_study.png"), dpi=300, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close()
    print("[OK] fig_ablation_study.png")


def fig_overlap_trend():
    """图 5-2: 不同重叠度下 mIoU 变化趋势图"""
    fig, ax = plt.subplots(figsize=(9, 5))

    overlap_ratios = ["0%\n(并排)", "25%", "50%", "75%", "100%\n(居中)", "100%\n(偏移)", "Cross\n(十字)", "Mixed\n(混合)"]
    pp_mious = [0.9971, 0.9933, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    baseline_mious = [0.9985, 0.9943, 1.0, 1.0, 1.0, 1.0, 1.0, 0.9908]

    x = np.arange(len(overlap_ratios))

    ax.plot(x, [m * 100 for m in pp_mious], color=COLORS["bar1"], linewidth=2.5, marker="o", markersize=9,
            label="PP-Attention", zorder=4)
    ax.plot(x, [m * 100 for m in baseline_mious], color="#BDBDBD", linewidth=2.5, marker="s", markersize=9,
            linestyle="--", label="Baseline (标准 PointNet++)", zorder=3)

    ax.fill_between(x, [m * 100 for m in baseline_mious], [m * 100 for m in pp_mious],
                    alpha=0.12, color=COLORS["bar1"])

    for i, (pp, bl) in enumerate(zip(pp_mious, baseline_mious)):
        gap = (pp - bl) * 100
        mid_y = (pp + bl) / 2 * 100
        ax.annotate(f"+{gap:.1f}%", xy=(i, mid_y), fontsize=7.5, fontweight="bold",
                    color=COLORS["best"], ha="center", va="center",
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="#E8F5E9", alpha=0.85))

    ax.set_xticks(x)
    ax.set_xticklabels(overlap_ratios, fontsize=9)
    ax.set_ylabel("mIoU (%)", fontsize=12)
    ax.set_title("图 5-2  不同场景类型下 mIoU 变化趋势", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10, loc="lower right")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(98, 102)

    plt.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "fig_overlap_trend.png"), dpi=300, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close()
    print("[OK] fig_overlap_trend.png")


def fig_overall_confusion_matrix():
    """图 5-3: 测试集整体混淆矩阵（8 场景汇总，Full PP-Attention）"""
    total_cm = np.array([[2749475, 184], [183, 89964]], dtype=int)

    fig, ax = plt.subplots(figsize=(6, 5.5))
    cm_norm = total_cm.astype(float) / total_cm.sum(axis=1, keepdims=True)

    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)

    classes = ["桌面 (Table)", "物体 (Object)"]
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(classes, fontsize=11)
    ax.set_yticklabels(classes, fontsize=11)
    ax.set_xlabel("预测标签", fontsize=12)
    ax.set_ylabel("真实标签", fontsize=12)

    for i in range(2):
        for j in range(2):
            text_color = "white" if cm_norm[i, j] > 0.5 else "black"
            ax.text(j, i, f"{total_cm[i, j]:,}\n({cm_norm[i, j]:.2%})",
                    ha="center", va="center", fontsize=12, fontweight="bold", color=text_color)

    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("归一化比例", fontsize=10)

    ax.set_title("图 5-3  测试集整体混淆矩阵（8 场景汇总，Full PP-Attention）", fontsize=13, fontweight="bold")

    plt.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "fig_overall_confusion_matrix.png"), dpi=300, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close()
    print("[OK] fig_overall_confusion_matrix.png")


if __name__ == "__main__":
    print("Generating paper figures...")
    print(f"Output: {OUT_DIR}\n")

    fig_data_samples()
    fig_depth_noise_comparison()
    fig_train_val_samples()
    fig_loss_detail()
    fig_miou_detail()
    fig_per_class_iou()
    fig_training_curves()
    fig_test_miou()
    fig_test_latency()
    fig_confusion_matrices()
    fig_hierarchy_results()
    fig_realtime_accuracy()
    fig_realtime_scatter()
    fig_model_comparison()
    fig_ablation_study()
    fig_overlap_trend()
    fig_overall_confusion_matrix()

    print(f"\nDone! {len(os.listdir(OUT_DIR))} figures saved to {OUT_DIR}")
"""Pipeline diagram for the technical report -> report/fig_pipeline.pdf"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

plt.rcParams.update({"font.size": 5.6, "pdf.fonttype": 42})
fig, ax = plt.subplots(figsize=(7.0, 1.4))
ax.set_xlim(0, 100); ax.set_ylim(0, 20); ax.axis("off")
W, G = 14.6, 2.48
boxes = [
    ("H-$\\alpha$ test image\n2048$\\times$2048\n(native, unresized)", "#e8eef5"),
    ("20 YOLO11s detectors\n4 runs $\\times$ 5 folds\nimgsz 1280, conf$\\geq$.05", "#d6e4f0"),
    ("score = conf $\\times$\nagreement-class weight\n(1 / .75 / .5), NMS .5", "#d6e4f0"),
    ("weighted boxes\nfusion, IoU .55\nscore = $\\Sigma$best / 20", "#fbe5d6"),
    ("5 crop refiners\nU-Net/ResNet34, 256 px\n[img, CLAHE, seed], TTA", "#e2efda"),
    ("keep score$\\geq$.25\nmask p$\\geq$.6, $\\geq$100 px\nno overlap $\\rightarrow$ RLE", "#fff2cc"),
]
xs = [0.6 + i * (W + G) for i in range(6)]
for x, (t, c) in zip(xs, boxes):
    ax.add_patch(FancyBboxPatch((x, 10), W, 9, boxstyle="round,pad=0,rounding_size=1.2", fc=c, ec="#555", lw=0.6))
    ax.text(x + W / 2, 14.5, t, ha="center", va="center", linespacing=1.3)
for x in xs[:-1]:
    ax.annotate("", xy=(x + W + G - 0.15, 14.5), xytext=(x + W + 0.15, 14.5),
                arrowprops=dict(arrowstyle="-|>", lw=0.7, color="#333", shrinkA=0, shrinkB=0))


def note(x0, x1, t):
    ax.add_patch(FancyBboxPatch((x0, 0.6), x1 - x0, 7, boxstyle="round,pad=0,rounding_size=1.2",
                                fc="white", ec="#888", lw=0.5, ls="--"))
    ax.text((x0 + x1) / 2, 4.1, t, ha="center", va="center", fontsize=5.3, color="#333", linespacing=1.3)


note(xs[1], xs[2] + W, "training: union of annotator records (dedup. at mask IoU .5);\n"
     "class = share of annotators who drew the filament:\n$\\geq$2/3 (or single annotator), $\\geq$1/2, minority")
note(xs[4], xs[5] + W, "training: every (record, filament) crop;\nseed = jittered GT box grown 25 %;\n"
     "BCE + Dice, 12 epochs, best by val. Dice")
fig.tight_layout(pad=0.05)
fig.savefig(Path(__file__).resolve().parents[1] / "report" / "fig_pipeline.pdf")

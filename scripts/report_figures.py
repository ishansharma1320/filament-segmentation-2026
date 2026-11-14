"""Numbers and figures for the technical report (report/), from the pulled v13 OOF crop
records (5 grouped folds, every annotator record) and the training images.

    python scripts/report_figures.py runs/20261002-204530
Writes report/fig_*.pdf and report/stats.json.
"""
import json
import sys
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import cascade  # noqa: E402
from dataset import load_gt_rles, load_meta  # noqa: E402
from metric import iou_matrix  # noqa: E402
from pycocotools import mask as mu  # noqa: E402
from rle import masks_to_rles  # noqa: E402

CUT, THR, AREA = 0.25, 0.6, 100
OUT = ROOT / "report"
plt.rcParams.update({"font.size": 7, "axes.titlesize": 7, "axes.labelsize": 7, "xtick.labelsize": 6,
                     "ytick.labelsize": 6, "legend.fontsize": 6, "pdf.fonttype": 42})


def per_file_stats(run):
    meta, gt = load_meta(), load_gt_rles()
    crops = cascade.load_crops(Path(run) / "oof_crops")
    files, preds_by_file = {}, {}
    ious, dices, gt_deg, pred_deg = [], [], [], []
    for f, cr in crops.items():
        masks = cascade.crop_instances(cr, CUT, THR, AREA)
        preds = masks_to_rles(masks)
        preds_by_file[f] = masks
        s = np.zeros(4)  # sum_iou, tp, fp, fn
        for rid in meta["records_by_file"][f]:
            g = gt[rid]
            if not g:
                s[2] += len(preds); continue
            if not preds:
                s[3] += len(g); gt_deg.extend([0] * len(g)); continue
            m = iou_matrix(g, preds)
            hit = m > 0.5
            s += [m[hit].sum(), hit.sum(), (hit.sum(0) == 0).sum(), (hit.sum(1) == 0).sum()]
            for i, j in zip(*np.nonzero(hit)):
                ious.append(m[i, j]); dices.append(2 * m[i, j] / (1 + m[i, j]))
            gt_deg.extend((m > 0).sum(1).tolist()); pred_deg.extend((m > 0).sum(0).tolist())
        files[f] = s
    return meta, gt, crops, files, preds_by_file, np.array(ious), np.array(dices), np.array(gt_deg), np.array(pred_deg)


def pq(s):
    d = s[1] + 0.5 * s[2] + 0.5 * s[3]
    return (s[0] / d if d else 0.0), (s[0] / s[1] if s[1] else 0.0), (s[1] / d if d else 0.0)


def human_stats(meta, gt):
    """One annotator's masks scored against another annotator of the same file (all ordered pairs)."""
    import itertools
    s = np.zeros(4); ious = []; gt_deg = []; pred_deg = []
    for f, rids in meta["records_by_file"].items():
        for a, b in itertools.permutations(rids, 2):
            P, G = gt[a], gt[b]
            if not P or not G:
                s[2] += len(P); s[3] += len(G); continue
            m = iou_matrix(G, P); hit = m > 0.5
            s += [m[hit].sum(), hit.sum(), (hit.sum(0) == 0).sum(), (hit.sum(1) == 0).sum()]
            ious.extend(m[hit].tolist())
            gt_deg.extend((m > 0).sum(1).tolist()); pred_deg.extend((m > 0).sum(0).tolist())
    return s, np.array(ious), np.array(gt_deg), np.array(pred_deg)


def main(run):
    meta, gt, crops, files, preds, ious, dices, gt_deg, pred_deg = per_file_stats(run)
    folds = meta["folds"]
    names = sorted(files)
    S = np.array([files[f] for f in names])
    stats = {"pooled": dict(zip(("pq", "sq", "rq"), pq(S.sum(0)))), "counts": dict(zip(("sum_iou", "tp", "fp", "fn"), S.sum(0).tolist()))}
    stats["folds"] = []
    for k in range(5):
        idx = [i for i, f in enumerate(names) if folds[f] == k]
        stats["folds"].append(dict(zip(("pq", "sq", "rq"), pq(S[idx].sum(0)))))
    fp = np.array([x["pq"] for x in stats["folds"]])
    stats["fold_mean"], stats["fold_std"] = float(fp.mean()), float(fp.std(ddof=1))
    rng = np.random.default_rng(2026)
    boot = [pq(S[rng.integers(0, len(S), len(S))].sum(0)) for _ in range(1000)]
    stats["boot95"] = {k: [float(np.percentile([b[i] for b in boot], q)) for q in (2.5, 97.5)] for i, k in enumerate(("pq", "sq", "rq"))}
    stats["iou"] = {"median": float(np.median(ious)), "mean": float(ious.mean()), "n": int(len(ious))}
    stats["dice"] = {"median": float(np.median(dices)), "mean": float(dices.mean())}
    stats["one_to_many_gt"] = {str(d): int((gt_deg == d).sum()) for d in range(0, 4)} | {"4+": int((gt_deg >= 4).sum())}
    stats["many_to_one_pred"] = {str(d): int((pred_deg == d).sum()) for d in range(0, 4)} | {"4+": int((pred_deg >= 4).sum())}
    hs, hious, hgd, hpd = human_stats(meta, gt)
    stats["human"] = dict(zip(("pq", "sq", "rq"), pq(hs))) | {"counts": hs.tolist()}
    json.dump(stats, open(OUT / "stats.json", "w"), indent=1)
    print(json.dumps({k: stats[k] for k in ("pooled", "fold_mean", "fold_std", "boot95", "iou", "dice", "human")}, indent=1))

    # --- evaluation figure: per-fold PQ/SQ/RQ, IoU distribution, match degrees -----------------
    fig, ax = plt.subplots(1, 3, figsize=(7.0, 1.75))
    x = np.arange(5); w = 0.26
    for j, (key, col) in enumerate((("pq", "#1f4e79"), ("sq", "#7fa7d0"), ("rq", "#c55a11"))):
        ax[0].bar(x + (j - 1) * w, [s[key] for s in stats["folds"]], w, color=col, label=key.upper())
    ax[0].set_xticks(x, [f"{k}" for k in range(5)]); ax[0].set_xlabel("fold"); ax[0].set_ylim(0, 0.95)
    ax[0].axhline(stats["human"]["pq"], color="gray", ls="--", lw=0.8)
    ax[0].text(4.55, stats["human"]["pq"] + 0.012, "2nd-annotator PQ", ha="right", va="bottom", color="dimgray", fontsize=5.5, bbox=dict(fc="white", ec="none", pad=0.4))
    ax[0].legend(ncol=3, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.02), columnspacing=0.8, handlelength=1.0); ax[0].set_title("(a) per-fold scores (detector v13)")
    bins = np.linspace(0.5, 1.0, 26)
    ax[1].hist(ious, bins, density=True, color="#1f4e79", alpha=0.8, label=f"model (median {np.median(ious):.2f})")
    ax[1].hist(hious, bins, density=True, histtype="step", color="#c55a11", lw=1.0, label=f"2nd annotator (median {np.median(hious):.2f})")
    ax[1].set_xlabel("IoU of matched pairs (TP)"); ax[1].set_ylim(0, ax[1].get_ylim()[1] * 1.45); ax[1].legend(frameon=True, framealpha=0.95, edgecolor="none", loc="upper right", bbox_to_anchor=(1.0, 1.02)); ax[1].set_title("(b) IoU distribution")
    deg = np.arange(0, 5)
    def frac(a):
        return np.array([(a == d).mean() if d < 4 else (a >= 4).mean() for d in deg])
    ax[2].bar(deg - 0.2, frac(gt_deg), 0.4, color="#1f4e79", label="GT: # overlapping predictions")
    ax[2].bar(deg + 0.2, frac(pred_deg), 0.4, color="#c55a11", label="pred.: # overlapping GT")
    ax[2].set_xticks(deg, ["0", "1", "2", "3", "4+"]); ax[2].set_ylabel("fraction")
    ax[2].set_ylim(0, 1.25); ax[2].legend(frameon=False, loc="upper right"); ax[2].set_title("(c) one-to-many / many-to-one")
    for a in ax:
        a.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(pad=0.3)
    fig.savefig(OUT / "fig_eval.pdf")

    # --- examples: GT (all annotators) vs predictions on held-out disks ---------------------------
    from config import find_data_root
    tr = find_data_root() / "train" / "train_images"
    multi = [f for f in names if len(meta["records_by_file"][f]) == 3]
    pqs = {f: pq(files[f])[0] for f in multi}
    order = sorted(multi, key=lambda f: pqs[f])
    picks = [order[int(len(order) * q)] for q in (0.85, 0.5, 0.15)]
    fig, ax = plt.subplots(1, 3, figsize=(7.0, 2.4))
    for a, f in zip(ax, picks):
        img = cv2.imread(str(tr / f), cv2.IMREAD_GRAYSCALE)
        rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        gmask = np.zeros(img.shape, np.uint8)
        for rid in meta["records_by_file"][f]:
            for r in gt[rid]:
                m = mu.decode(r).astype(np.uint8); gmask |= m
                cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                cv2.drawContours(rgb, cs, -1, (0, 200, 255), 2)
        for m in preds[f]:
            cs, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            cv2.drawContours(rgb, cs, -1, (255, 40, 20), 2)
        # zoom on the 760 px window holding the most annotated filament pixels
        k = cv2.boxFilter(gmask.astype(np.float32), -1, (760, 760), normalize=False)
        cy, cx = np.unravel_index(int(k.argmax()), k.shape)
        y0 = int(np.clip(cy - 380, 0, 2048 - 760)); x0 = int(np.clip(cx - 380, 0, 2048 - 760))
        a.imshow(rgb[y0:y0 + 760, x0:x0 + 760]); a.axis("off")
        a.text(8, 30, f"{f.rsplit('.', 1)[0]}  PQ {pqs[f]:.2f}", color="white", fontsize=6.5, bbox=dict(fc="black", alpha=0.6, ec="none", pad=1.5))
    fig.tight_layout(pad=0.2)
    fig.savefig(OUT / "fig_examples.pdf", dpi=300)
    print("examples:", picks)


if __name__ == "__main__":
    main(sys.argv[1])

"""
Compare AMICI interaction-weight matrices across every finished condition: each method run on the clean
data and on the induced data.

Methods (whichever have finished runs are included):
  amici   original AMICI                          ~/runs_final/                 (train_clean.py, train_induced.py)
  pre     centering as preprocessing              ~/experiments/runs/          (train_corrected.py)
  idea1   AMICIBatch, per-group baseline + centered residual    ~/batch_correction/runs_arch/  (train_architecture.py)
  idea2   AMICIBatch, centered neighbor inputs                  ~/batch_correction/runs_arch/
  both    AMICIBatch, all three                                 ~/batch_correction/runs_arch/
Conditions are named <data>_<method>, e.g. clean_amici, induced_both.

Reads <run dir>/seed*/interaction_weights.csv (rows = sender, columns = receiver; each entry is the
number of genes significant in the neighbor ablation). Every run is compared with every run of a
reference condition that has a DIFFERENT seed, then averaged, so all comparisons sit on the same
seed-to-seed noise:

  vs clean_amici   (reference = original AMICI on clean data)
      clean_amici                  -> noise floor
      induced_amici                -> damage from the batch effect
      induced_<method>             -> does the method undo it?
      clean_<method>               -> do-no-harm: does the method change results on clean data?
  vs same method on clean data
      induced_<method> vs clean_<method>  -> the cleanest test that batch is removed

Also reports the rows of the senders that became hubs (Endo, Micro by default), the known edges that
should survive, and the Frobenius distance to the mean clean matrix.

Usage: python ~/batch_correction/compare_correction_batch.py [--hubs Endo Micro] [--top 20]
"""
import argparse

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import sys
from pathlib import Path

EXP = Path(__file__).resolve().parent          # ~/batch_correction/ (outputs go to EXP/correction_comparison)
sys.path.insert(0, str(EXP.parent))
from train_clean import HOME

METHOD_DIRS = {
    "amici": lambda tag: HOME / "runs_final" / tag,                  # original AMICI, sweep config
    "both": lambda tag: EXP / "runs_arch" / f"{tag}_arch_both",     # new architecture
}
DATA_TAGS = {"clean": "slices162_170_clean", "induced": "slices162_clean_170_induced"}
CONDS = {f"{d}_{m}": f(tag) for m, f in METHOD_DIRS.items() for d, tag in DATA_TAGS.items()}
REF = "clean_amici"


def same_method_ref(cond):
    return "clean_" + cond.split("_", 1)[1]
KNOWN_EDGES = [("L5 ET", "L4/5 IT"), ("L4/5 IT", "L2/3 IT")]

ap = argparse.ArgumentParser()
ap.add_argument("--hubs", nargs="+", default=["Endo", "Micro"])
ap.add_argument("--top", type=int, default=20)
a = ap.parse_args()
OUT = EXP / "correction_comparison"
OUT.mkdir(parents=True, exist_ok=True)

runs = {}
for cond, run_dir in CONDS.items():
    for f in sorted(run_dir.glob("seed*/interaction_weights.csv")):
        runs[(cond, int(f.parent.name.replace("seed", "")))] = pd.read_csv(f, index_col=0)
if not runs:
    raise SystemExit("no interaction_weights.csv found in any run folder")
print("runs found:", {c: sorted(s for cc, s in runs if cc == c) for c in CONDS if any(cc == c for cc, _ in runs)})

# common sender/receiver order, missing pairs = 0
senders = sorted(set().union(*[W.index for W in runs.values()]))
receivers = sorted(set().union(*[W.columns for W in runs.values()]))
runs = {k: W.reindex(index=senders, columns=receivers).fillna(0) for k, W in runs.items()}
offdiag = np.array([[s != r for r in receivers] for s in senders])
pairs = np.array([[f"{s} -> {r}" for r in receivers] for s in senders])[offdiag]


def vec(W):
    return pd.Series(W.values[offdiag], index=pairs)


def compare(W1, W2):
    x, y = vec(W1), vec(W2)
    t1, t2 = set(x.nlargest(a.top).index), set(y.nlargest(a.top).index)
    return spearmanr(x, y)[0], len(t1 & t2) / len(t1 | t2)


mean_by = {c: np.mean([W.values for (cc, _), W in runs.items() if cc == c], axis=0)
           for c in CONDS if any(cc == c for cc, _ in runs)}

rows = []
for (cond, seed), W in runs.items():
    row = dict(condition=cond, seed=seed, total_sig_genes=int(W.values[offdiag].sum()))
    for label, ref in (("clean_amici", REF), ("same_method_clean", same_method_ref(cond))):
        sc = [compare(W, R) for (rc, rs), R in runs.items() if rc == ref and rs != seed]
        row[f"spearman_vs_{label}"] = np.mean([s for s, _ in sc]) if sc else np.nan
        row[f"top{a.top}_jaccard_vs_{label}"] = np.mean([j for _, j in sc]) if sc else np.nan
        row[f"frobenius_to_mean_{label}"] = (float(np.linalg.norm((W.values - mean_by[ref])[offdiag]))
                                             if ref in mean_by else np.nan)
    for h in a.hubs:
        row[f"{h}_row_sum"] = int(W.loc[h].values.sum()) if h in W.index else np.nan
    for s, r in KNOWN_EDGES:
        row[f"{s} -> {r}"] = int(W.loc[s, r]) if s in W.index and r in W.columns else np.nan
    rows.append(row)

per_run = pd.DataFrame(rows)
order = [c for c in CONDS if c in set(per_run.condition)]
per_run["condition"] = pd.Categorical(per_run.condition, categories=order, ordered=True)
per_run = per_run.sort_values(["condition", "seed"])
per_run.to_csv(OUT / "per_run.csv", index=False)
summary = per_run.drop(columns="seed").groupby("condition", observed=True).agg(["mean", "std"]).round(3)
summary.to_csv(OUT / "by_condition.csv")
pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 40)
print("\nMean (std) over seeds; 'vs X' = averaged over runs of X with a different seed\n")
print(summary.T.to_string())

# Largest pair changes vs original AMICI on clean data, per condition (mean over seeds)
changes = []
for c in order:
    if c == REF or REF not in mean_by:
        continue
    d = pd.Series((mean_by[c] - mean_by[REF])[offdiag], index=pairs)
    for p, v in d.reindex(d.abs().sort_values(ascending=False).index[:10]).items():
        changes.append(dict(condition=c, pair=p, change_vs_clean=round(float(v), 1)))
pd.DataFrame(changes).to_csv(OUT / "largest_changes_vs_clean.csv", index=False)
print("\nLargest mean changes vs clean (top 5 per condition):")
print(pd.DataFrame(changes).groupby("condition").head(5).to_string(index=False))

# Figure
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    panels = [("spearman_vs_clean_amici", "Spearman vs AMICI on clean\n(other seeds)"),
              (f"top{a.top}_jaccard_vs_clean_amici", f"Top-{a.top} overlap vs AMICI on clean"),
              ("spearman_vs_same_method_clean", "Spearman vs same method\non clean (other seeds)")]
    panels += [(f"{h}_row_sum", f"{h} as sender\n(total significant genes)") for h in a.hubs]
    fig, axes = plt.subplots(1, len(panels), figsize=(3.4 * len(panels), 4.0))
    for ax, (col, title) in zip(axes, panels):
        for i, c in enumerate(order):
            v = per_run.loc[per_run.condition == c, col].to_numpy(dtype=float)
            col_ = "#534AB7" if c.startswith("clean") else "#D85A30"
            ax.scatter(np.full(len(v), i), v, color=col_, alpha=0.7, s=28)
            if np.isfinite(v).any():
                ax.hlines(np.nanmean(v), i - 0.25, i + 0.25, color="#26215C", lw=2)
        ax.set_xticks(range(len(order)), order, rotation=30, ha="right")
        ax.set_title(title, fontsize=10)
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "correction_comparison.png", dpi=150)
    print(f"\nfigure: {OUT / 'correction_comparison.png'}")
except ImportError:
    pass
print(f"tables: {OUT}")
"""
Figures: did the AMICIBatch architecture remove the induced batch effect?
Original AMICI (sweep config, ~/runs_final) vs AMICIBatch with all switches on (~/batch_correction/runs_arch).

  fig1_interaction_heatmaps.png   sender x receiver interaction weights (mean over seeds):
                                  rows = original AMICI / new architecture, columns = clean, induced, induced - clean.
                                  Worked if the new architecture's "induced - clean" panel is flat where AMICI's
                                  shows bright Endo / Micro rows.
  fig2_hub_senders.png            total significant genes with Endo / Micro as sender, per run (dots = seeds).
                                  Worked if induced (new) sits with the clean runs, not with induced (AMICI).
  fig3_pair_scatter.png           every sender -> receiver pair, clean (x) vs induced (y), one panel per model.
                                  Worked if the new model's points hug the diagonal; Endo / Micro senders in orange.
  fig4_recovered_shift.png        sanity check: the technical-shift term in the new model (slide 170 - slide 162,
                                  induced run minus clean run) vs the shift actually injected, per cell type x gene.
                                  The per-(slide, type) baselines are group means of the data, so this is close to
                                  the diagonal by construction; it confirms the shift is held in the technical term,
                                  it is not by itself evidence that the interactions were fixed (figs 1-3 are).
  fig5_test_loss.png              held-out reconstruction loss per slide (lower = better fit). All models predict the
                                  same uncorrected data, so these are directly comparable.

Usage: python ~/batch_correction/make_arch_figures.py [--hubs Endo Micro]
Output: ~/batch_correction/figures/
"""
import argparse, json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, pearsonr

HOME = Path.home()
EXP = Path(__file__).resolve().parent
OUT = EXP / "figures"
DATA = {"clean": "slices162_170_clean", "induced": "slices162_clean_170_induced"}
MODELS = {
    "AMICI": lambda tag: HOME / "runs_final" / tag,
    "New architecture": lambda tag: EXP / "runs_arch" / f"{tag}_arch_both",
}
SLIDES = ["mouse1_slice162", "mouse1_slice170"]
GRAY, PURPLE, ORANGE, INK = "#888780", "#534AB7", "#D85A30", "#2C2C2A"
COLORS = {"AMICI": GRAY, "New architecture": PURPLE}

ap = argparse.ArgumentParser()
ap.add_argument("--hubs", nargs="+", default=["Endo", "Micro"])
a = ap.parse_args()
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.titlesize": 11, "axes.titleweight": "regular"})

# ---------- load interaction weights ----------
runs = {}   # (model, data) -> list of DataFrames
for m, f in MODELS.items():
    for d, tag in DATA.items():
        files = sorted(f(tag).glob("seed*/interaction_weights.csv"))
        if files:
            runs[(m, d)] = [pd.read_csv(x, index_col=0) for x in files]
print("runs found:", {f"{m} / {d}": len(v) for (m, d), v in runs.items()})
if not runs:
    raise SystemExit("no interaction_weights.csv found")

types = sorted(set().union(*[set(W.index) | set(W.columns) for v in runs.values() for W in v]))
runs = {k: [W.reindex(index=types, columns=types).fillna(0) for W in v] for k, v in runs.items()}
mean = {k: np.mean([W.values for W in v], axis=0) for k, v in runs.items()}
offdiag = ~np.eye(len(types), dtype=bool)
hub_rows = [types.index(h) for h in a.hubs if h in types]


def saved(name):
    plt.savefig(OUT / name, dpi=200, bbox_inches="tight")
    plt.close()
    print("saved", OUT / name)


# ---------- fig1: heatmaps ----------
models_here = [m for m in MODELS if (m, "clean") in mean and (m, "induced") in mean]
if models_here:
    vmax = max(np.percentile(mean[k][offdiag], 99) for k in mean)
    dmax = max(np.percentile(np.abs((mean[(m, "induced")] - mean[(m, "clean")])[offdiag]), 99) for m in models_here)
    fig, axes = plt.subplots(len(models_here), 3, figsize=(17, 5.6 * len(models_here)), squeeze=False)
    for r, m in enumerate(models_here):
        panels = [("clean", mean[(m, "clean")], "viridis", 0, vmax),
                  ("induced", mean[(m, "induced")], "viridis", 0, vmax),
                  ("induced − clean", mean[(m, "induced")] - mean[(m, "clean")], "RdBu_r", -dmax, dmax)]
        for c, (title, M, cmap, lo, hi) in enumerate(panels):
            ax = axes[r, c]
            M = np.where(offdiag, M, np.nan)
            im = ax.imshow(M, cmap=cmap, vmin=lo, vmax=hi, aspect="equal")
            ax.set_xticks(range(len(types)), types, rotation=90, fontsize=7)
            ax.set_yticks(range(len(types)), types, fontsize=7)
            for h in hub_rows:
                ax.get_yticklabels()[h].set_color(ORANGE)
            ax.set_title(f"{m}: {title}")
            if c == 0:
                ax.set_ylabel("sender")
            ax.set_xlabel("receiver")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03,
                         label="significant genes" if c < 2 else "change in significant genes")
    fig.suptitle("Interaction weights (mean over seeds); hub senders in orange", y=1.0)
    fig.tight_layout()
    saved("fig1_interaction_heatmaps.png")

# ---------- fig2: hub senders ----------
conds = [(m, d) for m in MODELS for d in DATA if (m, d) in runs]
fig, axes = plt.subplots(1, len(hub_rows) + 1, figsize=(4.2 * (len(hub_rows) + 1), 4), squeeze=False)
for ax, (label, rows) in zip(axes[0], [(types[h], [h]) for h in hub_rows] +
                             [("all other senders", [i for i in range(len(types)) if i not in hub_rows])]):
    for i, (m, d) in enumerate(conds):
        vals = [W.values[rows][:, :].sum() - sum(W.values[r, r] for r in rows) for W in runs[(m, d)]]
        filled = d == "clean"
        ax.scatter(np.full(len(vals), i) + np.linspace(-0.08, 0.08, len(vals)), vals, s=40,
                   facecolor=COLORS[m] if filled else "white", edgecolor=COLORS[m], linewidth=1.5, zorder=3)
        ax.hlines(np.mean(vals), i - 0.25, i + 0.25, color=COLORS[m], lw=2)
    short = {"AMICI": "AMICI", "New architecture": "New arch."}
    ax.set_xticks(range(len(conds)), [f"{short[m]}\n{d}" for m, d in conds], fontsize=8)
    ax.set_title(f"{label} as sender")
    ax.set_ylabel("total significant genes")
fig.suptitle("Hub senders: induced runs should sit with the clean runs (filled = clean, open = induced)", y=1.03)
fig.tight_layout()
saved("fig2_hub_senders.png")

# ---------- fig3: pair scatter ----------
if models_here:
    fig, axes = plt.subplots(1, len(models_here), figsize=(5.2 * len(models_here), 5), squeeze=False)
    is_hub = np.zeros_like(offdiag)
    is_hub[hub_rows, :] = True
    lim = max(max(mean[(m, d)][offdiag].max() for d in DATA) for m in models_here) * 1.05
    for ax, m in zip(axes[0], models_here):
        x, y = mean[(m, "clean")], mean[(m, "induced")]
        for mask, col, lab in [(offdiag & ~is_hub, COLORS[m], "other senders"),
                               (offdiag & is_hub, ORANGE, " / ".join(a.hubs) + " as sender")]:
            ax.scatter(x[mask], y[mask], s=12, color=col, alpha=0.6, label=lab, edgecolor="none")
        ax.plot([0, lim], [0, lim], color=INK, lw=0.8, ls="--")
        rho = spearmanr(x[offdiag], y[offdiag])[0]
        ax.set_title(f"{m}\nSpearman ρ = {rho:.2f}")
        ax.set_xlabel("clean data: significant genes per pair")
        ax.set_ylabel("induced data: significant genes per pair")
        ax.set_xlim(0, lim); ax.set_ylim(0, lim)
        ax.legend(frameon=False, fontsize=8, loc="upper left")
    fig.suptitle("Each sender → receiver pair, clean vs induced (mean over seeds); diagonal = unaffected by batch", y=1.03)
    fig.tight_layout()
    saved("fig3_pair_scatter.png")

# ---------- fig4: recovered vs injected shift ----------
try:
    import anndata as ad
    shift_files = {d: sorted((EXP / "runs_arch" / f"{DATA[d]}_arch_both").glob("seed*/technical_shift.csv"))
                   for d in DATA}
    clean_h5, ind_h5 = HOME / "data" / f"{DATA['clean']}.h5ad", HOME / "data" / f"{DATA['induced']}.h5ad"
    if all(shift_files.values()) and clean_h5.exists() and ind_h5.exists():
        def learned(files):   # slide 170 - slide 162 per cell type, mean over seeds
            ds = []
            for f in files:
                T = pd.read_csv(f, index_col=[0, 1, 2]).droplevel(2)
                ds.append(T.xs(SLIDES[1], level=0) - T.xs(SLIDES[0], level=0))
            return sum(ds) / len(ds)
        rec = learned(shift_files["induced"]) - learned(shift_files["clean"])

        c, i = ad.read_h5ad(clean_h5), ad.read_h5ad(ind_h5)
        common = c.obs_names.intersection(i.obs_names)
        c, i = c[common], i[common]
        dense = lambda M: M.toarray() if hasattr(M, "toarray") else np.asarray(M)
        is170 = (c.obs["slice_id"].astype(str) == SLIDES[1]).to_numpy()
        diff = pd.DataFrame(dense(i.X)[is170] - dense(c.X)[is170], columns=c.var_names)
        diff["t"] = c.obs["subclass"].astype(str).to_numpy()[is170]
        inj = diff.groupby("t").mean()
        n170 = diff["t"].value_counts()

        rec = rec.reindex(index=inj.index, columns=inj.columns).dropna(how="all")
        inj = inj.loc[rec.index]
        keep = [t for t in rec.index if n170.get(t, 0) >= 20]
        x, y = inj.loc[keep].values.ravel(), rec.loc[keep].values.ravel()
        r = pearsonr(x, y)[0]
        slope = np.polyfit(x, y, 1)[0]
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), gridspec_kw={"width_ratios": [1.1, 1]})
        ax = axes[0]
        ax.scatter(x, y, s=5, alpha=0.35, color=PURPLE, edgecolor="none")
        lim = np.abs(np.r_[x, y]).max() * 1.05
        ax.plot([-lim, lim], [-lim, lim], color=INK, lw=0.8, ls="--")
        ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
        ax.set_xlabel("injected shift (induced − clean, slide 170 mean)")
        ax.set_ylabel("learned technical shift δ (170 − 162, induced − clean run)")
        ax.set_title(f"Every cell type × gene (types with ≥ 20 cells)\nPearson r = {r:.3f}, slope = {slope:.2f}")
        per_t = pd.Series({t: pearsonr(inj.loc[t], rec.loc[t])[0] for t in keep}).sort_values()
        ax = axes[1]
        ax.barh(per_t.index, per_t.values, color=PURPLE)
        ax.set_xlim(min(0, per_t.min() - 0.05), 1)
        ax.set_xlabel("Pearson r across genes")
        ax.set_title("Per cell type")
        fig.suptitle("Sanity check: the model's technical-shift term holds the injected batch effect", y=1.02)
        fig.tight_layout()
        saved("fig4_recovered_shift.png")
        print("  shrinkage (kappa) pulls small groups toward 0, so a slope a little under 1 is expected")
    else:
        print("fig4 skipped: needs technical_shift.csv from both new-architecture runs and the clean / induced h5ad in ~/data")
except Exception as e:
    print(f"fig4 skipped ({type(e).__name__}: {e})")

# ---------- fig5: test loss ----------
rows = []
for m, f in MODELS.items():
    for d, tag in DATA.items():
        for p in sorted(f(tag).glob("seed*/metrics.json")):
            j = json.loads(p.read_text())
            for s in SLIDES:
                k = f"test_reconstruction_loss_{s}"
                if k in j:
                    rows.append(dict(model=m, data=d, slide=s.replace("mouse1_", ""), loss=j[k]))
if rows:
    df = pd.DataFrame(rows)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, d in zip(axes, DATA):
        sub = df[df.data == d]
        for j_, (m, s) in enumerate([(m, s) for s in sorted(sub.slide.unique()) for m in MODELS]):
            v = sub[(sub.model == m) & (sub.slide == s)].loss.values
            if len(v):
                ax.scatter(np.full(len(v), j_), v, color=COLORS[m], s=40, zorder=3)
                ax.hlines(v.mean(), j_ - 0.25, j_ + 0.25, color=COLORS[m], lw=2)
        short = {"AMICI": "AMICI", "New architecture": "New arch."}
        labels = [f"{short[m]}\n{s}" for s in sorted(sub.slide.unique()) for m in MODELS]
        ax.set_xticks(range(len(labels)), labels, fontsize=8)
        ax.set_title(f"{d} data")
        ax.set_ylabel("test reconstruction loss (lower = better)")
    fig.suptitle("Held-out fit per slide", y=1.02)
    fig.tight_layout()
    saved("fig5_test_loss.png")
else:
    print("fig5 skipped: no metrics.json found")

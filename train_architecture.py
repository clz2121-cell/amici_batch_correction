"""
Train AMICIBatch (amici_batch.py: AMICI with the technical / biological decomposition built into the
architecture) with the same sweep hyperparameters, seeds, train/test split, coordinates and evaluation
as train_clean.py / train_induced.py. Inputs are the ORIGINAL, uncorrected datasets; all correction
happens inside the model.

Variants (switches in AMICIBatch):
  idea1  batch_baseline + center_residual      per-(slide, type) baseline, residual centered per group
  idea2  center_inputs                         neighbors enter as deviation from their group mean + type embedding
  both   all three                             the full method

Datasets:
  clean    ~/data/slices162_170_clean.h5ad                     (same file as train_clean.py)
  induced  ~/data/slices162_clean_170_induced.h5ad             (same file as train_induced.py; built if missing)

Lives in ~/batch_correction/ next to amici_batch.py and writes there:
  ~/batch_correction/runs_arch/<dataset>_<variant>/seed*/   model, history.csv, metrics.json,
                                                     interaction_weights.csv, ablation_residuals.pkl,
                                                     technical_shift.csv (delta per slide x type x gene),
                                                     residual_group_means.csv (check: should be ~0 with idea1/both)

Usage (on the VM, in tmux, with the amici env active):
    python ~/batch_correction/train_architecture.py                                 # full method on clean + induced, seeds 4 7 13
    python ~/batch_correction/train_architecture.py --variants both idea1 idea2     # add the ablations (18 runs)
    python ~/batch_correction/train_architecture.py --datasets induced --variants both --seeds 4
    python ~/batch_correction/train_architecture.py --smoke                         # 2 epochs, 1 seed, no ablation: checks the code runs
Then: python ~/batch_correction/compare_correction_batch.py
"""
import argparse, json, pickle, sys, time
from pathlib import Path

EXP = Path(__file__).resolve().parent
sys.path.insert(0, str(EXP))           # amici_batch.py
sys.path.insert(0, str(EXP.parent))    # train_clean.py / train_induced.py

import numpy as np
import pandas as pd
import scanpy as sc
import torch
import pytorch_lightning as pl
from amici.callbacks import AttentionPenaltyMonitor

from amici_batch import AMICIBatch
from train_clean import load_config, DATA, SLIDES, LABELS_KEY
import train_induced

BATCH_KEY = "slice_id"
VARIANTS = {
    "idea1": dict(batch_baseline=True, center_residual=True, center_inputs=False),
    "idea2": dict(batch_baseline=False, center_residual=False, center_inputs=True),
    "both": dict(batch_baseline=True, center_residual=True, center_inputs=True),
}
DATASETS = {"clean": "slices162_170_clean", "induced": "slices162_clean_170_induced"}


def load_dataset(name):
    if name == "clean":
        adata = sc.read_h5ad(DATA)
    else:
        path = train_induced.COMBINED
        adata = sc.read_h5ad(path) if path.exists() else train_induced.build()
    adata.obs["in_test"] = adata.obs["in_test"].astype(bool)
    adata.obs[BATCH_KEY] = adata.obs[BATCH_KEY].astype(str).astype("category")
    print(f"{name}: {adata.shape}, test cells {int(adata.obs['in_test'].sum())}")
    return adata


def setup(adata, n_neighbors):
    AMICIBatch.setup_anndata(adata, batch_key=BATCH_KEY, labels_key=LABELS_KEY,
                             coord_obsm_key="spatial", n_neighbors=n_neighbors)


def interpret(model, adata, run_dir):
    t1 = time.time()
    abl = model.get_neighbor_ablation_scores(adata=adata, compute_z_value=True)
    abl._get_interaction_weight_matrix().to_csv(run_dir / "interaction_weights.csv")
    try:
        with open(run_dir / "ablation_residuals.pkl", "wb") as f:
            pickle.dump(abl, f)
    except Exception as e:
        print(f"  (could not pickle ablation results: {type(e).__name__}: {e})")
    print(f"interaction weights saved ({(time.time() - t1) / 60:.1f} min)")


def run(adata, cfg, variant, seed, out_dir, do_interpret, threads, tag, max_epochs=None, kappa=10.0):
    mp, ep, pp = cfg["model_params"], cfg["exp_params"], cfg["penalty_schedule_params"]
    run_dir = out_dir / f"seed{seed}"
    metrics_path, weights_path = run_dir / "metrics.json", run_dir / "interaction_weights.csv"
    torch.set_num_threads(threads)
    pl.seed_everything(seed)

    if metrics_path.exists():   # resume: finish a run that trained but was interrupted before the readout
        res = json.loads(metrics_path.read_text())
        if do_interpret and not weights_path.exists():
            print(f"{tag} seed {seed}: trained already, computing interaction weights from the saved model")
            setup(adata, ep["n_neighbors"])
            model = AMICIBatch.load(str(run_dir / "model"), adata=adata)
            interpret(model, adata, run_dir)
        else:
            print(f"{tag} seed {seed} already done, skipping")
        return res

    run_dir.mkdir(parents=True, exist_ok=True)
    # 1. train on the non-test cells (group means come from these cells only)
    adata_train = adata[~adata.obs["in_test"]].copy()
    setup(adata_train, ep["n_neighbors"])
    model = AMICIBatch(adata_train, kappa=kappa, **VARIANTS[variant], **mp)
    print(model)
    t0 = time.time()
    model.train(
        max_epochs=int(max_epochs or ep["epochs"]), batch_size=int(ep["batch_size"]), plan_kwargs={"lr": ep["lr"]},
        early_stopping=ep["early_stopping"], early_stopping_monitor=ep["early_stopping_monitor"],
        early_stopping_patience=ep["early_stopping_patience"], check_val_every_n_epoch=1,
        callbacks=[AttentionPenaltyMonitor(**pp)],
    )
    train_min = (time.time() - t0) / 60
    model.calibrate_residual_centering(adata_train)   # exact group means of r for inference (idea1 / both)
    model.save(str(run_dir / "model"), overwrite=True)
    pd.concat([v for v in model.history.values()], axis=1).to_csv(run_dir / "history.csv")
    val = model.history["elbo_validation"].iloc[:, 0]
    model.technical_shift_table().to_csv(run_dir / "technical_shift.csv")

    # 2. evaluate on the held-out strips, neighbors from the full slides (same as train_clean.py)
    setup(adata, ep["n_neighbors"])
    test = adata.obs["in_test"].to_numpy()
    res = dict(tag=tag, variant=variant, seed=seed, best_val_loss=float(val.min()), final_val_loss=float(val.iloc[-1]),
               epochs_trained=int(len(val)), train_minutes=round(train_min, 1))
    idx = np.where(test)[0]
    res["test_elbo"] = float(model.get_elbo(adata, indices=idx, batch_size=128))
    res["test_reconstruction_loss"] = float(
        model.get_reconstruction_error(adata, indices=idx, batch_size=128)["reconstruction_loss"])
    for s in SLIDES:
        idx_s = np.where(test & (adata.obs[BATCH_KEY].astype(str) == s).to_numpy())[0]
        res[f"test_elbo_{s}"] = float(model.get_elbo(adata, indices=idx_s, batch_size=128))
        res[f"test_reconstruction_loss_{s}"] = float(
            model.get_reconstruction_error(adata, indices=idx_s, batch_size=128)["reconstruction_loss"])
    rg = model.residual_group_means(adata)
    rg.to_csv(run_dir / "residual_group_means.csv", index=False)
    res["max_abs_residual_group_mean"] = float(rg.loc[rg.n_cells >= 50, "mean_abs_group_mean"].max())
    metrics_path.write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))

    # 3. interaction weights (sender x receiver), same readout as train_clean.py
    if do_interpret:
        interpret(model, adata, run_dir)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", choices=list(DATASETS), default=["clean", "induced"])
    ap.add_argument("--variants", nargs="+", choices=list(VARIANTS), default=["both"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[4, 7, 13])
    ap.add_argument("--kappa", type=float, default=10.0, help="shrinkage of small groups toward the pooled mean")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--no-interpret", action="store_true", help="skip neighbor-ablation scores")
    ap.add_argument("--smoke", action="store_true", help="2 epochs, first seed, no ablation, separate output folder")
    a = ap.parse_args()

    cfg = load_config()
    root = EXP / ("runs_arch_smoke" if a.smoke else "runs_arch")
    seeds = a.seeds[:1] if a.smoke else a.seeds
    for d in a.datasets:
        adata = load_dataset(d)
        for v in a.variants:
            tag = f"{DATASETS[d]}_arch_{v}"
            out = root / tag
            out.mkdir(parents=True, exist_ok=True)
            (out / "config_used.json").write_text(json.dumps(dict(cfg, variant=VARIANTS[v], kappa=a.kappa), indent=2))
            rows = [run(adata, cfg, v, s, out, not (a.no_interpret or a.smoke), a.threads, tag,
                        max_epochs=2 if a.smoke else None, kappa=a.kappa) for s in seeds]
            pd.DataFrame(rows).to_csv(out / "metrics_all_seeds.csv", index=False)
    print("\nNote: test ELBO is comparable across variants here (all predict the same uncorrected X).")


if __name__ == "__main__":
    main()
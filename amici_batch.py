"""
AMICIBatch: AMICI with an architecture that separates technical (batch) shifts from biological
(neighborhood) shifts. Subclasses amici-st 0.1.0's AMICI model and AMICIModule; AMICI itself is
not edited.

Original AMICI (amici/_module.py):
    prediction_c = mu_t                       + W_o[ attention over neighbors( MLP(x_j) ) ]
                   pooled cell-type mean        neighborhood residual r(c)

AMICIBatch, three switches:

  Idea 1a  batch_baseline   baseline = mu_{b,t} (mean of type t on slide b) instead of the pooled mu_t.
                            Stored as mu_t + s_{b,t} (mu_{b,t} - mu_t), s_{b,t} = n_{b,t} / (n_{b,t} + kappa),
                            so tiny groups shrink back toward the pooled mean.
  Idea 1b  center_residual  r_bio(c) = r(c) - mean of r over the cells of c's (slide, type) group.
                            Training: the minibatch group mean (groups with >= min_group_count cells in the
                            minibatch), otherwise a running mean; the running mean is also used at inference.
                            Removes the degeneracy between the baseline and r: nothing shared by a whole
                            group can sit in r.
  Idea 2   center_inputs    neighbor j enters the neighbor MLP as [x_j - mu_{b,t_j} ; e_{t_j}]: its deviation
                            from its own (slide, type) mean, plus a learned neighbor cell-type embedding
                            (centering erases type identity; the embedding restores it, shared across slides).

  prediction_c = baseline + r_bio(c)
  technical shift   delta_{b,t} = baseline_{b,t} - mu_t        (saved by technical_shift_table())
  biological shift  r_bio(c)                                   (what attention / ablation read out)

With all three off, AMICIBatch computes exactly what AMICI computes.

Requires a batch column registered with setup_anndata(..., batch_key="slice_id").
Neighbors are assumed to come from the receiver's own slide (true when slides are spatially separated,
as in the sweep's data prep); a neighbor's group is (receiver's slide, neighbor's type).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scvi import REGISTRY_KEYS
from scvi.data import AnnDataManager
from scvi.data.fields import CategoricalObsField, LayerField, ObsmField
from scvi.model.base import BaseModelClass
from scvi.module.base import auto_move_data

from amici import AMICI
from amici._components import ResNetMLP
from amici._constants import NN_REGISTRY_KEYS
from amici._module import AMICIModule


class AMICIBatchModule(AMICIModule):
    def __init__(
        self,
        n_genes: int,
        n_labels: int,
        empirical_ct_means: torch.Tensor,
        n_batch: int,
        empirical_bt_means: torch.Tensor,   # n_batch x n_labels x n_genes (already shrunk)
        batch_baseline: bool = True,
        center_residual: bool = True,
        center_inputs: bool = True,
        min_group_count: int = 8,
        ema_momentum: float = 0.1,
        **kwargs,
    ):
        super().__init__(n_genes=n_genes, n_labels=n_labels, empirical_ct_means=empirical_ct_means, **kwargs)
        self.n_batch = n_batch
        self.batch_baseline = batch_baseline
        self.center_residual = center_residual
        self.center_inputs = center_inputs
        self.min_group_count = min_group_count
        self.ema_momentum = ema_momentum

        self.register_buffer("bt_profiles", empirical_bt_means.float())
        self.register_buffer("resid_ema", torch.zeros(n_batch * n_labels, n_genes))
        self.register_buffer("resid_ema_seen", torch.zeros(n_batch * n_labels, dtype=torch.bool))

        if center_inputs:
            # neighbor MLP now takes [centered expression ; neighbor cell-type embedding]
            self.nn_ct_embed = nn.Embedding(n_labels, self.n_label_embed)
            self.nn_embed = ResNetMLP(
                n_input=n_genes + self.n_label_embed,
                n_output=self.n_nn_embed,
                n_layers=2,
                n_hidden=self.n_nn_embed_hidden,
                dropout=0.0,
            )
        self._nn_context = None  # used only by AMICIBatch's interpretation wrappers (see below)
        self.setup()

    # ---------------- inference: neighbor embedding (Idea 2) ----------------
    def _get_inference_input(self, tensors):
        out = super()._get_inference_input(tensors)
        if self.center_inputs:
            nn_labels = tensors[NN_REGISTRY_KEYS.NN_LABELS_KEY]                    # B x n x 1
            batch = tensors[REGISTRY_KEYS.BATCH_KEY]                               # B x 1
            out["nn_labels"] = nn_labels
            out["nn_batch"] = batch.expand(-1, nn_labels.shape[1]).unsqueeze(-1)  # same slide as receiver
        return out

    def inference(self, labels, nn_X, nn_batch=None, nn_labels=None):
        label_embed = self.hook_label_embed(self.ct_embed(labels).reshape(labels.shape[0], -1))
        x = nn_X
        if self.center_inputs:
            if nn_batch is None or nn_labels is None:
                nn_batch, nn_labels = self._next_context(nn_X)
            b = nn_batch.long().reshape(nn_X.shape[0], nn_X.shape[1])
            t = nn_labels.long().reshape(nn_X.shape[0], nn_X.shape[1])
            x = torch.cat([nn_X - self.bt_profiles[b, t], self.nn_ct_embed(t)], dim=-1)
        nn_embed = self.hook_nn_embed(self.nn_embed(x))
        return {"label_embed": label_embed, "nn_embed": nn_embed}

    def _next_context(self, nn_X):
        """Neighbor slide/type for interpretation helpers that call inference() without them."""
        if self._nn_context is None:
            raise RuntimeError(
                "center_inputs=True needs each neighbor's slide and cell type. Call this through an "
                "AMICIBatch method (training, get_elbo, get_neighbor_ablation_scores, "
                "get_attention_patterns, get_counterfactual_attention_patterns, get_nn_embed, "
                "get_gene_residual_contributions all supply them).")
        b, t = next(self._nn_context)
        if b.numel() != nn_X.shape[1]:
            raise RuntimeError("neighbor context out of sync with the data loader")
        dev = nn_X.device
        return b.to(dev).reshape(1, -1), t.to(dev).reshape(1, -1)

    # ---------------- generative: baseline + centered residual (Idea 1) ----------------
    def _get_generative_input(self, tensors, inference_outputs):
        out = super()._get_generative_input(tensors, inference_outputs)
        out["batch_index"] = tensors[REGISTRY_KEYS.BATCH_KEY]
        return out

    @auto_move_data
    def generative(self, labels, label_embed, nn_embed, nn_dist, batch_index=None, **kwargs):
        outs = super().generative(labels, label_embed, nn_embed, nn_dist, **kwargs)
        if batch_index is None:  # interpretation helpers that only need attention / values
            return outs
        t = labels.long().reshape(-1)
        b = batch_index.long().reshape(-1)
        residual = outs["residual"].reshape(t.shape[0], -1)

        if self.center_residual:
            g = b * self.n_labels + t
            if self.training:
                n_groups = self.n_batch * self.n_labels
                sums = torch.zeros(n_groups, residual.shape[1], device=residual.device).index_add(0, g, residual)
                counts = torch.bincount(g, minlength=n_groups).float()
                mb_mean = sums / counts.clamp(min=1).unsqueeze(1)
                with torch.no_grad():
                    present = counts > 0
                    first = present & ~self.resid_ema_seen
                    upd = present & self.resid_ema_seen
                    self.resid_ema[first] = mb_mean[first].detach()
                    self.resid_ema[upd] = ((1 - self.ema_momentum) * self.resid_ema[upd]
                                           + self.ema_momentum * mb_mean[upd].detach())
                    self.resid_ema_seen |= present
                use_mb = (counts >= self.min_group_count)[g].unsqueeze(1)
                residual = residual - torch.where(use_mb, mb_mean[g], self.resid_ema[g])
            else:
                residual = residual - self.resid_ema[g]

        baseline = self.bt_profiles[b, t] if self.batch_baseline else self.ct_profiles[t]
        outs["residual"] = residual
        outs["prediction"] = (baseline + residual).float()
        return outs


def _with_neighbor_context(method_name):
    """Wrap an AMICI interpretation method that calls module.inference() without neighbor slide/type."""
    def wrapper(self, *args, adata=None, indices=None, batch_size=None, **kwargs):
        parent = getattr(super(AMICIBatch, self), method_name)
        if not self.module.center_inputs:
            return parent(*args, adata=adata, indices=indices, batch_size=batch_size, **kwargs)
        ad_v = self._validate_anndata(adata)
        idx = indices if indices is not None else list(range(ad_v.n_obs))
        scdl = self._make_data_loader(adata=ad_v, indices=idx, batch_size=batch_size)
        ctx = [(t[REGISTRY_KEYS.BATCH_KEY].reshape(-1), t[REGISTRY_KEYS.LABELS_KEY].reshape(-1)) for t in scdl]
        self.module._nn_context = iter(ctx)
        try:
            return parent(*args, adata=adata, indices=indices, batch_size=batch_size, **kwargs)
        finally:
            self.module._nn_context = None
    wrapper.__name__ = method_name
    return wrapper


class AMICIBatch(AMICI):
    _module_cls = AMICIBatchModule

    def __init__(self, adata, batch_baseline: bool = True, center_residual: bool = True,
                 center_inputs: bool = True, kappa: float = 10.0, min_group_count: int = 8,
                 **model_kwargs):
        self._data_splitter_cls.data_loader_cls = self._data_loader_cls  # same hack as AMICI.__init__
        BaseModelClass.__init__(self, adata)
        self.n_neighbors = adata.uns[NN_REGISTRY_KEYS.NUM_NEIGHBORS_KEY]

        X = self.adata_manager.get_from_registry(REGISTRY_KEYS.X_KEY)
        X = X.toarray() if hasattr(X, "toarray") else np.asarray(X)
        lab = np.asarray(self.adata_manager.get_from_registry(REGISTRY_KEYS.LABELS_KEY)).ravel()
        bat = np.asarray(self.adata_manager.get_from_registry(REGISTRY_KEYS.BATCH_KEY)).ravel()
        n_labels, n_batch = self.summary_stats.n_labels, self.summary_stats.n_batch

        ct = np.zeros((n_labels, X.shape[1]), dtype=np.float32)
        bt = np.zeros((n_batch, n_labels, X.shape[1]), dtype=np.float32)
        n_bt = np.zeros((n_batch, n_labels), dtype=int)
        for t in range(n_labels):
            mt = lab == t
            ct[t] = X[mt].mean(0) if mt.any() else 0
            for b in range(n_batch):
                m = mt & (bat == b)
                n_bt[b, t] = int(m.sum())
                s = n_bt[b, t] / (n_bt[b, t] + kappa) if n_bt[b, t] else 0.0
                bt[b, t] = ct[t] + s * ((X[m].mean(0) if m.any() else ct[t]) - ct[t])
        self.group_counts_ = n_bt

        self.module = self._module_cls(
            n_genes=adata.n_vars, n_labels=n_labels, empirical_ct_means=torch.tensor(ct),
            n_batch=n_batch, empirical_bt_means=torch.tensor(bt),
            batch_baseline=batch_baseline, center_residual=center_residual, center_inputs=center_inputs,
            min_group_count=min_group_count, **model_kwargs,
        )
        self._model_summary_string = (f"AMICIBatch (batch_baseline={batch_baseline}, "
                                      f"center_residual={center_residual}, center_inputs={center_inputs})")
        self.init_params_ = self._get_init_params(locals())

    @classmethod
    def setup_anndata(cls, adata, batch_key: str, layer=None, labels_key=None, coord_obsm_key=None,
                      nn_dist_key="_nn_dist", nn_idx_key="_nn_idx", cell_radius_key=None,
                      n_neighbors=None, **kwargs):
        if n_neighbors is None:
            n_neighbors = adata.uns.get(NN_REGISTRY_KEYS.NUM_NEIGHBORS_KEY, 30)
        setup_method_args = cls._get_setup_method_args(**locals())
        cls._compute_nn(adata, coord_obsm_key, nn_idx_key, nn_dist_key, n_neighbors,
                        labels_obs_key=labels_key, exclude_self_labels=True, cell_radius_key=cell_radius_key)
        fields = [
            LayerField(REGISTRY_KEYS.X_KEY, layer, is_count_data=False),
            CategoricalObsField(REGISTRY_KEYS.LABELS_KEY, labels_key),
            CategoricalObsField(REGISTRY_KEYS.BATCH_KEY, batch_key),
            ObsmField(NN_REGISTRY_KEYS.COORD_KEY, coord_obsm_key),
            ObsmField(NN_REGISTRY_KEYS.NN_IDX_KEY, nn_idx_key),
            ObsmField(NN_REGISTRY_KEYS.NN_DIST_KEY, nn_dist_key),
        ]
        manager = AnnDataManager(fields=fields, setup_method_args=setup_method_args)
        manager.register_fields(adata, **kwargs)
        cls.register_manager(manager)

    # interpretation helpers that call module.inference() directly
    get_counterfactual_attention_patterns = _with_neighbor_context("get_counterfactual_attention_patterns")
    get_nn_embed = _with_neighbor_context("get_nn_embed")
    get_gene_residual_contributions = _with_neighbor_context("get_gene_residual_contributions")

    @torch.no_grad()
    def calibrate_residual_centering(self, adata=None, batch_size=256):
        """Set the inference-time group means of r to their exact values on adata (call after training).

        During training the group means come from minibatches and a running average, with neighbor
        dropout on, so they lag behind the final model. This recomputes them once in eval mode.
        """
        if not self.module.center_residual:
            return
        ad_v = self._validate_anndata(adata)
        self.module.eval()
        saved = self.module.resid_ema.clone()
        self.module.resid_ema.zero_()                      # eval-mode residual minus 0 = raw residual
        r = self.get_predictions(adata=ad_v, batch_size=batch_size, get_residuals=True, prog_bar=False)
        lab_key = self.adata_manager.get_state_registry(REGISTRY_KEYS.LABELS_KEY).original_key
        bat_key = self.adata_manager.get_state_registry(REGISTRY_KEYS.BATCH_KEY).original_key
        lab_map = list(self.adata_manager.get_state_registry(REGISTRY_KEYS.LABELS_KEY).categorical_mapping)
        bat_map = list(self.adata_manager.get_state_registry(REGISTRY_KEYS.BATCH_KEY).categorical_mapping)
        t = np.array([lab_map.index(x) for x in ad_v.obs[lab_key].astype(str)])
        b = np.array([bat_map.index(x) for x in ad_v.obs[bat_key].astype(str)])
        new = saved.cpu().numpy()
        for g in np.unique(b * self.module.n_labels + t):
            m = (b * self.module.n_labels + t) == g
            new[g] = r[m].mean(0)
        self.module.resid_ema.copy_(torch.tensor(new, device=self.module.resid_ema.device))

    # ---------------- outputs specific to the decomposition ----------------
    def technical_shift_table(self) -> pd.DataFrame:
        """delta_{b,t} = baseline_{b,t} - mu_t per (slide, cell type, gene): the learned-out technical shift."""
        bt = self.module.bt_profiles.cpu().numpy()
        ct = self.module.ct_profiles.cpu().numpy()
        slides = self.adata_manager.get_state_registry(REGISTRY_KEYS.BATCH_KEY).categorical_mapping
        types = self.adata_manager.get_state_registry(REGISTRY_KEYS.LABELS_KEY).categorical_mapping
        rows, idx = [], []
        for b, s in enumerate(slides):
            for t, ty in enumerate(types):
                rows.append(bt[b, t] - ct[t])
                idx.append((s, ty, int(self.group_counts_[b, t]) if hasattr(self, "group_counts_") else -1))
        genes = self.adata.var_names if self.adata is not None else None
        return pd.DataFrame(rows, index=pd.MultiIndex.from_tuples(idx, names=["slice_id", "cell_type", "n_cells"]),
                            columns=genes)

    def residual_group_means(self, adata=None, batch_size=128) -> pd.DataFrame:
        """Mean of r_bio per (slide, cell type) on adata: ~0 when center_residual holds."""
        ad_v = self._validate_anndata(adata)
        self.module.eval()
        r = self.get_predictions(adata=ad_v, batch_size=batch_size, get_residuals=True, prog_bar=False)
        lab_key = self.adata_manager.get_state_registry(REGISTRY_KEYS.LABELS_KEY).original_key
        bat_key = self.adata_manager.get_state_registry(REGISTRY_KEYS.BATCH_KEY).original_key
        df = pd.DataFrame({"slice_id": ad_v.obs[bat_key].astype(str).values,
                           "cell_type": ad_v.obs[lab_key].astype(str).values,
                           "mean_abs_group_mean": 0.0})
        out = []
        for (s, t), grp in df.groupby(["slice_id", "cell_type"]).groups.items():
            gm = r[np.asarray(list(grp))].mean(0)
            out.append(dict(slice_id=s, cell_type=t, n_cells=len(grp),
                            mean_abs_group_mean=float(np.abs(gm).mean()),
                            mean_abs_residual=float(np.abs(r[np.asarray(list(grp))]).mean())))
        return pd.DataFrame(out)
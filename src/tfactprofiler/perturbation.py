import warnings
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge


def perturbation_predict(
    RNA_data: pd.DataFrame,
    prior_knowledge: pd.DataFrame,
    perturb_tf: str,
    perturb_value: float,
    alpha: float = 0.01,
    step_limit: int = 3,
) -> pd.DataFrame:
    """Per-sample perturbation prediction"""

    # ---- validate & collapse duplicate edges ----
    need = {"source", "target", "weight"}
    if not need.issubset(prior_knowledge.columns):
        raise ValueError(f"prior_knowledge must contain {need}")
    pk = (prior_knowledge.groupby(["target", "source"], as_index=False)["weight"].mean())

    # intersect with genes present in RNA_data
    genes = set(RNA_data.index.astype(str))
    pk = pk[pk["target"].isin(genes) & pk["source"].isin(genes)].copy()
    if pk.empty:
        raise ValueError("No prior edges remain after intersecting with RNA_data genes.")

    # consistent ordering
    target_names = sorted(pk["target"].unique().tolist())
    tf_names = sorted(pk["source"].unique().tolist())
    # dense W and make sure it's finite
    W = (pk.pivot(index="target", columns="source", values="weight")
            .reindex(index=target_names, columns=tf_names, fill_value=0.0))
    W = W.astype(float)
    if not np.isfinite(W.values).all():
        W = W.fillna(0.0)
    W_np_master = W.values

    # target indices for TFs that are also in targets
    target_idx_map = {g: i for i, g in enumerate(W.index)}
    tf_to_tidx = {tf: target_idx_map[tf] for tf in tf_names if tf in target_idx_map}
    results = pd.DataFrame({"Gene_Symbol": target_names})

    sample_name=RNA_data.columns.astype(str)[0]
    # y (observed) — allow NaN for now; drop rows only in training step
    y_all = RNA_data.loc[target_names, sample_name].astype(float).values

    # x (TF vector) aligned to W columns
    x_full = RNA_data.loc[tf_names, sample_name].astype(float)

    # drop TFs with NaN/Inf in x and drop corresponding columns in W
    keep_tf = np.isfinite(x_full.values)
    if keep_tf.sum() == 0:
        warnings.warn(f"[{sample_name}] all TF values are NaN/Inf; skipping.")
    #    continue
    tf_names_local = [tf for tf, k in zip(tf_names, keep_tf) if k]
    x = x_full[keep_tf].values
    W_local = W.loc[:, tf_names_local]
    W_np = W_local.values

    # safety: ensure W is finite
    if not np.isfinite(W_np).all():
        W_np = np.nan_to_num(W_np, nan=0.0, posinf=0.0, neginf=0.0)
    # design matrix for ALL genes (used for prediction later)
    X_all = W_np * x[np.newaxis, :]
    row_ok = np.isfinite(X_all).all(axis=1) & np.isfinite(y_all)
    X = X_all[row_ok, :]
    y = y_all[row_ok]

    # fit scaling factors (a) with intercept
    model = Ridge(alpha=alpha, fit_intercept=True)
    model.fit(X, y)
    a_hat = model.coef_.astype(float)   # shape (n_tfs_kept,)
    b0 = float(model.intercept_)

    # original prediction for ALL genes
    original_pred = X_all @ a_hat + b0

    # ---- perturbation with iterative propagation (spec maintained) ----
    if perturb_tf not in tf_names_local:
        warnings.warn(f"[{sample_name}] perturb_tf '{perturb_tf}' not found among TFs or downstream genes.")
        perturbed_pred = original_pred.copy()
    else:
        # pairs (j_tf, i_target) to update TFs from predicted mRNA
        tf_overlap_pairs = []
        for j, tf in enumerate(tf_names_local):
            if tf in tf_to_tidx:
                tf_overlap_pairs.append((j, tf_to_tidx[tf]))

        coef=a_hat.copy()
        intercept=b0
        x_vec=x
        tf_names=tf_names_local.copy()

        name_to_idx = {tf: i for i, tf in enumerate(tf_names)} 
        k = name_to_idx[perturb_tf]        
        current_tf = x_vec.copy()
        current_tf[k] = perturb_value
        base_norm = np.linalg.norm(x_vec) + 1e-12
        growth_cap = 3.0
        for _ in range(max(1, int(step_limit))):
            new_mrna = W_np @ (coef * current_tf) + intercept     # (n_genes,)
            new_tf = current_tf.copy()
            for j_tf, i_target in tf_overlap_pairs:
                new_tf[j_tf] = new_mrna[i_target]
            new_tf[k] = perturb_value

            # simple damping to avoid blow-up
            new_norm = np.linalg.norm(new_tf)
            if new_norm > growth_cap * base_norm:
                new_tf *= (growth_cap * base_norm) / new_norm
            current_tf = new_tf
            current_tf[k] = perturb_value
        perturbed_pred = W_np @ (coef * current_tf) + intercept
    delta = perturbed_pred - original_pred
    predicted = y_all + delta
    results[f"Predicted_{sample_name}"] = predicted
    return results


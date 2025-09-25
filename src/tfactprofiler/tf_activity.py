import pandas as pd
import numpy as np

def estimate_reliability(
    adata,
    prior_df,
    cluster_key: str = "celltype",
    alpha: float = 0.1,
    alphas: "Optional[Union[List[float], np.ndarray]]" = None,
    cv_folds: "Optional[int]" = None,
    use_hvg_for_targets: bool = True,
    avoid_self_pred: bool = True,
    verbose: bool = True,
):
    """
    Parameters
    ----------
    adata : anndata.AnnData
        Expression matrix with `adata.X`, `adata.var_names`, `adata.obs_names`, and
        `adata.obs[cluster_key]`. If `use_hvg_for_targets=True`, requires `adata.var["highly_variable"]`.
    prior_df : pandas.DataFrame
        DataFrame with columns {"source", "target", "weight"} describing prior edges (TF->gene).
    cluster_key : str, default "celltype"
        Observation column that defines clusters.
    alpha : float, default 0.1
        Ridge penalty when `Ridge` (no CV) is used.
    alphas : Optional[List[float] or np.ndarray], default None
        Candidate alphas for `RidgeCV`. Effective only if `cv_folds >= 2`.
    cv_folds : Optional[int], default None
        Number of CV folds for `RidgeCV`. Use >=2 to activate CV; otherwise uses `Ridge`.
    use_hvg_for_targets : bool, default True
        Restrict target genes to highly variable genes if available.
    avoid_self_pred : bool, default True
        If True, drop the target gene from predictors when it is also a TF (self-prediction).
    verbose : bool, default True
        Print progress and matrix stats.

    Returns
    -------
    The input `adata` with:
      - adata.obs["mse"]: per-cell residual MSE
    """
    # Local imports to keep the function self-contained.
    from typing import Optional, Union, List
    import numpy as np
    import pandas as pd
    from sklearn.linear_model import Ridge, RidgeCV
    from sklearn.model_selection import KFold

    # === Basic input checks ===
    # Ensure the cluster key exists in adata.obs.
    if cluster_key not in adata.obs.columns:
        raise KeyError(f"cluster_key '{cluster_key}' not found in adata.obs")

    # Ensure prior_df has required columns.
    req_cols = {"source", "target", "weight"}
    if not req_cols.issubset(prior_df.columns):
        missing = req_cols - set(prior_df.columns)
        raise ValueError(f"prior_df missing required columns: {missing}")

    # Normalize var/obs names to strings to avoid dtype mismatches.
    var_names = list(map(str, adata.var_names))
    obs_names = list(map(str, adata.obs_names))

    # === Optionally restrict target genes to highly variable genes ===
    # If HVG gating is requested, require 'highly_variable' in adata.var.
    if use_hvg_for_targets:
        if "highly_variable" not in adata.var.columns:
            raise ValueError("use_hvg_for_targets=True but 'highly_variable' not found in adata.var")
        allowed_targets = set(
            np.asarray(adata.var_names)[adata.var["highly_variable"].astype(bool)].tolist()
        )
    else:
        allowed_targets = set(var_names)

    # Keep only prior edges whose targets are allowed and both ends exist in expression genes.
    pk = prior_df[
        (prior_df["target"].isin(allowed_targets))
        & (prior_df["source"].isin(var_names))
        & (prior_df["target"].isin(var_names))
    ].copy()

    # === Build prior matrix W as a dense, labeled DataFrame ===
    # If multiple edges exist for the same (source, target), aggregate by mean weight.
    pk_agg = pk.groupby(["source", "target"], as_index=False)["weight"].mean()

    # Rows = TF (source), Columns = gene (target). Missing pairs filled with 0.
    W_df = (
        pk_agg.pivot(index="source", columns="target", values="weight")
        .fillna(0.0)
        .sort_index(axis=0)
        .sort_index(axis=1)
    )

    tfs_all = [tf for tf in W_df.index.tolist() if tf in var_names]
    targets_all = [g for g in W_df.columns.tolist() if g in var_names]
    W_df = W_df.loc[tfs_all, targets_all]  # shape: (TF x gene)

    if W_df.shape[0] == 0 or W_df.shape[1] == 0:
        raise ValueError("W_df is empty after filtering. Check prior_df and allowed targets/genes.")

    if verbose:
        print(f"[prior] #TFs in W (present): {len(tfs_all)}")
        print(f"[prior] #targets in W (present{' +HVG' if use_hvg_for_targets else ''}): {len(targets_all)}")

    # === Output containers ===
    # Global per-cell diagnostics (indexed by the full adata.obs_names).
    fp_mse_global = pd.Series(index=obs_names, dtype=float, name="fp_mse")
    fp_corr_global = pd.Series(index=obs_names, dtype=float, name="fp_corr")
    #diag_per_cluster = {}
    #K_store = {}
    name_to_col = {g: idx for idx, g in enumerate(var_names)}

    # === Iterate over clusters ===
    clusters = list(pd.unique(adata.obs[cluster_key]))
    for i, cluster in enumerate(clusters):
        if verbose:
            print(f"\n[cluster {i+1}/{len(clusters)}] {cluster}")

        # Subset AnnData to the current cluster.
        mask = (adata.obs[cluster_key] == cluster).to_numpy()
        adata_c = adata[mask, :]
        X_c = adata_c.X
        X_c = X_c.toarray() if hasattr(X_c, "toarray") else np.asarray(X_c)
        X_c = np.asarray(X_c, dtype=float, order="C")

        n_cells_cluster = X_c.shape[0]
        if verbose:
            print(f"  cells: {n_cells_cluster}")

        # Design matrix for TF expression (cells x TFs).
        try:
            tf_cols = [name_to_col[tf] for tf in tfs_all]
        except KeyError as e:
            # Should not happen because we filtered tfs_all against var_names.
            raise ValueError(f"Some TFs not present in var_names: {e}")

        X_TF_cells = X_c[:, tf_cols]  # (cells, n_tfs)
        n_tfs = len(tfs_all)
        n_targets = len(targets_all)

        # Column-wise standard deviations of TFs (no centering).
        # Zero-variance TFs will be marked with 0.0 to exclude from predictors later.
        sigma_x = X_TF_cells.std(axis=0, ddof=0).astype(float)
        sigma_x[sigma_x == 0] = 0.0
        
        keep_mask = sigma_x > float(0.0) ###
        keep_idx = np.where(keep_mask)[0] ###
        # Containers for per-target model: K (genes x TFs), intercepts b, and in-sample R^2.
        K = np.zeros((n_targets, n_tfs), dtype=float)
        b = np.zeros(n_targets, dtype=float)
        R2 = np.full(n_targets, np.nan, dtype=float)

        # === Fit masked ridge per target gene using only TFs connected in W_df ===
        # For each target gene, use the nonzero TFs in the corresponding W column.
        for gi, g in enumerate(targets_all):
            if g not in name_to_col:
                continue

            # Candidate TFs are those with nonzero prior weight to this target.
            w_col_vals = W_df[g].to_numpy()
            cand_tfs = np.where(w_col_vals != 0.0)[0]
            if cand_tfs.size == 0:
                continue

            pred_tf_idx = cand_tfs.tolist()

            # Optionally avoid self-prediction: if the target gene is itself a TF.
            if avoid_self_pred and g in tfs_all:
                self_idx = tfs_all.index(g)
                if self_idx in pred_tf_idx:
                    pred_tf_idx.remove(self_idx)
                    if len(pred_tf_idx) == 0:
                        continue

            # Exclude TFs with zero variance (cannot be used as predictors).
            pred_tf_idx = [j for j in pred_tf_idx if sigma_x[j] > 0]
            if len(pred_tf_idx) == 0:
                continue

            # Build standardized design matrix: divide each TF column by its std (no centering).
            Xp = X_TF_cells[:, pred_tf_idx]  # (cells, p)
            Xp_std = Xp / sigma_x[np.array(pred_tf_idx)]

            # Response vector y for the target gene; scale by its std to aid conditioning.
            yi = X_c[:, name_to_col[g]].ravel()
            sigma_y = float(np.std(yi))
            if sigma_y <= 0:
                sigma_y = 1.0
            y_std = yi / sigma_y

            # Choose RidgeCV if alphas and cv_folds are provided; otherwise fixed-alpha Ridge.
            if (alphas is not None) and (cv_folds is not None) and (cv_folds >= 2):
                mdl = RidgeCV(
                    alphas=alphas,
                    cv=KFold(cv_folds, shuffle=True, random_state=0),
                    fit_intercept=True,
                )
            else:
                mdl = Ridge(alpha=alpha, fit_intercept=True)

            # Fit on standardized predictors/response.
            mdl.fit(Xp_std, y_std)

            # Convert coefficients back to the original (raw) scale:
            #   beta_raw = (sigma_y * beta_std) / sigma_x
            beta = np.asarray(mdl.coef_).ravel()  # (p,)
            raw_coef = (sigma_y * beta) / sigma_x[np.array(pred_tf_idx)]
            K[gi, np.array(pred_tf_idx, dtype=int)] = raw_coef

            # Intercept on the original scale.
            b[gi] = float(mdl.intercept_) * sigma_y

            # In-sample R^2 for diagnostics.
            y_pred_raw = (Xp_std @ beta) * sigma_y + b[gi]
            ss_res = float(np.sum((yi - y_pred_raw) ** 2))
            ss_tot = float(np.sum((yi - yi.mean()) ** 2)) + 1e-12
            R2[gi] = 1.0 - ss_res / ss_tot

        # === Labeled outputs for this cluster (not returned; preserved for parity) ===
        # K_df: rows = target genes (mRNA), columns = TFs.
        K_df = pd.DataFrame(K, index=targets_all, columns=tfs_all)
        b_se = pd.Series(b, index=targets_all, name="intercept_raw")
        R2_se = pd.Series(R2, index=targets_all, name="R2")

        # === Fixed-point residual MSE and correlation per cell ===
        # Predicted TF expression follows: x_hat = W (K x + b) = (W K) x + W b
        W_vals = W_df.values
        WK = W_vals @ K_df.values                     # (TF x TF)
        Wb = (W_vals @ b_se.values.reshape(-1, 1)).ravel().astype(float)  # (TF,)

        X_all = X_TF_cells if isinstance(X_TF_cells, np.ndarray) else X_TF_cells.toarray()
        X_all = X_all.T  # (TF x N_cells)

        X_pred = WK @ X_all + Wb[:, None]  # (TF x N_cells)
        res = X_pred - X_all               # (TF x N_cells)
        res = res[keep_idx, :]     ###
        # Per-cell MSE over TF dimensions.
        mse_vals = np.mean(res * res, axis=0)  # (N_cells,)
        fp_mse_series = pd.Series(mse_vals, index=list(map(str, adata_c.obs_names)), name="mse")
        fp_mse_global.loc[fp_mse_series.index] = fp_mse_series.values

        # Per-cell Pearson correlation between predicted and observed TF expressions.
        Xc = X_all - X_all.mean(axis=0, keepdims=True)
        Xc = Xc[keep_idx, :]     ###
        Xpc = X_pred - X_pred.mean(axis=0, keepdims=True)
        Xpc = Xpc[keep_idx, :]     ###
        num = np.sum(Xc * Xpc, axis=0)                                # (N_cells,)
        den = np.sqrt(np.sum(Xc**2, axis=0) * np.sum(Xpc**2, axis=0)) # (N_cells,)
        corr_vals = np.divide(num, den, out=np.full_like(num, np.nan, dtype=float), where=den > 0)
        fp_corr_series = pd.Series(corr_vals, index=list(map(str, adata_c.obs_names)), name="corr")
        fp_corr_global.loc[fp_corr_series.index] = fp_corr_series.values

    # Write diagnostics back to the full AnnData object.
    adata.obs["mse"] = fp_mse_global.reindex(adata.obs_names).astype(float)
    #adata.obs["corr"] = fp_corr_global.reindex(adata.obs_names).astype(float)
    return adata
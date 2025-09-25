from __future__ import annotations
from typing import Tuple, Union
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

ArrayLike = Union[np.ndarray, pd.DataFrame]

def train_W(Y: pd.DataFrame, P: pd.DataFrame, K: int=10, lambda_ridge: float = 0.1):
    """Training W using gene embedding (G) and perturbation embedding (P).

    Parameters
    ----------
    Y : pd.DataFrame
        Gene expression matrix (genes x samples).
    K : int
        Number of components for PCA (gene embeddings).
    P : pd.DataFrame
        Perturbation embedding (samples x L).
    lambda_ridge : float
        Ridge penalty (λ).

    Returns
    -------
    tuple
        (W, b, G, P_)
    """
    Y_np = Y.values
    b = Y_np.mean(axis=1, keepdims=True)
    Y_centered = Y_np - b

    # G
    K_eff = min(K, Y_centered.shape[0], Y_centered.shape[1])
    pca = PCA(n_components=K_eff)
    G = pca.fit_transform(Y_centered)
    
    # Convert inputs
    K = len(pd.DataFrame(G).T)
    G_arr = G.values if isinstance(G, pd.DataFrame) else np.asarray(G)
    P_np = P.values

    GTG = G_arr.T @ G_arr
    PTP = P_np.T @ P_np
    L = P_np.shape[1]
    #W = np.linalg.inv(GTG + lambda_ridge * np.eye(K)) @ (G_arr.T @ Y_centered @ P_np) @ np.linalg.inv(PTP + lambda_ridge * np.eye(L))
    lamG = lambda_ridge
    lamP = lambda_ridge
    A = GTG + lamG * np.eye(K)                             # (K x K)
    B = PTP + lamP * np.eye(L)                             # (L x L)
    C = G_arr.T @ Y_centered @ P_np                        # (K x L)
    M = np.kron(B.T, A)                                    # ((K*L) x (K*L))
    w = np.linalg.solve(M, C.reshape(-1, order='F'))       # (K*L,)
    W = w.reshape(K, L, order='F')                         # (K x L)
    return W, b, G, P


def predict_withW(G: ArrayLike, W: np.ndarray, P_block: ArrayLike, b: np.ndarray):
    """Predict (reconstruct) Y block given learned W and intercept b.

    Parameters
    ----------
    G : array-like (genes x K)
        Gene embedding (numpy array or DataFrame).
    W : np.ndarray (K x L)
        Learned weights.
    P_block : array-like (samples x L)
        Perturbation embedding (DataFrame or ndarray).
    b : np.ndarray (genes x 1)
        Per-gene intercept.

    Returns
    -------
    np.ndarray
        Predicted Y (genes x samples).
    """
    G_arr = G.values if isinstance(G, pd.DataFrame) else np.asarray(G)
    P_np = P_block.to_numpy() if isinstance(P_block, pd.DataFrame) else np.asarray(P_block)
    return G_arr @ W @ P_np.T + b


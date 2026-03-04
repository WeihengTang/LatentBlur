"""
pca_baseline.py
---------------
PCA baseline for PSF representation quality.

For each n_components in [8, 16, 32, 64, 128]:
  1. Fit PCA on the flattened train kernels.
  2. Project-and-reconstruct both ID-Test and OOD-Test kernels.
  3. Compute per-sample MSE and PSNR, then report means.

Results are saved to results/pca_results.csv.
"""

import os
import csv
import numpy as np
from sklearn.decomposition import PCA

# ── config ─────────────────────────────────────────────────────────────────────
DATA_DIR    = "data"
RESULTS_DIR = "results"
DIMS        = [8, 16, 32, 64, 128]
KERNEL_SIZE = 64
FLAT_DIM    = KERNEL_SIZE * KERNEL_SIZE   # 4096


# ── metric helpers ─────────────────────────────────────────────────────────────

def mse(x: np.ndarray, y: np.ndarray) -> float:
    """Mean squared error between two arrays."""
    return float(np.mean((x - y) ** 2))


def psnr(x: np.ndarray, y: np.ndarray, data_range: float = 1.0) -> float:
    """
    Peak Signal-to-Noise Ratio.
    PSNR = 10 * log10(data_range^2 / MSE)
    Returns -inf when MSE == 0 (perfect reconstruction).
    """
    err = mse(x, y)
    if err == 0:
        return float("inf")
    return 10.0 * np.log10(data_range ** 2 / err)


def batch_metrics(originals: np.ndarray,
                  reconstructions: np.ndarray) -> tuple[float, float]:
    """
    Compute mean MSE and mean PSNR across a batch.
    Both arrays should be shaped (N, H, W) or (N, D).
    """
    assert originals.shape == reconstructions.shape
    n = originals.shape[0]
    orig_flat  = originals.reshape(n, -1)
    recon_flat = reconstructions.reshape(n, -1)

    mse_vals  = np.mean((orig_flat - recon_flat) ** 2, axis=1)   # (N,)
    psnr_vals = 10.0 * np.log10(1.0 / np.maximum(mse_vals, 1e-12))  # (N,)

    return float(mse_vals.mean()), float(psnr_vals.mean())


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # load data
    print("Loading data…")
    train_k  = np.load(os.path.join(DATA_DIR, "train.npz"))["kernels"]
    id_k     = np.load(os.path.join(DATA_DIR, "id_test.npz"))["kernels"]
    ood_k    = np.load(os.path.join(DATA_DIR, "ood_test.npz"))["kernels"]

    # flatten  →  (N, 4096)
    train_flat = train_k.reshape(len(train_k), -1).astype(np.float64)
    id_flat    = id_k.reshape(len(id_k),    -1).astype(np.float64)
    ood_flat   = ood_k.reshape(len(ood_k),  -1).astype(np.float64)

    rows = []

    for dim in DIMS:
        print(f"  PCA dim={dim:3d} …", end="", flush=True)

        pca = PCA(n_components=dim, random_state=42)
        pca.fit(train_flat)

        # reconstruct: project then inverse-project
        id_recon  = pca.inverse_transform(pca.transform(id_flat)).astype(np.float32)
        ood_recon = pca.inverse_transform(pca.transform(ood_flat)).astype(np.float32)

        # clip to [0,1] (tiny negative values can appear from floating-point arithmetic)
        id_recon  = np.clip(id_recon,  0.0, 1.0)
        ood_recon = np.clip(ood_recon, 0.0, 1.0)

        id_mse,  id_psnr  = batch_metrics(id_flat.astype(np.float32),  id_recon)
        ood_mse, ood_psnr = batch_metrics(ood_flat.astype(np.float32), ood_recon)

        var_explained = float(pca.explained_variance_ratio_.sum()) * 100

        rows.append({
            "model": "PCA",
            "dim": dim,
            "id_mse":   round(id_mse,   8),
            "id_psnr":  round(id_psnr,  4),
            "ood_mse":  round(ood_mse,  8),
            "ood_psnr": round(ood_psnr, 4),
            "var_explained_pct": round(var_explained, 3),
        })

        print(f"  ID PSNR={id_psnr:.2f} dB  |  OOD PSNR={ood_psnr:.2f} dB  "
              f"  (var explained: {var_explained:.1f}%)")

    # save CSV
    csv_path = os.path.join(RESULTS_DIR, "pca_results.csv")
    fieldnames = ["model", "dim", "id_mse", "id_psnr",
                  "ood_mse", "ood_psnr", "var_explained_pct"]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nResults saved to {csv_path}")


if __name__ == "__main__":
    main()

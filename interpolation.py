"""
interpolation.py
----------------
Compares latent-space interpolation (via the trained AE, dim=32) against
naive pixel-space interpolation for PSF mid-point estimation.

Experiment:
  PSF_A  : L=10, theta=0°
  PSF_B  : L=20, theta=90°
  GT_mid : L=15, theta=45°  (physical parameter midpoint)

  Pixel interp  : (PSF_A + PSF_B) / 2
  Latent interp : decode( (encode(PSF_A) + encode(PSF_B)) / 2 )

Saves:
  results/interpolation_results.csv
  plots/plot_interpolation.png        (visual grid — produced by report_gen.py)
  results/interp_kernels.npz          (kernels for the plot)
"""

import os
import csv
import numpy as np
import torch
from autoencoder import ConvAutoencoder, SCALE

CKPT_PATH   = os.path.join("checkpoints", "ae_dim32.pt")
RESULTS_DIR = "results"
LATENT_DIM  = 32
KERNEL_SIZE = 64
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── re-use dataset helper to generate a single kernel ─────────────────────────

def make_linear_blur_kernel(length: int, angle_deg: float,
                            size: int = KERNEL_SIZE) -> np.ndarray:
    k = np.zeros((size, size), dtype=np.float32)
    cx, cy = size // 2, size // 2
    angle_rad = np.deg2rad(angle_deg)
    cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
    half = length / 2.0
    n_steps = max(length * 4, 32)
    for i in range(n_steps + 1):
        t = -half + i * (length / n_steps)
        x = cx + t * cos_a
        y = cy + t * sin_a
        xi, yi = int(round(x)), int(round(y))
        if 0 <= xi < size and 0 <= yi < size:
            k[yi, xi] += 1.0
    total = k.sum()
    if total > 0:
        k /= total
    return k


def mse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean((a - b) ** 2))


def psnr(a: np.ndarray, b: np.ndarray, data_range: float = 1.0) -> float:
    err = mse(a, b)
    if err == 0:
        return float("inf")
    return 10.0 * np.log10(data_range ** 2 / err)


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # ── generate the three canonical kernels ──────────────────────────────────
    psf_a  = make_linear_blur_kernel(10,  0.0)   # L=10, theta=0°
    psf_b  = make_linear_blur_kernel(20, 90.0)   # L=20, theta=90°
    psf_gt = make_linear_blur_kernel(15, 45.0)   # L=15, theta=45° (GT)

    print(f"PSF_A  : L=10, theta=0°   — sum={psf_a.sum():.4f}")
    print(f"PSF_B  : L=20, theta=90°  — sum={psf_b.sum():.4f}")
    print(f"PSF_GT : L=15, theta=45°  — sum={psf_gt.sum():.4f}")

    # ── pixel interpolation ───────────────────────────────────────────────────
    psf_pixel = (psf_a + psf_b) / 2.0

    # ── latent interpolation ──────────────────────────────────────────────────
    model = ConvAutoencoder(LATENT_DIM).to(DEVICE)
    state = torch.load(CKPT_PATH, map_location=DEVICE)
    model.load_state_dict(state)
    model.eval()

    def to_tensor(k: np.ndarray) -> torch.Tensor:
        return torch.tensor(k * SCALE).unsqueeze(0).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        z_a = model.encode(to_tensor(psf_a))
        z_b = model.encode(to_tensor(psf_b))
        z_mid = (z_a + z_b) / 2.0
        psf_latent = model.decode(z_mid).squeeze().cpu().numpy() / SCALE

    psf_latent = np.clip(psf_latent, 0.0, None)

    # ── metrics ───────────────────────────────────────────────────────────────
    pixel_mse  = mse(psf_pixel,  psf_gt)
    pixel_psnr = psnr(psf_pixel, psf_gt)
    latent_mse  = mse(psf_latent, psf_gt)
    latent_psnr = psnr(psf_latent, psf_gt)

    print(f"\nPixel  interpolation — MSE={pixel_mse:.2e}  PSNR={pixel_psnr:.2f} dB")
    print(f"Latent interpolation — MSE={latent_mse:.2e}  PSNR={latent_psnr:.2f} dB")

    improvement = latent_psnr - pixel_psnr
    winner = "Latent" if latent_mse < pixel_mse else "Pixel"
    print(f"\n{winner} interpolation wins  (PSNR delta: {improvement:+.2f} dB)")

    # ── save CSV ──────────────────────────────────────────────────────────────
    csv_path = os.path.join(RESULTS_DIR, "interpolation_results.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "method", "mse", "psnr_db"
        ])
        writer.writeheader()
        writer.writerow({"method": "pixel_interp",
                         "mse": round(pixel_mse,  8),
                         "psnr_db": round(pixel_psnr, 4)})
        writer.writerow({"method": "latent_interp",
                         "mse": round(latent_mse, 8),
                         "psnr_db": round(latent_psnr, 4)})
    print(f"Saved {csv_path}")

    # ── save kernels for plotting ─────────────────────────────────────────────
    np.savez(
        os.path.join(RESULTS_DIR, "interp_kernels.npz"),
        psf_a=psf_a, psf_b=psf_b,
        psf_gt=psf_gt,
        psf_pixel=psf_pixel,
        psf_latent=psf_latent,
    )
    print(f"Saved results/interp_kernels.npz")


if __name__ == "__main__":
    main()

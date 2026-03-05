"""
autoencoder.py
--------------
Convolutional Autoencoder for PSF representation.

Root cause of previous collapse:
  PSF kernels are extremely sparse (most pixels are 0, nonzero values ~0.001–0.27).
  With raw inputs + Sigmoid output + MSE loss, predicting all-zeros achieves very
  low MSE immediately — the model sticks there and gradients vanish.

Fixes applied:
  1. Global input scaling (SCALE=100): brings nonzero pixel values into [0, ~27]
     so gradients are strong and meaningful.
  2. LeakyReLU (alpha=0.1) + BatchNorm throughout: prevents dying neurons.
  3. ReLU output (not Sigmoid): kernels are non-negative, no upper-bound needed.
  4. Lower LR (1e-4) + gradient clipping + cosine LR schedule.
  5. Evaluation reverts to original scale for fair MSE/PSNR comparison with PCA.

Architecture (encoder):
  64×64  →  Conv(32, 3, s2)+BN+LReLU  →  32×32
         →  Conv(64, 3, s2)+BN+LReLU  →  16×16
         →  Conv(128,3, s2)+BN+LReLU  →   8×8
         →  Flatten  →  Linear(latent_dim)

Decoder mirrors encoder with ConvTranspose2d.

Results saved to results/ae_results.csv.
Trained models saved under checkpoints/.
"""

import os
import csv
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# ── config ─────────────────────────────────────────────────────────────────────
DATA_DIR    = "data"
RESULTS_DIR = "results"
CKPT_DIR    = "checkpoints"
DIMS        = [8, 16, 32, 64, 128]

SCALE       = 100.0        # input pre-scaling factor (reverted for eval)
BATCH_SIZE  = 128
MAX_EPOCHS  = 300
LR          = 1e-4
CLIP_GRAD   = 1.0          # max gradient norm
PATIENCE    = 20           # early-stopping patience
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── model ──────────────────────────────────────────────────────────────────────

class Encoder(nn.Module):
    def __init__(self, latent_dim: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32,  3, stride=2, padding=1),   # → 32×32
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),   # → 16×16
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(64, 128,3, stride=2, padding=1),   # →  8×8
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1, inplace=True),
        )
        self.fc = nn.Linear(128 * 8 * 8, latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.conv(x).flatten(1))


class Decoder(nn.Module):
    def __init__(self, latent_dim: int):
        super().__init__()
        self.fc = nn.Linear(latent_dim, 128 * 8 * 8)
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1, inplace=True),
            nn.ConvTranspose2d(64, 32,  3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1, inplace=True),
            nn.ConvTranspose2d(32, 1,   3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),          # PSF values are non-negative; no upper bound needed
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.deconv(self.fc(z).view(-1, 128, 8, 8))


class ConvAutoencoder(nn.Module):
    def __init__(self, latent_dim: int):
        super().__init__()
        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder(latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)


# ── metrics ────────────────────────────────────────────────────────────────────

def batch_metrics_np(orig: np.ndarray,
                     recon: np.ndarray) -> tuple[float, float]:
    """Mean MSE and mean PSNR on original (unscaled) kernels."""
    n = orig.shape[0]
    o = orig.reshape(n, -1).astype(np.float32)
    r = recon.reshape(n, -1).astype(np.float32)
    mse_vals  = np.mean((o - r) ** 2, axis=1)
    psnr_vals = 10.0 * np.log10(1.0 / np.maximum(mse_vals, 1e-12))
    return float(mse_vals.mean()), float(psnr_vals.mean())


# ── data helpers ───────────────────────────────────────────────────────────────

def load_tensor(path: str, scale: float = 1.0) -> torch.Tensor:
    k = np.load(path)["kernels"].astype(np.float32) * scale
    return torch.tensor(k).unsqueeze(1)          # (N,1,64,64)


# ── training ───────────────────────────────────────────────────────────────────

def train_one_model(latent_dim: int,
                    train_loader: DataLoader,
                    val_loader:   DataLoader) -> ConvAutoencoder:
    model = ConvAutoencoder(latent_dim).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=MAX_EPOCHS, eta_min=LR * 0.01
    )
    criterion = nn.MSELoss()

    best_val_loss  = float("inf")
    patience_cnt   = 0
    best_state     = None

    for epoch in range(1, MAX_EPOCHS + 1):
        # ── train ──
        model.train()
        train_loss = 0.0
        for (x,) in train_loader:
            x = x.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(x), x)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), CLIP_GRAD)
            optimizer.step()
            train_loss += loss.item() * x.size(0)
        train_loss /= len(train_loader.dataset)
        scheduler.step()

        # ── validate ──
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for (x,) in val_loader:
                x = x.to(DEVICE)
                val_loss += criterion(model(x), x).item() * x.size(0)
        val_loss /= len(val_loader.dataset)

        if val_loss < best_val_loss - 1e-9:
            best_val_loss = val_loss
            patience_cnt  = 0
            best_state    = {k: v.cpu().clone()
                             for k, v in model.state_dict().items()}
        else:
            patience_cnt += 1

        if epoch % 20 == 0 or epoch == 1:
            print(f"    epoch {epoch:3d}  train={train_loss:.6f}  "
                  f"val={val_loss:.6f}  lr={scheduler.get_last_lr()[0]:.2e}")

        if patience_cnt >= PATIENCE:
            print(f"    → Early stop at epoch {epoch}  "
                  f"(best val={best_val_loss:.6f})")
            break

    model.load_state_dict(best_state)
    return model


@torch.no_grad()
def reconstruct_scaled(model: ConvAutoencoder,
                        loader: DataLoader,
                        scale: float) -> np.ndarray:
    """Reconstruct and revert to original kernel scale."""
    model.eval()
    chunks = []
    for (x,) in loader:
        x = x.to(DEVICE)
        out = model(x).cpu().numpy()
        chunks.append(out)
    recon_scaled = np.concatenate(chunks, axis=0).squeeze(1)   # (N,64,64)
    return recon_scaled / scale


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(CKPT_DIR,    exist_ok=True)

    print(f"Device: {DEVICE}  |  Input scale: ×{SCALE}")

    # load & scale tensors
    train_t = load_tensor(os.path.join(DATA_DIR, "train.npz"),   SCALE)
    id_t    = load_tensor(os.path.join(DATA_DIR, "id_test.npz"), SCALE)
    ood_t   = load_tensor(os.path.join(DATA_DIR, "ood_test.npz"),SCALE)

    # 10% of train used for early-stopping validation
    n_val   = max(1, len(train_t) // 10)
    val_t   = train_t[:n_val]
    train_t = train_t[n_val:]

    train_loader = DataLoader(TensorDataset(train_t), batch_size=BATCH_SIZE,
                              shuffle=True,  num_workers=0)
    val_loader   = DataLoader(TensorDataset(val_t),   batch_size=BATCH_SIZE,
                              shuffle=False, num_workers=0)
    id_loader    = DataLoader(TensorDataset(id_t),    batch_size=BATCH_SIZE,
                              shuffle=False, num_workers=0)
    ood_loader   = DataLoader(TensorDataset(ood_t),   batch_size=BATCH_SIZE,
                              shuffle=False, num_workers=0)

    # original-scale arrays for evaluation
    id_orig  = np.load(os.path.join(DATA_DIR, "id_test.npz"))["kernels"]
    ood_orig = np.load(os.path.join(DATA_DIR, "ood_test.npz"))["kernels"]

    rows = []

    for dim in DIMS:
        print(f"\n── Latent dim = {dim} ──────────────────────────────")
        model = train_one_model(dim, train_loader, val_loader)

        ckpt_path = os.path.join(CKPT_DIR, f"ae_dim{dim}.pt")
        torch.save(model.state_dict(), ckpt_path)

        id_recon  = reconstruct_scaled(model, id_loader,  SCALE)
        ood_recon = reconstruct_scaled(model, ood_loader, SCALE)

        id_mse,  id_psnr  = batch_metrics_np(id_orig,  id_recon)
        ood_mse, ood_psnr = batch_metrics_np(ood_orig, ood_recon)

        print(f"  ID  PSNR={id_psnr:.2f} dB   MSE={id_mse:.2e}")
        print(f"  OOD PSNR={ood_psnr:.2f} dB   MSE={ood_mse:.2e}")

        rows.append({
            "model":    "AE",
            "dim":      dim,
            "id_mse":   round(id_mse,   8),
            "id_psnr":  round(id_psnr,  4),
            "ood_mse":  round(ood_mse,  8),
            "ood_psnr": round(ood_psnr, 4),
        })

    csv_path = os.path.join(RESULTS_DIR, "ae_results.csv")
    fieldnames = ["model", "dim", "id_mse", "id_psnr", "ood_mse", "ood_psnr"]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nResults saved to {csv_path}")


if __name__ == "__main__":
    main()

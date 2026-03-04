"""
autoencoder.py
--------------
Convolutional Autoencoder for PSF representation.

Architecture (encoder):
  64×64  →  Conv(32, 3, s2)  →  32×32
         →  Conv(64, 3, s2)  →  16×16
         →  Conv(128,3, s2)  →   8×8
         →  Flatten           →  128*8*8 = 8192
         →  Linear(latent_dim)

Decoder mirrors the encoder using ConvTranspose2d.

Ablation over latent_dim in [8, 16, 32, 64, 128].
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

BATCH_SIZE  = 128
MAX_EPOCHS  = 200
LR          = 1e-3
PATIENCE    = 15           # early-stopping patience (epochs without improvement)
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── model ──────────────────────────────────────────────────────────────────────

class Encoder(nn.Module):
    def __init__(self, latent_dim: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32,  3, stride=2, padding=1),   # → 32×32
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),   # → 16×16
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128,3, stride=2, padding=1),   # → 8×8
            nn.ReLU(inplace=True),
        )
        self.fc = nn.Linear(128 * 8 * 8, latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv(x)
        h = h.flatten(1)
        return self.fc(h)


class Decoder(nn.Module):
    def __init__(self, latent_dim: int):
        super().__init__()
        self.fc = nn.Linear(latent_dim, 128 * 8 * 8)
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),  # → 16×16
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32,  3, stride=2, padding=1, output_padding=1),  # → 32×32
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 1,   3, stride=2, padding=1, output_padding=1),  # → 64×64
            nn.Sigmoid(),   # output in [0,1]
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = self.fc(z).view(-1, 128, 8, 8)
        return self.deconv(h)


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


# ── metric helpers ─────────────────────────────────────────────────────────────

def batch_metrics_np(orig: np.ndarray, recon: np.ndarray) -> tuple[float, float]:
    """Mean MSE and mean PSNR over a batch of (H,W) or flattened samples."""
    n = orig.shape[0]
    o = orig.reshape(n, -1).astype(np.float32)
    r = recon.reshape(n, -1).astype(np.float32)
    mse_vals  = np.mean((o - r) ** 2, axis=1)
    psnr_vals = 10.0 * np.log10(1.0 / np.maximum(mse_vals, 1e-12))
    return float(mse_vals.mean()), float(psnr_vals.mean())


# ── training ───────────────────────────────────────────────────────────────────

def load_tensor(path: str) -> torch.Tensor:
    k = np.load(path)["kernels"].astype(np.float32)
    return torch.tensor(k).unsqueeze(1)   # (N,1,64,64)


def train_one_model(latent_dim: int,
                    train_loader: DataLoader,
                    val_loader:   DataLoader) -> ConvAutoencoder:
    model = ConvAutoencoder(latent_dim).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.MSELoss()

    best_val_loss = float("inf")
    patience_cnt  = 0
    best_state    = None

    for epoch in range(1, MAX_EPOCHS + 1):
        # ── train ──
        model.train()
        train_loss = 0.0
        for (x,) in train_loader:
            x = x.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(x), x)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * x.size(0)
        train_loss /= len(train_loader.dataset)

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
            best_state    = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_cnt += 1

        if epoch % 20 == 0:
            print(f"    epoch {epoch:3d}  train={train_loss:.6f}  val={val_loss:.6f}")

        if patience_cnt >= PATIENCE:
            print(f"    → Early stop at epoch {epoch} (best val={best_val_loss:.6f})")
            break

    model.load_state_dict(best_state)
    return model


@torch.no_grad()
def reconstruct(model: ConvAutoencoder,
                loader: DataLoader) -> np.ndarray:
    model.eval()
    chunks = []
    for (x,) in loader:
        x = x.to(DEVICE)
        chunks.append(model(x).cpu().numpy())
    return np.concatenate(chunks, axis=0).squeeze(1)   # (N,64,64)


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(CKPT_DIR,    exist_ok=True)

    print(f"Device: {DEVICE}")

    # load tensors
    train_t = load_tensor(os.path.join(DATA_DIR, "train.npz"))
    id_t    = load_tensor(os.path.join(DATA_DIR, "id_test.npz"))
    ood_t   = load_tensor(os.path.join(DATA_DIR, "ood_test.npz"))

    # 10% of train used for validation / early-stopping
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

    id_orig  = id_t.squeeze(1).numpy()
    ood_orig = ood_t.squeeze(1).numpy()

    rows = []

    for dim in DIMS:
        print(f"\n── Latent dim = {dim} ──────────────────────────────")
        model = train_one_model(dim, train_loader, val_loader)

        # save checkpoint
        ckpt_path = os.path.join(CKPT_DIR, f"ae_dim{dim}.pt")
        torch.save(model.state_dict(), ckpt_path)

        # evaluate
        id_recon  = reconstruct(model, id_loader)
        ood_recon = reconstruct(model, ood_loader)

        id_mse,  id_psnr  = batch_metrics_np(id_orig,  id_recon)
        ood_mse, ood_psnr = batch_metrics_np(ood_orig, ood_recon)

        print(f"  ID  PSNR={id_psnr:.2f} dB   MSE={id_mse:.2e}")
        print(f"  OOD PSNR={ood_psnr:.2f} dB   MSE={ood_mse:.2e}")

        rows.append({
            "model": "AE",
            "dim": dim,
            "id_mse":   round(id_mse,   8),
            "id_psnr":  round(id_psnr,  4),
            "ood_mse":  round(ood_mse,  8),
            "ood_psnr": round(ood_psnr, 4),
        })

    # save CSV
    csv_path = os.path.join(RESULTS_DIR, "ae_results.csv")
    fieldnames = ["model", "dim", "id_mse", "id_psnr", "ood_mse", "ood_psnr"]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nResults saved to {csv_path}")


if __name__ == "__main__":
    main()

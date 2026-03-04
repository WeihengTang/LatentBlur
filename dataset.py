"""
dataset.py
----------
Procedurally generates a dataset of 2D motion blur PSF kernels (64x64 pixels).

Splits:
  - Train    (80%): Standard range L in [5,30], theta in [0,180]
  - ID-Test  (10%): Unseen samples from same standard range
  - OOD-Test (10%): Out-of-distribution — long-blur (L in [35,45]) AND
                    random-walk trajectory blur kernels

Output files (under ./data/):
  data/train.npz      — keys: kernels (N,64,64), lengths, angles
  data/id_test.npz
  data/ood_test.npz
"""

import os
import numpy as np
from numpy.random import default_rng


# ── constants ──────────────────────────────────────────────────────────────────
KERNEL_SIZE = 64
TOTAL_STANDARD = 9000       # 8000 train + 1000 id-test  (will be split 8:1)
N_TRAIN       = 8000
N_ID_TEST     = 1000
N_OOD_LINEAR  = 500         # long L OOD
N_OOD_WALK    = 500         # random-walk OOD
SEED          = 42
DATA_DIR      = "data"


# ── kernel helpers ─────────────────────────────────────────────────────────────

def make_linear_blur_kernel(length: int, angle_deg: float,
                            size: int = KERNEL_SIZE) -> np.ndarray:
    """Generate a normalised linear motion blur kernel."""
    k = np.zeros((size, size), dtype=np.float32)
    cx, cy = size // 2, size // 2
    angle_rad = np.deg2rad(angle_deg)
    cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)

    half = length / 2.0
    # draw line with sub-pixel accumulation
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


def make_random_walk_kernel(steps: int, step_size: float = 1.5,
                            size: int = KERNEL_SIZE,
                            rng: np.random.Generator = None) -> np.ndarray:
    """
    Generate a non-linear random-walk motion blur kernel.
    The blur trajectory is a 2-D Brownian path centred in the kernel.
    """
    if rng is None:
        rng = default_rng()

    k = np.zeros((size, size), dtype=np.float32)
    cx, cy = size / 2.0, size / 2.0
    x, y = cx, cy

    # smooth walk: each step has some angular persistence
    angle = rng.uniform(0, 2 * np.pi)
    for _ in range(steps):
        angle += rng.normal(0, 0.4)          # angular jitter
        dx = step_size * np.cos(angle)
        dy = step_size * np.sin(angle)
        x = np.clip(x + dx, 1, size - 2)
        y = np.clip(y + dy, 1, size - 2)
        xi, yi = int(round(x)), int(round(y))
        k[yi, xi] += 1.0

    total = k.sum()
    if total > 0:
        k /= total
    return k


# ── generation ─────────────────────────────────────────────────────────────────

def generate_standard_set(n: int, rng: np.random.Generator):
    """Standard range: L in [5,30], theta in [0,180)."""
    lengths = rng.integers(5, 31, size=n)           # inclusive [5,30]
    angles  = rng.uniform(0, 180, size=n)
    kernels = np.stack([
        make_linear_blur_kernel(int(l), float(a))
        for l, a in zip(lengths, angles)
    ])
    return kernels, lengths.astype(np.float32), angles.astype(np.float32)


def generate_ood_linear(n: int, rng: np.random.Generator):
    """OOD linear: L in [35,45], theta in [0,180)."""
    lengths = rng.integers(35, 46, size=n)
    angles  = rng.uniform(0, 180, size=n)
    kernels = np.stack([
        make_linear_blur_kernel(int(l), float(a))
        for l, a in zip(lengths, angles)
    ])
    return kernels, lengths.astype(np.float32), angles.astype(np.float32)


def generate_ood_walk(n: int, rng: np.random.Generator):
    """OOD random-walk: non-linear trajectory blur."""
    steps_arr = rng.integers(20, 50, size=n)       # variable walk length
    kernels = np.stack([
        make_random_walk_kernel(int(s), step_size=1.5, rng=rng)
        for s in steps_arr
    ])
    lengths = steps_arr.astype(np.float32)          # 'length' == steps for walk
    angles  = np.full(n, -1, dtype=np.float32)      # undefined for walk
    return kernels, lengths, angles


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    rng = default_rng(SEED)

    print("Generating standard set (train + id-test)…")
    std_kernels, std_lengths, std_angles = generate_standard_set(
        N_TRAIN + N_ID_TEST, rng
    )

    train_k   = std_kernels[:N_TRAIN]
    train_l   = std_lengths[:N_TRAIN]
    train_a   = std_angles[:N_TRAIN]

    id_k      = std_kernels[N_TRAIN:]
    id_l      = std_lengths[N_TRAIN:]
    id_a      = std_angles[N_TRAIN:]

    print("Generating OOD set (long-blur + random-walk)…")
    ood_lin_k, ood_lin_l, ood_lin_a = generate_ood_linear(N_OOD_LINEAR, rng)
    ood_wlk_k, ood_wlk_l, ood_wlk_a = generate_ood_walk(N_OOD_WALK, rng)

    ood_k = np.concatenate([ood_lin_k, ood_wlk_k], axis=0)
    ood_l = np.concatenate([ood_lin_l, ood_wlk_l], axis=0)
    ood_a = np.concatenate([ood_lin_a, ood_wlk_a], axis=0)

    # shuffle OOD
    idx = rng.permutation(len(ood_k))
    ood_k, ood_l, ood_a = ood_k[idx], ood_l[idx], ood_a[idx]

    # save
    np.savez_compressed(
        os.path.join(DATA_DIR, "train.npz"),
        kernels=train_k, lengths=train_l, angles=train_a
    )
    np.savez_compressed(
        os.path.join(DATA_DIR, "id_test.npz"),
        kernels=id_k, lengths=id_l, angles=id_a
    )
    np.savez_compressed(
        os.path.join(DATA_DIR, "ood_test.npz"),
        kernels=ood_k, lengths=ood_l, angles=ood_a
    )

    print(f"  Train   : {len(train_k):>5} kernels  → data/train.npz")
    print(f"  ID-Test : {len(id_k):>5} kernels  → data/id_test.npz")
    print(f"  OOD-Test: {len(ood_k):>5} kernels  → data/ood_test.npz")
    print("Done.")


if __name__ == "__main__":
    main()

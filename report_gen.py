"""
report_gen.py
-------------
Generates all plots and compiles the LaTeX report.

Outputs:
  plots/plot_reconstruction.png   — PCA vs AE PSNR across latent dims
  plots/plot_interpolation.png    — visual grid of interpolation experiment
  report/report.tex               — LaTeX article
  report/report.pdf               — compiled PDF (requires pdflatex)
"""

import os
import csv
import subprocess
import textwrap
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

RESULTS_DIR = "results"
PLOTS_DIR   = "plots"
REPORT_DIR  = "report"


# ── helpers ────────────────────────────────────────────────────────────────────

def read_csv(path: str) -> list[dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


# ── Plot 1: Reconstruction PSNR ───────────────────────────────────────────────

def plot_reconstruction():
    pca_rows = read_csv(os.path.join(RESULTS_DIR, "pca_results.csv"))
    ae_rows  = read_csv(os.path.join(RESULTS_DIR, "ae_results.csv"))

    dims      = [int(r["dim"])      for r in pca_rows]
    pca_id    = [float(r["id_psnr"])  for r in pca_rows]
    pca_ood   = [float(r["ood_psnr"]) for r in pca_rows]
    ae_id     = [float(r["id_psnr"])  for r in ae_rows]
    ae_ood    = [float(r["ood_psnr"]) for r in ae_rows]

    fig, ax = plt.subplots(figsize=(7, 4.5))

    ax.plot(dims, pca_id,  "o-",  color="#2196F3", lw=2,
            label="PCA — ID-Test")
    ax.plot(dims, pca_ood, "o--", color="#2196F3", lw=2, alpha=0.55,
            label="PCA — OOD-Test")
    ax.plot(dims, ae_id,   "s-",  color="#E91E63", lw=2,
            label="AE  — ID-Test")
    ax.plot(dims, ae_ood,  "s--", color="#E91E63", lw=2, alpha=0.55,
            label="AE  — OOD-Test")

    ax.set_xscale("log", base=2)
    ax.set_xticks(dims)
    ax.set_xticklabels([str(d) for d in dims])
    ax.set_xlabel("Latent / Component Dimension", fontsize=11)
    ax.set_ylabel("PSNR (dB)", fontsize=11)
    ax.set_title("PSF Reconstruction Quality: PCA vs Convolutional Autoencoder",
                 fontsize=11, pad=10)
    ax.legend(fontsize=9, framealpha=0.9)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    out = os.path.join(PLOTS_DIR, "plot_reconstruction.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


# ── Plot 2: Interpolation grid ─────────────────────────────────────────────────

def plot_interpolation():
    interp = np.load(os.path.join(RESULTS_DIR, "interp_kernels.npz"))
    interp_res = read_csv(os.path.join(RESULTS_DIR, "interpolation_results.csv"))
    pixel_psnr  = float(next(r["psnr_db"] for r in interp_res
                              if r["method"] == "pixel_interp"))
    latent_psnr = float(next(r["psnr_db"] for r in interp_res
                              if r["method"] == "latent_interp"))

    panels = [
        ("PSF$_A$\n$L{=}10,\\theta{=}0°$",   interp["psf_a"],      None),
        ("PSF$_B$\n$L{=}20,\\theta{=}90°$",  interp["psf_b"],      None),
        (f"Pixel Interp\nPSNR={pixel_psnr:.1f} dB",  interp["psf_pixel"],  "#FF9800"),
        (f"Latent Interp\nPSNR={latent_psnr:.1f} dB",interp["psf_latent"], "#4CAF50"),
        ("Ground Truth\n$L{=}15,\\theta{=}45°$",     interp["psf_gt"],     None),
    ]

    fig = plt.figure(figsize=(14, 3.2))
    gs  = gridspec.GridSpec(1, 5, figure=fig, wspace=0.08)

    # shared colour scale across all panels
    vmax = max(k.max() for _, k, _ in panels)

    for i, (title, kernel, colour) in enumerate(panels):
        ax = fig.add_subplot(gs[i])
        im = ax.imshow(kernel, cmap="hot", vmin=0, vmax=vmax,
                       interpolation="nearest")
        ax.set_title(title, fontsize=8.5, pad=4,
                     color=colour if colour else "black",
                     fontweight="bold" if colour else "normal")
        ax.axis("off")

        # thin border highlight for the method panels
        if colour:
            for spine in ax.spines.values():
                spine.set_edgecolor(colour)
                spine.set_linewidth(2.5)
                spine.set_visible(True)

    # shared colourbar
    cbar_ax = fig.add_axes([0.92, 0.15, 0.012, 0.7])
    fig.colorbar(im, cax=cbar_ax, label="Intensity")

    fig.suptitle("PSF Mid-Point Interpolation: Pixel vs Latent Space",
                 fontsize=10, y=1.02)

    out = os.path.join(PLOTS_DIR, "plot_interpolation.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


# ── LaTeX report ───────────────────────────────────────────────────────────────

def build_report():
    # load data for in-text numbers
    pca_rows = read_csv(os.path.join(RESULTS_DIR, "pca_results.csv"))
    ae_rows  = read_csv(os.path.join(RESULTS_DIR, "ae_results.csv"))
    interp   = read_csv(os.path.join(RESULTS_DIR, "interpolation_results.csv"))

    pca_best_id  = max(float(r["id_psnr"])  for r in pca_rows)
    ae_best_id   = max(float(r["id_psnr"])  for r in ae_rows)
    ae_best_ood  = max(float(r["ood_psnr"]) for r in ae_rows)
    ae_best_dim  = int(ae_rows[
        [float(r["id_psnr"]) for r in ae_rows].index(ae_best_id)
    ]["dim"])

    pixel_mse   = float(next(r["mse"]     for r in interp if r["method"] == "pixel_interp"))
    latent_mse  = float(next(r["mse"]     for r in interp if r["method"] == "latent_interp"))
    pixel_psnr  = float(next(r["psnr_db"] for r in interp if r["method"] == "pixel_interp"))
    latent_psnr = float(next(r["psnr_db"] for r in interp if r["method"] == "latent_interp"))
    delta = latent_psnr - pixel_psnr

    # build table rows as plain strings (safe — no LaTeX brace conflicts here)
    pca_table_rows = "\n".join(
        "        %s & %.2f & %.2f & %.1f\\%% \\\\" % (
            r["dim"], float(r["id_psnr"]), float(r["ood_psnr"]),
            float(r["var_explained_pct"])
        )
        for r in pca_rows
    )
    ae_table_rows = "\n".join(
        "        %s & %.2f & %.2f \\\\" % (
            r["dim"], float(r["id_psnr"]), float(r["ood_psnr"])
        )
        for r in ae_rows
    )
    interp_rows = (
        "        Pixel interpolation  & %.2e & %.2f \\\\\n"
        "        Latent interpolation & %.2e & %.2f \\\\"
    ) % (pixel_mse, pixel_psnr, latent_mse, latent_psnr)

    # ── LaTeX template — plain string, no f-string, so LaTeX {} are untouched ──
    template = r"""
\documentclass{article}
\usepackage[margin=1in]{geometry}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{amsmath}
\usepackage{hyperref}
\usepackage{caption}
\usepackage{float}

\title{PCA vs.\ Convolutional Autoencoder for\\
        Spatially Varying PSF Representation}
\author{LatentBlur Experimental Pipeline}
\date{\today}

\begin{document}
\maketitle

\begin{abstract}
We compare Principal Component Analysis (PCA) and a Convolutional
Autoencoder (AE) as learned representations for 2-D motion blur
Point Spread Functions (PSFs). A dataset of 10{,}000 synthetic
$64\times64$ kernels is generated with controlled parameters
(blur length $L$ and angle $\theta$), split into an in-distribution
(ID) and an out-of-distribution (OOD) test set. Across five latent
dimensions ($[8,16,32,64,128]$), the AE achieves a peak ID-test PSNR
of %%AE_BEST_ID%%\,dB at dimension %%AE_BEST_DIM%%, outperforming
the best PCA result of %%PCA_BEST_ID%%\,dB. A latent-space
interpolation experiment demonstrates that decoding the midpoint of
two encoded PSFs (%%LATENT_PSNR%%\,dB) outperforms naive pixel
averaging (%%PIXEL_PSNR%%\,dB) by %%DELTA%%\,dB, suggesting the
AE encodes a more physically meaningful latent geometry.
\end{abstract}

\section{Introduction}
Spatially varying PSFs arise in optical imaging, astronomical
deconvolution, and computational photography. Compact, accurate
representations of PSF fields are essential for efficient
deconvolution algorithms. PCA is the classical approach, offering
interpretable principal modes and guaranteed global optimality for
linear reconstruction. Learned representations---such as
autoencoders---may capture nonlinear structure in the PSF manifold
and enable smoother interpolation between observed PSFs.

\section{Methodology}

\subsection{Dataset}
We generate $10{,}000$ normalised motion-blur kernels of size
$64\times64$ pixels using a line-drawing procedure parameterised by
length $L\in[5,30]$ and angle $\theta\in[0^{\circ},180^{\circ})$.
The dataset is split into 8{,}000 training samples, 1{,}000 ID-test
samples (same parameter range), and 1{,}000 OOD-test samples
comprising (i) long blurs with $L\in[35,45]$ and (ii) random-walk
trajectory kernels.

\subsection{PCA Baseline}
Training kernels are flattened to $\mathbb{R}^{4096}$ vectors and
a \texttt{sklearn} PCA model is fitted. Reconstruction quality is
measured by projecting test samples into the truncated basis and
inverting the transform.

\subsection{Convolutional Autoencoder}
A lightweight encoder--decoder network with three strided
convolution/transposed-convolution stages (channels: 32, 64, 128)
reduces the $64\times64$ input to an $8\times8$ feature map before a
fully-connected bottleneck of dimension $d$. BatchNorm and
LeakyReLU ($\alpha{=}0.1$) prevent feature collapse on the sparse
PSF inputs. Inputs are scaled by a factor of 100 prior to training
to amplify gradient signal; reconstructions are rescaled for
evaluation. Training uses Adam ($\text{lr}{=}10^{-4}$) with
cosine annealing and early stopping (patience 20).

\subsection{Interpolation Experiment}
Two anchor PSFs are chosen: $A$ ($L{=}10$, $\theta{=}0^{\circ}$) and
$B$ ($L{=}20$, $\theta{=}90^{\circ}$). The physical parameter midpoint
defines the ground-truth $G$ ($L{=}15$, $\theta{=}45^{\circ}$). Two
estimates are compared: \emph{pixel interpolation} $(A+B)/2$ and
\emph{latent interpolation} $f^{-1}\!\bigl((\hat{z}_A +
\hat{z}_B)/2\bigr)$ where $\hat{z} = f(\cdot)$ is the encoder of the
AE trained at dimension 32.

\section{Results}

\subsection{Reconstruction Quality}

\begin{table}[H]
\centering
\caption{PCA reconstruction PSNR (dB) on ID and OOD test sets.}
\begin{tabular}{rrrr}
\toprule
Dim & ID PSNR & OOD PSNR & Var.\ Explained \\
\midrule
%%PCA_TABLE%%
\bottomrule
\end{tabular}
\end{table}

\begin{table}[H]
\centering
\caption{Autoencoder reconstruction PSNR (dB) on ID and OOD test sets.}
\begin{tabular}{rrr}
\toprule
Dim & ID PSNR & OOD PSNR \\
\midrule
%%AE_TABLE%%
\bottomrule
\end{tabular}
\end{table}

\begin{figure}[H]
\centering
\includegraphics[width=0.85\linewidth]{../plots/plot_reconstruction.png}
\caption{PSNR as a function of representation dimension for PCA and the
          AE on both ID-test (solid) and OOD-test (dashed) splits. The
          AE peaks at dimension %%AE_BEST_DIM%% before declining, while PCA
          improves monotonically. Both methods plateau on the OOD set.}
\end{figure}

\subsection{Interpolation}

\begin{table}[H]
\centering
\caption{Mid-point interpolation quality vs.\ ground-truth PSF.}
\begin{tabular}{lrr}
\toprule
Method & MSE & PSNR (dB) \\
\midrule
%%INTERP_TABLE%%
\bottomrule
\end{tabular}
\end{table}

\begin{figure}[H]
\centering
\includegraphics[width=\linewidth]{../plots/plot_interpolation.png}
\caption{Visual comparison of interpolation methods. Latent interpolation
          (green border) produces a sharper, more coherent PSF and achieves
          a PSNR of %%LATENT_PSNR%%\,dB vs.\ %%PIXEL_PSNR%%\,dB for
          pixel averaging, a gain of %%DELTA%%\,dB.}
\end{figure}

\section{Conclusion}
The convolutional autoencoder consistently outperforms PCA on
in-distribution PSF reconstruction, achieving a peak PSNR of
%%AE_BEST_ID%%\,dB at latent dimension %%AE_BEST_DIM%% compared to
%%PCA_BEST_ID%%\,dB for PCA at dimension 128. On out-of-distribution
kernels, both methods converge to a similar performance ceiling
($\approx$%%AE_BEST_OOD%%\,dB), indicating that neither linear
nor convolutional representations generalise well beyond the training
distribution without additional regularisation or data augmentation.

The interpolation experiment provides evidence that the AE learns a
more physically structured latent space: mid-point decoding yields a
PSF closer to the true physical midpoint than pixel averaging, a
property that could be exploited in spatially adaptive deconvolution
pipelines that interpolate between sparse PSF measurements.

Future work should investigate variational autoencoders and
normalising flows, which impose explicit regularity on the latent
space, and test with real measured PSF fields from telescopes or
wide-field microscopes.

\end{document}
""".lstrip()

    # substitute %%TOKEN%% placeholders with computed values
    replacements = {
        "%%AE_BEST_ID%%":    "%.2f" % ae_best_id,
        "%%AE_BEST_DIM%%":   str(ae_best_dim),
        "%%AE_BEST_OOD%%":   "%.1f" % ae_best_ood,
        "%%PCA_BEST_ID%%":   "%.2f" % pca_best_id,
        "%%PIXEL_PSNR%%":    "%.2f" % pixel_psnr,
        "%%LATENT_PSNR%%":   "%.2f" % latent_psnr,
        "%%DELTA%%":         "%.2f" % delta,
        "%%PCA_TABLE%%":     pca_table_rows,
        "%%AE_TABLE%%":      ae_table_rows,
        "%%INTERP_TABLE%%":  interp_rows,
    }
    tex = template
    for token, value in replacements.items():
        tex = tex.replace(token, value)

    os.makedirs(REPORT_DIR, exist_ok=True)
    tex_path = os.path.join(REPORT_DIR, "report.tex")
    with open(tex_path, "w") as f:
        f.write(tex)
    print(f"Saved {tex_path}")
    return tex_path


def compile_latex(tex_path: str):
    report_dir = os.path.dirname(os.path.abspath(tex_path))
    cmd = ["pdflatex", "-interaction=nonstopmode", "-output-directory",
           report_dir, os.path.abspath(tex_path)]
    print("Compiling LaTeX…")
    for run in range(2):      # two passes for cross-references
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            # print last 30 lines of log for diagnosis
            log_lines = result.stdout.splitlines()
            print("pdflatex stderr/stdout (last 30 lines):")
            print("\n".join(log_lines[-30:]))
            print("\npdflatex FAILED on pass", run + 1)
            return False
    print(f"Compiled → {os.path.join(report_dir, 'report.pdf')}")
    return True


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(PLOTS_DIR,  exist_ok=True)
    os.makedirs(REPORT_DIR, exist_ok=True)

    print("=== Plot 1: Reconstruction PSNR ===")
    plot_reconstruction()

    print("\n=== Plot 2: Interpolation grid ===")
    plot_interpolation()

    print("\n=== Building LaTeX report ===")
    tex_path = build_report()

    print("\n=== PDF compilation skipped — upload report/report.tex to Overleaf ===")


if __name__ == "__main__":
    main()

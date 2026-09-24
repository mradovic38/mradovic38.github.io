"""New animation (not in the paper): two "straight lines" from P to Q.

Top: the mixture path (1 - t) P + t Q. Mass fades out in one place and fades in
somewhere else. It is the straight line for total variation, since
TV(P, M_t) = t * TV(P, Q): TV grows at a constant speed along it.
Bottom: the optimal transport (displacement) path, where every quantile moves in a
straight line x_t = (1 - t) x_P + t x_Q. It is the straight line for W_2, since
W_2(P, rho_t) = t * W_2(P, Q). This is the kind of path a flow model can produce.

Each panel shows how far along the path we are, measured in TV and in W_2.
Writes images/f-divergences/ot_vs_mixture.mp4 (needs ffmpeg) and a poster PNG.
"""

import shutil
import subprocess
import tempfile
from math import erf, sqrt
from pathlib import Path

import numpy as np

from style import BLUE, GRAY, ORANGE, OUT, RED, setup

import matplotlib.pyplot as plt


def normal_pdf(x, mu, s):
    return np.exp(-((x - mu) ** 2) / (2 * s**2)) / (s * np.sqrt(2 * np.pi))


def normal_cdf(x, mu, s):
    return np.array([0.5 * (1 + erf((v - mu) / (s * sqrt(2)))) for v in x])


X = np.linspace(-5.0, 6.0, 3000)
P = normal_pdf(X, -2.2, 0.55)
Q = 0.5 * normal_pdf(X, 1.3, 0.45) + 0.5 * normal_pdf(X, 3.3, 0.45)
FP = normal_cdf(X, -2.2, 0.55)
FQ = 0.5 * normal_cdf(X, 1.3, 0.45) + 0.5 * normal_cdf(X, 3.3, 0.45)

U = np.linspace(1e-4, 1 - 1e-4, 4000)
XP = np.interp(U, FP, X)  # quantile functions
XQ = np.interp(U, FQ, X)
PARTICLES = np.linspace(0.03, 0.97, 24)
TV_PQ = 0.5 * np.trapezoid(np.abs(P - Q), X)
W2_PQ = np.sqrt(np.mean((XQ - XP) ** 2))


def ease(s):
    return 0.5 - 0.5 * np.cos(np.pi * s)


def progress(t):
    """Fraction of the way from P to Q, measured in TV and in W_2, for both paths."""
    # Mixture path M_t = (1 - t) P + t Q.
    mix = (1 - t) * P + t * Q
    tv_mix = 0.5 * np.trapezoid(np.abs(mix - P), X) / TV_PQ
    q_mix = np.interp(U, (1 - t) * FP + t * FQ, X)
    w2_mix = np.sqrt(np.mean((q_mix - XP) ** 2)) / W2_PQ
    # Transport path rho_t: quantiles move in straight lines.
    xt = (1 - t) * XP + t * XQ
    dens = np.interp(X, xt, 1.0 / np.gradient(xt, U), left=0.0, right=0.0)
    tv_ot = 0.5 * np.trapezoid(np.abs(dens - P), X) / TV_PQ
    w2_ot = np.sqrt(np.mean((xt - XP) ** 2)) / W2_PQ
    return (tv_mix, w2_mix), (tv_ot, w2_ot)


def readout(ax, tv, w2):
    ax.text(0.01, 0.97, f"progress measured by TV: {100 * tv:5.1f}%", transform=ax.transAxes,
            ha="left", va="top", fontsize=9, family="DejaVu Sans Mono", color="#444")
    ax.text(0.01, 0.84, f"progress measured by W2: {100 * w2:5.1f}%", transform=ax.transAxes,
            ha="left", va="top", fontsize=9, family="DejaVu Sans Mono", color="#444")


def draw(t, path):
    fig, (top, bottom) = plt.subplots(2, 1, figsize=(7.2, 4.6), sharex=True)

    mix = (1 - t) * P + t * Q
    top.fill_between(X, mix, color=ORANGE, alpha=0.35, linewidth=0)
    top.plot(X, mix, color=ORANGE)
    top.plot(X, P, color=BLUE, linewidth=0.9, alpha=0.5, linestyle="--")
    top.plot(X, Q, color=RED, linewidth=0.9, alpha=0.5, linestyle="--")
    top.set_title(r"Mixture path $(1-t)P+tQ$: mass fades (a straight line for TV)", fontsize=11)

    xt = (1 - t) * XP + t * XQ
    dens = 1.0 / np.gradient(xt, U)
    bottom.fill_between(xt, dens, color=ORANGE, alpha=0.35, linewidth=0)
    bottom.plot(xt, dens, color=ORANGE)
    bottom.plot(X, P, color=BLUE, linewidth=0.9, alpha=0.5, linestyle="--")
    bottom.plot(X, Q, color=RED, linewidth=0.9, alpha=0.5, linestyle="--")
    px = (1 - t) * np.interp(PARTICLES, U, XP) + t * np.interp(PARTICLES, U, XQ)
    bottom.plot(px, np.full_like(px, -0.045), "o", color="black", markersize=3)
    bottom.set_title(r"Transport path: mass slides (a straight line for $W_2$)", fontsize=11)
    bottom.set_xlabel(r"$x$")

    (tv_mix, w2_mix), (tv_ot, w2_ot) = progress(t)
    readout(top, tv_mix, w2_mix)
    readout(bottom, tv_ot, w2_ot)

    for ax in (top, bottom):
        ax.set_ylim(-0.08, 0.95)
        ax.set_yticks([])
        ax.text(0.99, 0.85, f"t = {t:.2f}", transform=ax.transAxes, ha="right", fontsize=10, color=GRAY)
    top.text(-3.3, 0.55, "P", color=BLUE, ha="center", fontsize=11)
    top.text(2.3, 0.55, "Q", color=RED, ha="center", fontsize=11)

    fig.savefig(path, dpi=110)
    plt.close(fig)


def main() -> None:
    setup()
    OUT.mkdir(parents=True, exist_ok=True)
    n_move, n_hold = 60, 18
    ts = [ease(i / (n_move - 1)) for i in range(n_move)]
    schedule = [0.0] * n_hold + ts + [1.0] * n_hold + ts[::-1]

    tmp = Path(tempfile.mkdtemp())
    for i, t in enumerate(schedule):
        draw(t, tmp / f"{i:04d}.png")
    draw(0.5, OUT / "ot_vs_mixture.png")

    subprocess.run(
        ["ffmpeg", "-loglevel", "error", "-y", "-framerate", "30", "-i", str(tmp / "%04d.png"),
         "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2,format=yuv420p", "-c:v", "libx264", "-crf", "26",
         "-preset", "slow", "-an", "-movflags", "+faststart", str(OUT / "ot_vs_mixture.mp4")],
        check=True,
    )
    shutil.rmtree(tmp)
    print(f"saved: {OUT / 'ot_vs_mixture.mp4'}")


if __name__ == "__main__":
    main()

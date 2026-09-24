"""Figure: Wasserstein distance vs f-divergences.

English version of scripts/vasershtajn.py. Left: point masses at 0 and theta;
every f-divergence is constant in theta (saturated) while W_1 grows linearly.
Right: in 1D, W_1 is the area between the two CDFs.
"""

import numpy as np

from style import COLORS, GRAY, ORANGE, RED, BLUE, save, setup

import matplotlib.pyplot as plt


def norm_cdf(x, mu=0.0, s=1.0):
    from math import erf, sqrt

    return np.array([0.5 * (1 + erf((v - mu) / (s * sqrt(2)))) for v in np.atleast_1d(x)])


def main() -> None:
    setup()
    fig, (left, right) = plt.subplots(1, 2, figsize=(9.2, 3.5))

    # --- left: saturated f-divergences vs linear W_1 ---
    theta = np.linspace(0.02, 3.0, 400)
    ones = np.ones_like(theta)

    left.plot(theta, theta, color=BLUE, label=r"$W_1(\delta_0,\delta_\theta)=\theta$")
    left.plot(theta, ones, color=RED, label=r"$\mathrm{TV}=1$")
    left.plot(theta, 2 * ones, color=COLORS[2], label=r"$H^2=2$")
    left.plot(theta, 3.0 * ones, color=ORANGE, linestyle="--", label=r"$\mathrm{KL}=\chi^2=+\infty$")

    left.annotate(
        "$f$-divergences\ndon't depend on $\\theta$",
        xy=(2.35, 1.0),
        xytext=(1.55, 0.35),
        fontsize=8,
        ha="center",
        arrowprops=dict(arrowstyle="->", color=GRAY, linewidth=0.8),
    )
    left.set_xlabel(r"$\theta$")
    left.set_ylabel("value")
    left.set_ylim(0.0, 3.5)
    left.set_title("Disjoint supports")
    left.legend(loc="upper left", fontsize=8)

    # --- right: W_1 as the area between the CDFs ---
    x = np.linspace(-3.6, 5.6, 1500)
    F = norm_cdf(x, 0.0)
    G = norm_cdf(x, 1.5)

    right.plot(x, F, color=BLUE, label=r"$F$")
    right.plot(x, G, color=RED, label=r"$G$")
    right.fill_between(x, F, G, color=ORANGE, alpha=0.35, linewidth=0)

    w1 = np.trapezoid(np.abs(F - G), x)
    right.annotate(
        rf"$W_1=\int |F-G|\,dx \approx {w1:.2f}$",
        xy=(0.75, 0.5),
        xytext=(1.35, 0.18),
        fontsize=8,
        arrowprops=dict(arrowstyle="->", color=GRAY, linewidth=0.8),
    )

    right.set_xlabel(r"$x$")
    right.set_ylabel(r"$F(x)$")
    right.set_ylim(-0.04, 1.08)
    right.set_title("Area between the CDFs")
    right.legend(loc="upper left")

    save(fig, "wasserstein")


if __name__ == "__main__":
    main()

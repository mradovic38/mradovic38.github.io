"""Figure: Le Cam's two-point method.

English version of scripts/le_kam.py. Two parameter values 2*delta apart whose
data distributions overlap a lot cannot be told apart reliably. As the sample
size n grows the overlap shrinks, so the bound 1 - TV drops; the hypotheses stay
indistinguishable while the gap is of order 1/sqrt(n).
"""

import numpy as np

from style import BLUE, GRAY, ORANGE, RED, save, setup

import matplotlib.pyplot as plt


def density(x, mu, sigma=1.0):
    return np.exp(-((x - mu) ** 2) / (2 * sigma**2)) / (sigma * np.sqrt(2 * np.pi))


def tv_normals(delta, n=1):
    """TV between N(0, 1/n) and N(2*delta, 1/n): 2*Phi(|b - a| / (2s)) - 1."""
    from math import erf, sqrt

    s = 1.0 / np.sqrt(n)
    z = (2 * delta) / (2 * s)
    return np.array([2 * (0.5 * (1 + erf(v / sqrt(2)))) - 1 for v in np.atleast_1d(z)])


def main() -> None:
    setup()
    fig, (left, right) = plt.subplots(1, 2, figsize=(9.2, 3.4))

    # --- left: two nearby distributions and their overlap ---
    delta = 0.45
    x = np.linspace(-3.5, 4.4, 1500)
    p0 = density(x, 0.0)
    p1 = density(x, 2 * delta)

    left.plot(x, p0, color=BLUE, label=r"$P_{\theta_0}$")
    left.plot(x, p1, color=RED, label=r"$P_{\theta_1}$")
    left.fill_between(x, np.minimum(p0, p1), color=GRAY, alpha=0.30, linewidth=0)

    y = 0.445
    left.annotate("", xy=(0.0, y), xytext=(2 * delta, y),
                  arrowprops=dict(arrowstyle="<->", color=ORANGE, linewidth=1.4))
    left.text(delta, y + 0.018, r"$2\delta$", ha="center", fontsize=9, color=ORANGE)
    left.plot([0.0, 2 * delta], [0, 0], marker="|", markersize=9, color=ORANGE, linestyle="none")
    left.text(0.0, -0.045, r"$\theta_0$", ha="center", fontsize=9)
    left.text(2 * delta, -0.045, r"$\theta_1$", ha="center", fontsize=9)
    left.annotate(
        "overlap: hard to tell\nthe hypotheses apart",
        xy=(0.45, 0.12),
        xytext=(1.75, 0.30),
        fontsize=8,
        arrowprops=dict(arrowstyle="->", color=GRAY, linewidth=0.8),
    )

    left.set_xlabel(r"$x$")
    left.set_ylabel("density")
    left.set_ylim(-0.07, 0.52)
    left.set_title("Two nearby parameter values")
    left.legend(loc="upper right")

    # --- right: the bound 1 - TV as a function of the gap, for several n ---
    deltas = np.linspace(0.0, 1.2, 400)
    for n, color in zip((1, 5, 25), (BLUE, RED, ORANGE)):
        right.plot(deltas, 1 - tv_normals(deltas, n), color=color, label=rf"$n={n}$")
        right.axvline(1.0 / np.sqrt(n), color=color, linewidth=0.8, linestyle=":")

    right.set_xlabel(r"gap $\delta$ between parameter values")
    right.set_ylabel(r"$1-\mathrm{TV}$ (best total error)")
    right.set_title("Indistinguishability bound")
    right.set_ylim(0.0, 1.04)
    right.legend(loc="upper right")
    right.annotate(
        r"dotted: $\delta\sim 1/\sqrt{n}$",
        xy=(1.0 / np.sqrt(5), 0.30),
        xytext=(0.62, 0.60),
        fontsize=8,
        arrowprops=dict(arrowstyle="->", color=GRAY, linewidth=0.8),
    )

    save(fig, "le_cam")


if __name__ == "__main__":
    main()

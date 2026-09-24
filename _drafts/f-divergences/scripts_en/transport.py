"""Figure: intuition for optimal transport ("moving mass").

English version of scripts/transport_ilustracija.py (made for the presentation).
Left: two piles of mass with arrows for the transport plan. Right: two ways of
comparing distributions -- f-divergences compare densities at the same point
(vertically), Wasserstein measures how far mass travels (horizontally).
"""

import numpy as np

from style import BLUE, GRAY, ORANGE, RED, save, setup

import matplotlib.pyplot as plt


def density(x, mu, s=0.55):
    return np.exp(-((x - mu) ** 2) / (2 * s**2)) / (s * np.sqrt(2 * np.pi))


def main() -> None:
    setup()
    fig, (left, right) = plt.subplots(1, 2, figsize=(9.6, 3.4))

    # --- left: moving mass from P to Q ---
    x = np.linspace(-2.5, 6.5, 1200)
    p = density(x, 0.0)
    q = density(x, 3.6)

    left.fill_between(x, p, color=BLUE, alpha=0.30, linewidth=0)
    left.fill_between(x, q, color=RED, alpha=0.30, linewidth=0)
    left.plot(x, p, color=BLUE, label=r"$P$")
    left.plot(x, q, color=RED, label=r"$Q$")

    for x0, frac in ((-0.75, 0.55), (0.0, 0.85), (0.75, 0.55)):
        y = density(x0, 0.0) * frac
        left.annotate(
            "",
            xy=(x0 + 3.6, y),
            xytext=(x0, y),
            arrowprops=dict(arrowstyle="->", color=ORANGE, linewidth=1.6, connectionstyle="arc3,rad=-0.25"),
        )
    left.text(1.8, 0.80, "moving mass", ha="center", fontsize=9, color=ORANGE)

    left.set_xlabel(r"$x$")
    left.set_ylabel("density")
    left.set_ylim(0, 0.95)
    left.set_title("Optimal transport")
    left.legend(loc="upper right")

    # --- right: two ways of comparing ---
    x2 = np.linspace(-2.5, 6.5, 1200)
    p2 = density(x2, 1.0)
    q2 = density(x2, 2.6)

    right.plot(x2, p2, color=BLUE, label=r"$p$")
    right.plot(x2, q2, color=RED, label=r"$q$")

    for xv in (0.6, 1.2, 1.8, 2.4, 3.0):
        right.plot([xv, xv], [density(xv, 1.0), density(xv, 2.6)], color=GRAY, linewidth=1.0, linestyle=":")
    right.text(-2.2, 0.90, "$f$-divergences:\ncompare $p(x)$ and $q(x)$\nat the same point",
               ha="left", va="top", fontsize=8, color=GRAY)

    right.annotate("", xy=(2.6, -0.055), xytext=(1.0, -0.055),
                   arrowprops=dict(arrowstyle="<->", color=ORANGE, linewidth=1.6))
    right.text(1.8, -0.135, r"$W_1$: how far", ha="center", fontsize=8, color=ORANGE)

    right.set_xlabel(r"$x$")
    right.set_ylim(-0.17, 1.02)
    right.set_title("Two ways to compare")
    right.legend(loc="upper right")

    save(fig, "transport")


if __name__ == "__main__":
    main()

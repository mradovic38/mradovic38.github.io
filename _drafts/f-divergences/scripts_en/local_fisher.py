"""New figure (not in the paper): local behavior of f-divergences.

Family P_t = N(t, 1), whose Fisher information is J_F = 1. Left: each divergence
D_f(P_t || P_0) next to its quadratic approximation f''(1)/2 * J_F * t^2 (dashed);
TV is the exception and grows linearly, like t / sqrt(2 pi). Right: after
dividing by f''(1)/2 * t^2, every smooth f-divergence starts at the same value J_F = 1.
"""

import numpy as np
from math import erf, sqrt

from style import COLORS, GRAY, save, setup

import matplotlib.pyplot as plt


def density(x, mu):
    return np.exp(-((x - mu) ** 2) / 2) / np.sqrt(2 * np.pi)


def js(t, grid):
    """Jensen-Shannon in the paper's convention: KL(P||M) + KL(Q||M), M = (P+Q)/2."""
    p, q = density(grid, t), density(grid, 0.0)
    m = 0.5 * (p + q)
    return np.trapezoid(p * np.log(p / m) + q * np.log(q / m), grid)


def main() -> None:
    setup()
    t = np.linspace(1e-3, 1.6, 300)
    grid = np.linspace(-12, 14, 6000)

    # (label, exact values, f''(1)/2)
    curves = [
        (r"$\chi^2$", np.exp(t**2) - 1, 1.0),
        (r"KL", t**2 / 2, 0.5),
        (r"$H^2$", 2 - 2 * np.exp(-(t**2) / 8), 0.25),
        (r"JS", np.array([js(v, grid) for v in t]), 0.25),
    ]
    tv = np.array([2 * (0.5 * (1 + erf(v / 2 / sqrt(2)))) - 1 for v in t])

    fig, (left, right) = plt.subplots(1, 2, figsize=(9.2, 3.5))

    for (label, vals, c), color in zip(curves, COLORS):
        left.plot(t, vals, color=color, label=label)
        left.plot(t, c * t**2, color=color, linestyle="--", linewidth=1.1)
        right.plot(t, vals / (c * t**2), color=color, label=label)
    left.plot(t, tv, color=COLORS[4], label="TV")
    left.plot(t, t / np.sqrt(2 * np.pi), color=COLORS[4], linestyle="--", linewidth=1.1)

    left.set_xlabel(r"shift $t$ in $N(t,1)$ vs $N(0,1)$")
    left.set_ylabel("divergence")
    left.set_title(r"Solid: exact, dashed: $\frac{f''(1)}{2}\,J_F\,t^2$")
    left.set_ylim(0, 1.6)
    left.legend(loc="upper left", fontsize=8)

    right.axhline(1.0, color=GRAY, linewidth=0.8, linestyle=":")
    right.set_xlabel(r"$t$")
    right.set_ylabel(r"$D_f \,/\, (\frac{f''(1)}{2}\,t^2)$")
    right.set_title(r"All start at the Fisher information $J_F=1$")
    right.set_ylim(0.6, 2.2)
    right.legend(loc="upper left", fontsize=8)

    save(fig, "local_fisher")


if __name__ == "__main__":
    main()

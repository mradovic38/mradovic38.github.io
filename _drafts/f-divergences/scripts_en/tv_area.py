"""Figure: total variation as an area, and bounded vs unbounded divergences.

English version of scripts/ukupna_varijacija.py. Left: TV(P,Q) is half the area
between the densities (the complement of their overlap). Right: as two Gaussians
move apart, TV and H^2 saturate while KL and chi^2 grow without bound.
"""

import numpy as np

from style import BLUE, COLORS, GRAY, RED, save, setup

import matplotlib.pyplot as plt


def density(x, mu, sigma=1.0):
    return np.exp(-((x - mu) ** 2) / (2 * sigma**2)) / (sigma * np.sqrt(2 * np.pi))


def main() -> None:
    setup()
    fig, (left, right) = plt.subplots(1, 2, figsize=(9.2, 3.5))

    # --- left: area between two densities ---
    x = np.linspace(-4.0, 6.0, 1500)
    p = density(x, 0.0)
    q = density(x, 1.6)

    left.plot(x, p, color=BLUE, label=r"$p$")
    left.plot(x, q, color=RED, label=r"$q$")
    left.fill_between(x, np.minimum(p, q), color=GRAY, alpha=0.30, linewidth=0)
    left.fill_between(x, np.minimum(p, q), np.maximum(p, q), color=COLORS[3], alpha=0.35, linewidth=0)

    tv = 0.5 * np.trapezoid(np.abs(p - q), x)
    left.annotate(
        rf"area $=2\,\mathrm{{TV}}(P,Q)\approx {2*tv:.2f}$",
        xy=(2.35, 0.10),
        xytext=(-3.9, 0.335),
        fontsize=8,
        arrowprops=dict(arrowstyle="->", color=GRAY, linewidth=0.8),
    )
    left.annotate(
        "overlap",
        xy=(0.85, 0.045),
        xytext=(-3.9, 0.20),
        fontsize=8,
        arrowprops=dict(arrowstyle="->", color=GRAY, linewidth=0.8),
    )
    left.set_xlabel(r"$x$")
    left.set_ylabel("density")
    left.set_title("Total variation as an area")
    left.legend(loc="upper right", frameon=False)

    # --- right: divergences as the distributions move apart ---
    deltas = np.linspace(0.0, 4.0, 240)
    grid = np.linspace(-12.0, 16.0, 4000)

    tv_vals, h2_vals, kl_vals, chi_vals = [], [], [], []
    for d in deltas:
        p = density(grid, 0.0)
        q = density(grid, d)
        tv_vals.append(0.5 * np.trapezoid(np.abs(p - q), grid))
        h2_vals.append(np.trapezoid((np.sqrt(p) - np.sqrt(q)) ** 2, grid))
        # Closed forms for N(0,1) vs N(d,1).
        kl_vals.append(d**2 / 2)
        chi_vals.append(np.exp(d**2) - 1)

    right.plot(deltas, tv_vals, color=COLORS[0], label=r"$\mathrm{TV}$")
    right.plot(deltas, h2_vals, color=COLORS[1], label=r"$H^2$")
    right.plot(deltas, kl_vals, color=COLORS[2], label=r"$\mathrm{KL}(P\|Q)$")
    right.plot(deltas, chi_vals, color=COLORS[3], label=r"$\chi^2(P\|Q)$")

    right.axhline(1.0, color=GRAY, linewidth=0.8, linestyle=":")
    right.axhline(2.0, color=GRAY, linewidth=0.8, linestyle=":")
    right.text(3.85, 1.06, r"$\mathrm{TV}\leq 1$", fontsize=8, ha="right", color=GRAY)
    right.text(3.85, 2.06, r"$H^2\leq 2$", fontsize=8, ha="right", color=GRAY)

    right.set_xlabel(r"gap $\delta$ between $N(0,1)$ and $N(\delta,1)$")
    right.set_ylabel("divergence")
    right.set_title("Bounded vs unbounded divergences")
    right.set_ylim(0.0, 4.5)
    right.legend(loc="upper left", frameon=False)

    save(fig, "tv_area")


if __name__ == "__main__":
    main()

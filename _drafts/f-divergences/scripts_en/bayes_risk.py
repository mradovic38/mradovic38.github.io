"""Figure: Bayes risk as an area, and the matching generator f.

English version of scripts/bajesov_rizik.py. Left: R_pi = integral of
min{pi0 p, pi1 q}, the area under the lower envelope of the weighted densities;
the optimal test threshold is where they cross. Right: f_pi(t) = min{pi0 t, pi1}
is concave, and flipping the sign and adding a constant gives the convex
h_pi with h_pi(1) = 0, a valid f-divergence generator.
"""

import numpy as np

from style import BLUE, GRAY, GREEN, ORANGE, RED, save, setup

import matplotlib.pyplot as plt

PI0, PI1 = 0.3, 0.7


def density(x, mu, sigma=1.0):
    return np.exp(-((x - mu) ** 2) / (2 * sigma**2)) / (sigma * np.sqrt(2 * np.pi))


def main() -> None:
    setup()
    fig, (left, right) = plt.subplots(1, 2, figsize=(9.2, 3.5))

    # --- left: R_pi as the area under the minimum of the weighted densities ---
    x = np.linspace(-4.0, 6.0, 2000)
    pp = PI0 * density(x, 0.0)
    qq = PI1 * density(x, 1.8)
    lower = np.minimum(pp, qq)

    left.plot(x, pp, color=BLUE, label=r"$\pi_0\,p(x)$")
    left.plot(x, qq, color=RED, label=r"$\pi_1\,q(x)$")
    left.fill_between(x, lower, color=ORANGE, alpha=0.40, linewidth=0)

    risk = np.trapezoid(lower, x)

    # Threshold: where the two weighted densities cross (inside the overlap).
    k = int(np.argmin(np.abs(pp - qq)[(x > -1) & (x < 4)]) + np.sum(x <= -1))
    left.axvline(x[k], color=GRAY, linewidth=0.9, linestyle=":")
    left.annotate(
        "optimal test threshold",
        xy=(x[k], lower[k]),
        xytext=(x[k] - 4.3, 0.20),
        fontsize=8,
        arrowprops=dict(arrowstyle="->", color=GRAY, linewidth=0.8),
    )
    left.annotate(
        rf"$R_\pi \approx {risk:.3f}$",
        xy=(x[k] + 0.55, lower[k] * 0.45),
        xytext=(x[k] + 1.5, 0.17),
        fontsize=8,
        arrowprops=dict(arrowstyle="->", color=GRAY, linewidth=0.8),
    )

    left.set_xlabel(r"$x$")
    left.set_ylabel("weighted density")
    left.set_title(rf"Bayes risk, $\pi_0={PI0}$, $\pi_1={PI1}$")
    left.legend(loc="upper right")

    # --- right: f_pi (concave) and h_pi (convex, h_pi(1) = 0) ---
    t = np.linspace(0.0, 4.0, 800)
    f_pi = np.minimum(PI0 * t, PI1)
    h_pi = np.minimum(PI0 * 1.0, PI1) - f_pi

    right.plot(t, f_pi, color=GREEN, label=r"$f_\pi(t)=\min\{\pi_0 t,\ \pi_1\}$")
    right.plot(t, h_pi, color=BLUE, label=r"$h_\pi(t)=f_\pi(1)-f_\pi(t)$")

    right.axhline(0.0, color=GRAY, linewidth=0.8)
    right.axvline(1.0, color=GRAY, linewidth=0.8, linestyle=":")
    right.plot([1.0], [0.0], marker="o", markersize=4.5, color=BLUE, zorder=6)
    right.annotate(
        r"$h_\pi(1)=0$",
        xy=(1.0, 0.0),
        xytext=(1.85, 0.16),
        fontsize=8,
        arrowprops=dict(arrowstyle="->", color=GRAY, linewidth=0.8),
    )
    right.annotate(
        "kink: $h_\\pi$ is convex",
        xy=(2.55, -0.40),
        xytext=(1.75, -0.62),
        fontsize=8,
        ha="center",
        arrowprops=dict(arrowstyle="->", color=GRAY, linewidth=0.8),
    )

    right.set_xlabel(r"$t$")
    right.set_title("Concave $f_\\pi$ and convex $h_\\pi$")
    right.set_ylim(-0.78, 0.88)
    right.legend(loc="upper left", fontsize=8)

    save(fig, "bayes_risk")


if __name__ == "__main__":
    main()

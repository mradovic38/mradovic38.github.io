"""Figure: convex generator functions f of the classic f-divergences.

English version of scripts/konveksne_funkcije.py. All generators are convex and
pass through (1, 0); they differ in how fast they grow at infinity, which decides
whether the divergence can be infinite.
"""

import numpy as np

from style import COLORS, GRAY, save, setup

import matplotlib.pyplot as plt


def kl(t):
    return t * np.log(t)


def chi_squared(t):
    return (t - 1) ** 2


def total_variation(t):
    return 0.5 * np.abs(t - 1)


def hellinger(t):
    return (1 - np.sqrt(t)) ** 2


def jensen_shannon(t):
    return t * np.log(2 * t / (t + 1)) + np.log(2 / (t + 1))


GENERATORS = [
    (r"$t\log t$  (KL)", kl),
    (r"$(t-1)^2$  ($\chi^2$)", chi_squared),
    (r"$\frac{1}{2}|t-1|$  (TV)", total_variation),
    (r"$(1-\sqrt{t})^2$  ($H^2$)", hellinger),
    (r"$t\log\frac{2t}{t+1}+\log\frac{2}{t+1}$  (JS)", jensen_shannon),
]


def main() -> None:
    setup()
    fig, (left, right) = plt.subplots(1, 2, figsize=(9.2, 3.6))

    # --- left: all generators on [0, 3] ---
    t = np.linspace(1e-6, 3.0, 800)
    for (label, f), color in zip(GENERATORS, COLORS):
        left.plot(t, f(t), color=color, label=label)

    left.axhline(0.0, color=GRAY, linewidth=0.8, zorder=1)
    left.axvline(1.0, color=GRAY, linewidth=0.8, linestyle=":", zorder=1)
    left.plot([1.0], [0.0], marker="o", markersize=4.5, color="black", zorder=5)
    left.set_xlabel(r"$t = p(x)/q(x)$")
    left.set_ylabel(r"$f(t)$")
    left.set_title("Generator functions")
    left.set_ylim(-0.4, 2.6)
    left.legend(loc="upper left", fontsize=8)

    # --- right: f(t)/t for large t (the recession constant) ---
    t2 = np.linspace(1.0, 60.0, 800)
    for (label, f), color in zip(GENERATORS, COLORS):
        short = label.split("  ")[-1].strip("()")
        right.plot(t2, f(t2) / t2, color=color, label=short)

    right.set_xlabel(r"$t$")
    right.set_ylabel(r"$f(t)\,/\,t$")
    right.set_title("Growth at infinity")
    right.set_ylim(-0.1, 3.0)
    right.legend(loc="upper right", ncols=2)
    right.annotate(
        "finite limit\n" + r"$\Rightarrow$ divergence stays finite",
        xy=(48, 0.62),
        xytext=(24, 1.55),
        fontsize=8,
        ha="center",
        arrowprops=dict(arrowstyle="->", color=GRAY, linewidth=0.8),
    )

    save(fig, "generators")


if __name__ == "__main__":
    main()

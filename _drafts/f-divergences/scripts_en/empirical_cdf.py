"""Figure: the empirical CDF converges to the true CDF (Glivenko-Cantelli).

English version of scripts/empirijska_raspodela.py (convergence panel).
"""

import numpy as np

from style import BLUE, GRAY, RED, save, setup

import matplotlib.pyplot as plt

# Fixed seed: the figure has to be reproducible.
SEED = 20250910


def true_cdf(x):
    """CDF of N(0, 1) via the error function."""
    from math import erf

    return np.array([0.5 * (1 + erf(v / np.sqrt(2))) for v in np.atleast_1d(x)])


def empirical_cdf(sample, x):
    """F_n(x) = (number of sample points <= x) / n."""
    sample = np.sort(sample)
    return np.searchsorted(sample, x, side="right") / sample.size


def main() -> None:
    setup()
    rng = np.random.default_rng(SEED)
    grid = np.linspace(-3.2, 3.2, 1200)
    f0 = true_cdf(grid)

    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.1), sharey=True)
    for ax, n in zip(axes, (10, 100)):
        sample = rng.standard_normal(n)
        fn = empirical_cdf(sample, grid)

        ax.step(grid, fn, where="post", color=BLUE, label=rf"$F_n$, $n={n}$")
        ax.plot(grid, f0, color=RED, linestyle="--", label=r"$F_0$")
        ax.set_xlabel(r"$x$")
        ax.set_title(rf"$n={n}$")
        ax.set_ylim(-0.04, 1.08)
        ax.axhline(0.0, color=GRAY, linewidth=0.6)
        ax.axhline(1.0, color=GRAY, linewidth=0.6)
        ax.legend(loc="upper left")

    axes[0].set_ylabel(r"$F(x)$")
    save(fig, "empirical_cdf")


if __name__ == "__main__":
    main()

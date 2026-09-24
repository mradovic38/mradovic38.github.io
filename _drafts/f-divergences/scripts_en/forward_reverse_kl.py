"""New figure (not in the paper): forward vs reverse KL.

Fit a single Gaussian q to a two-mode target p. Minimizing the forward
KL(p||q) (what maximum likelihood does) matches moments and covers both modes;
minimizing the reverse KL(q||p) locks onto one mode.
"""

import numpy as np

from style import BLUE, GRAY, RED, save, setup

import matplotlib.pyplot as plt

MODES = (-2.0, 2.0)
SIGMA = 0.6


def log_normal(x, mu, s):
    return -0.5 * ((x - mu) / s) ** 2 - np.log(s * np.sqrt(2 * np.pi))


def log_target(x):
    a = np.log(0.5) + log_normal(x, MODES[0], SIGMA)
    b = np.log(0.5) + log_normal(x, MODES[1], SIGMA)
    return np.logaddexp(a, b)


def main() -> None:
    setup()
    x = np.linspace(-8.0, 8.0, 6000)
    logp = log_target(x)
    p = np.exp(logp)

    # Forward KL over Gaussians = moment matching (exact).
    mu_f = 0.0
    s_f = np.sqrt(SIGMA**2 + MODES[1] ** 2)

    # Reverse KL: grid search over (mu, sigma).
    best = (np.inf, None, None)
    for mu in np.arange(0.0, 3.0, 0.01):
        for s in np.arange(0.2, 3.0, 0.01):
            logq = log_normal(x, mu, s)
            q = np.exp(logq)
            d = np.trapezoid(q * (logq - logp), x)
            if d < best[0]:
                best = (d, mu, s)
    _, mu_r, s_r = best
    print(f"forward KL fit: mu={mu_f:.2f}, sigma={s_f:.2f}")
    print(f"reverse KL fit: mu={mu_r:.2f}, sigma={s_r:.2f}")

    fig, ax = plt.subplots(figsize=(7.4, 3.2))
    ax.fill_between(x, p, color=GRAY, alpha=0.30, linewidth=0, label=r"target $p$ (two modes)")
    ax.plot(x, np.exp(log_normal(x, mu_f, s_f)), color=BLUE,
            label=r"$\arg\min_q \mathrm{KL}(p\,\|\,q)$: covers both modes")
    ax.plot(x, np.exp(log_normal(x, mu_r, s_r)), color=RED,
            label=r"$\arg\min_q \mathrm{KL}(q\,\|\,p)$: picks one mode")
    ax.set_xlim(-6.5, 6.5)
    ax.set_ylim(0, 0.75)
    ax.set_xlabel(r"$x$")
    ax.set_ylabel("density")
    ax.set_title("Fitting one Gaussian to a two-mode target")
    ax.legend(loc="upper left", fontsize=8.5)

    save(fig, "forward_reverse_kl")


if __name__ == "__main__":
    main()

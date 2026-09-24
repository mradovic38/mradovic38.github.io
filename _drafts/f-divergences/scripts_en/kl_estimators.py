"""New figure (not in the paper): Monte Carlo estimators of KL used in RLHF.

Setting from Schulman's "Approximating KL Divergence": samples x ~ q, we want
KL[q, p] = E_q[log q/p]. With r = p(x)/q(x):
    k1 = -log r              (unbiased, high variance, often negative)
    k2 = 0.5 * (log r)^2     (biased, low variance)
    k3 = (r - 1) - log r     (unbiased, low variance, always >= 0)
Here q = N(0, 1) plays the current policy and p = N(mu, 1) the reference model,
so KL[q, p] = mu^2 / 2.
"""

import numpy as np

from style import BLUE, GRAY, GREEN, RED, save, setup

import matplotlib.pyplot as plt

SEED = 0


def estimators(x, mu):
    log_r = mu * x - mu**2 / 2  # log p(x) - log q(x)
    r = np.exp(log_r)
    return -log_r, 0.5 * log_r**2, (r - 1) - log_r


def main() -> None:
    setup()
    rng = np.random.default_rng(SEED)

    # Table for the text: bias and standard deviation relative to the true KL.
    print(f"{'mu':>4} {'true KL':>8} | {'k1 bias':>8} {'k1 std':>7} | {'k2 bias':>8} {'k2 std':>7} | {'k3 bias':>8} {'k3 std':>7}")
    for mu in (0.1, 0.5, 1.0):
        x = rng.standard_normal(2_000_000)
        true = mu**2 / 2
        row = [f"{mu:>4} {true:>8.4f} |"]
        for k in estimators(x, mu):
            row.append(f"{(k.mean() - true) / true:>8.3f} {k.std() / true:>7.2f} |")
        print(" ".join(row))

    fig, (left, right) = plt.subplots(1, 2, figsize=(9.2, 3.5))

    # --- left: each estimator as a function of the ratio r ---
    r = np.linspace(0.05, 4.0, 800)
    left.plot(r, -np.log(r), color=RED, label=r"$k_1=-\log r$")
    left.plot(r, 0.5 * np.log(r) ** 2, color=BLUE, label=r"$k_2=\frac{1}{2}(\log r)^2$")
    left.plot(r, (r - 1) - np.log(r), color=GREEN, label=r"$k_3=(r-1)-\log r$")
    left.axhline(0.0, color=GRAY, linewidth=0.8)
    left.axvline(1.0, color=GRAY, linewidth=0.8, linestyle=":")
    left.annotate(r"$k_1<0$ whenever $r>1$", xy=(2.6, -0.95), xytext=(1.9, -1.9), fontsize=8,
                  arrowprops=dict(arrowstyle="->", color=GRAY, linewidth=0.8))
    left.set_xlabel(r"$r = p(x)/q(x)$  (reference / policy)")
    left.set_ylabel("per-sample estimate")
    left.set_title("Three per-sample KL estimators")
    left.set_ylim(-2.2, 3.0)
    left.legend(loc="upper center", fontsize=8)

    # --- right: distribution of single-sample estimates, mu = 0.5 ---
    mu = 0.5
    x = rng.standard_normal(400_000)
    k1, k2, k3 = estimators(x, mu)
    bins = np.linspace(-1.5, 1.5, 121)
    for k, color, name in ((k1, RED, "k_1"), (k2, BLUE, "k_2"), (k3, GREEN, "k_3")):
        label = rf"${name}$: mean {k.mean():.3f}, std {k.std():.3f}"
        right.hist(k, bins=bins, density=True, histtype="step", linewidth=1.6, color=color, label=label)
    right.axvline(mu**2 / 2, color="black", linewidth=1.0, linestyle="--", label=r"true KL $=0.125$")
    right.set_xlabel("single-sample estimate")
    right.set_ylabel("density (log scale)")
    right.set_yscale("log")
    right.set_title(r"Single-sample estimates, reference $N(0.5,1)$")
    right.legend(loc="upper right", fontsize=7.5)

    save(fig, "kl_estimators")


if __name__ == "__main__":
    main()

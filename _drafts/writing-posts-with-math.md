---
layout: post
title: Writing posts with Markdown and LaTeX
description: A quick tour of everything a post can contain — Markdown, code, and LaTeX math.
tags: [meta]
---

This post doubles as a reference for writing on this blog. Every post is a plain Markdown file in
`_posts/`, named `YYYY-MM-DD-some-title.md`. Push it to GitHub and it shows up on the
[blog page]({{ '/blog/' | relative_url }}).

## Text

Regular **bold**, *italic*, ~~strikethrough~~, [links](https://jonbarron.info), and footnotes[^1].

> Block quotes look like this.

- Lists
- work
  1. and nest

| Syntax   | Result        |
| -------- | ------------- |
| `$x^2$`  | $x^2$         |
| `$$x^2$$` | display math |

## Math

Inline math goes between single dollars: the Gaussian density is
$p(x) = \frac{1}{\sqrt{2\pi\sigma^2}} e^{-\frac{(x-\mu)^2}{2\sigma^2}}$, where $x_i$ and $y_i$
can have subscripts and $a * b * c$ can use asterisks without turning into italics.

Display math goes between double dollars:

$$
\mathbb{E}_{x \sim p}\left[ f(x) \right] = \int_{\mathbb{R}^d} f(x)\, p(x)\, dx
$$

Multi-line derivations with `aligned` (line breaks with `\\` survive intact). For example, the
KL divergence is non-negative, by Jensen's inequality:

$$
\begin{aligned}
D_{\mathrm{KL}}(p \,\|\, q)
  &= \mathbb{E}_{x \sim p}\left[ \log \frac{p(x)}{q(x)} \right] \\
  &= -\mathbb{E}_{x \sim p}\left[ \log \frac{q(x)}{p(x)} \right] \\
  &\geq -\log \mathbb{E}_{x \sim p}\left[ \frac{q(x)}{p(x)} \right] \\
  &= -\log \int q(x)\, dx = 0.
\end{aligned}
$$

Numbered equations work with `equation` or `align`, and can be referenced with `\eqref`:

\begin{align}
\nabla_\theta \mathcal{L}(\theta) &= \frac{1}{N} \sum_{i=1}^{N} \nabla_\theta \ell(f_\theta(x_i), y_i) \label{eq:grad} \\
\theta_{t+1} &= \theta_t - \eta \, \nabla_\theta \mathcal{L}(\theta_t) \label{eq:sgd}
\end{align}

Equation \eqref{eq:sgd} is plain gradient descent using the gradient from \eqref{eq:grad}.

Matrices, cases, and anything else MathJax supports:

$$
A = \begin{bmatrix} a_{11} & a_{12} \\ a_{21} & a_{22} \end{bmatrix}, \qquad
|x| = \begin{cases} x & \text{if } x \geq 0 \\ -x & \text{otherwise} \end{cases}
$$

A literal dollar sign is written as `\$`: this costs \$5, not $5.

## Code

Inline `code` and fenced blocks with syntax highlighting. Dollar signs in code are left alone:

```python
import numpy as np

def softmax(z: np.ndarray) -> np.ndarray:
    """Numerically stable softmax. Costs $0."""
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)
```

## Images

Put images in `images/` and reference them like this:

```markdown
![Alt text](/images/figure.png)
```

[^1]: Footnotes end up at the bottom of the post.

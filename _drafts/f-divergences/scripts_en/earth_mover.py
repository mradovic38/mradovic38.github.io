"""New animation (not in the paper): the earth mover's distance as shovelling dirt.

P and Q are made of equal "units" of mass (balls). A shovel moves them one at a time
from P to Q using the optimal 1D plan (sort both sides and match them in order), and a
counter adds up the work, distance times mass. At the end, the total work divided by the
number of units is W_1(P, Q) for these discrete distributions (not shown; it's an illustration).

Writes images/f-divergences/earth_mover.gif (needs ffmpeg), a looping GIF at 20 fps.
"""

import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from style import BLUE, GRAY, OUT, RED, setup

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Polygon

R = 0.19  # ball radius
D = 2 * R + 0.02  # spacing between stacked balls
SRC_X = np.round(np.arange(-3.6, -1.1, D), 3)
SRC_H = [1, 2, 3, 4, 3, 2, 1]
DST_X = np.round(np.arange(1.2, 4.6, D), 3)
DST_H = [1, 2, 2, 3, 3, 2, 2, 1]
assert sum(SRC_H) == sum(DST_H) and len(SRC_X) >= len(SRC_H) and len(DST_X) >= len(DST_H)
SRC_X, DST_X = SRC_X[: len(SRC_H)], DST_X[: len(DST_H)]
N = sum(SRC_H)


def envelope(xs, hs):
    """Smooth curve drawn over the stacks (a Gaussian fitted to the stack profile)."""
    xs, hs = np.asarray(xs), np.asarray(hs, dtype=float)
    mu = np.average(xs, weights=hs)
    sd = np.sqrt(np.average((xs - mu) ** 2, weights=hs)) * 1.15
    grid = np.linspace(mu - 4 * sd, mu + 4 * sd, 400)
    peak = (hs.max() + 0.45) * D
    return grid, peak * np.exp(-0.5 * ((grid - mu) / sd) ** 2)


def plan():
    """Optimal 1D plan: the i-th unit of P from the left goes to the i-th slot of Q from the left.

    Moves are ordered right to left, taking the top ball of a source column and dropping it on
    the next free spot of its destination column, so nothing ever floats.
    """
    src_units = [c for c, h in enumerate(SRC_H) for _ in range(h)]  # source column per unit, left to right
    dst_units = [c for c, h in enumerate(DST_H) for _ in range(h)]
    pairs = list(zip(src_units, dst_units))[::-1]
    src_left = list(SRC_H)
    dst_fill = [0] * len(DST_H)
    moves = []
    for s, d in pairs:
        src_left[s] -= 1
        start = (SRC_X[s], R + src_left[s] * D)
        end = (DST_X[d], R + dst_fill[d] * D)
        dst_fill[d] += 1
        moves.append((s, src_left[s], start, end))
    return moves


SHOVEL_ANGLE = np.deg2rad(155)  # direction from the blade tip toward the handle


def draw_shovel(ax, bx, by):
    """Draw a spade whose blade is centred at (bx, by), handle trailing up and to the left."""
    u = np.array([np.cos(SHOVEL_ANGLE), np.sin(SHOVEL_ANGLE)])  # along the shovel
    n = np.array([-u[1], u[0]])  # across the shovel
    if n[1] < 0:
        n = -n
    tip = np.array([bx, by]) - 0.3 * u

    def world(s, w):
        return tip + s * u + w * n

    blade = [world(*pt) for pt in [(0.0, -0.07), (0.1, -0.2), (0.55, -0.23), (0.55, 0.23), (0.1, 0.2), (0.0, 0.07)]]
    ax.add_patch(Polygon(blade, closed=True, facecolor="#8a8f94", edgecolor="#3f4347", linewidth=1.2, zorder=6))
    collar = [world(*pt) for pt in [(0.55, -0.07), (0.72, -0.05), (0.72, 0.05), (0.55, 0.07)]]
    ax.add_patch(Polygon(collar, closed=True, facecolor="#4b4f53", edgecolor="#3f4347", zorder=6))
    shaft = np.array([world(0.72, 0), world(2.05, 0)])
    ax.plot(shaft[:, 0], shaft[:, 1], color="#9c6b3c", linewidth=4.0, solid_capstyle="round", zorder=5)
    grip = np.array([world(2.05, -0.16), world(2.05, 0.16)])
    ax.plot(grip[:, 0], grip[:, 1], color="#7a4f28", linewidth=4.5, solid_capstyle="round", zorder=5)


def draw(state, path):
    """state: (moved, current_index, s) where s in [0, 1] is the progress of the current move."""
    moved, cur, s = state
    moves = plan()
    fig, ax = plt.subplots(figsize=(8.0, 3.4))

    gx, gy = envelope(SRC_X, SRC_H)
    ax.plot(gx, gy, color=BLUE, linewidth=1.6)
    hx, hy = envelope(DST_X, DST_H)
    ax.plot(hx, hy, color=RED, linewidth=1.6)
    ax.text(gx[np.argmax(gy)], gy.max() + 0.12, "P", color=BLUE, ha="center", fontsize=13)
    ax.text(hx[np.argmax(hy)], hy.max() + 0.12, "Q", color=RED, ha="center", fontsize=13)
    ax.axhline(0, color=GRAY, linewidth=0.8)

    # Balls still in P: every source ball that hasn't been moved (or started moving).
    in_flight = set((m[0], m[1]) for m in moves[:moved + (1 if cur is not None else 0)])
    for c, h in enumerate(SRC_H):
        for k in range(h):
            if (c, k) not in in_flight:
                ax.add_patch(Circle((SRC_X[c], R + k * D), R * 0.95, color=BLUE, alpha=0.55))
    # Balls already delivered to Q.
    for m in moves[:moved]:
        ax.add_patch(Circle(m[3], R * 0.95, color=RED, alpha=0.55))

    work = sum(abs(m[3][0] - m[2][0]) for m in moves[:moved])
    if cur is not None:
        (x0, y0), (x1, y1) = moves[cur][2], moves[cur][3]
        x = x0 + (x1 - x0) * s
        y = y0 + (y1 - y0) * s + 1.3 * np.sin(np.pi * s)  # arc
        color = BLUE if s < 0.5 else RED
        ax.add_patch(Circle((x, y), R * 0.95, color=color, alpha=0.9, zorder=7))
        up = np.array([-np.sin(SHOVEL_ANGLE), np.cos(SHOVEL_ANGLE)])
        up = up if up[1] > 0 else -up
        draw_shovel(ax, x - (R + 0.03) * up[0], y - (R + 0.03) * up[1])
        work += abs(x1 - x0) * s

    ax.text(-4.3, 2.55, f"work so far = {work:5.2f}", fontsize=11, family="DejaVu Sans Mono")

    ax.set_xlim(-4.5, 5.2)
    ax.set_ylim(-0.1, 2.9)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main() -> None:
    setup()
    OUT.mkdir(parents=True, exist_ok=True)
    frames_per_move, hold_start, hold_end = 10, 10, 40  # 0.5 s per unit at 20 fps
    states = [(0, None, 0.0)] * hold_start
    for i in range(N):
        for f in range(frames_per_move):
            s = 0.5 - 0.5 * np.cos(np.pi * (f + 1) / frames_per_move)
            states.append((i, i, s) if f < frames_per_move - 1 else (i + 1, None, 0.0))
    states += [(N, None, 0.0)] * hold_end

    tmp = Path(tempfile.mkdtemp())
    for i, st in enumerate(states):
        draw(st, tmp / f"{i:04d}.png")
    # Two-pass GIF: build an optimal palette, then encode with it (sharp edges, small file).
    palette = tmp / "palette.png"
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-framerate", "20", "-i", str(tmp / "%04d.png"),
                    "-vf", "palettegen=max_colors=64:stats_mode=diff", str(palette)], check=True)
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-framerate", "20", "-i", str(tmp / "%04d.png"),
                    "-i", str(palette), "-lavfi", "paletteuse=dither=none:diff_mode=rectangle",
                    "-loop", "0", str(OUT / "earth_mover.gif")], check=True)
    shutil.rmtree(tmp)
    moves = plan()
    total = sum(abs(m[3][0] - m[2][0]) for m in moves)
    print(f"saved: {OUT / 'earth_mover.gif'}  (N={N}, total work={total:.3f}, W1={total / N:.3f})")


if __name__ == "__main__":
    main()

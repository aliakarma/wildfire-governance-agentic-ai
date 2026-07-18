"""Server-side GIF/MP4 export of an episode (PyroRL-style).

Re-runs the real simulation, then renders each frame as a green grid with a warm
fire gradient (yellow→orange→red by fire age) and visible agents — matching the
in-browser canvas — and stitches the frames with imageio.
"""
from __future__ import annotations

import base64
from typing import Any, Dict, List, Tuple

import numpy as np

from .simulation_service import collect_episode

# Warm fire ramp (t, RGB), matching frontend lib/colormap.ts.
_FIRE_STOPS: List[Tuple[float, Tuple[int, int, int]]] = [
    (0.0, (255, 241, 148)),
    (0.15, (255, 202, 40)),
    (0.35, (255, 143, 26)),
    (0.6, (233, 58, 30)),
    (1.0, (140, 26, 12)),
]
_AGE_MAX = 45


def _fire_lut() -> np.ndarray:
    lut = np.zeros((256, 3), dtype=np.uint8)
    for age in range(256):
        t = min(age, _AGE_MAX) / _AGE_MAX
        lo, hi = _FIRE_STOPS[0], _FIRE_STOPS[-1]
        for i in range(len(_FIRE_STOPS) - 1):
            if _FIRE_STOPS[i][0] <= t <= _FIRE_STOPS[i + 1][0]:
                lo, hi = _FIRE_STOPS[i], _FIRE_STOPS[i + 1]
                break
        span = (hi[0] - lo[0]) or 1
        f = (t - lo[0]) / span
        lut[age] = [int(lo[1][k] + (hi[1][k] - lo[1][k]) * f) for k in range(3)]
    return lut


_FIRE = _fire_lut()


def episode_to_gif(
    params: Dict[str, Any] | None,
    fps: int = 12,
    cmap: str = "inferno",  # kept for API compatibility; unused
    theme: str = "dark",
    max_frames: int = 240,
) -> bytes:
    """Render an episode to an animated GIF and return the raw bytes."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import imageio.v2 as imageio

    grass = (23, 168, 119) if theme == "dark" else (37, 196, 141)
    gridline = (11, 14, 20) if theme == "dark" else (255, 255, 255)
    uav_c = "#39C6FF" if theme == "dark" else "#0B6FB8"
    bg = "#0B0E14" if theme == "dark" else "#F7F6F3"
    txt = "#E8EDF4" if theme == "dark" else "#1B1E24"

    out = collect_episode(params)
    frames: List[Dict[str, Any]] = out["frames"]
    if len(frames) > max_frames:
        step = -(-len(frames) // max_frames)
        frames = frames[::step]

    images: List[np.ndarray] = []
    for fr in frames:
        grid = fr["grid_size"]
        fire = np.frombuffer(base64.b64decode(fr["fire_b64"]), dtype=np.uint8).reshape(grid, grid)
        img = np.empty((grid, grid, 3), dtype=np.uint8)
        img[:] = grass
        ys, xs = np.nonzero(fire)
        img[ys, xs] = _FIRE[fire[ys, xs]]

        fig, ax = plt.subplots(figsize=(5, 5), dpi=110)
        fig.patch.set_facecolor(bg)
        ax.imshow(img, interpolation="nearest")
        # gridlines
        if grid <= 120:
            ax.set_xticks(np.arange(-0.5, grid, 1), minor=True)
            ax.set_yticks(np.arange(-0.5, grid, 1), minor=True)
            ax.grid(which="minor", color=np.array(gridline) / 255.0, linewidth=0.4)
        ax.tick_params(which="both", length=0)
        ax.set_xticklabels([]); ax.set_yticklabels([])
        if fr["uavs"]:
            ax.scatter([u["x"] for u in fr["uavs"]], [u["y"] for u in fr["uavs"]],
                       s=22, c=uav_c, edgecolors="white", linewidths=0.5, zorder=3)
        m = fr["metrics"]
        ax.set_title(f"t={fr['t']}   L_d={m['ld']}   F_p={m['fp_pct']}%   comp={m['compliance']}%",
                     color=txt, fontsize=9)
        ax.set_xlim(-0.5, grid - 0.5); ax.set_ylim(grid - 0.5, -0.5)
        buf = __import__("io").BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0.05, facecolor=bg)
        plt.close(fig)
        buf.seek(0)
        images.append(imageio.imread(buf))

    gif_buf = __import__("io").BytesIO()
    imageio.mimsave(gif_buf, images, format="GIF", duration=1.0 / max(1, fps), loop=0)
    return gif_buf.getvalue()

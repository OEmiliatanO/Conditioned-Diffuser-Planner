import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

from typing import List

def visualize_maze_and_trajectories(
    trajectories: torch.FloatTensor,
    MAZE_ARR: np.ndarray,
    num_show: int = 5,
    dangerous_zones: List = [],
    save_path: str = "diffusion_trajectories.png",
    use_scatter: bool = False
):
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches

    cell = 2.0
    rows, cols = MAZE_ARR.shape

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.set_facecolor("#1a1a2e")
    fig.patch.set_facecolor("#1a1a2e")

    for r in range(rows):
        for c in range(cols):
            x = c * cell
            y = (rows - 1 - r) * cell
            color = "#0f3460" if MAZE_ARR[r, c] == 1 else "#16213e"
            rect = patches.Rectangle((x, y), cell, cell, linewidth=0.5,
                                      edgecolor="#e94560", facecolor=color)
            ax.add_patch(rect)

    for zone in dangerous_zones:
        center = zone['center']
        radius = zone['radius']
        circle = patches.Circle(center, radius, color='red', alpha=0.3, zorder=4)
        ax.add_patch(circle)

    colors = plt.cm.plasma(np.linspace(0.2, 0.95, num_show))
    for i in range(min(num_show, len(trajectories))):
        traj = trajectories[i].numpy()
        xs, ys = traj[:, 0], traj[:, 1]
        if use_scatter:
            ax.scatter(xs, ys, color=colors[i], s=20, zorder=5)
        else:
            ax.quiver(xs[:-1], ys[:-1], xs[1:]-xs[:-1], ys[1:]-ys[:-1], angles='xy', scale_units='xy', scale=1, color=colors[i], alpha=0.7, zorder=5, width=0.003)
        ax.scatter(xs[0], ys[0], color=colors[i], s=60, zorder=5, marker='o')
        ax.scatter(xs[-1], ys[-1], color=colors[i], s=80, zorder=5, marker='*')

    ax.set_xlim(0, cols * cell)
    ax.set_ylim(0, rows * cell)
    ax.set_aspect('equal')
    ax.set_title("U-Maze Sample Trajectories", color='white', fontsize=13, pad=12)
    ax.tick_params(colors='#aaaaaa')
    for spine in ax.spines.values():
        spine.set_edgecolor('#e94560')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Visualized trajectories saved to {save_path}")

def set_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
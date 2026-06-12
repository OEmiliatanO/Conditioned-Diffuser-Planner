import random

import numpy as np
import torch
from collections import deque
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from typing import List, Tuple
import argparse

from MAZE import UMAZE_ARR, MEDIUM_MAZE_ARR, LARGE_MAZE_ARR
from utilis import visualize_maze_and_trajectories

# coordinate system (cell_size=2.0)：
#
#   y=10 ┌─────────────────────────┐
#        │  W    W    W    W    W  │  row 0
#   y=8  ├──────────────────────────┤
#        │  W    .    .    .    W  │  row 1  ← empty cells (1,1) (1,2) (1,3)
#   y=6  ├──────────────────────────┤
#        │  W    W    W    .    W  │  row 2  ← empty cells (2,3)
#   y=4  ├──────────────────────────┤
#        │  W    .    .    .    W  │  row 3  ← empty cells (3,1) (3,2) (3,3)
#   y=2  ├──────────────────────────┤
#        │  W    W    W    W    W  │  row 4
#   y=0  └─────────────────────────┘
#         x=0  x=2  x=4  x=6  x=8
#         col0 col1 col2 col3 col4


class UMazeDataGenerator:

    def __init__(
        self,
        maze: np.ndarray = UMAZE_ARR,
        cell_size: float = 2.0,
        dt: float = 0.1,
        friction: float = 0.3,
        wall_margin: float = 0.15,
        kp: float = 8.0,
        kd: float = 3.0,
        noise_scale: float = 0.2,
        max_vel: float = 2.0,
    ):
        self.maze = maze
        self.cell_size = cell_size
        self.dt = dt
        self.friction = friction
        self.margin = wall_margin * cell_size
        self.kp = kp
        self.kd = kd
        self.noise_scale = noise_scale
        self.max_vel = max_vel

        self.rows, self.cols = maze.shape
        self.state_dim = 4   # [x, y, vx, vy]
        self.action_dim = 2  # [ax, ay]

        self.free_cells = [
            (r, c)
            for r in range(self.rows)
            for c in range(self.cols)
            if maze[r, c] == 0
        ]
        assert len(self.free_cells) > 1, "no free cells in the maze!"

    # ------------------------------------------------------------------
    # coordinate conversion
    # ------------------------------------------------------------------

    def cell_to_world(self, row: int, col: int) -> np.ndarray:
        x = col * self.cell_size + self.cell_size * 0.5
        y = (self.rows - 1 - row) * self.cell_size + self.cell_size * 0.5
        return np.array([x, y], dtype=np.float64)

    def world_to_cell(self, x: float, y: float) -> Tuple[int, int]:
        col = int(np.clip(x / self.cell_size, 0, self.cols - 1))
        row = int(np.clip((self.rows * self.cell_size - y) / self.cell_size, 0, self.rows - 1))
        return row, col

    def is_wall(self, x: float, y: float) -> bool:
        r, c = self.world_to_cell(x, y)
        return self.maze[r, c] == 1

    # ------------------------------------------------------------------
    # BFS
    # ------------------------------------------------------------------

    def bfs(self, start: Tuple[int, int], goal: Tuple[int, int]):
        if start == goal:
            return [start]

        queue: deque[Tuple[Tuple[int, int], List[Tuple[int, int]]]] = deque([(start, [start])])
        visited = {start}
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

        while queue:
            (r, c), path = queue.popleft()
            for dr, dc in directions:
                nr, nc = r + dr, c + dc
                if (nr, nc) not in visited and 0 <= nr < self.rows and 0 <= nc < self.cols:
                    if self.maze[nr, nc] == 0:
                        new_path = path + [(nr, nc)]
                        if (nr, nc) == goal:
                            return new_path
                        visited.add((nr, nc))
                        queue.append(((nr, nc), new_path))
        return None

    # ------------------------------------------------------------------
    # waypoint interpolation
    # ------------------------------------------------------------------

    def _interpolate_waypoints(
        self, waypoints: List[np.ndarray], horizon: int
    ) -> List[np.ndarray]:
        cum_len = [0.0]
        for i in range(1, len(waypoints)):
            cum_len.append(cum_len[-1] + np.linalg.norm(waypoints[i] - waypoints[i - 1]))
        total = cum_len[-1]

        targets = []
        for t in range(horizon):
            s = (t / max(horizon - 1, 1)) * total
            idx = np.searchsorted(cum_len, s, side='right') - 1
            idx = int(np.clip(idx, 0, len(waypoints) - 2))
            seg = cum_len[idx + 1] - cum_len[idx]
            alpha = (s - cum_len[idx]) / seg if seg > 1e-8 else 0.0
            targets.append((1.0 - alpha) * waypoints[idx] + alpha * waypoints[idx + 1])

        return targets

    # ------------------------------------------------------------------
    # collision
    # ------------------------------------------------------------------

    def _resolve_collision(
        self, pos: np.ndarray, next_pos: np.ndarray, vel: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        if not self.is_wall(next_pos[0], next_pos[1]):
            return next_pos, vel

        pos_x = np.array([next_pos[0], pos[1]])
        if not self.is_wall(pos_x[0], pos_x[1]):
            vel = vel.copy()
            vel[1] *= -0.3
            return pos_x, vel

        pos_y = np.array([pos[0], next_pos[1]])
        if not self.is_wall(pos_y[0], pos_y[1]):
            vel = vel.copy()
            vel[0] *= -0.3
            return pos_y, vel

        vel = vel.copy() * -0.3
        return pos.copy(), vel

    # ------------------------------------------------------------------
    # generate single trajectory
    # ------------------------------------------------------------------

    def generate_trajectory(
        self,
        horizon: int,
        start_cell: Tuple[int, int] = None,
        goal_cell: Tuple[int, int] = None,
    ) -> np.ndarray:
        if start_cell is not None and goal_cell is not None:
            path = self.bfs(start_cell, goal_cell)
        else:
            while True:
                start_cell = self.free_cells[np.random.randint(len(self.free_cells))]
                goal_cell = self.free_cells[np.random.randint(len(self.free_cells))]
                if start_cell != goal_cell:
                    path = self.bfs(start_cell, goal_cell)
                    if path:
                        break

        jitter = self.cell_size * 0.15
        waypoints = []
        for i, (r, c) in enumerate(path):
            center = self.cell_to_world(r, c)
            if 0 < i < len(path) - 1:
                center = center + np.random.uniform(-jitter, jitter, 2)
            waypoints.append(center)

        targets = self._interpolate_waypoints(waypoints, horizon)

        trajectory = np.zeros((horizon, self.state_dim + self.action_dim))
        pos = waypoints[0].copy()
        vel = np.zeros(2)

        total_time = 12.8
        effective_dt = total_time / horizon
        for t in range(horizon):
            target = targets[t]

            error = target - pos
            action = self.kp * error - self.kd * vel
            action += np.random.randn(2) * self.noise_scale
            action = np.clip(action, -3.0, 3.0)

            trajectory[t, 0:2] = pos
            trajectory[t, 2:4] = vel
            trajectory[t, 4:6] = action

            new_vel = (1.0 - self.friction) * vel + action * effective_dt
            new_vel = np.clip(new_vel, -self.max_vel, self.max_vel)
            next_pos = pos + new_vel * effective_dt

            pos, vel = self._resolve_collision(pos, next_pos, new_vel)

        return trajectory

    # ------------------------------------------------------------------
    # generate multiple trajectories
    # ------------------------------------------------------------------

    def generate_trajectories(self, num_samples: int, horizon: int) -> torch.FloatTensor:
        trajs = []
        for i in range(num_samples):
            if i % 500 == 0:
                print(f"  generating {i:>5d} / {num_samples} ...")
            trajs.append(self.generate_trajectory(horizon))

        return torch.FloatTensor(np.stack(trajs))

    def generate_trajectories_per_path(
        self, n_per_path: int, horizon: int
    ) -> torch.FloatTensor:
        valid_pairs = [
            (s, g)
            for s in self.free_cells
            for g in self.free_cells
            if s != g and self.bfs(s, g) is not None
        ]
        total = len(valid_pairs) * n_per_path

        print(f"  find {len(valid_pairs)} valid paths, {n_per_path} trajectories each, total {total} trajectories")

        trajs = []
        count = 0
        for start_cell, goal_cell in valid_pairs:
            for _ in range(n_per_path):
                if count % 500 == 0:
                    print(f"  generating trajectory {count:>5d} / {total} ...")
                trajs.append(self.generate_trajectory(horizon, start_cell, goal_cell))
                count += 1

        return torch.FloatTensor(np.stack(trajs))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--maze_size", type=str, default="small", choices=["small", "medium", "large"])
    parser.add_argument("--n_per_path", type=int, default=100)
    parser.add_argument("--total_trajectories", type=int, default=100)
    parser.add_argument("--horizon", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--brute_force_pairs", action="store_true")
    parser.add_argument("--save_path", type=str, default="simulated_maze_data.pth")
    args = parser.parse_args()

    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    generator = UMazeDataGenerator(
        maze={
            "small": UMAZE_ARR,
            "medium": MEDIUM_MAZE_ARR,
            "large": LARGE_MAZE_ARR
        }[args.maze_size],
        cell_size=2.0,
        dt=0.1,
        friction=0.3,
        kp=8.0,
        kd=3.0,
        noise_scale=0.2,
    )

    HORIZON = args.horizon

    if args.brute_force_pairs:
        N_PER_PATH = args.n_per_path
        data = generator.generate_trajectories_per_path(N_PER_PATH, HORIZON)
    else:
        TOTAL_TRAJECTORIES = args.total_trajectories
        data = generator.generate_trajectories(TOTAL_TRAJECTORIES, HORIZON)
    print(f"Data Shape: {data.shape}")

    mean = data.mean(dim=(0, 1), keepdim=True)   # [1, 1, 6]
    std  = data.std(dim=(0, 1), keepdim=True) + 1e-6
    normalized_data = (data - mean) / std
    print(f"NaN Check: {np.isnan(normalized_data.numpy()).any()}")
    
    visualize_maze_and_trajectories(data, MAZE_ARR=generator.maze, num_show=50, save_path=f"{args.maze_size}_maze_sample_trajectories.png")

    data = {
        "trajectories": data,
        "mean": mean,
        "std": std,
        "maze_size": args.maze_size,
    }
    torch.save(data, args.save_path)
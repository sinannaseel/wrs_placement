# env/gym_world.py
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from enum import Enum

from dqrobotics import *
from dqrobotics.interfaces.coppeliasim import DQ_CoppeliaSimInterfaceZMQ

CUBE_NAMES = ["short0", "short1", "long"]


class DependencyWorldEnv(gym.Env):

    metadata = {"render_modes": ["human"], "render_fps": 10}

    def __init__(self, host="localhost", port=23000):
        super().__init__()

        self.observation_space = spaces.Box(
            low=-10.0, high=10.0, shape=(12,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(len(CUBE_NAMES))

        self.vi = DQ_CoppeliaSimInterfaceZMQ()
        if not self.vi.connect(host, port, 500, 1):
            raise RuntimeError("Could not connect to CoppeliaSim")

        self.vi.set_synchronous(True)
        self.vi.start_simulation()

        # Cache poses
        self.initial_poses = {
            name: self.vi.get_object_pose(name) for name in CUBE_NAMES
        }
        self.limbo_pose = self.vi.get_object_pose("limbo")

        self.settle_steps = 20
        self.active = None
        self.step_count = 0
        self.max_steps = 3

    def _step_sim(self, n=1):
        for _ in range(n):
            self.vi.trigger_next_simulation_step()
            self.vi.wait_for_simulation_step_to_end()

    def _get_obs(self):
        obs = []
        for name in CUBE_NAMES:
            pose = self.vi.get_object_pose(name)
            t = translation(pose)
            obs.extend(vec3(t))

        obs.extend([1.0 if a else 0.0 for a in self.active])
        return np.array(obs, dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        for name, pose in self.initial_poses.items():
            self.vi.set_object_pose(name, pose)

        self.active = [True] * len(CUBE_NAMES)
        self.step_count = 0

        self._step_sim(self.settle_steps)
        return self._get_obs(), {}

    def step(self, action):

        if not self.active[action]:
            return self._get_obs(), -5.0, True, False, {"invalid": True}

        obs_before = self._get_obs()
        before = obs_before[:3 * len(CUBE_NAMES)].reshape(len(CUBE_NAMES), 3)

        picked_name = CUBE_NAMES[action]
        self.vi.set_object_pose(picked_name, self.limbo_pose)
        self.active[action] = False

        self._step_sim(self.settle_steps)

        obs_after = self._get_obs()
        after = obs_after[:3 * len(CUBE_NAMES)].reshape(len(CUBE_NAMES), 3)

        disturbance = sum(
            np.linalg.norm(after[i] - before[i])
            for i in range(len(CUBE_NAMES))
            if self.active[i]
        )

        reward = -10.0 * disturbance

        self.step_count += 1
        terminated = self.step_count >= self.max_steps
        truncated = False

        info = {
            "picked": picked_name,
            "active": self.active.copy(),
            "disturbance": disturbance,
        }

        return self._get_obs(), reward, terminated, truncated, info

    def close(self):
        self.vi.stop_simulation()

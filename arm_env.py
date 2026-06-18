import numpy as np
import torch as th
import motornet as mn

dt = 0.01
duration = 2.0
n_steps = int(duration / dt)

amplitude = 0.08
frequency = 0.5
x_center = 0.25
y_center = 0.25


class SinusoidalArmEnv(mn.environment.Environment):
    def __init__(self, effector, max_ep_duration):
        super().__init__(effector=effector, max_ep_duration=max_ep_duration)

    def make_target(self, batch_size, device="cpu"):
        t = th.arange(n_steps, device=device) * dt
        x = x_center + 0.05 * th.cos(2 * np.pi * frequency * t)
        y = y_center + amplitude * th.sin(2 * np.pi * frequency * t)
        target = th.stack([x, y], dim=-1)
        return target.unsqueeze(0).repeat(batch_size, 1, 1)

    def get_cartesian_position(self, info):
        if "cartesian" in info:
            return info["cartesian"][:, :2]
        if "cartesian_state" in info:
            return info["cartesian_state"][:, :2]
        if "states" in info:
            for key in ["cartesian", "cartesian_state"]:
                if key in info["states"]:
                    return info["states"][key][:, :2]
        raise KeyError(f"Could not find cartesian position. Info keys: {info.keys()}")

    def get_joint_angles(self, info):
        if "joint" in info:
            return info["joint"][:, :2]
        if "joint_state" in info:
            return info["joint_state"][:, :2]
        if "states" in info:
            for key in ["joint", "joint_state"]:
                if key in info["states"]:
                    return info["states"][key][:, :2]
        raise KeyError(f"Could not find joint angles. Info keys: {info.keys()}")


class Policy(th.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.gru = th.nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.fc = th.nn.Linear(hidden_dim, output_dim)
        self.sigmoid = th.nn.Sigmoid()

    def forward(self, obs, hidden):
        y, hidden = self.gru(obs[:, None, :], hidden)
        action = self.sigmoid(self.fc(y)).squeeze(1)
        return action, hidden

    def init_hidden(self, batch_size, device="cpu"):
        return th.zeros(1, batch_size, self.hidden_dim, device=device)


def make_env():
    muscle = mn.muscle.RigidTendonHillMuscle()
    effector = mn.effector.RigidTendonArm26(muscle=muscle)
    return SinusoidalArmEnv(effector=effector, max_ep_duration=duration)

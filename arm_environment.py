import numpy as np
import torch as th
import motornet as mn

dt = 0.01
duration = 2.0
n_steps = int(duration / dt)

amplitude = 0.08
frequency = 0.5

# Hand position when both joints are at mid-range (theta_max / 2)
# theta1 = 2.3562/2 = 1.1781 rad, theta2 = 2.7053/2 = 1.3526 rad
x_center = -0.1545
y_center = 0.4765


class SinusoidalArmEnv(mn.environment.Environment):
    def __init__(self, effector, max_ep_duration):
        super().__init__(effector=effector, max_ep_duration=max_ep_duration)

    def make_target(self, batch_size, device="cpu", freq=None):
        f = freq if freq is not None else frequency
        t = th.arange(n_steps, device=device) * dt
        disp = amplitude * th.sin(2 * np.pi * f * t)
        x = x_center + disp * np.cos(np.pi / 4)
        y = y_center + disp * np.sin(np.pi / 4)
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


def midrange_start_joint_state(batch_size, device="cpu"):
    """Start arm at theta1=theta1_max/2, theta2=theta2_max/2 — both joints mid-range."""
    t1 = 2.3562 / 2
    t2 = 2.7053 / 2
    base = th.tensor([t1, t2, 0.0, 0.0], dtype=th.float32, device=device)
    return base.unsqueeze(0).expand(batch_size, -1).clone()

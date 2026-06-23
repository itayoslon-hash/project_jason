import numpy as np
import torch as th
import motornet as mn

dt = 0.01
duration = 2.0
n_steps = int(duration / dt)

amplitude = 0.08
frequency = 0.5
x_center = 0.3
y_center = 0.3


class SinusoidalArmEnv(mn.environment.Environment):
    def __init__(self, effector, max_ep_duration):
        super().__init__(effector=effector, max_ep_duration=max_ep_duration)

    def make_target(self, batch_size, device="cpu", freq=None):
        f = freq if freq is not None else frequency
        t = th.arange(n_steps, device=device) * dt
        disp = amplitude * th.sin(2 * np.pi * f * t)
        x = x_center + disp * np.cos(3 * np.pi / 4)
        y = y_center + disp * np.sin(3 * np.pi / 4)
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


def _ik_near(tx, ty, n_iter=500):
    """Gradient-descent IK: find valid joint angles closest to cartesian target."""
    l1, l2 = 0.309, 0.333
    lb = th.tensor([0.0, 0.0])
    ub = th.tensor([2.3562, 2.7053])
    theta = th.tensor([0.3, 1.5], dtype=th.float32, requires_grad=True)
    opt = th.optim.Adam([theta], lr=0.02)
    for _ in range(n_iter):
        opt.zero_grad()
        x = l1 * th.cos(theta[0]) + l2 * th.cos(theta[0] + theta[1])
        y = l1 * th.sin(theta[0]) + l2 * th.sin(theta[0] + theta[1])
        loss = (x - tx) ** 2 + (y - ty) ** 2
        loss.backward()
        opt.step()
        with th.no_grad():
            theta.clamp_(lb, ub)
    return theta.detach()


def ellipse_start_joint_state(batch_size, noise_std=0.05, device="cpu"):
    """
    Finds joint angles closest to ellipse t=0 point via IK, adds per-batch noise.
    Returns joint state [theta1, theta2, dtheta1, dtheta2].
    """
    lb = th.tensor([0.0, 0.0, 0.0, 0.0], device=device)
    ub = th.tensor([2.3562, 2.7053, 1.0, 1.0], device=device)

    t = _ik_near(x_center, y_center)  # t=0 on rail: (x_center, y_center)
    base = th.tensor([t[0], t[1], 0.0, 0.0], dtype=th.float32, device=device)

    noise = th.randn(batch_size, 4, device=device) * noise_std
    noise[:, 2:] = 0.0  # start stationary

    return (base.unsqueeze(0).expand(batch_size, -1) + noise).clamp(lb, ub)

import time
import numpy as np
import torch as th
import matplotlib.pyplot as plt
import motornet as mn


# -----------------------------
# Settings
# -----------------------------
device = th.device("cuda" if th.cuda.is_available() else "cpu")
print("Using device:", device)

dt = 0.01
duration = 2.0
n_steps = int(duration / dt)

batch_size = 32
n_batches = 1000

amplitude = 0.08
frequency = 0.5
x_center = 0.25
y_center = 0.25


# -----------------------------
# Effector
# -----------------------------
muscle = mn.muscle.RigidTendonHillMuscle()
effector = mn.effector.RigidTendonArm26(muscle=muscle)


# -----------------------------
# Environment
# -----------------------------
class SinusoidalArmEnv(mn.environment.Environment):
    def __init__(self, effector, max_ep_duration):
        super().__init__(effector=effector, max_ep_duration=max_ep_duration)

    def make_target(self, batch_size):
        t = th.arange(n_steps, device=device) * dt

        x = x_center + 0.05 * th.cos(2 * np.pi * frequency * t)
        y = y_center + amplitude * th.sin(2 * np.pi * frequency * t)

        target = th.stack([x, y], dim=-1)
        return target.unsqueeze(0).repeat(batch_size, 1, 1)

    def get_position(self, info):
        if "cartesian" in info:
            return info["cartesian"][:, :2]

        if "cartesian_state" in info:
            return info["cartesian_state"][:, :2]

        if "states" in info:
            for key in ["cartesian", "cartesian_state"]:
                if key in info["states"]:
                    return info["states"][key][:, :2]

        raise KeyError(f"Could not find cartesian position. Info keys: {info.keys()}")


env = SinusoidalArmEnv(effector=effector, max_ep_duration=duration)


# -----------------------------
# Policy network
# -----------------------------
class Policy(th.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.gru = th.nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.fc = th.nn.Linear(hidden_dim, output_dim)
        self.sigmoid = th.nn.Sigmoid()

        th.nn.init.xavier_uniform_(self.gru.weight_ih_l0)
        th.nn.init.orthogonal_(self.gru.weight_hh_l0)
        th.nn.init.zeros_(self.gru.bias_ih_l0)
        th.nn.init.zeros_(self.gru.bias_hh_l0)

        th.nn.init.xavier_uniform_(self.fc.weight)
        th.nn.init.constant_(self.fc.bias, -5.0)

    def forward(self, obs, hidden):
        y, hidden = self.gru(obs[:, None, :], hidden)
        action = self.sigmoid(self.fc(y)).squeeze(1)
        return action, hidden

    def init_hidden(self, batch_size):
        return th.zeros(1, batch_size, self.hidden_dim, device=device)


policy = Policy(
    input_dim=env.observation_space.shape[0],
    hidden_dim=64,
    output_dim=env.n_muscles,
).to(device)

optimizer = th.optim.Adam(policy.parameters(), lr=1e-3)


# -----------------------------
# Rollout
# -----------------------------
def rollout(batch_size):
    obs, info = env.reset(options={"batch_size": batch_size})
    obs = obs.to(device)

    hidden = policy.init_hidden(batch_size)

    positions = []
    actions = []

    for _ in range(n_steps):
        action, hidden = policy(obs, hidden)

        obs, reward, terminated, truncated, info = env.step(action)
        obs = obs.to(device)

        pos = env.get_position(info).to(device)

        positions.append(pos)
        actions.append(action)

    positions = th.stack(positions, dim=1)
    actions = th.stack(actions, dim=1)

    return positions, actions


# -----------------------------
# Training
# -----------------------------
loss_history = []
start_training = time.time()

for batch in range(n_batches):
    batch_start = time.time()

    target = env.make_target(batch_size)
    positions, actions = rollout(batch_size)

    tracking_loss = th.mean((positions - target) ** 2)
    effort_loss = th.mean(actions ** 2)
    smooth_loss = th.mean((actions[:, 1:] - actions[:, :-1]) ** 2)

    loss = tracking_loss + 1e-4 * effort_loss + 1e-3 * smooth_loss

    optimizer.zero_grad()
    loss.backward()
    th.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
    optimizer.step()

    batch_time = time.time() - batch_start
    remaining_batches = n_batches - batch - 1
    eta_min = batch_time * remaining_batches / 60

    loss_history.append(loss.item())

    print(
        f"Batch {batch + 1}/{n_batches} | "
        f"Loss: {loss.item():.6f} | "
        f"Track: {tracking_loss.item():.6f} | "
        f"Effort: {effort_loss.item():.6f} | "
        f"Smooth: {smooth_loss.item():.6f} | "
        f"Time: {batch_time:.2f}s | "
        f"ETA: {eta_min:.1f} min"
    )


# -----------------------------
# Validation plot
# -----------------------------
policy.eval()

with th.no_grad():
    target = env.make_target(1)
    positions, actions = rollout(1)

target_np = target[0].cpu().numpy()
pos_np = positions[0].cpu().numpy()

plt.figure()
plt.plot(target_np[:, 0], target_np[:, 1], "--", label="target")
plt.plot(pos_np[:, 0], pos_np[:, 1], label="arm endpoint")
plt.axis("equal")
plt.xlabel("x [m]")
plt.ylabel("y [m]")
plt.legend()
plt.title("RigidTendonArm26 sinusoidal movement")
plt.show()

plt.figure()
plt.plot(loss_history)
plt.xlabel("batch")
plt.ylabel("loss")
plt.title("Training loss")
plt.show()


# -----------------------------
# Save model
# -----------------------------
th.save(policy.state_dict(), "rigid_tendon_arm26_sinusoidal.pt")
print("Saved model to rigid_tendon_arm26_sinusoidal.pt")
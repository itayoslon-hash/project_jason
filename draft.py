import os
import time
import torch as th
import matplotlib.pyplot as plt

import numpy as np
from arm_env import make_env, Policy, n_steps, x_center, dt, ellipse_start_joint_state

# -----------------------------
# Settings
# -----------------------------
device = th.device("cuda" if th.cuda.is_available() else "cpu")
print("Using device:", device)

batch_size = 32
n_batches = 1000
checkpoint_interval = 100
checkpoint_path = "checkpoint.pt"
model_path = "rigid_tendon_arm26_sinusoidal.pt"

# -----------------------------
# Environment and policy
# -----------------------------
env = make_env()

policy = Policy(
    input_dim=env.observation_space.shape[0] + 2,  # +2 for (target_x, target_y)
    hidden_dim=64,
    output_dim=env.n_muscles,
).to(device)

th.nn.init.xavier_uniform_(policy.gru.weight_ih_l0)
th.nn.init.orthogonal_(policy.gru.weight_hh_l0)
th.nn.init.zeros_(policy.gru.bias_ih_l0)
th.nn.init.zeros_(policy.gru.bias_hh_l0)
th.nn.init.xavier_uniform_(policy.fc.weight)
th.nn.init.constant_(policy.fc.bias, -5.0)

optimizer = th.optim.Adam(policy.parameters(), lr=1e-3)

start_batch = 0
loss_history = []

# Precompute IK for ellipse start once (same t=0 point for all frequencies)
print("Computing IK for ellipse start position...")
_ik_base = ellipse_start_joint_state(1, noise_std=0.0)
print(f"  Joint angles: theta1={_ik_base[0,0]:.3f}, theta2={_ik_base[0,1]:.3f}")

# Resume from checkpoint if available
if os.path.exists(checkpoint_path):
    checkpoint = th.load(checkpoint_path, map_location=device)
    policy.load_state_dict(checkpoint["policy"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    start_batch = checkpoint["batch"] + 1
    loss_history = checkpoint["loss_history"]
    print(f"Resumed from checkpoint at batch {start_batch}")


# -----------------------------
# Rollout
# -----------------------------
def rollout(batch_size, target):
    joint_state = _ik_base.to(device).expand(batch_size, -1).clone()
    obs, info = env.reset(options={"batch_size": batch_size, "joint_state": joint_state})
    obs = obs.to(device)
    hidden = policy.init_hidden(batch_size, device)

    positions = []
    actions = []

    for step in range(n_steps):
        target_now = target[:, step, :]
        obs_aug = th.cat([obs, target_now], dim=-1)
        action, hidden = policy(obs_aug, hidden)
        obs, reward, terminated, truncated, info = env.step(action)
        obs = obs.to(device)
        pos = env.get_cartesian_position(info).to(device)
        positions.append(pos)
        actions.append(action)

    return th.stack(positions, dim=1), th.stack(actions, dim=1)


# -----------------------------
# Training
# -----------------------------
for batch in range(start_batch, n_batches):
    batch_start = time.time()

    progress = batch / n_batches
    max_freq = 0.5 + 2.0 * progress
    freq = np.random.uniform(0.25, max_freq)
    target = env.make_target(batch_size, device, freq=freq)
    positions, actions = rollout(batch_size, target)

    pos_x = positions[:, :, 0]
    target_x = target[:, :, 0]
    pos_y = positions[:, :, 1]
    target_y = target[:, :, 1]

    y_loss = th.mean((pos_y - target_y) ** 2)
    vel_pos_y = (pos_y[:, 1:] - pos_y[:, :-1]) / dt
    vel_target_y = (target_y[:, 1:] - target_y[:, :-1]) / dt
    vel_loss = th.mean((vel_pos_y - vel_target_y) ** 2)
    x_loss = th.mean((pos_x - x_center) ** 2)
    tracking_loss = y_loss + 0.3 * vel_loss + 10.0 * x_loss

    loss = tracking_loss

    optimizer.zero_grad()
    loss.backward()
    th.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
    optimizer.step()

    loss_history.append(loss.item())

    batch_time = time.time() - batch_start
    eta_min = batch_time * (n_batches - batch - 1) / 60

    print(
        f"Batch {batch + 1}/{n_batches} | "
        f"Loss: {loss.item():.6f} | "
        f"freq: {freq:.2f} | "
        f"Y: {y_loss.item():.6f} | "
        f"Vel: {vel_loss.item():.6f} | "
        f"X: {x_loss.item():.6f} | "
        f"Time: {batch_time:.2f}s | "
        f"ETA: {eta_min:.1f} min"
    )

    if (batch + 1) % checkpoint_interval == 0:
        th.save({
            "batch": batch,
            "policy": policy.state_dict(),
            "optimizer": optimizer.state_dict(),
            "loss_history": loss_history,
        }, checkpoint_path)
        print(f"  -> Checkpoint saved at batch {batch + 1}")


# -----------------------------
# Validation plot
# -----------------------------
policy.eval()

with th.no_grad():
    target = env.make_target(1, device)
    positions, actions = rollout(1, target)

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
# Mean Square Jerk (MSJ)
# MSJ = (1/N) * sum((d³x/dt³)²)
# Third derivative approximated via finite differences:
# jerk[i] = (x[i+3] - 3x[i+2] + 3x[i+1] - x[i]) / dt³
# -----------------------------
pos_x = pos_np[:, 0]
jerk = (pos_x[3:] - 3 * pos_x[2:-1] + 3 * pos_x[1:-2] - pos_x[:-3]) / (dt ** 3)
msj = np.mean(jerk ** 2)
print(f"\nMean Square Jerk (MSJ): {msj:.4f} m^2/s^6")

# -----------------------------
# Save final model
# -----------------------------
th.save(policy.state_dict(), model_path)
print(f"Saved model to {model_path}")

if os.path.exists(checkpoint_path):
    os.remove(checkpoint_path)
    print("Removed checkpoint file")

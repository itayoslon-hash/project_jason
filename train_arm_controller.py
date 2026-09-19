import os
import time
import torch as th
import matplotlib.pyplot as plt

import numpy as np
from arm_environment import make_env, Policy, n_steps, dt, midrange_start_joint_state

# -----------------------------
# Settings
# -----------------------------
if not th.cuda.is_available():
    raise RuntimeError(
        "CUDA is required for training. Install a CUDA-enabled PyTorch build "
        "and verify that an NVIDIA GPU is available."
    )

device = th.device("cuda")
print(f"Using device: {device} ({th.cuda.get_device_name(device)})")

# The rollout is a sequential loop of tiny ops, so the GPU is launch-bound
# rather than compute-bound. A larger batch adds parallel work per step
# without adding more sequential steps, which is the main lever to raise
# GPU utilization here.
th.backends.cudnn.benchmark = True
th.backends.cuda.matmul.allow_tf32 = True
th.backends.cudnn.allow_tf32 = True

batch_size = 8192
n_batches = 3000
checkpoint_interval = 100
checkpoint_path = "checkpoint.pt"
model_path = "rigid_tendon_arm26_sinusoidal.pt"
dist_penalty_scale = 20.0  # steepness of the exponential distance penalty
# -----------------------------
# Environment and policy
# -----------------------------
env = make_env(device=device)

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

_start_state = midrange_start_joint_state(1)
print(f"Start joint angles: theta1={_start_state[0,0]:.4f} rad, theta2={_start_state[0,1]:.4f} rad")

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
    joint_state = _start_state.to(device).expand(batch_size, -1).clone()
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

    # exponential penalty grows sharply with distance, unlike squared error
    dist = th.sqrt((positions[:, :, 0] - target[:, :, 0]) ** 2
                 + (positions[:, :, 1] - target[:, :, 1]) ** 2 + 1e-8)
    pos_loss = th.mean(th.exp(dist_penalty_scale * dist) - 1.0)

    loss = pos_loss

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    th.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
    optimizer.step()

    loss_history.append(loss.item())

    batch_time = time.time() - batch_start
    eta_min = batch_time * (n_batches - batch - 1) / 60

    if (batch + 1) % 10 == 0 or batch == start_batch:
        print(
            f"Batch {batch + 1}/{n_batches} | "
            f"Loss: {loss.item():.6f} | "
            f"freq: {freq:.2f} | "
            f"Pos: {pos_loss.item():.6f} | "
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

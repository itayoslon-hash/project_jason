import os
import time
import torch as th
import matplotlib.pyplot as plt

from arm_env import make_env, Policy, n_steps

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
    input_dim=env.observation_space.shape[0],
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
def rollout(batch_size):
    obs, info = env.reset(options={"batch_size": batch_size})
    obs = obs.to(device)
    hidden = policy.init_hidden(batch_size, device)

    positions = []
    actions = []

    for _ in range(n_steps):
        action, hidden = policy(obs, hidden)
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

    target = env.make_target(batch_size, device)
    positions, actions = rollout(batch_size)

    tracking_loss = th.mean((positions - target) ** 2)
    effort_loss = th.mean(actions ** 2)
    smooth_loss = th.mean((actions[:, 1:] - actions[:, :-1]) ** 2)
    loss = tracking_loss + 1e-4 * effort_loss + 1e-3 * smooth_loss

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
        f"Track: {tracking_loss.item():.6f} | "
        f"Effort: {effort_loss.item():.6f} | "
        f"Smooth: {smooth_loss.item():.6f} | "
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
# Save final model
# -----------------------------
th.save(policy.state_dict(), model_path)
print(f"Saved model to {model_path}")

if os.path.exists(checkpoint_path):
    os.remove(checkpoint_path)
    print("Removed checkpoint file")

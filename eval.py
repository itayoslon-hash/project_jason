import numpy as np
import torch as th
import matplotlib.pyplot as plt

from arm_env import make_env, Policy, n_steps, x_center, dt, ellipse_start_joint_state

device = th.device("cuda" if th.cuda.is_available() else "cpu")
checkpoint_path = "checkpoint.pt"

env = make_env()

policy = Policy(
    input_dim=env.observation_space.shape[0] + 2,
    hidden_dim=64,
    output_dim=env.n_muscles,
).to(device)

checkpoint = th.load(checkpoint_path, map_location=device)
policy.load_state_dict(checkpoint["policy"])
batch_reached = checkpoint["batch"] + 1
loss_history = checkpoint["loss_history"]
print(f"Loaded checkpoint at batch {batch_reached}")

policy.eval()

def rollout(target):
    joint_state = ellipse_start_joint_state(1, noise_std=0.0, device=device)
    obs, info = env.reset(options={"batch_size": 1, "joint_state": joint_state})
    obs = obs.to(device)
    hidden = policy.init_hidden(1, device)
    positions, actions = [], []
    for step in range(n_steps):
        target_now = target[:, step, :]
        obs_aug = th.cat([obs, target_now], dim=-1)
        action, hidden = policy(obs_aug, hidden)
        obs, _, _, _, info = env.step(action)
        obs = obs.to(device)
        positions.append(env.get_cartesian_position(info).to(device))
        actions.append(action)
    return th.stack(positions, dim=1), th.stack(actions, dim=1)

with th.no_grad():
    target = env.make_target(1, device)
    positions, actions = rollout(target)

target_np = target[0].cpu().numpy()
pos_np = positions[0].cpu().numpy()

# dX metric
pos_x = pos_np[:, 0]
target_x = target_np[:, 0]
numerator = np.abs(pos_x - target_x)
denominator = np.abs(pos_x + target_x - 2.0 * x_center)
mask = denominator > 0.005
dx_metric = 2.0 * np.mean(numerator[mask] / denominator[mask]) if mask.any() else float("nan")

# MSJ
jerk = (pos_x[3:] - 3 * pos_x[2:-1] + 3 * pos_x[1:-2] - pos_x[:-3]) / (dt ** 3)
msj = np.mean(jerk ** 2)

print(f"dX:  {dx_metric:.4f}")
print(f"MSJ: {msj:.4f} m^2/s^6")

# Tracking plot
plt.figure()
plt.plot(target_np[:, 0], target_np[:, 1], "--", label="target")
plt.plot(pos_np[:, 0], pos_np[:, 1], label="arm endpoint")
plt.axis("equal")
plt.xlabel("x [m]")
plt.ylabel("y [m]")
plt.legend()
plt.title(f"RigidTendonArm26 at batch {batch_reached} | dX={dx_metric:.3f} | MSJ={msj:.1f}")
plt.show()

# Loss curve
plt.figure()
plt.plot(loss_history)
plt.xlabel("batch")
plt.ylabel("loss")
plt.title("Training loss")
plt.show()

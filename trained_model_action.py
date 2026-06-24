import numpy as np
import torch as th
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from arm_env import make_env, Policy, n_steps, dt, midrange_start_joint_state

# -----------------------------
# Settings
# -----------------------------
device = th.device("cuda" if th.cuda.is_available() else "cpu")
model_path = "rigid_tendon_arm26_sinusoidal.pt"

frequencies = np.linspace(0.25, 2.5, 10)

# -----------------------------
# Environment and policy
# -----------------------------
env = make_env()

policy = Policy(
    input_dim=env.observation_space.shape[0] + 2,  # +2 for (target_x, target_y)
    hidden_dim=64,
    output_dim=env.n_muscles,
).to(device)

policy.load_state_dict(th.load(model_path, map_location=device))
policy.eval()


# -----------------------------
# Arm geometry
# -----------------------------
def arm_points_from_angles(theta):
    l1, l2 = 0.309, 0.333
    shoulder = np.array([0.0, 0.0])
    elbow = np.array([l1 * np.cos(theta[0]), l1 * np.sin(theta[0])])
    hand = elbow + np.array([
        l2 * np.cos(theta[0] + theta[1]),
        l2 * np.sin(theta[0] + theta[1])
    ])
    return shoulder, elbow, hand


# -----------------------------
# Rollout for a given frequency
# -----------------------------
def run_freq(freq):
    joint_state = midrange_start_joint_state(1, device=device)

    obs, info = env.reset(options={"batch_size": 1, "joint_state": joint_state})
    obs = obs.to(device)
    hidden = policy.init_hidden(1, device)

    target_tensor = env.make_target(1, device=device, freq=freq)  # [1, n_steps, 2]
    target = target_tensor[0].cpu().numpy()
    positions, joint_angles = [], []

    with th.no_grad():
        for step in range(n_steps):
            target_now = target_tensor[:, step, :]  # [1, 2]
            obs_aug = th.cat([obs, target_now], dim=-1)
            action, hidden = policy(obs_aug, hidden)
            obs, _, _, _, info = env.step(action)
            obs = obs.to(device)
            positions.append(env.get_cartesian_position(info)[0].cpu().numpy())
            joint_angles.append(env.get_joint_angles(info)[0].cpu().numpy())

    return target, np.array(positions), np.array(joint_angles)


# -----------------------------
# Run all frequencies
# -----------------------------
print("Running rollouts...")
all_targets, all_positions, all_joints = [], [], []
for freq in frequencies:
    target, positions, joints = run_freq(freq)
    all_targets.append(target)
    all_positions.append(positions)
    all_joints.append(joints)
print("Done. Starting animation...")


# -----------------------------
# Animation
# -----------------------------
fig, axes = plt.subplots(2, 5, figsize=(18, 7))
axes = axes.flatten()

arm_lines, target_points, endpoint_traces, target_traces = [], [], [], []

for i, ax in enumerate(axes):
    ax.set_aspect("equal")
    ax.set_xlim(-0.5, 0.5)
    ax.set_ylim(0.1, 0.85)
    ax.set_title(f"f = {frequencies[i]:.2f} Hz", fontsize=9)
    ax.set_xlabel("x [m]", fontsize=7)
    ax.set_ylabel("y [m]", fontsize=7)
    ax.tick_params(labelsize=6)

    arm_line, = ax.plot([], [], "ko-", lw=2, markersize=4)
    target_point, = ax.plot([], [], "rx", markersize=8)
    endpoint_trace, = ax.plot([], [], "b-", lw=1, alpha=0.7)
    target_trace, = ax.plot(all_targets[i][:, 0], all_targets[i][:, 1], "r--", lw=1, alpha=0.5)

    arm_lines.append(arm_line)
    target_points.append(target_point)
    endpoint_traces.append(endpoint_trace)
    target_traces.append(target_trace)

plt.suptitle("RigidTendonArm26: tracking at 10 frequencies", fontsize=12)
plt.tight_layout()


def init():
    for i in range(10):
        arm_lines[i].set_data([], [])
        target_points[i].set_data([], [])
        endpoint_traces[i].set_data([], [])
    return arm_lines + target_points + endpoint_traces


def update(frame):
    for i in range(10):
        shoulder, elbow, hand = arm_points_from_angles(all_joints[i][frame])
        arm_lines[i].set_data(
            [shoulder[0], elbow[0], hand[0]],
            [shoulder[1], elbow[1], hand[1]]
        )
        target_points[i].set_data([all_targets[i][frame, 0]], [all_targets[i][frame, 1]])
        endpoint_traces[i].set_data(
            all_positions[i][:frame + 1, 0],
            all_positions[i][:frame + 1, 1]
        )
    return arm_lines + target_points + endpoint_traces


ani = FuncAnimation(fig, update, frames=n_steps, init_func=init, interval=dt * 1000, blit=True)
plt.show()

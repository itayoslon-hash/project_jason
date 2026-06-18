import numpy as np
import torch as th
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from arm_env import make_env, Policy, n_steps, dt

# -----------------------------
# Settings
# -----------------------------
device = th.device("cuda" if th.cuda.is_available() else "cpu")
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


def get_muscle_segments(shoulder, elbow, hand):
    return [
        (shoulder + np.array([0.00,  0.06]), elbow + np.array([0.00,  0.03])),
        (shoulder + np.array([0.00, -0.06]), elbow + np.array([0.00, -0.03])),
        (elbow    + np.array([0.02,  0.04]), hand  + np.array([0.02,  0.02])),
        (elbow    + np.array([-0.02, -0.04]), hand + np.array([-0.02, -0.02])),
        (shoulder + np.array([0.04,  0.05]), hand  + np.array([0.03,  0.02])),
        (shoulder + np.array([0.04, -0.05]), hand  + np.array([0.03, -0.02])),
    ]


def activation_color(a):
    a = float(np.clip(a, 0.0, 1.0))
    return (a, 0.0, 1.0 - a)


# -----------------------------
# Run trained model
# -----------------------------
obs, info = env.reset(options={"batch_size": 1})
obs = obs.to(device)
hidden = policy.init_hidden(1, device)
target = env.make_target(1, device)[0].detach().cpu().numpy()

joint_angles, endpoint_positions, actions_list = [], [], []

with th.no_grad():
    for _ in range(n_steps):
        action, hidden = policy(obs, hidden)
        obs, reward, terminated, truncated, info = env.step(action)
        obs = obs.to(device)
        joint_angles.append(env.get_joint_angles(info)[0].detach().cpu().numpy())
        endpoint_positions.append(env.get_cartesian_position(info)[0].detach().cpu().numpy())
        actions_list.append(action[0].detach().cpu().numpy())

joint_angles = np.array(joint_angles)
endpoint_positions = np.array(endpoint_positions)
actions_list = np.array(actions_list)


# -----------------------------
# Animation
# -----------------------------
fig, ax = plt.subplots(figsize=(7, 7))
ax.set_aspect("equal")
ax.set_xlim(-0.4, 0.75)
ax.set_ylim(-0.4, 0.75)
ax.set_xlabel("x [m]")
ax.set_ylabel("y [m]")
ax.set_title("RigidTendonArm26: sinusoidal movement with muscle activation")

arm_line, = ax.plot([], [], "ko-", lw=4, markersize=8, label="arm skeleton")

muscle_lines = []
for i in range(6):
    line, = ax.plot([], [], lw=5, alpha=0.85, label=f"muscle {i}")
    muscle_lines.append(line)

target_point, = ax.plot([], [], "rx", markersize=12, label="current target")
endpoint_trace, = ax.plot([], [], "k-", lw=1, alpha=0.6, label="endpoint trace")
ax.plot(target[:, 0], target[:, 1], "r--", lw=1, alpha=0.5, label="target path")

time_text = ax.text(0.02, 0.95, "", transform=ax.transAxes)
activation_text = ax.text(0.02, 0.90, "", transform=ax.transAxes)
ax.legend(loc="upper right")


def init():
    arm_line.set_data([], [])
    target_point.set_data([], [])
    endpoint_trace.set_data([], [])
    time_text.set_text("")
    activation_text.set_text("")
    for line in muscle_lines:
        line.set_data([], [])
    return arm_line, target_point, endpoint_trace, time_text, activation_text, *muscle_lines


def update(frame):
    shoulder, elbow, hand = arm_points_from_angles(joint_angles[frame])

    arm_line.set_data([shoulder[0], elbow[0], hand[0]], [shoulder[1], elbow[1], hand[1]])

    activations = actions_list[frame]
    for i, line in enumerate(muscle_lines):
        start, end = get_muscle_segments(shoulder, elbow, hand)[i]
        line.set_data([start[0], end[0]], [start[1], end[1]])
        line.set_color(activation_color(activations[i]))

    target_point.set_data([target[frame, 0]], [target[frame, 1]])
    endpoint_trace.set_data(endpoint_positions[:frame + 1, 0], endpoint_positions[:frame + 1, 1])
    time_text.set_text(f"t = {frame * dt:.2f} s")
    activation_text.set_text("activation: " + ", ".join([f"{a:.2f}" for a in activations]))

    return arm_line, target_point, endpoint_trace, time_text, activation_text, *muscle_lines


ani = FuncAnimation(fig, update, frames=n_steps, init_func=init, interval=dt * 1000, blit=True)
plt.show()

# Optional save
# ani.save("arm_muscle_activation.mp4", fps=int(1 / dt))
# ani.save("arm_muscle_activation.gif", fps=30)

print("Finished animation.")

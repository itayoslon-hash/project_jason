import numpy as np
import torch as th
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import motornet as mn


# -----------------------------
# Settings: must match training
# -----------------------------
device = th.device("cuda" if th.cuda.is_available() else "cpu")

dt = 0.01
duration = 2.0
n_steps = int(duration / dt)

amplitude = 0.08
frequency = 0.5
x_center = 0.25
y_center = 0.25

model_path = "rigid_tendon_arm26_sinusoidal.pt"


# -----------------------------
# Effector and environment
# -----------------------------
muscle = mn.muscle.RigidTendonHillMuscle()
effector = mn.effector.RigidTendonArm26(muscle=muscle)


class SinusoidalArmEnv(mn.environment.Environment):
    def __init__(self, effector, max_ep_duration):
        super().__init__(effector=effector, max_ep_duration=max_ep_duration)

    def make_target(self, batch_size):
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


env = SinusoidalArmEnv(effector=effector, max_ep_duration=duration)


# -----------------------------
# Policy network: must match training
# -----------------------------
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

    def init_hidden(self, batch_size):
        return th.zeros(1, batch_size, self.hidden_dim, device=device)


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
    l1 = 0.309
    l2 = 0.333

    shoulder = np.array([0.0, 0.0])

    elbow = np.array([
        l1 * np.cos(theta[0]),
        l1 * np.sin(theta[0])
    ])

    hand = elbow + np.array([
        l2 * np.cos(theta[0] + theta[1]),
        l2 * np.sin(theta[0] + theta[1])
    ])

    return shoulder, elbow, hand


def get_muscle_segments(shoulder, elbow, hand):
    return [
        # 0 shoulder flexor
        (shoulder + np.array([0.00, 0.06]), elbow + np.array([0.00, 0.03])),

        # 1 shoulder extensor
        (shoulder + np.array([0.00, -0.06]), elbow + np.array([0.00, -0.03])),

        # 2 elbow flexor
        (elbow + np.array([0.02, 0.04]), hand + np.array([0.02, 0.02])),

        # 3 elbow extensor
        (elbow + np.array([-0.02, -0.04]), hand + np.array([-0.02, -0.02])),

        # 4 biarticular flexor
        (shoulder + np.array([0.04, 0.05]), hand + np.array([0.03, 0.02])),

        # 5 biarticular extensor
        (shoulder + np.array([0.04, -0.05]), hand + np.array([0.03, -0.02])),
    ]


def activation_color(a):
    a = float(np.clip(a, 0.0, 1.0))
    return (a, 0.0, 1.0 - a)


# -----------------------------
# Run trained model
# -----------------------------
obs, info = env.reset(options={"batch_size": 1})
obs = obs.to(device)

hidden = policy.init_hidden(1)

target = env.make_target(1)[0].detach().cpu().numpy()

joint_angles = []
endpoint_positions = []
actions_list = []

with th.no_grad():
    for step in range(n_steps):
        action, hidden = policy(obs, hidden)

        obs, reward, terminated, truncated, info = env.step(action)
        obs = obs.to(device)

        theta = env.get_joint_angles(info)[0].detach().cpu().numpy()
        pos = env.get_cartesian_position(info)[0].detach().cpu().numpy()

        joint_angles.append(theta)
        endpoint_positions.append(pos)
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
target_trace, = ax.plot(target[:, 0], target[:, 1], "r--", lw=1, alpha=0.5, label="target path")

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

    return (
        arm_line,
        target_point,
        endpoint_trace,
        time_text,
        activation_text,
        *muscle_lines
    )


def update(frame):
    shoulder, elbow, hand = arm_points_from_angles(joint_angles[frame])

    # skeleton
    arm_x = [shoulder[0], elbow[0], hand[0]]
    arm_y = [shoulder[1], elbow[1], hand[1]]
    arm_line.set_data(arm_x, arm_y)

    # muscles
    muscle_segments = get_muscle_segments(shoulder, elbow, hand)
    activations = actions_list[frame]

    for i, line in enumerate(muscle_lines):
        start, end = muscle_segments[i]

        line.set_data(
            [start[0], end[0]],
            [start[1], end[1]]
        )

        line.set_color(activation_color(activations[i]))

    # target
    target_point.set_data(
        [target[frame, 0]],
        [target[frame, 1]]
    )

    # endpoint trace
    endpoint_trace.set_data(
        endpoint_positions[:frame + 1, 0],
        endpoint_positions[:frame + 1, 1]
    )

    time_text.set_text(f"t = {frame * dt:.2f} s")

    activation_text.set_text(
        "activation: "
        + ", ".join([f"{a:.2f}" for a in activations])
    )

    return (
        arm_line,
        target_point,
        endpoint_trace,
        time_text,
        activation_text,
        *muscle_lines
    )


ani = FuncAnimation(
    fig,
    update,
    frames=n_steps,
    init_func=init,
    interval=dt * 1000,
    blit=True
)

plt.show()


# Optional save
# ani.save("arm_muscle_activation.mp4", fps=int(1 / dt))
# ani.save("arm_muscle_activation.gif", fps=30)

print("Finished animation.")
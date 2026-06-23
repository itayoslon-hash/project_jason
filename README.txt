project_jason
=============

A GRU-based neural network controller for a 2D biomechanical arm model (RigidTendonArm26)
trained to track sinusoidal trajectories at varying frequencies.


SETUP
-----
Install dependencies:
    pip install -r requirements.txt


FILES
-----
arm_env.py
    Shared module. Defines the motornet environment, Policy (GRU network),
    IK utilities, and trajectory generation. All other scripts import from here.

draft.py
    Training script. Runs curriculum learning over sinusoidal trajectories
    (frequency ramps from 0.5 Hz to 2.5 Hz). Saves checkpoints every 100 batches
    and a final model to rigid_tendon_arm26_sinusoidal.pt.

trained_model_action.py
    Visualization script. Loads the trained model and runs it at 10 different
    frequencies, displaying all results as a 2x5 animated grid.

eval.py
    Evaluation script. Loads a checkpoint and prints the dX tracking metric
    and Mean Square Jerk (MSJ), plus a tracking plot and loss curve.


USAGE
-----
1. Train the model:
    python draft.py

   Set n_batches in draft.py to control training length.
   Training resumes automatically from checkpoint.pt if it exists.

2. Visualize the trained model:
    python trained_model_action.py

3. Evaluate a checkpoint mid-training:
    python eval.py


TRAJECTORY
----------
The arm tracks a diagonal rail (135 degrees) centered at (0.3, 0.3) meters,
with amplitude 0.08 m. The arm starts stationary at the center of the rail.
The policy is goal-conditioned: at each timestep it receives the current
observation plus the target (x, y) position for that timestep.


ARM MODEL
---------
RigidTendonArm26 from motornet: a 2-DOF planar arm with 6 Hill-type muscles.
  Segment lengths: upper arm 0.309 m, forearm 0.333 m
  Joint bounds:    shoulder [0, 2.356 rad], elbow [0, 2.705 rad]


LOSS FUNCTION
-------------
loss = mean((pos_x - target_x)^2 + (pos_y - target_y)^2)
     + 0.3 * mean((vel_x - target_vel_x)^2 + (vel_y - target_vel_y)^2)

The velocity term reduces phase lag between the arm and the target.

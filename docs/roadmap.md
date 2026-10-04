# Continuing the project

## First milestone

The current deliverable includes planar rigid-body/motor dynamics, manual flight,
isolated autonomous modes, waypoint PD control, reproducible trajectories,
two-stage JEPA training, physical probing, and learned-model planning.

Keep the manual simulator useful independently of training. Extend the physics
only when a testable question requires it; preserve the state/action/time contract.

## Next experiments

1. Compare model predictions with the zero-residual prior at increasing horizons.
2. Measure waypoint success and settling time over seeds and initial conditions.
3. Hold out particular masses, wind levels, or actuator delays to measure transfer.
4. Increase data excitation gradually, inspect coverage, and add smooth reference paths.
5. Train at the same horizon used by the planner; tune samples against measured latency.
6. Add observation noise and state-estimation delays as separately logged observations.

## Expand to 3D

Introduce a new state/action specification instead of overwriting the planar one:
3D position, velocity, rotation, body angular velocity, and four motor commands.
Add geometry-aware attitude integration, a matching prober, and controller costs.
Version the dataset schema. Existing recording, split, training, mode-isolation,
and experiment patterns remain useful, but the model weights need retraining.

## Connect to RAM Engineering hardware

Resolve the available command interface and state estimator first. Position
commands, body-rate/thrust commands, and motor commands are different control
levels. Calibrate parameters, establish timing/units/frame conventions, and add
an adapter rather than letting hardware protocols enter the simulator.

A 2D demonstration measures the software/learning pipeline. It does not establish
flight performance on the real platform. Real-platform testing should follow a
separate project plan based on the actual hardware and team procedures.

## Vision extension

Add a renderer/camera observation interface, preserve synchronized actions and
state labels for evaluation, and revisit encoder capacity and data coverage.
Pixel input adds appearance and partial-observation problems beyond planar
state-based dynamics. Keep it a separately evaluated extension.


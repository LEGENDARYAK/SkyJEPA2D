# SkyJEPA 2D: Planar Flight Lab

A small Python drone simulator for AI@UNC's RAM Engineering project. Fly manually,
click a waypoint for a conventional controller, collect state/action trajectories,
and train a compact JEPA model for learned-model waypoint control.

This is a **planar adaptation inspired by SkyJEPA**, with its own simulator and
training implementation. It is an educational research starting point, not an
official reproduction, calibrated RAM Engineering drone model, or demonstration
of real-world transfer. It has no camera input or obstacle avoidance.

## Start on your Mac

Use Python 3.10 or newer; Python 3.11 or 3.12 is a good starting choice. Unzip the
project, open Terminal in its directory, and run:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
skyjepa2d play
```

After installation, the Mac launcher `Launch-Simulator.command` starts the
baseline interface. You can also launch it from Terminal with mode/checkpoint flags.

The simulation and display need only NumPy and Pygame. PyTorch is an optional
dependency for training and learned control. Nothing is downloaded during flights.
On Windows, activate with `.venv\Scripts\activate` instead.

To open manual mode immediately:

```bash
skyjepa2d play --mode manual
```

## Controls and flight modes

| Input | Behavior |
| --- | --- |
| `1`, `2`, `3` or mode buttons | Manual, baseline waypoint, learned waypoint; switching starts a fresh episode |
| `Enter` or `P` | Start or pause the simulation |
| `R` | Reset the episode; unlock the waypoint |
| Click in the flight area | Set a waypoint before an autonomous episode starts |
| Left / Right arrows | Manual only: request left/right lean; release to level |
| Up / Down arrows | Manual only: increase/decrease collective thrust around hover trim |
| `Space` | Manual only: cut motor commands; existing motor thrust decays with motor lag |
| `Esc` | Exit |

Manual control is attitude-assisted. The drone levels when horizontal arrows are
released, but releasing the lift keys does not hold altitude or cancel momentum.
This makes thrust direction, gravity, and braking visible and intuitive.

In both autonomous modes, the waypoint locks when you first start an episode,
including while paused. Arrow keys, motor-cut input, and waypoint clicks cannot
alter a running autonomous flight. Reset explicitly starts a new episode. The
headless collector and training code have no input-device dependency.

**Baseline** uses an ordinary cascaded PD controller. **Learned** uses a completed
JEPA checkpoint and sampling-based planning through that model. Learned mode
requires a checkpoint and never silently switches to the baseline.

Arrival means distance below 0.25 m, speed below 0.35 m/s, and absolute pitch below
0.15 rad for one simulated second. Ground/boundary contact ends the episode.

## Try the included research checkpoint

Install the learning dependency in the same environment:

```bash
python -m pip install -e '.[train]'
skyjepa2d play --mode learned --checkpoint examples/demo_model.pt --device cpu
```

The bundled checkpoint is a small starter experiment. The validation report in
`docs/validation.md` records its actual results and limitations. The conventional
controller is useful immediately, regardless of training quality.

On Apple Silicon, `--device mps` requests the Apple GPU; `--device cpu` is also
supported. Training defaults to `auto` (CUDA, then MPS, then CPU). Interactive
learned control defaults to CPU because small recurrent models may benefit from
lower overhead. Benchmark on your actual machine; the package makes no Mac timing
guarantee. If planning runs slowly, reduce `--samples` or `--horizon`; simulation
time stays fixed and may advance slower than wall-clock time.

## Generate data and train your own model

```bash
skyjepa2d collect --out data/local --episodes 100 --seconds 12 --seed 42
skyjepa2d train --data data/local --out runs/local --device auto
skyjepa2d play --mode learned --checkpoint runs/local/model.pt
```

Collection is entirely headless. It changes the waypoint on a predefined schedule
and adds small correlated force perturbations to a conventional tracking controller.
Mass, inertia, motor time constant, gains, drag, and wind vary between episodes.
These are modest educational ranges, editable in `data.randomized_config`.
Use `--no-randomize` for a fixed-parameter experiment.

The trainer first optimizes the shared temporal encoders and latent GRU with
multi-step consistency and SIGReg. It selects a validation checkpoint with a
minimum representation-spread check, freezes it, and then trains the physical
residual prober. Test episodes are evaluated after checkpoint selection.

For a shorter development experiment:

```bash
skyjepa2d train --data examples/dataset --out runs/experiment \
  --latent-epochs 20 --prober-epochs 10 --batch-size 64 --batches-per-epoch 40 --device cpu
```

Each run saves its settings, epoch metrics, latent checkpoint, completed model,
and held-out evaluation. Use a new output directory for each run. The trainer
refuses to overwrite existing experiments and stops if no checkpoint passes its
basic latent-spread check. This check rejects near-constant embeddings; it does
not prove that a representation has learned all useful dynamics.

## Evaluate and visualize

```bash
skyjepa2d benchmark --mode baseline --trials 10 --out reports/baseline.json
skyjepa2d benchmark --mode learned --checkpoint examples/demo_model.pt \
  --trials 10 --out reports/learned.json
skyjepa2d benchmark --mode learned --checkpoint examples/demo_model.pt \
  --zero-residual --trials 10 --out reports/physics-prior-control.json
skyjepa2d evaluate --data examples/dataset --checkpoint examples/demo_model.pt \
  --out reports/prediction.json
python -m pip install -e '.[preview]'
skyjepa2d preview --out previews/my-preview.gif
```

Prediction reports include physical-state error, a zero-residual physics-only
ablation, time-shuffled actions, latent spread, and residual magnitude. They help
distinguish a functioning data/control pipeline from a demonstrably useful learned
model. Waypoint benchmarks use goals fixed before flight and no human input.

## Change the physics

```bash
skyjepa2d play --config configs/default.json
skyjepa2d collect --config configs/default.json --out data/custom
```

| Parameter | Meaning / units |
| --- | --- |
| `mass`, `inertia` | kg; planar moment of inertia in kg m² |
| `arm` | Center-to-motor-group distance, m |
| `max_motor_force` | Maximum force command per group, N |
| `motor_tau` | First-order motor response time, s; zero gives instantaneous response |
| `linear_drag`, `angular_drag` | Translational and rotational damping coefficients |
| `thrust_gains` | Per-group force calibration multipliers |
| `wind_velocity` | Horizontal/vertical air velocity, m/s |
| `physics_dt`, `control_dt` | Integration and command intervals, s |
| `width`, `height`, `collision_radius` | Flight area and simple collision bounds, m |

Defaults integrate physics at 200 Hz, apply commands at 20 Hz, and redraw up to
60 Hz. Rendering does not change integration timesteps. Configurations are
validated, including a sufficiently small timestep for motor dynamics. Motor
commands are clipped; position and pitch are never manipulated by the controls.

## Project structure

| Module | Responsibility |
| --- | --- |
| `config.py`, `physics.py` | Validated SI parameters, RK4 rigid-body and motor dynamics |
| `controllers.py` | Conventional waypoint controller and manual thrust/lean control |
| `session.py` | Episode lifecycle, waypoint lock, mode/input isolation |
| `view.py` | Pygame interface and read-only renderer |
| `data.py`, `dataset.py` | Reproducible trajectories, episode splits, aligned temporal windows |
| `model.py`, `training.py` | Compact JEPA, SIGReg, frozen-model physical probing, evaluation |
| `planning.py` | Learned-model MPPI waypoint controller |
| `experiments.py`, `cli.py` | Benchmarks, animated previews, command-line workflows |

Read `docs/physics.md` for the equations, `docs/learning.md` for training details
and deviations from the paper, `docs/validation.md` for measured results, and
`docs/roadmap.md` for the path toward 3D and the real drone.

## Development

```bash
python -m pip install -e '.[dev,train,preview]'
python -m pytest
ruff check src tests
ruff format --check src tests
```

Core simulation tests remain runnable without PyTorch; learning tests skip if it
is absent. NumPy simulation is deterministic for a fixed seed and configuration.
Floating-point training results can differ across CPU, MPS, and CUDA.

The code is provided under the MIT license. Please cite the research when using
the architecture in academic work; see `docs/learning.md`.

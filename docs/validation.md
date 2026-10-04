# Validation of the initial deliverable

## Environment and scope

Checks used Linux x86_64, Python 3.12.14, NumPy 2.3.5, Pygame 2.6.1, and
PyTorch 2.14.1+cpu. See reports/environment.json for exact installed versions.
Apple M2, MPS, and CUDA execution were not available for direct testing.

The UI was rendered and visually reviewed. Synthetic Pygame events exercised its
event loop under a headless display. Mac installation, keyboard focus, and GPU
performance still need local verification.

## Physics and software checks

There are 26 passing tests. They cover hover, analytic freefall, thrust direction,
motor response, manual movement, four waypoint flights, command limits, nonfinite
input, contacts, timestep convergence, configuration validation/immutability,
copied state access, and episode lifecycle.

They also cover autonomous input isolation, waypoint locking while paused, mode
resets, missing-checkpoint rejection, causal window alignment, episode splits,
repeatable collection, SIGReg versus collapse, action-conditioned prediction,
periodic angle features, prober hover, frozen encoder gradients, read-only
rendering, and GUI event routing. Formatting and lint checks pass.

## Waypoint benchmark

Both controllers used ten identical goals drawn from seed 91 and started at
[4, 4, 0, 0, 0, 0] in the default environment. Goals were fixed before flight.
No human inputs were used.

| Controller | Arrived | Crashed | Other outcomes |
| --- | --- | --- | --- |
| Conventional PD baseline | 10 / 10 | 0 | None |
| JEPA + trained prober + MPPI | 8 / 10 | 0 | Two time limits, 0.345 m and 0.509 m from target |

MPPI used 128 candidates, 25 prediction steps, two iterations, and generator seed
13. Arrival required distance <0.25 m, speed <0.35 m/s, and pitch magnitude <0.15
rad for one simulated second. Episodes end after 30 simulated seconds.

The preview's learned flight reaches [14, 6] after about 6.25 simulated seconds
with approximately 0.23 m final position error. The preview comes from the actual
simulator; its manual section replays scripted manual inputs.

The full outcomes and compute times are in reports/baseline-benchmark.json and
reports/learned-benchmark.json. Python learned planning exceeded the nominal
50 ms decision interval in this environment, so interactive learned flights can
run slower than real time. Manual and baseline modes are much lighter. These
measurements are not Apple M2 performance estimates.

## Held-out prediction

The checkpoint used 64 training episodes, validation-based selection, and eight
independent test episodes supplying 344 windows. Errors are averaged over the
whole 20-step, one-second rollout, rather than only its final step.

| Metric | Result |
| --- | --- |
| Position-vector RMSE | 0.337 m |
| Velocity-vector RMSE | 1.282 m/s |
| Pitch RMSE | 24.227 degrees |
| Normalized physical-state loss | 0.2490 |
| Zero-residual physics-only loss | 0.2153 |
| Time-shuffled-action loss | 0.4607 |

The learned prober's physical-state loss is about 15.6% worse than its nominal
physics prior. The correction is nonzero and the selected representation varies
across examples, but this experiment does **not** establish a predictive advantage
from JEPA. The strong physical prior means navigation alone cannot establish it.

The action-shuffle diagnostic includes the known physical action pathway and
therefore does not isolate learned causal action understanding. Latent spread
is averaged across minibatches and can vary with evaluation batch size.

This experiment verifies recording, splitting, two-stage training, checkpoint
loading, prediction, and learned-mode control end to end. Improving representation
quality and residual prediction is a next research milestone. It makes no
real-drone or sim-to-real claim.

## Re-run checks

```bash
python -m pytest -q
ruff check src tests
ruff format --check src tests
skyjepa2d benchmark --mode baseline --trials 10 --seed 91 --out reports/new-baseline.json
skyjepa2d benchmark --mode learned --checkpoint examples/demo_model.pt \
  --trials 10 --seed 91 --samples 128 --horizon 25 --out reports/new-learned.json
skyjepa2d evaluate --data examples/dataset --checkpoint examples/demo_model.pt \
  --batch-size 64 --out reports/new-prediction.json
```

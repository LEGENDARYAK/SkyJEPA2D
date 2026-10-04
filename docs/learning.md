# Learning design and research grounding

## Sources

- Rao et al., *SkyJEPA: Learning Long-Horizon World Models for Zero-Shot Sim-to-Real
  Control of Quadrotors*, arXiv:2606.23444v2, June 2026.
  https://arxiv.org/abs/2606.23444
- Assran et al., *Self-Supervised Learning from Images with a Joint-Embedding
  Predictive Architecture*, 2023. https://arxiv.org/abs/2301.08243
- LeCun, *A Path Towards Autonomous Machine Intelligence*, 2022.
  https://openreview.net/forum?id=BZ5a1r-kVsf
- Balestriero and LeCun, *LeJEPA: Provable and Scalable Self-Supervised Learning
  Without the Heuristics*, 2025. https://arxiv.org/abs/2511.08544
- PyTorch Apple GPU backend: https://docs.pytorch.org/docs/stable/notes/mps.html
- Pygame input events: https://www.pygame.org/docs/ref/event.html

SkyJEPA uses estimated physical-state histories and motor actions. It learns
latent dynamics with multi-step predictive consistency and SIGReg, then freezes
the latent model and fits a physics-inspired physical-state prober. Its controller
evaluates candidate actions through the model. Camera-based operation is a future
extension in the paper. The project retains this state-first structure.

## What changes in this implementation

| Aspect | This project |
| --- | --- |
| Geometry and actions | Six planar observed variables; two motor-group force commands |
| Physical-state encoder input | Seven features: normalized position/velocity/rate and sin/cos of pitch |
| State encoder | Causal temporal convolutions with channels 8, 8, 16; projection to 24 dimensions |
| Action encoder | Causal temporal convolutions with channels 4, 4, 8; eight-dimensional output |
| Predictor | GRU cell, 24-dimensional hidden state; latent rollout remains autoregressive |
| Context and rollout defaults | Exactly 10 context samples and 20 future steps at 20 Hz |
| Data | Scheduled waypoint PD tracking with correlated excitation and modest randomization |
| Prober | Nominal planar dynamics and motor lag plus bounded learned acceleration corrections |
| Control | Python/PyTorch MPPI, fewer samples and no TensorRT/C++ dependency |

The implementation uses AdamW at fixed learning rate and smaller batches rather
than reproducing the paper's full optimization schedule or data volume. Random
waypoint references replace its Gaussian-process references. These choices are
documented adaptations, and the results cannot be presented as a SkyJEPA replication.

## Dataset contract

For N transitions, each NPZ episode contains:

| Array | Shape | Meaning |
| --- | --- | --- |
| `states` | (N+1, 6) | True physical observations, before and after each command |
| `actions` | (N, 2) | Clipped requested left/right forces in N |
| `motor_forces` | (N+1, 2) | Actual applied motor forces; diagnostics only, never a model input |
| `times` | (N+1,) | Actual integration times in seconds |
| `goals` | (N, 2) | Collector reference; not a dynamics-model input |

The manifest records schema version, parameter values, episode seed, split,
termination, and nominal configuration. State x[t] and command a[t] produce
x[t+1]. Files are loaded without pickle. Whole episodes, each with its own
randomized domain, receive an 80/10/10 train/validation/test assignment.

Context H means H state samples ending at x[t]. The first action window ends at
a[t]. For predicted step k, the action window includes the candidate command
that will produce x[t+k+1]. A target encoder sees the true state history ending
at x[t+k+1] only when computing the training loss. No true future states enter
the latent rollout. Windows never cross episode boundaries; a terminal collision
transition is excluded because it may have a shorter-than-normal sample interval.

## Stage 1: latent dynamics

The shared state encoder maps a context to s[t]. The action encoder maps the
corresponding history to z[t]. The GRU repeatedly computes the next latent
from the prior latent and encoded action history. Encoded true future contexts
supply targets, with gradients through the shared encoder on both paths.

The loss is mean squared multi-step latent error plus lambda times SIGReg.
No EMA teacher, reconstruction decoder, or stop-gradient target is used in this
stage. SIGReg projects latents onto 32 random unit directions, evaluates the
empirical characteristic function across the minibatch at each rollout time,
and compares it to the standard Gaussian characteristic function.

This implementation uses 17 quadrature points from 0 to 3, doubles the integral
by symmetry, and weights with exp(-t²/2). It includes the conventional batch-size
factor in the Epps-Pulley statistic. Consequently the coefficient 0.02 is specific
to this normalization. Real-valued sine/cosine operations avoid complex tensors.

Validation uses fixed projection directions. Checkpoints are eligible only when
mean latent standard deviation across examples reaches 0.1. This prevents an
early near-constant checkpoint from winning on prediction consistency alone.
The threshold is a basic diagnostic, not an isotropy or noncollapse proof.

## Stage 2: physical probing

Freeze all state encoder, action encoder, and predictor weights. The prober reads
predicted latents and outputs two translational acceleration residuals plus a
two-component angular coefficient K. The latter produces a residual pitch
acceleration K dot a. Tanh bounds the corrections.

The nominal integrator supplies gravity, thrust direction, angular mechanics,
drag, and motor lag. Actual hidden motor forces are not exposed: the prober
estimates initial forces from the executed command history, initialized at
nominal hover trim. It advances physical state using candidate commands and
the latent-conditioned corrections. Predicted physical states are never fed back
into the latent state encoder during the rollout.

This differs from the paper's particular prober, which learns its angular
acceleration mapping. Retaining nominal angular dynamics makes a transparent
starter prior and enables a zero-residual comparison. Evaluation must show
whether the learned correction actually improves that prior.

## Waypoint planning

The learned controller maintains observed-state and executed-action histories.
At each decision, MPPI samples smooth collective/differential force sequences,
rolls them out through the latent model and physical prober, scores distance,
velocity, attitude, effort, and boundary penalties, and computes a cost-weighted
plan. It applies only the first command and shifts the remainder for the next
decision. The waypoint changes the cost, not the learned dynamics inputs.

The default lookahead is 25 steps. The trained prediction horizon is 20 steps;
the planner therefore extrapolates five steps beyond the training rollout. Use
`--horizon 20` to avoid this extrapolation or train with a longer horizon. A longer
planning horizon can help braking and requires independent validation.

There is no baseline-controller fallback and no direct access to plant internals.
The nominal equations inside the trained prober provide a strong prior, so
successful control alone is not evidence that the JEPA representation is useful.

## Evaluation and reproducibility

Runs record seed, versions, nominal configuration, selected validation metrics,
per-epoch logs, and final held-out metrics. NumPy collection is deterministic.
PyTorch seeds are set, but cross-backend or cross-version bitwise equality is not
guaranteed. CPU, MPS, and CUDA are selected explicitly or by availability.

Reports include position/velocity/pitch rollout error, latent spread, residual
magnitude, zero-residual physical prediction, and time-shuffled candidate actions.
The latter checks the full prediction pipeline's action sensitivity; because
actions also enter the physical prior, it is not an isolated test of the latent
predictor's learned action understanding. The unit tests separately check that
the latent predictor responds to changed actions.

The loader does not concatenate episodes or use hidden actuator state, domain
metadata, waypoints, or future targets as predictor input. Future work should add
more rigorous learned-latent action interventions, long-horizon error curves,
domain holdouts, and learned-controller comparisons with zero residuals.


# Included development experiment

The dataset contains 80 randomized planar flights of 12 simulated seconds each:
64 training, 8 validation, and 8 test episodes. Collection used seed 42.

The completed checkpoint is demo_model.pt. The training directory records
settings, epoch logs, the selected latent checkpoint, and held-out evaluation.

Commands used:

```bash
skyjepa2d collect --out data/demo --episodes 80 --seconds 12 --seed 42
skyjepa2d train --data data/demo --out runs/validated \
  --latent-epochs 25 --prober-epochs 20 --batch-size 64 \
  --batches-per-epoch 40 --device cpu --threads 2
```

This is an experimental checkpoint. Its learned corrections do not improve the
physics prior's held-out prediction loss in this run. Read ../docs/validation.md
before interpreting waypoint success as evidence of a learning advantage.


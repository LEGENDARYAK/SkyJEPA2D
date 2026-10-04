"""Two-stage training, validation-selected checkpoints, and held-out evaluation."""

from __future__ import annotations

import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from .config import SimConfig
from .dataset import TrajectoryWindows
from .model import LatentDynamics, PhysicalProber, choose_device, physical_loss, sigreg

CHECKPOINT_VERSION = 1


def move_batch(batch, device):
    return {key: value.to(device) for key, value in batch.items()}


def atomic_save(payload, path):
    path = Path(path)
    temporary = path.with_suffix(".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def load_checkpoint(path, device="cpu", require_prober=True):
    device = choose_device(device)
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    if checkpoint.get("checkpoint_version") != CHECKPOINT_VERSION:
        raise ValueError("Unsupported checkpoint version")
    if require_prober and checkpoint.get("stage") != "prober":
        raise ValueError("Complete both training stages before using learned control")
    config = SimConfig.from_dict(checkpoint["config"])
    model = LatentDynamics(config, checkpoint["latent_dim"]).to(device)
    model.load_state_dict(checkpoint["model"])
    prober = PhysicalProber(config, checkpoint["latent_dim"]).to(device)
    if "prober" in checkpoint:
        prober.load_state_dict(checkpoint["prober"])
    model.eval()
    prober.eval()
    return model, prober, checkpoint, device


@torch.no_grad()
def validate_latent(model, loader, device, sig_weight, max_batches=None):
    model.eval()
    count = 0
    total, prediction, std = 0.0, 0.0, 0.0
    for i, batch in enumerate(loader):
        if max_batches and i >= max_batches:
            break
        batch = move_batch(batch, device)
        latent = model(batch["context"], batch["action_windows"])
        target = model.targets(batch["target_contexts"])
        loss_pred = F.mse_loss(latent, target)
        loss = loss_pred + sig_weight * sigreg(latent, seed=0)
        b = len(latent)
        total += b * loss.item()
        prediction += b * loss_pred.item()
        std += b * latent.std(dim=0).mean().item()
        count += b
    return {"loss": total / count, "prediction_loss": prediction / count, "latent_std": std / count}


@torch.no_grad()
def evaluate_physical(model, prober, loader, device, max_batches=None):
    """Physical accuracy plus zero-residual and shuffled-action diagnostics.

    These metrics expose whether the nominal prior or learned residual supplies
    performance; successful navigation alone does not validate a JEPA model.
    """
    model.eval()
    prober.eval()
    count = 0
    sums = {
        k: 0.0
        for k in (
            "loss",
            "position_sq",
            "velocity_sq",
            "pitch_sq",
            "prior_loss",
            "shuffled_action_loss",
            "residual_acc_sq",
            "latent_std",
        )
    }
    for i, batch in enumerate(loader):
        if max_batches and i >= max_batches:
            break
        batch = move_batch(batch, device)
        latent = model(batch["context"], batch["action_windows"])
        initial, actions = batch["context"][:, -1], batch["future_actions"]
        predicted = prober(latent, initial, actions, batch["past_actions"])
        prior = prober(latent, initial, actions, batch["past_actions"], zero_residual=True)
        # Reverse the time order of the candidate commands. Perturb the predictor
        # and physical prober consistently; targets remain unchanged.
        wrong_actions = actions.flip(1)
        wrong_latent = model.from_actions(batch["context"], batch["past_actions"], wrong_actions)
        wrong = prober(wrong_latent, initial, wrong_actions, batch["past_actions"])
        target = batch["future_states"]
        delta = predicted - target
        angle = torch.atan2(delta[..., 4].sin(), delta[..., 4].cos())
        b = len(initial)
        sums["loss"] += b * physical_loss(predicted, target).item()
        sums["position_sq"] += b * delta[..., :2].square().sum(-1).mean().item()
        sums["velocity_sq"] += b * delta[..., 2:4].square().sum(-1).mean().item()
        sums["pitch_sq"] += b * angle.square().mean().item()
        sums["prior_loss"] += b * physical_loss(prior, target).item()
        sums["shuffled_action_loss"] += b * physical_loss(wrong, target).item()
        sums["residual_acc_sq"] += (
            b * (3 * torch.tanh(prober.network(latent))[..., :2]).square().mean().item()
        )
        sums["latent_std"] += b * latent.std(dim=0).mean().item()
        count += b
    means = {k: v / count for k, v in sums.items()}
    return {
        "physical_loss": means["loss"],
        "position_rmse_m": means["position_sq"] ** 0.5,
        "velocity_rmse_m_s": means["velocity_sq"] ** 0.5,
        "pitch_rmse_deg": np.degrees(means["pitch_sq"] ** 0.5).item(),
        "physics_only_loss": means["prior_loss"],
        "time_shuffled_actions_loss": means["shuffled_action_loss"],
        "residual_acc_rms_m_s2": means["residual_acc_sq"] ** 0.5,
        "latent_std": means["latent_std"],
        "windows": count,
    }


def train(
    dataset,
    output,
    history=10,
    horizon=20,
    latent_epochs=15,
    prober_epochs=15,
    batch_size=128,
    batches_per_epoch=0,
    device="auto",
    seed=7,
    learning_rate=0.001,
    sig_weight=0.02,
    threads=2,
):
    if min(history, horizon, latent_epochs, prober_epochs, threads) < 1 or batch_size < 2:
        raise ValueError("Positive training sizes and batch_size >= 2 are required")
    if (
        not np.isfinite(learning_rate)
        or learning_rate <= 0
        or not np.isfinite(sig_weight)
        or sig_weight <= 0
        or batches_per_epoch < 0
    ):
        raise ValueError("learning-rate and sig-weight must be positive; batches-per-epoch >= 0")
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Run output must be empty; choose a new directory")
    output.mkdir(parents=True, exist_ok=True)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(threads)
    device = choose_device(device)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    train_set = TrajectoryWindows(dataset, "train", history, horizon)
    val_set = TrajectoryWindows(dataset, "validation", history, horizon, stride=5)
    config = SimConfig.from_dict(train_set.manifest["config"])
    if len(train_set) < batch_size or len(val_set) < 2:
        raise ValueError("Dataset is too small for this batch size and window length")
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        generator=generator,
        num_workers=0,
    )
    # Avoid a final singleton, which cannot support batch-distribution regularization.
    val_batch = min(batch_size, len(val_set))
    val_loader = DataLoader(
        val_set, batch_size=val_batch, drop_last=len(val_set) % val_batch == 1, num_workers=0
    )
    model, prober = LatentDynamics(config).to(device), PhysicalProber(config).to(device)
    metadata = {
        "checkpoint_version": CHECKPOINT_VERSION,
        "config": config.to_dict(),
        "history": history,
        "horizon": horizon,
        "latent_dim": model.latent_dim,
        "control_dt": config.world.control_dt,
        "seed": seed,
        "training": {
            "latent_epochs": latent_epochs,
            "prober_epochs": prober_epochs,
            "batch_size": batch_size,
            "batches_per_epoch": batches_per_epoch,
            "learning_rate": learning_rate,
            "sig_weight": sig_weight,
        },
        "torch_version": str(torch.__version__),
        "device": str(device),
    }
    (output / "run_config.json").write_text(json.dumps(metadata, indent=2) + "\n")
    log_path = output / "metrics.jsonl"
    started = time.perf_counter()
    best = float("inf")
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    print(
        f"Device {device}; latent model {sum(p.numel() for p in model.parameters())} parameters; "
        f"{len(train_set)} training windows",
        flush=True,
    )
    for epoch in range(latent_epochs):
        model.train()
        losses = []
        for i, batch in enumerate(train_loader):
            if batches_per_epoch and i >= batches_per_epoch:
                break
            batch = move_batch(batch, device)
            predicted = model(batch["context"], batch["action_windows"])
            target = model.targets(batch["target_contexts"])
            loss = F.mse_loss(predicted, target) + sig_weight * sigreg(predicted)
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite latent training loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()
            losses.append(loss.item())
        validation = validate_latent(model, val_loader, device, sig_weight)
        record = {
            "stage": "latent",
            "epoch": epoch + 1,
            "train_loss": float(np.mean(losses)),
            "validation": validation,
        }
        with log_path.open("a") as log:
            log.write(json.dumps(record) + "\n")
        # Early near-constant representations can score deceptively well before
        # the anti-collapse term has created a useful spread across examples.
        if validation["latent_std"] >= 0.1 and validation["loss"] < best:
            best = validation["loss"]
            atomic_save(
                {
                    **metadata,
                    "stage": "latent",
                    "model": model.state_dict(),
                    "validation": validation,
                },
                output / "latent.pt",
            )
        print(
            f"Latent {epoch + 1}/{latent_epochs}: val {validation['loss']:.4f}, "
            f"latent std {validation['latent_std']:.3f}",
            flush=True,
        )
    if not (output / "latent.pt").exists():
        raise RuntimeError(
            "No latent checkpoint passed the minimum spread check (std >= 0.1). "
            "Increase --latent-epochs or --batches-per-epoch and inspect metrics.jsonl."
        )
    selected = torch.load(output / "latent.pt", map_location=device, weights_only=True)
    model.load_state_dict(selected["model"])
    model.eval()
    model.requires_grad_(False)
    optimizer = torch.optim.AdamW(prober.parameters(), lr=learning_rate, weight_decay=1e-5)
    best = float("inf")
    for epoch in range(prober_epochs):
        prober.train()
        losses = []
        for i, batch in enumerate(train_loader):
            if batches_per_epoch and i >= batches_per_epoch:
                break
            batch = move_batch(batch, device)
            with torch.no_grad():
                latent = model(batch["context"], batch["action_windows"])
            predicted = prober(
                latent.detach(),
                batch["context"][:, -1],
                batch["future_actions"],
                batch["past_actions"],
            )
            loss = physical_loss(predicted, batch["future_states"])
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite prober training loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(prober.parameters(), 0.5)
            optimizer.step()
            losses.append(loss.item())
        validation = evaluate_physical(model, prober, val_loader, device, max_batches=4)
        record = {
            "stage": "prober",
            "epoch": epoch + 1,
            "train_loss": float(np.mean(losses)),
            "validation": validation,
        }
        with log_path.open("a") as log:
            log.write(json.dumps(record) + "\n")
        if validation["physical_loss"] < best:
            best = validation["physical_loss"]
            atomic_save(
                {
                    **metadata,
                    "stage": "prober",
                    "model": model.state_dict(),
                    "prober": prober.state_dict(),
                    "validation": validation,
                },
                output / "model.pt",
            )
        print(
            f"Prober {epoch + 1}/{prober_epochs}: val {validation['physical_loss']:.5f}, "
            f"position RMSE {validation['position_rmse_m']:.3f} m",
            flush=True,
        )
    model, prober, _, device = load_checkpoint(output / "model.pt", str(device))
    test_set = TrajectoryWindows(dataset, "test", history, horizon, stride=5)
    test_loader = DataLoader(
        test_set,
        batch_size=min(batch_size, len(test_set)),
        drop_last=len(test_set) % min(batch_size, len(test_set)) == 1,
    )
    evaluation = evaluate_physical(model, prober, test_loader, device)
    evaluation["elapsed_seconds"] = time.perf_counter() - started
    evaluation["split"] = "test"
    (output / "evaluation.json").write_text(json.dumps(evaluation, indent=2) + "\n")
    print(json.dumps(evaluation, indent=2), flush=True)
    return evaluation

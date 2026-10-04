"""Command-line entry points; learning dependencies are loaded only when needed."""

from __future__ import annotations

import argparse
import json

from .config import load_config
from .session import Mode


def parser():
    root = argparse.ArgumentParser(description="SkyJEPA 2D - planar drone learning laboratory")
    commands = root.add_subparsers(dest="command", required=True)
    play = commands.add_parser("play", help="Open manual, baseline, or learned flight mode")
    play.add_argument("--mode", choices=[m.value for m in Mode], default="baseline")
    play.add_argument("--config")
    play.add_argument("--checkpoint")
    play.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="cpu")
    play.add_argument("--samples", type=int, default=128)
    play.add_argument("--horizon", type=int, default=25)
    collect = commands.add_parser("collect", help="Generate a headless state/action dataset")
    collect.add_argument("--out", required=True)
    collect.add_argument("--config")
    collect.add_argument("--episodes", type=int, default=100)
    collect.add_argument("--seconds", type=float, default=12)
    collect.add_argument("--seed", type=int, default=42)
    collect.add_argument("--action-noise", type=float, default=0.1)
    collect.add_argument("--no-randomize", action="store_true")
    train = commands.add_parser("train", help="Train latent model, then frozen-model prober")
    train.add_argument("--data", required=True)
    train.add_argument("--out", required=True)
    for name, default in [
        ("history", 10),
        ("horizon", 20),
        ("latent-epochs", 15),
        ("prober-epochs", 15),
        ("batch-size", 128),
        ("batches-per-epoch", 0),
        ("threads", 2),
        ("seed", 7),
    ]:
        train.add_argument("--" + name, type=int, default=default)
    train.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    train.add_argument("--learning-rate", type=float, default=0.001)
    train.add_argument("--sig-weight", type=float, default=0.02)
    evaluate = commands.add_parser("evaluate", help="Evaluate a checkpoint on held-out episodes")
    evaluate.add_argument("--data", required=True)
    evaluate.add_argument("--checkpoint", required=True)
    evaluate.add_argument("--out", required=True)
    evaluate.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="cpu")
    evaluate.add_argument("--batch-size", type=int, default=128)
    evaluate.add_argument("--threads", type=int, default=2)
    bench = commands.add_parser("benchmark", help="Run waypoint flights without human input")
    bench.add_argument("--mode", choices=["baseline", "learned"], default="baseline")
    bench.add_argument("--config")
    bench.add_argument("--checkpoint")
    bench.add_argument("--trials", type=int, default=5)
    bench.add_argument("--seed", type=int, default=91)
    bench.add_argument("--samples", type=int, default=128)
    bench.add_argument("--horizon", type=int, default=25)
    bench.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="cpu")
    bench.add_argument("--out", required=True)
    bench.add_argument(
        "--zero-residual",
        action="store_true",
        help="Ablate the learned prober corrections for a physics-prior comparison",
    )
    preview = commands.add_parser("preview", help="Export an animated simulator preview")
    preview.add_argument("--out", required=True)
    preview.add_argument("--config")
    preview.add_argument("--checkpoint")
    return root


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        config = load_config(getattr(args, "config", None))
        checkpoint = getattr(args, "checkpoint", None)
        if checkpoint and not getattr(args, "config", None):
            from .training import load_checkpoint

            model, _, _, _ = load_checkpoint(checkpoint)
            config = model.config
        if args.command in ("play", "benchmark", "preview"):
            learned = None
            if checkpoint:
                import torch

                from .planning import LearnedController

                torch.set_num_threads(2)
                learned = LearnedController(
                    checkpoint,
                    config,
                    getattr(args, "device", "cpu"),
                    getattr(args, "samples", 128),
                    getattr(args, "horizon", 25),
                    zero_residual=getattr(args, "zero_residual", False),
                )
            if getattr(args, "mode", "baseline") == "learned" and learned is None:
                raise ValueError("Learned mode requires --checkpoint from both training stages")
            if args.command == "play":
                from .view import run_app

                run_app(config, Mode(args.mode), learned)
            elif args.command == "benchmark":
                from .experiments import benchmark, write_report

                report = benchmark(config, Mode(args.mode), learned, args.trials, args.seed)
                write_report(report, args.out)
                print(json.dumps(report, indent=2))
            else:
                from .experiments import export_preview

                print(json.dumps(export_preview(args.out, config, learned), indent=2))
        elif args.command == "collect":
            from .data import collect_dataset

            manifest = collect_dataset(
                args.out,
                config,
                args.episodes,
                args.seconds,
                args.seed,
                not args.no_randomize,
                args.action_noise,
            )
            print(f"Saved {len(manifest['episodes'])} episodes to {args.out}")
        elif args.command == "train":
            from .training import train

            train(
                args.data,
                args.out,
                args.history,
                args.horizon,
                args.latent_epochs,
                args.prober_epochs,
                args.batch_size,
                args.batches_per_epoch,
                args.device,
                args.seed,
                args.learning_rate,
                args.sig_weight,
                args.threads,
            )
        elif args.command == "evaluate":
            import torch
            from torch.utils.data import DataLoader

            from .dataset import TrajectoryWindows
            from .experiments import write_report
            from .training import evaluate_physical, load_checkpoint

            if args.threads < 1 or args.batch_size < 2:
                raise ValueError("Evaluation requires threads >= 1 and batch-size >= 2")
            torch.set_num_threads(args.threads)

            model, prober, metadata, device = load_checkpoint(args.checkpoint, args.device)
            windows = TrajectoryWindows(
                args.data, "test", metadata["history"], metadata["horizon"], stride=5
            )
            if windows.manifest["config"] != metadata["config"]:
                raise ValueError("Evaluation data and checkpoint have different nominal configs")
            report = evaluate_physical(model, prober, DataLoader(windows, args.batch_size), device)
            write_report(report, args.out)
            print(json.dumps(report, indent=2))
    except ImportError as exc:
        raise SystemExit(
            f"Missing optional dependency. Install with pip install -e '.[train,preview]'\n{exc}"
        ) from exc
    except (ValueError, FileExistsError, FileNotFoundError) as exc:
        raise SystemExit(str(exc)) from exc

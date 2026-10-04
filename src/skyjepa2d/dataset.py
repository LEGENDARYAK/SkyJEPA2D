"""Episode-safe windows for learning; imports PyTorch only on the training path."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .data import read_manifest


class TrajectoryWindows(Dataset):
    """Return causal context, future action windows, and encoded-target contexts.

    H means exactly H state samples (not H+1). At time t, the context ends
    at x[t]. Action window k ends at a[t+k]; that command produces x[t+k+1].
    Future ground truth is used only as the training target, never as a predictor input.
    """

    def __init__(self, directory, split="train", history=10, horizon=20, stride=2):
        if min(history, horizon, stride) < 1:
            raise ValueError("history, horizon and stride must be positive")
        self.history, self.horizon = history, horizon
        self.manifest = read_manifest(directory)
        self.episodes, self.index = [], []
        for metadata in self.manifest["episodes"]:
            if metadata["split"] != split:
                continue
            with np.load(Path(directory) / metadata["file"], allow_pickle=False) as episode:
                states, actions = episode["states"].copy(), episode["actions"].copy()
                times = episode["times"]
                if len(states) != len(actions) + 1:
                    raise ValueError("Dataset contains misaligned states and actions")
                if states.shape[1:] != (6,) or actions.shape[1:] != (2,):
                    raise ValueError("Unexpected state or action shape")
                if not np.isfinite(states).all() or not np.isfinite(actions).all():
                    raise ValueError("Dataset contains nonfinite values")
                # A collision may interrupt a control interval: exclude that transition.
                valid = len(actions) - int(metadata["terminal"])
                if not np.allclose(
                    np.diff(times[: valid + 1]), self.manifest["control_dt"], atol=1e-7
                ):
                    raise ValueError("Learning requires uniformly sampled control intervals")
            ep_id = len(self.episodes)
            self.episodes.append((states, actions))
            self.index.extend((ep_id, t) for t in range(history - 1, valid - horizon + 1, stride))
        if not self.index:
            raise ValueError(f"No valid {split} windows; collect more or longer episodes")

    def __len__(self):
        return len(self.index)

    def __getitem__(self, index):
        episode, t = self.index[index]
        states, actions = self.episodes[episode]
        h, k = self.history, self.horizon
        past = actions[t - h + 1 : t]
        return {
            "context": torch.from_numpy(states[t - h + 1 : t + 1]),
            "past_actions": torch.from_numpy(past),
            "future_actions": torch.from_numpy(actions[t : t + k]),
            "action_windows": torch.from_numpy(
                np.stack([actions[t + j - h + 1 : t + j + 1] for j in range(k)])
            ),
            "target_contexts": torch.from_numpy(
                np.stack([states[t + j - h + 2 : t + j + 2] for j in range(k)])
            ),
            "future_states": torch.from_numpy(states[t + 1 : t + k + 1]),
        }

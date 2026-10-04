# Training is optional; skip before importing modules that depend on PyTorch.
# ruff: noqa: E402
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from skyjepa2d.config import SimConfig
from skyjepa2d.data import collect_dataset
from skyjepa2d.dataset import TrajectoryWindows
from skyjepa2d.model import LatentDynamics, PhysicalProber, sigreg


def test_episode_splits_and_window_alignment(tmp_path):
    first = collect_dataset(tmp_path / "a", SimConfig(), episodes=10, seconds=2, seed=3)
    second = collect_dataset(tmp_path / "b", SimConfig(), episodes=10, seconds=2, seed=3)
    assert first == second
    ids = {
        s: {e["file"] for e in first["episodes"] if e["split"] == s}
        for s in ["train", "validation", "test"]
    }
    assert not ids["train"] & ids["validation"]
    assert not ids["test"] & ids["train"]
    dataset = TrajectoryWindows(tmp_path / "a", history=3, horizon=4, stride=1)
    item = dataset[0]
    states, actions = dataset.episodes[dataset.index[0][0]]
    t = dataset.index[0][1]
    np.testing.assert_array_equal(item["context"][-1], states[t])
    np.testing.assert_array_equal(item["action_windows"][0, -1], actions[t])
    np.testing.assert_array_equal(item["target_contexts"][0, -1], states[t + 1])
    np.testing.assert_array_equal(item["future_states"][0], states[t + 1])
    with np.load(tmp_path / "a" / first["episodes"][0]["file"]) as a:
        with np.load(tmp_path / "b" / second["episodes"][0]["file"]) as b:
            np.testing.assert_array_equal(a["states"], b["states"])


def test_sigreg_penalizes_collapse():
    torch.manual_seed(0)
    normal = sigreg(torch.randn(256, 3, 24), seed=42)
    collapsed = sigreg(torch.zeros(256, 3, 24), seed=42)
    assert normal < collapsed / 10


def test_predictor_is_action_conditioned_and_angle_features_periodic():
    torch.manual_seed(3)
    model = LatentDynamics(SimConfig())
    context = torch.zeros(8, 10, 6)
    context[..., 0:2] = 4
    actions = torch.full((8, 5, 10, 2), 4.905)
    shifted = actions.clone()
    shifted[..., 0] += 1
    assert model(context, actions).shape == (8, 5, 24)
    assert not torch.allclose(model(context, actions), model(context, shifted))
    rotated = context.clone()
    rotated[..., 4] += 2 * torch.pi
    torch.testing.assert_close(model.state_features(context), model.state_features(rotated))


def test_zero_residual_prober_preserves_hover():
    config = SimConfig()
    prober = PhysicalProber(config)
    initial = torch.tensor([[4.0, 4.0, 0, 0, 0, 0]])
    actions = torch.full((1, 5, 2), 4.905)
    past = torch.full((1, 9, 2), 4.905)
    states = prober(torch.zeros(1, 5, 24), initial, actions, past)
    torch.testing.assert_close(states, initial[:, None].expand(-1, 5, -1), atol=1e-5, rtol=1e-5)


def test_prober_training_keeps_latent_model_frozen():
    model = LatentDynamics(SimConfig()).requires_grad_(False)
    prober = PhysicalProber(SimConfig())
    context = torch.zeros(4, 10, 6)
    context[..., :2] = 4
    commands = torch.full((4, 3, 10, 2), 5.0)
    latent = model(context, commands).detach()
    states = prober(latent, context[:, -1], commands[:, :, -1], commands[:, 0, :-1])
    states.square().mean().backward()
    assert all(p.grad is None for p in model.parameters())
    assert prober.network[-1].weight.grad.abs().sum() > 0

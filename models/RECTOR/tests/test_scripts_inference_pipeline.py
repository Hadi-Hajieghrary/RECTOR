from pathlib import Path
import sys

import torch


SCRIPTS_ROOT = Path(__file__).parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS_ROOT))

from inference.pipeline import InferenceConfig, RECTORInference


class _DummyTieredScorer:
    def score_trajectories(self, violations, applicability=None, mode="lexicographic"):
        return violations.sum(dim=-1)

    def select_best(self, violations, applicability=None, mode="lexicographic"):
        scores = violations.sum(dim=-1)
        idx = scores.argmin(dim=1)
        return idx, scores


class _DummyTrajectoryHead:
    def sample(self, scene_embedding, num_samples, temperature):
        b = scene_embedding.shape[0]
        t, d = 50, 4
        trajectories = torch.zeros(
            (b, num_samples, t, d), device=scene_embedding.device
        )
        confidences = torch.full(
            (b, num_samples), 1.0 / num_samples, device=scene_embedding.device
        )

        # Ensure different modes are distinguishable.
        for mode in range(num_samples):
            trajectories[:, mode, :, 0] = float(mode)
        return trajectories, confidences


class _DummyModel:
    def __init__(self):
        self.tiered_scorer = _DummyTieredScorer()
        self.trajectory_head = _DummyTrajectoryHead()

    def to(self, device):
        return self

    def eval(self):
        return self

    def encode_scene(
        self,
        ego_history,
        agent_states,
        lane_centers,
        agent_mask=None,
        lane_mask=None,
        agent_types=None,
    ):
        b = ego_history.shape[0]
        embed = torch.zeros((b, 16), device=ego_history.device)
        return embed, None, None

    def predict_applicability(
        self, scene_embedding, scene_elements=None, element_mask=None
    ):
        b = scene_embedding.shape[0]
        probs = torch.full((b, 28), 0.6, device=scene_embedding.device)
        return None, probs

    def evaluate_violations(self, trajectories, scene_features):
        b, m = trajectories.shape[:2]
        out = torch.zeros((b, m, 28), device=trajectories.device)
        # Mode index becomes proxy total violation magnitude.
        for mode in range(m):
            out[:, mode, :] = float(mode)
        return out


def test_inference_config_defaults_common_usage():
    cfg = InferenceConfig()
    assert cfg.num_samples > 0
    assert cfg.shortlist_k > 0
    assert cfg.selection_mode in {"lexicographic", "weighted"}


def test_shortlist_candidates_corner_k_lower_than_modes():
    model = _DummyModel()
    cfg = InferenceConfig(device="cpu", shortlist_k=3, num_samples=6)
    engine = RECTORInference(model=model, config=cfg)

    b, m, t, d, r = 2, 6, 50, 4, 28
    trajectories = torch.randn((b, m, t, d))
    violations = torch.randn((b, m, r)).abs()
    applicability = torch.ones((b, r))

    shortlisted_traj, shortlisted_viol, shortlist_idx = engine.shortlist_candidates(
        trajectories=trajectories,
        violations=violations,
        applicability=applicability,
        k=3,
    )

    assert shortlisted_traj.shape == (b, 3, t, d)
    assert shortlisted_viol.shape == (b, 3, r)
    assert shortlist_idx.shape == (b, 3)


def test_infer_common_usage_return_all_shapes():
    model = _DummyModel()
    cfg = InferenceConfig(
        device="cpu", num_samples=6, shortlist_k=4, use_shortlisting=True
    )
    engine = RECTORInference(model=model, config=cfg)

    b = 2
    ego_history = torch.zeros((b, 11, 4))
    agent_states = torch.zeros((b, 8, 11, 4))
    lane_centers = torch.zeros((b, 16, 20, 2))

    result = engine.infer(
        ego_history=ego_history,
        agent_states=agent_states,
        lane_centers=lane_centers,
        return_all=True,
    )

    assert result["trajectory"].shape == (b, 50, 4)
    assert result["applicability"].shape == (b, 28)
    assert result["violations"].shape == (b, 28)
    assert result["all_trajectories"].shape[1] == cfg.shortlist_k
    assert result["all_confidences"].shape == (b, cfg.shortlist_k)

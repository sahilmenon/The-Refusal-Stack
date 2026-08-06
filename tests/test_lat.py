"""CPU-only tests for §8E LAT latent adversarial training (arXiv:2407.15549).

No model / GPU: exercises the eps-ball projection and the PGD inner step. The
min-max training loop is pod-only (needs a backdoored checkpoint) and not tested.
"""

import numpy as np
import pytest

from refusal_stack.generalize.harden.lat import LATConfig, pgd_step, project_to_ball


def test_project_to_ball_clamps_large_vector():
    d = np.array([3.0, 4.0])  # norm 5
    p = project_to_ball(d, epsilon=1.0)
    assert np.linalg.norm(p) == pytest.approx(1.0)
    # direction preserved
    assert np.allclose(p, d / 5.0)


def test_project_to_ball_leaves_small_vector():
    d = np.array([0.1, 0.1])
    assert np.allclose(project_to_ball(d, epsilon=1.0), d)


def test_project_to_ball_rowwise():
    d = np.array([[3.0, 4.0], [0.0, 0.1]])  # row0 norm 5, row1 small
    p = project_to_ball(d, epsilon=1.0)
    assert np.linalg.norm(p[0]) == pytest.approx(1.0)
    assert np.allclose(p[1], d[1])


def test_pgd_step_ascends_and_stays_bounded():
    delta = np.zeros(2)
    grad = np.array([10.0, 0.0])
    out = pgd_step(delta, grad, lr=1.0, epsilon=1.0)
    # moved toward the gradient but clamped to the ball
    assert out[0] > 0
    assert np.linalg.norm(out) <= 1.0 + 1e-9


def test_lat_config_defaults():
    cfg = LATConfig()
    assert cfg.epsilon > 0
    assert cfg.inner_steps >= 1
    assert cfg.perturb_layer == 15

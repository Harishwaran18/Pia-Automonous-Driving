"""
mpc_planner.py
==============
Uncertainty-aware Model Predictive Path Integral (MPPI) planner.

The planner samples ``N`` candidate action sequences of length ``H``, rolls each
of them out with the probabilistic ensemble world model and re-weights the
samples with a soft-max over their trajectory cost.  Three cost ingredients are
directly inherited from the PIA design:

1. **Tracking** - lateral deviation from the centre line and heading error.
2. **Progress** - speed tracking against a friction-limited reference profile
   computed with the *nominal* friction coefficient.  In the real environment
   this profile is optimistic, which is the root cause of the sim-to-real
   performance drop.
3. **Uncertainty** - the predictive standard deviation of the ensemble is
   penalised along the rollout, so that plans that drive the model into
   out-of-distribution states are rejected before they are executed.

The rollout is written with ``torch`` and no autograd, and the reference
geometry is pre-computed once as a GPU/CPU tensor, so a full ``plan()`` call is
a handful of batched matrix products.

Author: PIA reference implementation.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from dynamics_sim import GRAVITY, IDX_HEADING, IDX_V, Track
from world_model import ProbabilisticEnsemble


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #


@dataclass
class MPCConfig:
    """MPPI hyper-parameters and cost weights.

    All cost weights operate on *per-step* quantities and the total cost is the
    mean over the horizon, which keeps the weighting scale independent of
    ``horizon``.

    Attributes
    ----------
    horizon:
        Planning horizon ``H`` in control steps (``H * dt`` seconds).
    num_samples:
        Number of sampled action sequences ``N`` per MPPI iteration.
    num_iterations:
        Number of MPPI refinement iterations per control step.
    temperature:
        Scale of the soft-max temperature ``lambda``.  ``lambda`` is adapted
        per iteration as ``temperature * std(costs)``.
    init_std_accel, init_std_steer:
        Initial exploration standard deviations of the sampling distribution.
    min_std_accel, min_std_steer:
        Lower bounds on the sampling standard deviations.
    smoothing:
        Exponential moving average coefficient applied to the MPPI mean and
        standard deviation (higher = more inertia).
    accel_bounds, steer_bounds:
        Box constraints of the action space.
    w_lateral:
        Weight of the squared lateral error.
    w_heading:
        Weight of the squared heading error.
    w_speed:
        Weight of the squared speed-tracking error.
    w_steer_rate:
        Weight of the squared steering-rate penalty.
    w_accel:
        Weight of the squared acceleration penalty (energy regulariser).
    w_uncertainty:
        Weight of the predictive standard deviation penalty.
    w_terminal_lateral:
        Extra terminal penalty on the final lateral error.
    target_speed:
        Upper speed cap of the reference profile in m/s.
    assumed_mu:
        Friction coefficient used to build the reference speed profile.  This
        is the *prior* the controller brings to the real track.
    uncertainty_members:
        Number of ensemble members evaluated during rollouts.  Two members are
        enough to obtain a mean plus an epistemic variance estimate.
    seed:
        Seed of the planner's internal RNG.
    """

    horizon: int = 10
    num_samples: int = 64
    num_iterations: int = 2
    temperature: float = 0.6
    init_std_accel: float = 2.5
    init_std_steer: float = 0.20
    min_std_accel: float = 0.20
    min_std_steer: float = 0.015
    smoothing: float = 0.0
    accel_bounds: Tuple[float, float] = (-8.0, 6.5)
    steer_bounds: Tuple[float, float] = (-0.50, 0.50)
    w_lateral: float = 8.0
    w_heading: float = 1.5
    w_speed: float = 0.30
    w_steer_rate: float = 1.5
    w_accel: float = 0.02
    w_uncertainty: float = 6.0
    w_terminal_lateral: float = 15.0
    target_speed: float = 11.5
    assumed_mu: float = 1.0
    uncertainty_members: int = 2
    seed: int = 0


# --------------------------------------------------------------------------- #
# Torch reference geometry
# --------------------------------------------------------------------------- #


class TrackReference:
    """Vectorised track reference used inside the MPPI cost function.

    The centre line is subsampled and stored as tensors so that the lateral /
    heading / speed errors of a whole batch of predicted positions can be
    computed with two batched matrix products.

    Parameters
    ----------
    track:
        Source track geometry.
    stride:
        Subsampling factor applied to the centre line (``1`` = every sample).
    assumed_mu:
        Friction coefficient used for the reference speed profile.
    max_speed:
        Speed cap of the reference profile.
    device:
        Torch device of the cached tensors.
    """

    def __init__(
        self,
        track: Track,
        *,
        stride: int = 3,
        assumed_mu: float = 1.0,
        max_speed: float = 16.0,
        device: str | torch.device = "cpu",
    ) -> None:
        self.track = track
        self.device = torch.device(device)
        self.stride = int(stride)
        indices = np.arange(0, track.num_points, self.stride, dtype=np.int64)
        self.num = int(indices.size)
        self.spacing = float(track.length / self.num)

        self.points = torch.as_tensor(track.centerline[indices], dtype=torch.float32, device=self.device)
        self.tangents = torch.as_tensor(track.tangents[indices], dtype=torch.float32, device=self.device)
        self.headings = torch.as_tensor(track.headings[indices], dtype=torch.float32, device=self.device)
        speed_limit = track.curvature_speed_limit(
            mu=assumed_mu, safety_factor=0.92, max_speed=max_speed, min_speed=4.0
        )
        self.v_ref = torch.as_tensor(speed_limit[indices], dtype=torch.float32, device=self.device)

    def index_of(self, progress: float) -> int:
        """Reference index corresponding to an arc-length position."""
        return int(float(progress) / self.spacing) % self.num

    def evaluate(
        self,
        states: torch.Tensor,
        start_index: int,
        window: int = 220,
    ) -> Dict[str, torch.Tensor]:
        """Compute tracking terms for a batch of predicted states.

        Parameters
        ----------
        states:
            ``(B, 5)`` predicted states.
        start_index:
            Reference index of the current vehicle position.  The search window
            starts slightly behind it and extends ``window`` samples ahead.
        window:
            Number of reference samples considered.

        Returns
        -------
        dict with ``lateral`` (signed, m), ``heading_error`` (rad), ``v_ref``
        (m/s) and ``progress`` (m travelled along the reference window).
        """
        device = self.device
        offset = torch.arange(window, device=device)
        idx = (start_index - 4 + offset) % self.num
        pts = self.points[idx]                     # (W, 2)
        tangents = self.tangents[idx]              # (W, 2)
        v_ref = self.v_ref[idx]                    # (W,)

        diff = states[:, None, :2] - pts[None, :, :]        # (B, W, 2)
        distance2 = diff.pow(2).sum(dim=-1)                 # (B, W)
        nearest = distance2.argmin(dim=1)                   # (B,)

        tangent = tangents[nearest]                         # (B, 2)
        closest = pts[nearest]                              # (B, 2)
        offset_vec = states[:, :2] - closest                # (B, 2)
        lateral = tangent[:, 0] * offset_vec[:, 1] - tangent[:, 1] * offset_vec[:, 0]
        heading = torch.atan2(tangent[:, 1], tangent[:, 0])
        heading_error = torch.remainder(heading - states[:, IDX_HEADING] + math.pi, 2.0 * math.pi) - math.pi
        progress = (idx[nearest].float() - float(start_index)) * self.spacing
        return {
            "lateral": lateral,
            "heading_error": heading_error,
            "v_ref": v_ref[nearest],
            "progress": progress,
        }


# --------------------------------------------------------------------------- #
# Plan result
# --------------------------------------------------------------------------- #


@dataclass
class PlanResult:
    """Output of one :meth:`MPCPlanner.plan` call.

    Attributes
    ----------
    action:
        ``(2,)`` action to apply: ``[acceleration, steering]``.
    nominal_actions:
        ``(H, 2)`` optimised (mean) action sequence.
    predicted_states:
        ``(H + 1, 5)`` ensemble-mean rollout of the nominal sequence.
    predicted_lateral_error:
        ``(H + 1,)`` lateral error along the nominal rollout in metres.
    predicted_uncertainty:
        ``(H,)`` predictive standard deviation along the nominal rollout.
    best_cost:
        Lowest sampled trajectory cost.
    mean_cost:
        Weighted mean trajectory cost.
    evaluation_ms:
        Wall-clock planning time in milliseconds.
    """

    action: np.ndarray
    nominal_actions: np.ndarray
    predicted_states: np.ndarray
    predicted_lateral_error: np.ndarray
    predicted_uncertainty: np.ndarray
    best_cost: float
    mean_cost: float
    evaluation_ms: float


# --------------------------------------------------------------------------- #
# Planner
# --------------------------------------------------------------------------- #


class MPCPlanner:
    """Uncertainty-aware MPPI planner over a learned ensemble world model.

    Parameters
    ----------
    world_model:
        Trained :class:`~world_model.ProbabilisticEnsemble`.  The planner keeps
        a reference to the module, so PIA can fine-tune it in place and the
        planner will automatically use the adapted dynamics.
    track:
        Track geometry used to build the reference.
    config:
        Planner hyper-parameters.
    device:
        Device on which rollouts are computed.
    seed:
        Seed for the sampling RNG.

    Examples
    --------
    >>> planner = MPCPlanner(model, track)
    >>> result = planner.plan(state)
    >>> env.step(result.action)
    """

    def __init__(
        self,
        world_model: ProbabilisticEnsemble,
        track: Track,
        config: Optional[MPCConfig] = None,
        *,
        device: str | torch.device = "cpu",
        seed: int = 0,
    ) -> None:
        self.model = world_model
        self.track = track
        self.config = config if config is not None else MPCConfig()
        self.device = torch.device(device)
        self.model.to(self.device)
        self.model.eval()

        self.reference = TrackReference(
            track,
            stride=3,
            assumed_mu=self.config.assumed_mu,
            max_speed=self.config.target_speed,
            device=self.device,
        )
        self.assumed_mu: float = float(self.config.assumed_mu)
        self.member_ids: List[int] = list(range(min(self.config.uncertainty_members, self.model.n_members)))

        self._generator = torch.Generator(device=self.device).manual_seed(int(seed))
        self._mean = torch.zeros(self.config.horizon, 2, device=self.device)
        self._std = torch.tensor(
            [self.config.init_std_accel, self.config.init_std_steer], device=self.device
        ).expand(self.config.horizon, 2).clone()
        self._last_action = torch.zeros(2, device=self.device)
        self._best_sequence = torch.zeros(self.config.horizon, 2, device=self.device)

        low = torch.tensor([self.config.accel_bounds[0], self.config.steer_bounds[0]], device=self.device)
        high = torch.tensor([self.config.accel_bounds[1], self.config.steer_bounds[1]], device=self.device)
        self._low = low.expand(self.config.horizon, 2)
        self._high = high.expand(self.config.horizon, 2)

    # ------------------------------------------------------------------ #
    def set_world_model(self, model: ProbabilisticEnsemble) -> None:
        """Replace the dynamics model (used when PIA swaps in its adapted copy)."""
        self.model = model
        self.model.to(self.device)
        self.model.eval()
        self.member_ids = list(range(min(self.config.uncertainty_members, self.model.n_members)))

    def reset(self) -> None:
        """Clear the warm-started action distribution."""
        self._mean.zero_()
        self._std = torch.tensor(
            [self.config.init_std_accel, self.config.init_std_steer], device=self.device
        ).expand(self.config.horizon, 2).clone()
        self._last_action.zero_()

    def set_assumed_mu(self, mu: float) -> None:
        """Rebuild the reference speed profile with a new friction estimate.

        PIA calls this when its online friction estimator concludes that the
        prior's ``mu`` was optimistic.  Only the reference geometry is rebuilt;
        the sampling distribution and the warm start are preserved.
        """
        self.assumed_mu = float(mu)
        self.reference = TrackReference(
            self.track,
            stride=3,
            assumed_mu=self.assumed_mu,
            max_speed=self.config.target_speed,
            device=self.device,
        )

    # ------------------------------------------------------------------ #
    def plan(self, state: Sequence[float] | np.ndarray) -> PlanResult:
        """Compute the MPPI action for a single state.

        Parameters
        ----------
        state:
            ``[x, y, v, heading, yaw_rate]`` observation.

        Returns
        -------
        PlanResult
        """
        started = time.perf_counter()
        cfg = self.config
        state_tensor = torch.as_tensor(np.asarray(state, dtype=np.float32), device=self.device).reshape(1, -1)
        projection = self.track.project(np.asarray(state, dtype=np.float64)[:2])
        start_index = self.reference.index_of(projection.progress)

        mean = self._mean
        std = self._std
        best_costs = math.inf
        mean_cost = math.inf
        diagnostics: Optional[Dict[str, torch.Tensor]] = None

        with torch.no_grad():
            for iteration in range(int(cfg.num_iterations)):
                noise = torch.randn(
                    cfg.num_samples,
                    cfg.horizon,
                    2,
                    generator=self._generator,
                    device=self.device,
                )
                samples = torch.clamp(mean.unsqueeze(0) + noise * std.unsqueeze(0), self._low, self._high)
                samples[0] = mean  # always evaluate the current nominal plan
                if iteration > 0:
                    # Re-evaluate the previous best sequence alongside the new samples.
                    samples[1] = self._best_sequence

                costs, diagnostics = self._evaluate(
                    state_tensor,
                    samples,
                    start_index,
                    record=(iteration == cfg.num_iterations - 1),
                )
                spread = float(costs.std().item())
                lam = max(cfg.temperature * spread, 1e-4)
                weights = torch.softmax(-(costs - costs.min()) / lam, dim=0)

                new_mean = (weights[:, None, None] * samples).sum(dim=0)
                new_mean = torch.clamp(new_mean, self._low, self._high)
                variance = (weights[:, None, None] * (samples - new_mean) ** 2).sum(dim=0)
                new_std = torch.sqrt(variance.clamp_min(1e-12))
                new_std[:, 0] = new_std[:, 0].clamp_min(cfg.min_std_accel)
                new_std[:, 1] = new_std[:, 1].clamp_min(cfg.min_std_steer)

                mean = cfg.smoothing * mean + (1.0 - cfg.smoothing) * new_mean
                std = cfg.smoothing * std + (1.0 - cfg.smoothing) * new_std
                self._mean, self._std = mean, std
                best_index = int(costs.argmin().item())
                self._best_sequence = samples[best_index].clone()
                best_costs = float(costs.min().item())
                mean_cost = float((weights * costs).sum().item())

            action = torch.clamp(self._mean[0], self._low[0], self._high[0])
            self._last_action = action.clone()

            # The last iteration records the nominal (sample 0) rollout for free.
            assert diagnostics is not None
            trajectory = diagnostics["states"]
            uncertainty = diagnostics["uncertainty"]
            lateral = diagnostics["lateral"]
            nominal_actions = self._mean.clone()
            self._mean = torch.cat([self._mean[1:], self._mean[-1:]], dim=0)

        evaluation_ms = (time.perf_counter() - started) * 1e3
        return PlanResult(
            action=action.detach().cpu().numpy().astype(np.float64),
            nominal_actions=nominal_actions.detach().cpu().numpy().astype(np.float64),
            predicted_states=trajectory.detach().cpu().numpy()[0].astype(np.float64),
            predicted_lateral_error=lateral.detach().cpu().numpy()[0].astype(np.float64),
            predicted_uncertainty=uncertainty.detach().cpu().numpy()[0].astype(np.float64),
            best_cost=best_costs,
            mean_cost=mean_cost,
            evaluation_ms=evaluation_ms,
        )

    # ------------------------------------------------------------------ #
    def _evaluate(
        self,
        state0: torch.Tensor,
        samples: torch.Tensor,
        start_index: int,
        record: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Dict[str, torch.Tensor]]]:
        """Roll out ``samples`` with the world model and return their costs.

        Parameters
        ----------
        state0:
            ``(1, 5)`` current state, broadcast over the samples.
        samples:
            ``(N, H, 2)`` candidate action sequences.  Sample ``0`` is always the
            current nominal plan, so recording it here is free.
        start_index:
            Reference index of the current position.
        record:
            Whether to also return the diagnostics of the nominal rollout.

        Returns
        -------
        ``(costs, diagnostics)`` where ``costs`` has shape ``(N,)`` and
        ``diagnostics`` is ``None`` unless ``record`` is ``True``.
        """
        cfg = self.config
        num_samples, horizon, _ = samples.shape
        states = state0.expand(num_samples, -1)
        total = torch.zeros(num_samples, device=self.device)
        steer_prev = self._last_action[1].expand(num_samples)

        record_states: List[torch.Tensor] = [state0] if record else []
        record_lateral: List[torch.Tensor] = []
        record_uncertainty: List[torch.Tensor] = []

        reference: Dict[str, torch.Tensor] = {}
        for step in range(horizon):
            actions = samples[:, step]
            states, uncertainty = self.model.predict_step(states, actions, members=self.member_ids)
            reference = self.reference.evaluate(states, start_index)

            lateral = reference["lateral"]
            heading_error = reference["heading_error"]
            v_ref = reference["v_ref"]

            step_cost = cfg.w_lateral * lateral.pow(2)
            step_cost = step_cost + cfg.w_heading * heading_error.pow(2)
            step_cost = step_cost + cfg.w_speed * (states[:, IDX_V] - v_ref).pow(2)
            step_cost = step_cost + cfg.w_uncertainty * uncertainty
            step_cost = step_cost + cfg.w_steer_rate * (actions[:, 1] - steer_prev).pow(2)
            step_cost = step_cost + cfg.w_accel * actions[:, 0].pow(2)
            total = total + step_cost
            steer_prev = actions[:, 1]

            if record:
                record_states.append(states[0:1])
                record_lateral.append(lateral[0:1])
                record_uncertainty.append(uncertainty[0:1])

        total = total / float(horizon)
        total = total + cfg.w_terminal_lateral * reference["lateral"].pow(2)
        if not record:
            return total, None
        diagnostics = {
            "states": torch.cat(record_states, dim=0).unsqueeze(0),
            "lateral": torch.cat(record_lateral, dim=0).unsqueeze(0),
            "uncertainty": torch.cat(record_uncertainty, dim=0).unsqueeze(0),
        }
        return total, diagnostics


# --------------------------------------------------------------------------- #
# Self-test
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    from pathlib import Path

    from dataset_generator import StanleyController, generate_prior_dataset
    from dynamics_sim import RealEnv, SimEnv, build_track

    torch.manual_seed(0)
    np.random.seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    track = build_track()

    model_path = Path("artifacts/world_model.pt")
    if model_path.exists():
        print(f"loading world model from {model_path}")
        model = ProbabilisticEnsemble.load(model_path, device=device)
    else:
        print("no cached model found - training a small one for the self-test")
        env = SimEnv(track, seed=0)
        data = generate_prior_dataset(
            env, StanleyController(track, seed=1), episodes=6, steps_per_episode=150, seed=0, verbose=False
        )
        model = ProbabilisticEnsemble(n_members=3, hidden_dims=(128, 128), seed=1)
        model.fit(data, epochs=5, batch_size=256, device=device, verbose=False)
        model.save(model_path)

    cfg = MPCConfig(horizon=10, num_samples=96, num_iterations=2, seed=0)
    planner = MPCPlanner(model, track, cfg, device=device, seed=0)
    print(planner.reference.points.shape, "reference samples")

    for env_cls in (SimEnv, RealEnv):
        env = env_cls(track, seed=5)
        planner.reset()
        env.reset(progress=0.0, speed=9.0)
        errors: List[float] = []
        times: List[float] = []
        for _ in range(260):
            result = planner.plan(env.observe())
            _s, _r, done, info = env.step(result.action)
            errors.append(info["lateral_error"])
            times.append(result.evaluation_ms)
            if done:
                break
        arr = np.asarray(errors)
        print(
            f"{env.config.name:7s} | steps={len(arr):3d} | lateral RMSE={np.sqrt(np.mean(arr**2)):.3f} m "
            f"| max={np.abs(arr).max():.3f} m | plan={np.mean(times):.1f} ms"
        )
    print("mpc_planner.py self-test OK")

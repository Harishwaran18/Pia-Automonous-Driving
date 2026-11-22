"""
dataset_generator.py
====================
Offline prior-data generation for the PIA study.

A noisy Stanley / PID baseline controller drives :class:`~dynamics_sim.SimEnv`
(the nominal, high-grip simulator) and every ``(state, action, next_state,
delta_state)`` transition is logged.  The resulting dataset is the *only*
information the probabilistic ensemble world model ever receives about
vehicle dynamics before deployment, so its coverage determines how well the
prior generalises to the real track.

The exported :class:`PriorDataset` is deliberately lightweight: plain NumPy
arrays plus helpers that materialise ``torch.utils.data.TensorDataset`` objects
for training.

Typical use
-----------
>>> from dynamics_sim import SimEnv, build_track
>>> from dataset_generator import StanleyController, generate_prior_dataset
>>> track = build_track()
>>> env = SimEnv(track, seed=0)
>>> controller = StanleyController(track, seed=0)
>>> dataset = generate_prior_dataset(env, controller, episodes=40, steps_per_episode=450)
>>> len(dataset)
18000

Author: PIA reference implementation.
"""
from __future__ import annotations

import argparse
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import TensorDataset

from dynamics_sim import (
    ACTION_DIM,
    IDX_HEADING,
    IDX_V,
    STATE_DIM,
    GRAVITY,
    SimEnv,
    Track,
    VehicleParams,
    build_track,
    wrap_angle,
)

# --------------------------------------------------------------------------- #
# Baseline controller
# --------------------------------------------------------------------------- #


@dataclass
class StanleyConfig:
    """Tuning parameters of the Stanley baseline.

    Attributes
    ----------
    target_speed:
        Cruising speed in m/s (overridden per episode during data collection).
    lateral_gain:
        Stanley cross-track gain ``k`` in ``atan(k * e / (v + eps))``.
    heading_gain:
        Gain applied to the look-ahead heading error.
    speed_gain:
        Proportional gain of the speed controller (m/s^2 per m/s error).
    brake_gain:
        Extra gain used when the car is faster than the target.
    lookahead_base:
        Minimum look-ahead distance in metres.
    lookahead_gain:
        Speed-dependent look-ahead gain in seconds.
    assumed_mu:
        Friction coefficient assumed by the controller (nominal 1.0).
    steer_margin:
        Fraction of the grip-limited steering angle the controller is allowed
        to use (keeps the baseline out of permanent saturation).
    """

    target_speed: float = 15.0
    lateral_gain: float = 1.50
    heading_gain: float = 1.10
    speed_gain: float = 1.40
    brake_gain: float = 2.20
    lookahead_base: float = 3.0
    lookahead_gain: float = 0.55
    assumed_mu: float = 1.0
    steer_margin: float = 0.85


class StanleyController:
    """Stanley-style path tracker with curvature-aware speed regulation.

    The controller is intentionally simple and *deterministic*; exploration
    noise is injected by :func:`generate_prior_dataset` so that the controller
    can also be reused as a clean evaluation baseline.

    Parameters
    ----------
    track:
        Track geometry used for path projection and look-ahead sampling.
    vehicle:
        Vehicle parameters (wheelbase, steering limits).
    config:
        Controller tuning parameters.
    seed:
        Seed of the internal RNG (only used by :meth:`act_noisy`).
    """

    def __init__(
        self,
        track: Track,
        vehicle: Optional[VehicleParams] = None,
        config: Optional[StanleyConfig] = None,
        *,
        seed: int = 0,
    ) -> None:
        self.track = track
        self.vehicle = vehicle if vehicle is not None else VehicleParams()
        self.config = config if config is not None else StanleyConfig()
        self._rng = np.random.default_rng(seed)
        self._speed_limits: np.ndarray = track.curvature_speed_limit(
            mu=self.config.assumed_mu,
            safety_factor=0.90,
            max_speed=self.config.target_speed,
            min_speed=4.0,
        )

    # ------------------------------------------------------------------ #
    def set_target_speed(self, speed: float) -> None:
        """Update the cruising speed and rebuild the local speed profile."""
        self.config.target_speed = float(speed)
        self._speed_limits = self.track.curvature_speed_limit(
            mu=self.config.assumed_mu,
            safety_factor=0.90,
            max_speed=self.config.target_speed,
            min_speed=4.0,
        )

    # ------------------------------------------------------------------ #
    def act(self, state: Sequence[float]) -> np.ndarray:
        """Compute the deterministic control action for a state.

        Parameters
        ----------
        state:
            ``[x, y, v, heading, yaw_rate]`` in SI units.

        Returns
        -------
        ``np.ndarray`` of shape ``(2,)`` with ``[acceleration, steering]``.
        """
        s = np.asarray(state, dtype=np.float64).reshape(STATE_DIM)
        speed = float(s[IDX_V])
        heading = float(s[IDX_HEADING])
        proj = self.track.project(s[:2])
        lookahead = float(
            np.clip(
                self.config.lookahead_base + self.config.lookahead_gain * speed,
                self.config.lookahead_base,
                18.0,
            )
        )

        # --- curvature feed-forward + Stanley heading / cross-track terms -- #
        # Pure Stanley has no cornering authority in steady state, so the local
        # path curvature is fed forward exactly as a real racing controller
        # would do.  The remaining two terms are the textbook Stanley law.
        heading_error = float(wrap_angle(proj.heading - heading))
        feedforward = math.atan(self.vehicle.wheelbase * proj.curvature)
        # ``signed_lateral_error`` is positive when the vehicle is left of the
        # path, so the correction must steer right (negative angle).
        cross_track = -math.atan2(
            self.config.lateral_gain * proj.signed_lateral_error,
            speed + 1.0,
        )
        steer = feedforward + self.config.heading_gain * heading_error + cross_track

        # --- grip- and geometry-limited steering --------------------------- #
        grip = self.config.assumed_mu * GRAVITY
        tan_limit = self.vehicle.wheelbase * grip / max(speed * speed, 4.0)
        steer_limit = min(self.vehicle.max_steer, math.atan(tan_limit))
        steer = float(
            np.clip(steer, -self.config.steer_margin * steer_limit, self.config.steer_margin * steer_limit)
        )

        # --- curvature-aware speed target ---------------------------------- #
        look_index = int(proj.index + max(2, int(round(lookahead / self.track.ds)))) % self.track.num_points
        v_des = min(self.config.target_speed, float(self._speed_limits[look_index]))
        error = v_des - speed
        gain = self.config.speed_gain if error >= 0.0 else self.config.brake_gain
        accel = gain * error

        # Feed-forward drag compensation so the car can actually reach v_des.
        accel += self.vehicle.drag_coefficient * speed * speed
        accel += self.vehicle.rolling_resistance * GRAVITY
        accel = float(np.clip(accel, self.vehicle.max_decel, self.vehicle.max_accel))

        return np.array([accel, steer], dtype=np.float64)

    # ------------------------------------------------------------------ #
    def act_noisy(self, state: Sequence[float], noise_std: float) -> np.ndarray:
        """Return :meth:`act` perturbed by Gaussian exploration noise.

        Noise is scaled as ``[4.0, 0.35] * noise_std`` so that a single scalar
        controls both channels with physically sensible units.
        """
        action = self.act(state)
        if noise_std > 0.0:
            scale = np.array([4.0, 0.35])
            action = action + self._rng.normal(0.0, noise_std, size=ACTION_DIM) * scale
        action[0] = float(np.clip(action[0], self.vehicle.max_decel, self.vehicle.max_accel))
        action[1] = float(np.clip(action[1], -self.vehicle.max_steer, self.vehicle.max_steer))
        return action


# --------------------------------------------------------------------------- #
# Dataset container
# --------------------------------------------------------------------------- #


@dataclass
class PriorDataset:
    """Offline transition dataset collected in the nominal simulator.

    Attributes
    ----------
    states:
        ``(N, 5)`` current states.
    actions:
        ``(N, 2)`` applied actions.
    next_states:
        ``(N, 5)`` successor states.
    deltas:
        ``(N, 5)`` state residuals ``next_state - state`` with the heading
        component wrapped to ``(-pi, pi]``.
    episode_ids:
        ``(N,)`` integer identifier of the episode each transition came from.
    meta:
        Free-form metadata (environment parameters, controller settings, ...).
    """

    states: np.ndarray
    actions: np.ndarray
    next_states: np.ndarray
    deltas: np.ndarray
    episode_ids: np.ndarray
    meta: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        n = int(self.states.shape[0])
        if self.states.shape != (n, STATE_DIM):
            raise ValueError(f"states must have shape (N, {STATE_DIM})")
        if self.actions.shape != (n, ACTION_DIM):
            raise ValueError(f"actions must have shape (N, {ACTION_DIM})")
        if self.next_states.shape != (n, STATE_DIM):
            raise ValueError(f"next_states must have shape (N, {STATE_DIM})")
        if self.deltas.shape != (n, STATE_DIM):
            raise ValueError(f"deltas must have shape (N, {STATE_DIM})")

    def __len__(self) -> int:
        return int(self.states.shape[0])

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"PriorDataset(N={len(self)}, speed=[{self.states[:, IDX_V].min():.1f}, "
            f"{self.states[:, IDX_V].max():.1f}] m/s, "
            f"episodes={int(self.episode_ids.max()) + 1 if len(self) else 0})"
        )

    # ------------------------------------------------------------------ #
    def tensors(self, device: str | torch.device = "cpu") -> Dict[str, torch.Tensor]:
        """Materialise every field as a ``torch`` tensor on ``device``."""
        dev = torch.device(device)
        return {
            "states": torch.as_tensor(self.states, dtype=torch.float32, device=dev),
            "actions": torch.as_tensor(self.actions, dtype=torch.float32, device=dev),
            "next_states": torch.as_tensor(self.next_states, dtype=torch.float32, device=dev),
            "deltas": torch.as_tensor(self.deltas, dtype=torch.float32, device=dev),
        }

    def as_tensor_dataset(self, device: str | torch.device = "cpu") -> TensorDataset:
        """Export the full dataset as a :class:`torch.utils.data.TensorDataset`.

        The tuple order is ``(state, action, next_state, delta_state)``.
        """
        t = self.tensors(device)
        return TensorDataset(t["states"], t["actions"], t["next_states"], t["deltas"])

    def split(
        self,
        val_fraction: float = 0.15,
        *,
        seed: int = 0,
        device: str | torch.device = "cpu",
    ) -> Tuple[TensorDataset, TensorDataset]:
        """Split into train / validation :class:`TensorDataset` objects.

        The split is performed *by episode* when more than one episode is
        present, which avoids leaking temporally correlated consecutive
        transitions into the validation set.
        """
        if not 0.0 < val_fraction < 1.0:
            raise ValueError("val_fraction must be in (0, 1)")

        rng = np.random.default_rng(seed)
        unique_eps = np.unique(self.episode_ids)
        if unique_eps.size > 1:
            n_val_eps = max(1, int(round(val_fraction * unique_eps.size)))
            val_eps = rng.choice(unique_eps, size=n_val_eps, replace=False)
            val_mask = np.isin(self.episode_ids, val_eps)
        else:
            indices = rng.permutation(len(self))
            n_val = max(1, int(round(val_fraction * len(self))))
            val_mask = np.zeros(len(self), dtype=bool)
            val_mask[indices[:n_val]] = True
        train_mask = ~val_mask

        t = self.tensors(device)
        train_ds = TensorDataset(
            t["states"][train_mask], t["actions"][train_mask],
            t["next_states"][train_mask], t["deltas"][train_mask],
        )
        val_ds = TensorDataset(
            t["states"][val_mask], t["actions"][val_mask],
            t["next_states"][val_mask], t["deltas"][val_mask],
        )
        return train_ds, val_ds

    def subset(self, indices: np.ndarray) -> "PriorDataset":
        """Return a new dataset containing only ``indices``."""
        idx = np.asarray(indices, dtype=np.int64)
        return PriorDataset(
            states=self.states[idx],
            actions=self.actions[idx],
            next_states=self.next_states[idx],
            deltas=self.deltas[idx],
            episode_ids=self.episode_ids[idx],
            meta=dict(self.meta),
        )

    # ------------------------------------------------------------------ #
    def statistics(self) -> Dict[str, Any]:
        """Summary statistics used for logging and for the README report."""
        speed = self.states[:, IDX_V]
        steer = self.actions[:, 1]
        accel = self.actions[:, 0]
        delta_norm = np.linalg.norm(self.deltas, axis=1)
        return {
            "num_transitions": int(len(self)),
            "num_episodes": int(self.episode_ids.max()) + 1 if len(self) else 0,
            "speed_mean": float(speed.mean()),
            "speed_max": float(speed.max()),
            "speed_min": float(speed.min()),
            "steer_abs_mean": float(np.abs(steer).mean()),
            "steer_abs_max": float(np.abs(steer).max()),
            "accel_mean": float(accel.mean()),
            "accel_min": float(accel.min()),
            "accel_max": float(accel.max()),
            "delta_norm_mean": float(delta_norm.mean()),
            "delta_norm_p95": float(np.percentile(delta_norm, 95)),
        }

    def summary(self) -> str:
        """Human readable multi-line summary."""
        stats = self.statistics()
        lines = ["Offline prior dataset summary", "----------------------------"]
        for key, value in stats.items():
            lines.append(f"{key:>22s}: {value:.4f}" if isinstance(value, float) else f"{key:>22s}: {value}")
        return "\n".join(lines)

    # ------------------------------------------------------------------ #
    def save(self, path: str | Path) -> Path:
        """Persist the dataset to ``path`` with :func:`torch.save` (``.npz``-free)."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "states": torch.as_tensor(self.states, dtype=torch.float32),
                "actions": torch.as_tensor(self.actions, dtype=torch.float32),
                "next_states": torch.as_tensor(self.next_states, dtype=torch.float32),
                "deltas": torch.as_tensor(self.deltas, dtype=torch.float32),
                "episode_ids": torch.as_tensor(self.episode_ids, dtype=torch.int64),
                "meta": self.meta,
            },
            target,
        )
        return target

    @classmethod
    def load(cls, path: str | Path) -> "PriorDataset":
        """Load a dataset previously stored with :meth:`save`."""
        payload = torch.load(Path(path), map_location="cpu", weights_only=False)
        return cls(
            states=payload["states"].numpy(),
            actions=payload["actions"].numpy(),
            next_states=payload["next_states"].numpy(),
            deltas=payload["deltas"].numpy(),
            episode_ids=payload["episode_ids"].numpy(),
            meta=payload.get("meta", {}),
        )


# --------------------------------------------------------------------------- #
# Generation
# --------------------------------------------------------------------------- #


def generate_prior_dataset(
    env: SimEnv,
    controller: StanleyController,
    *,
    episodes: int = 40,
    steps_per_episode: int = 450,
    seed: int = 0,
    noise_std: float = 0.08,
    speed_range: Tuple[float, float] = (10.0, 16.0),
    verbose: bool = True,
) -> PriorDataset:
    """Drive the nominal simulator and collect transitions.

    Each episode randomises the spawn point, the lateral offset, the initial
    speed, the controller's cruising speed and the exploration noise level.
    Transitions that occur after the vehicle has left the track are discarded,
    so the dataset only contains on-track dynamics.

    Parameters
    ----------
    env:
        Nominal simulator (:class:`~dynamics_sim.SimEnv`).  It is reset by this
        function and left in a valid state afterwards.
    controller:
        Baseline controller used to generate behaviour.
    episodes:
        Number of episodes to roll out.
    steps_per_episode:
        Maximum number of control steps per episode.
    seed:
        Master seed for spawn randomisation and exploration noise.
    noise_std:
        Maximum Gaussian exploration-noise scale (see
        :meth:`StanleyController.act_noisy`).
    speed_range:
        Range of cruising speeds sampled per episode.
    verbose:
        Whether to print progress.

    Returns
    -------
    PriorDataset
        The collected transitions.
    """
    rng = np.random.default_rng(seed)
    states: List[np.ndarray] = []
    actions: List[np.ndarray] = []
    next_states: List[np.ndarray] = []
    deltas: List[np.ndarray] = []
    episode_ids: List[int] = []

    track = env.track
    total = 0
    for episode in range(int(episodes)):
        spawn = float(rng.uniform(0.0, track.length))
        lateral = float(rng.uniform(-2.0, 2.0))
        cruising = float(rng.uniform(*speed_range))
        episode_noise = float(rng.uniform(0.3, 1.0) * noise_std)
        controller.set_target_speed(cruising)
        # Low initial speeds are included on purpose: the real vehicle may have
        # to recover from a spin, and the world model must cover that region.
        env.reset(progress=spawn, lateral_offset=lateral, speed=float(rng.uniform(3.5, 12.0)))

        collected = 0
        for step in range(int(steps_per_episode)):
            state = env.state.copy()
            action = controller.act_noisy(state, episode_noise)

            # Occasional steering pulse to excite transient dynamics.
            if rng.random() < 0.015:
                pulse = float(rng.normal(0.0, 0.10))
                action = action + np.array([0.0, pulse])
                action[1] = float(np.clip(action[1], -env.vehicle.max_steer, env.vehicle.max_steer))

            next_state, _reward, done, info = env.step(action)
            if not info["off_track"] and next_state[IDX_V] > 1.0:
                delta = next_state - state
                delta[IDX_HEADING] = float(wrap_angle(delta[IDX_HEADING]))
                states.append(state)
                actions.append(action.astype(np.float64))
                next_states.append(next_state.copy())
                deltas.append(delta.astype(np.float64))
                episode_ids.append(episode)
                collected += 1
                total += 1
            if done:
                break

        if verbose and (episode + 1) % 10 == 0:
            print(f"  [dataset] episode {episode + 1:3d}/{episodes} | transitions={total}")

    if total == 0:
        raise RuntimeError("no transitions were collected - check the controller / track")

    dataset = PriorDataset(
        states=np.asarray(states, dtype=np.float64),
        actions=np.asarray(actions, dtype=np.float64),
        next_states=np.asarray(next_states, dtype=np.float64),
        deltas=np.asarray(deltas, dtype=np.float64),
        episode_ids=np.asarray(episode_ids, dtype=np.int64),
        meta={
            "source_env": env.config.describe(),
            "controller": {
                "target_speed": controller.config.target_speed,
                "lateral_gain": controller.config.lateral_gain,
                "heading_gain": controller.config.heading_gain,
            },
            "noise_std": noise_std,
            "episodes": int(episodes),
            "steps_per_episode": int(steps_per_episode),
            "seed": int(seed),
            "track_length": float(track.length),
            "dt": float(env.dt),
        },
    )
    if verbose:
        print(dataset.summary())
    return dataset


def evaluate_controller(
    env: SimEnv,
    controller: StanleyController,
    *,
    steps: int = 600,
    seed: int = 0,
) -> Dict[str, float]:
    """Roll out the baseline controller and report tracking statistics."""
    env.reset(progress=0.0, speed=controller.config.target_speed * 0.6)
    errors: List[float] = []
    speeds: List[float] = []
    for _ in range(int(steps)):
        action = controller.act(env.observe())
        _state, _reward, done, info = env.step(action)
        errors.append(float(info["lateral_error"]))
        speeds.append(float(info["speed"]))
        if done:
            break
    arr = np.asarray(errors)
    return {
        "steps": float(len(arr)),
        "lateral_rmse": float(np.sqrt(np.mean(arr * arr))),
        "lateral_max": float(np.max(np.abs(arr))),
        "mean_speed": float(np.mean(speeds)),
        "distance": float(env.total_progress),
    }


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate the offline prior dataset (SimEnv).")
    parser.add_argument("--episodes", type=int, default=40)
    parser.add_argument("--steps-per-episode", type=int, default=450)
    parser.add_argument("--noise-std", type=float, default=0.08)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=str, default="artifacts/prior_dataset.pt")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    """CLI entry point: generate, report and save an offline prior dataset."""
    args = _build_arg_parser().parse_args(argv)
    track = build_track()
    env = SimEnv(track, seed=args.seed)
    controller = StanleyController(track, seed=args.seed + 1)

    baseline = evaluate_controller(env, controller, steps=900, seed=args.seed)
    print("Baseline controller on SimEnv:", {k: round(v, 3) for k, v in baseline.items()})

    dataset = generate_prior_dataset(
        env,
        controller,
        episodes=args.episodes,
        steps_per_episode=args.steps_per_episode,
        noise_std=args.noise_std,
        seed=args.seed,
    )
    path = dataset.save(args.out)
    print(f"Saved {len(dataset)} transitions to {path}")


if __name__ == "__main__":
    main()

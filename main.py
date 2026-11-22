"""
main.py
=======
End-to-end orchestrator and benchmark for the **Prior-Informed Adaptation (PIA)**
autonomous-racing study.

The script runs the full five-stage pipeline:

1. Collect an offline prior dataset by driving :class:`~dynamics_sim.SimEnv`
   with a noisy Stanley baseline controller.
2. Train the probabilistic ensemble world model on that dataset and log the
   validation negative log-likelihood.
3. Deploy the resulting MPPI planner on :class:`~dynamics_sim.RealEnv` *without*
   adaptation and measure the sim-to-real performance drop.
4. Re-deploy with the PIA engine enabled: the ensemble is fine-tuned online on
   the live residuals while the KL/L2 anchors preserve the offline prior.
5. Produce the comparison figures (trajectory, tracking error, speed, lap
   times, training and adaptation loss curves) and a JSON metrics report.

Usage
-----
::

    python main.py                 # full benchmark (a few minutes on a laptop)
    python main.py --quick         # fast smoke test of the whole pipeline
    python main.py --no-show       # write the figures without opening a window

Author: PIA reference implementation.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from dataset_generator import PriorDataset, StanleyController, generate_prior_dataset
from dynamics_sim import EnvConfig, RealEnv, SimEnv, Track, build_track
from mpc_planner import MPCConfig, MPCPlanner
from pia_adaptation import PIAdaptationEngine, PIAConfig
from world_model import ProbabilisticEnsemble, TrainHistory

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

NOMINAL_COLOR = "#c1121f"
PIA_COLOR = "#00798c"
TRACK_COLOR = "#8d99ae"
GRID_COLOR = "#d9d9d9"


# --------------------------------------------------------------------------- #
# Logging helpers
# --------------------------------------------------------------------------- #


def log(message: str) -> None:
    """Print a timestamped progress line (flushed for live monitoring)."""
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def banner(title: str) -> None:
    """Print a section banner."""
    line = "=" * 78
    print(f"\n{line}\n{title}\n{line}", flush=True)


def _safe_mean(values: Sequence[float] | np.ndarray) -> float:
    arr = np.asarray(values, dtype=np.float64)
    return float(arr.mean()) if arr.size else float("nan")


def _rolling_mean(values: np.ndarray, window: int) -> np.ndarray:
    """Centred rolling mean with edge padding (for readable plots)."""
    arr = np.asarray(values, dtype=np.float64)
    if window <= 1 or arr.size == 0:
        return arr
    window = min(int(window), arr.size)
    kernel = np.ones(window, dtype=np.float64) / float(window)
    padded = np.pad(arr, (window // 2, window - 1 - window // 2), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


# --------------------------------------------------------------------------- #
# Benchmark log
# --------------------------------------------------------------------------- #


@dataclass
class RunLog:
    """Everything recorded during one closed-loop benchmark run.

    Attributes
    ----------
    label:
        Human readable run name.
    time, states, actions, lateral_errors, speeds, uncertainties:
        Per-step arrays of the run.
    online_rmse:
        One-step prediction RMSE of the *current* world model measured on a
        fixed probe set of real transitions (metres).
    lap_ids:
        Lap index reported by the environment at every step.
    lap_times:
        Duration of every completed lap in seconds.
    lap_rmse:
        Lateral RMSE of every completed lap in metres.
    offtrack_events:
        Number of off-track recoveries.
    wall_time_s:
        Wall-clock duration of the run.
    plan_ms:
        Mean planning time per control step in milliseconds.
    adaptation:
        Telemetry of the PIA engine (empty for the nominal run).
    """

    label: str
    time: np.ndarray
    states: np.ndarray
    actions: np.ndarray
    lateral_errors: np.ndarray
    speeds: np.ndarray
    uncertainties: np.ndarray
    online_rmse: np.ndarray
    lap_ids: np.ndarray
    lap_times: List[float] = field(default_factory=list)
    lap_rmse: List[float] = field(default_factory=list)
    offtrack_events: int = 0
    wall_time_s: float = 0.0
    plan_ms: float = 0.0
    adaptation: Dict[str, np.ndarray] = field(default_factory=dict)

    # ------------------------------------------------------------------ #
    def metrics(self, *, steady_from_lap: int = 2) -> Dict[str, float]:
        """Summary metrics of the run.

        Parameters
        ----------
        steady_from_lap:
            1-based lap index from which the steady-state metrics are computed.
            Lap 1 is excluded because it contains the adaptation warm-up (and,
            for fairness, the same exclusion is applied to the nominal run).
        """
        errors = np.asarray(self.lateral_errors, dtype=np.float64)
        speeds = np.asarray(self.speeds, dtype=np.float64)
        steady_laps = [value for index, value in enumerate(self.lap_rmse) if index + 1 >= steady_from_lap]
        steady_errors = errors[np.asarray(self.lap_ids) >= steady_from_lap] if errors.size else errors
        online_tail = self.online_rmse[-1] if self.online_rmse.size else float("nan")
        return {
            "laps_completed": float(len(self.lap_times)),
            "lap_time_mean": _safe_mean(self.lap_times),
            "lap_time_best": float(np.min(self.lap_times)) if self.lap_times else float("nan"),
            "lateral_rmse": float(np.sqrt(np.mean(errors ** 2))) if errors.size else float("nan"),
            "lateral_mae": float(np.mean(np.abs(errors))) if errors.size else float("nan"),
            "lateral_max": float(np.max(np.abs(errors))) if errors.size else float("nan"),
            "steady_lateral_rmse": float(np.sqrt(np.mean(np.asarray(steady_errors) ** 2)))
            if np.size(steady_errors) else float("nan"),
            "steady_lap_rmse": _safe_mean(steady_laps),
            "mean_speed": _safe_mean(speeds),
            "offtrack_events": float(self.offtrack_events),
            "online_rmse_final": float(online_tail),
            "plan_ms": float(self.plan_ms),
            "wall_time_s": float(self.wall_time_s),
        }

    def summary(self) -> str:
        """Formatted multi-line summary used in the console report."""
        metrics = self.metrics()
        lines = [f"{self.label}:"]
        for key in ("laps_completed", "lap_time_mean", "lap_time_best", "lateral_rmse",
                    "lateral_mae", "lateral_max", "steady_lateral_rmse", "mean_speed",
                    "offtrack_events", "online_rmse_final"):
            value = metrics[key]
            lines.append(f"  {key:>20s}: {value:8.3f}")
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        """Serialisable summary (no per-step arrays)."""
        payload = {"label": self.label, **self.metrics()}
        payload["lap_times"] = [float(v) for v in self.lap_times]
        payload["lap_rmse"] = [float(v) for v in self.lap_rmse]
        return payload


# --------------------------------------------------------------------------- #
# Environment gap report
# --------------------------------------------------------------------------- #


def describe_gap(sim_config: EnvConfig, real_config: EnvConfig) -> str:
    """Human readable table of the sim-to-real mismatches."""
    rows = [
        ("friction coefficient mu", f"{sim_config.mu:.2f}", f"{real_config.mu:.2f}"),
        ("steering lag tau_s [s]", f"{sim_config.tire_lag_tau:.3f}", f"{real_config.tire_lag_tau:.3f}"),
        ("yaw-rate lag tau_y [s]", f"{sim_config.yaw_lag_tau:.3f}", f"{real_config.yaw_lag_tau:.3f}"),
        ("wind amplitude [m/s^2]", f"{sim_config.wind_amplitude:.2f}", f"{real_config.wind_amplitude:.2f}"),
        ("wind noise std [m/s^2]", f"{sim_config.wind_noise_std:.2f}", f"{real_config.wind_noise_std:.2f}"),
        ("steering bias [rad]", f"{sim_config.steer_bias:.4f}", f"{real_config.steer_bias:.4f}"),
        ("actuator delay [steps]", f"{sim_config.actuation_delay:d}", f"{real_config.actuation_delay:d}"),
        ("sensor noise std [m]", f"{sim_config.sensor_noise_std:.3f}", f"{real_config.sensor_noise_std:.3f}"),
    ]
    width = max(len(row[0]) for row in rows) + 2
    lines = [f"{'quantity':<{width}s}{'SimEnv (prior)':>16s}{'RealEnv (track)':>18s}"]
    lines.append("-" * (width + 34))
    for name, sim, real in rows:
        lines.append(f"{name:<{width}s}{sim:>16s}{real:>18s}")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Stage 1 - offline data
# --------------------------------------------------------------------------- #


def stage_generate_dataset(
    args: argparse.Namespace,
    track: Track,
    outdir: Path,
) -> PriorDataset:
    """Collect the offline prior dataset in the nominal simulator."""
    banner("STAGE 1/5 - Offline prior data collection (SimEnv)")
    cache = outdir / "prior_dataset.pt"
    if args.reuse_dataset and cache.exists():
        dataset = PriorDataset.load(cache)
        log(f"reused cached dataset: {len(dataset)} transitions ({cache})")
        return dataset

    env = SimEnv(track, seed=args.seed)
    controller = StanleyController(track, seed=args.seed + 1)
    started = time.perf_counter()
    dataset = generate_prior_dataset(
        env,
        controller,
        episodes=args.episodes,
        steps_per_episode=args.steps_per_episode,
        seed=args.seed,
        verbose=True,
    )
    path = dataset.save(cache)
    stats = dataset.statistics()
    log(
        f"collected {len(dataset)} transitions from {stats['num_episodes']} episodes in "
        f"{time.perf_counter() - started:.1f}s -> {path}"
    )
    log(f"speed coverage: {stats['speed_min']:.1f} .. {stats['speed_max']:.1f} m/s")
    return dataset


# --------------------------------------------------------------------------- #
# Stage 2 - world model
# --------------------------------------------------------------------------- #


def stage_train_world_model(
    args: argparse.Namespace,
    dataset: PriorDataset,
    outdir: Path,
    device: torch.device,
) -> Tuple[ProbabilisticEnsemble, TrainHistory]:
    """Train the probabilistic ensemble on the offline priors."""
    banner("STAGE 2/5 - Offline world-model training")
    model = ProbabilisticEnsemble(
        n_members=args.members,
        hidden_dims=(args.hidden, args.hidden),
        min_logvar=-8.0,
        max_logvar=1.0,
        learning_rate=2e-3,
        feature_mode="invariant",
        seed=args.seed,
    )
    log(model.describe())
    started = time.perf_counter()
    history = model.fit(
        dataset,
        epochs=args.epochs,
        batch_size=256,
        device=device,
        val_fraction=0.15,
        seed=args.seed,
        verbose=True,
    )
    model.save(outdir / "world_model.pt", extra={"history": history.to_dict()})
    log(history.summary())
    log(f"validation one-step position RMSE: {history.extra['position_rmse']:.4f} m")
    log(f"trained in {time.perf_counter() - started:.1f}s")
    return model, history


# --------------------------------------------------------------------------- #
# Stage 3 / 4 - closed-loop benchmark
# --------------------------------------------------------------------------- #


def _probe_rmse(
    model: ProbabilisticEnsemble,
    states: torch.Tensor,
    actions: torch.Tensor,
    deltas: torch.Tensor,
    device: torch.device,
) -> float:
    """One-step RMSE of the ensemble mean on a probe batch."""
    with torch.no_grad():
        mu, _var = model.predict_members(states.to(device), actions.to(device), members=[0])
        error = mu[0] - deltas.to(device)
        return float(torch.sqrt((error ** 2).mean()).item())


def run_benchmark(
    env: RealEnv,
    planner: MPCPlanner,
    *,
    label: str,
    laps: int,
    adapter: Optional[PIAdaptationEngine] = None,
    probe_size: int = 128,
    max_steps: Optional[int] = None,
) -> RunLog:
    """Run the closed-loop benchmark and record everything needed for plots.

    The controller observes a noisy state (``env.observe``) but the PIA engine
    learns from the ground-truth transition, mimicking a state estimator that
    is available on the real vehicle.

    Parameters
    ----------
    env:
        Real environment instance (reset by this function).
    planner:
        MPPI planner whose ``world_model`` is either the prior or the adapted
        ensemble.
    label:
        Name of the run (used in plots and reports).
    laps:
        Number of completed laps after which the run stops.
    adapter:
        Optional PIA engine.  When provided, every transition is fed to the
        engine and ``maybe_update`` is called after every step.
    probe_size:
        Number of live transitions used for the online one-step RMSE curve.
    max_steps:
        Safety cap on the number of control steps.  Defaults to
        ``laps * steps_per_lap_budget`` where the budget allows a very slow lap.
    """
    step_budget = int(max_steps) if max_steps is not None else int(laps) * 2400
    started = time.perf_counter()
    env.reset(progress=0.0, speed=min(9.0, planner.config.target_speed * 0.8))
    planner.reset()

    times: List[float] = []
    states: List[np.ndarray] = []
    actions: List[np.ndarray] = []
    errors: List[float] = []
    speeds: List[float] = []
    uncertainties: List[float] = []
    lap_ids: List[int] = []
    online_rmse: List[float] = []
    plan_times: List[float] = []

    probe_states: List[np.ndarray] = []
    probe_actions: List[np.ndarray] = []
    probe_deltas: List[np.ndarray] = []
    probe_tensors: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None

    lap_sq_error: List[float] = []
    current_lap = 0
    lap_times: List[float] = []
    lap_rmse: List[float] = []
    offtrack_events = 0
    stuck_steps = 0

    log(f"running '{label}' for {laps} lap(s) ...")
    steps = 0
    while len(lap_times) < laps and steps < step_budget:
        observation = env.observe()
        result = planner.plan(observation)
        state_before = env.state.copy()
        next_state, _reward, done, info = env.step(result.action)
        steps += 1

        # --- record ------------------------------------------------------- #
        times.append(env.time)
        states.append(next_state.copy())
        actions.append(result.action.copy())
        errors.append(info["lateral_error"])
        speeds.append(info["speed"])
        uncertainties.append(float(result.predicted_uncertainty[0]) if result.predicted_uncertainty.size else 0.0)
        lap_ids.append(int(info["lap"]))
        plan_times.append(result.evaluation_ms)
        lap_sq_error.append(info["lateral_error"] ** 2)

        delta = next_state - state_before
        delta[3] = (delta[3] + math.pi) % (2.0 * math.pi) - math.pi

        # --- PIA sees every live transition -------------------------------- #
        if adapter is not None:
            adapter.observe(state_before, result.action, next_state)
            adapter.maybe_update()

            # Adapt the *friction prior* as well: the reference speed profile
            # is an explicit input of the planner, so residual adaptation
            # alone can never correct an optimistic grip assumption.
            if adapter.config.grip_adaptation and steps % 20 == 0:
                mu_estimate = adapter.estimate_grip(prior_mu=planner.config.assumed_mu)
                if abs(mu_estimate - planner.assumed_mu) > 0.02:
                    planner.set_assumed_mu(mu_estimate)
                    log(f"  friction prior updated: mu={mu_estimate:.3f} (reference rebuilt)")

        # --- fixed probe for the online prediction-error curve ------------- #
        if probe_tensors is None:
            if len(probe_states) < probe_size:
                probe_states.append(state_before.copy())
                probe_actions.append(result.action.copy())
                probe_deltas.append(delta.copy())
                if len(probe_states) == probe_size:
                    probe_tensors = (
                        torch.as_tensor(np.asarray(probe_states), dtype=torch.float32),
                        torch.as_tensor(np.asarray(probe_actions), dtype=torch.float32),
                        torch.as_tensor(np.asarray(probe_deltas), dtype=torch.float32),
                    )
            online_rmse.append(float("nan"))
        else:
            online_rmse.append(_probe_rmse(planner.model, *probe_tensors, planner.device))

        # --- lap bookkeeping ------------------------------------------------ #
        if info["lap"] > current_lap:
            current_lap = int(info["lap"])
            if info["lap_time"] is not None:
                lap_times.append(float(info["lap_time"]))
            lap_rmse.append(float(np.sqrt(np.mean(lap_sq_error))))
            lap_sq_error = []
            log(
                f"  lap {current_lap} completed in {lap_times[-1]:.2f}s | "
                f"lateral RMSE {lap_rmse[-1]:.3f} m | v_mean {np.mean(speeds[-600:]):.2f} m/s"
            )

        # --- off-track recovery -------------------------------------------- #
        if done and len(lap_times) < laps:
            offtrack_events += 1
            proj = env.track.project(env.state[:2])
            env.recover(progress=proj.progress, speed=max(6.0, 0.4 * env.speed))
            # The MPPI mean and sampling covariance were built around the state
            # that just left the track; without a reset the collapsed sampling
            # distribution makes the planner unable to recover.
            planner.reset()
            stuck_steps = 0
            log(f"  off-track at t={env.time:.2f}s -> recovered at s={proj.progress:.1f} m")

        # --- standstill guard ------------------------------------------------ #
        # If the planner ever parks the car (e.g. after a spin) the learned
        # model is outside its training distribution; restart the lap segment
        # at the minimum modelled speed instead of idling forever.
        if env.speed < 0.5:
            stuck_steps += 1
            if stuck_steps == 80:
                proj = env.track.project(env.state[:2])
                env.recover(progress=proj.progress, speed=6.0)
                planner.reset()
                log(f"  standstill detected at t={env.time:.2f}s -> restarting at 6 m/s")
        else:
            stuck_steps = 0

        if steps % 250 == 0:
            log(
                f"  step {steps:4d} | lap {env.lap_count} | s={env.total_progress:6.1f} m | "
                f"v={env.speed:5.2f} m/s | e={info['lateral_error']:+.2f} m"
            )

    wall = time.perf_counter() - started
    if len(lap_times) < laps:
        # Record the partial lap so that the metrics still make sense.
        if lap_sq_error:
            lap_rmse.append(float(np.sqrt(np.mean(lap_sq_error))))

    run = RunLog(
        label=label,
        time=np.asarray(times, dtype=np.float64),
        states=np.asarray(states, dtype=np.float64) if states else np.zeros((0, 5)),
        actions=np.asarray(actions, dtype=np.float64) if actions else np.zeros((0, 2)),
        lateral_errors=np.asarray(errors, dtype=np.float64),
        speeds=np.asarray(speeds, dtype=np.float64),
        uncertainties=np.asarray(uncertainties, dtype=np.float64),
        online_rmse=np.asarray(online_rmse, dtype=np.float64),
        lap_ids=np.asarray(lap_ids, dtype=np.int64),
        lap_times=lap_times,
        lap_rmse=lap_rmse,
        offtrack_events=offtrack_events,
        wall_time_s=wall,
        plan_ms=_safe_mean(plan_times),
        adaptation=adapter.history_arrays() if adapter is not None else {},
    )
    log(
        f"'{label}' finished: {len(lap_times)} lap(s), {len(errors)} steps, "
        f"lateral RMSE {run.metrics()['lateral_rmse']:.3f} m, {wall:.1f}s wall-clock"
    )
    return run


# --------------------------------------------------------------------------- #
# Stage 5 - figures
# --------------------------------------------------------------------------- #


def _plot_track(ax: Any, track: Track, nominal: RunLog, adapted: Optional[RunLog]) -> None:
    """Draw the track and the driven trajectories."""
    render = track.render_data()
    ax.plot(render["left"][:, 0], render["left"][:, 1], color=TRACK_COLOR, lw=1.2, label="track boundary")
    ax.plot(render["right"][:, 0], render["right"][:, 1], color=TRACK_COLOR, lw=1.2)
    ax.plot(render["centerline"][:, 0], render["centerline"][:, 1], "--", color="#6c757d", lw=0.9,
            label="centre line")

    if nominal.states.size:
        ax.plot(nominal.states[:, 0], nominal.states[:, 1], color=NOMINAL_COLOR, lw=1.6,
                label=f"{nominal.label} (no adaptation)")
    if adapted is not None and adapted.states.size:
        ax.plot(adapted.states[:, 0], adapted.states[:, 1], color=PIA_COLOR, lw=1.6,
                label=f"{adapted.label} (PIA)")

    start = render["centerline"][0]
    ax.plot(start[0], start[1], marker="o", ms=7, color="black", label="start")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title("Trajectory tracking")
    ax.legend(loc="upper right", fontsize=7, framealpha=0.9)
    ax.grid(True, color=GRID_COLOR, lw=0.6)


def _plot_lateral_error(ax: Any, nominal: RunLog, adapted: Optional[RunLog]) -> None:
    ax.plot(nominal.time, nominal.lateral_errors, color=NOMINAL_COLOR, lw=0.6, alpha=0.35)
    ax.plot(nominal.time, _rolling_mean(nominal.lateral_errors, 25), color=NOMINAL_COLOR, lw=1.8,
            label=nominal.label)
    if adapted is not None:
        ax.plot(adapted.time, adapted.lateral_errors, color=PIA_COLOR, lw=0.6, alpha=0.35)
        ax.plot(adapted.time, _rolling_mean(adapted.lateral_errors, 25), color=PIA_COLOR, lw=1.8,
                label=adapted.label)
    ax.axhline(0.0, color="black", lw=0.6)
    ax.set_xlabel("time [s]")
    ax.set_ylabel("lateral error [m]")
    ax.set_title("Tracking error over time (25-sample rolling mean)")
    ax.legend(fontsize=8)
    ax.grid(True, color=GRID_COLOR, lw=0.6)


def _plot_speed(ax: Any, nominal: RunLog, adapted: Optional[RunLog], target_speed: float) -> None:
    ax.plot(nominal.time, _rolling_mean(nominal.speeds, 15), color=NOMINAL_COLOR, lw=1.6, label=nominal.label)
    if adapted is not None:
        ax.plot(adapted.time, _rolling_mean(adapted.speeds, 15), color=PIA_COLOR, lw=1.6, label=adapted.label)
    ax.axhline(target_speed, ls=":", color="#6c757d", lw=1.0, label="target speed")
    ax.set_xlabel("time [s]")
    ax.set_ylabel("speed [m/s]")
    ax.set_title("Speed profile")
    ax.legend(fontsize=8)
    ax.grid(True, color=GRID_COLOR, lw=0.6)


def _plot_lap_times(ax: Any, nominal: RunLog, adapted: Optional[RunLog]) -> None:
    max_laps = max(len(nominal.lap_times), len(adapted.lap_times) if adapted else 0)
    indices = np.arange(1, max_laps + 1)
    width = 0.38
    ax.bar(indices - width / 2, nominal.lap_times + [float("nan")] * (max_laps - len(nominal.lap_times)),
           width=width, color=NOMINAL_COLOR, label=nominal.label)
    if adapted is not None:
        ax.bar(indices + width / 2, adapted.lap_times + [float("nan")] * (max_laps - len(adapted.lap_times)),
               width=width, color=PIA_COLOR, label=adapted.label)
    ax.set_xlabel("lap")
    ax.set_ylabel("lap time [s]")
    ax.set_title("Lap times")
    ax.set_xticks(indices)
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", color=GRID_COLOR, lw=0.6)


def _plot_training(ax: Any, history: TrainHistory) -> None:
    train = np.asarray(history.train_nll, dtype=np.float64)
    val = np.asarray(history.val_nll, dtype=np.float64)
    epochs = np.arange(1, train.shape[0] + 1)
    ax.plot(epochs, train.mean(axis=1), color="#264653", lw=1.8, label="train NLL")
    ax.fill_between(epochs, train.min(axis=1), train.max(axis=1), color="#264653", alpha=0.15)
    ax.plot(epochs, val.mean(axis=1), color="#e76f51", lw=1.8, label="validation NLL")
    ax.fill_between(epochs, val.min(axis=1), val.max(axis=1), color="#e76f51", alpha=0.15)
    ax.set_xlabel("epoch")
    ax.set_ylabel("Gaussian NLL (normalised)")
    ax.set_title(f"World-model training ({train.shape[1]} members)")
    ax.legend(fontsize=8)
    ax.grid(True, color=GRID_COLOR, lw=0.6)


def _plot_adaptation(ax: Any, nominal: RunLog, adapted: Optional[RunLog]) -> None:
    if adapted is not None and adapted.adaptation:
        telemetry = adapted.adaptation
        observation = telemetry["observation"]
        ax.plot(observation, telemetry["prior_rmse"], color=NOMINAL_COLOR, lw=1.6,
                label="prior model on live data")
        ax.plot(observation, telemetry["buffer_rmse"], color=PIA_COLOR, lw=1.8,
                label="adapted model on live data")
        ax.set_xlabel("live observation")
        ax.set_ylabel("one-step RMSE [m]")
        ax.set_title("PIA adaptation: live prediction error")
        twin = ax.twinx()
        twin.plot(observation, telemetry["kl"], color="#6a4c93", lw=1.2, ls="--", label="KL anchor")
        twin.set_ylabel("KL divergence", color="#6a4c93")
        twin.tick_params(axis="y", colors="#6a4c93")
        lines, labels = ax.get_legend_handles_labels()
        lines2, labels2 = twin.get_legend_handles_labels()
        ax.legend(lines + lines2, labels + labels2, fontsize=7, loc="upper right")
    else:
        ax.plot(nominal.time, nominal.online_rmse, color=NOMINAL_COLOR, lw=1.6, label=nominal.label)
        if adapted is not None:
            ax.plot(adapted.time, adapted.online_rmse, color=PIA_COLOR, lw=1.6, label=adapted.label)
        ax.set_xlabel("time [s]")
        ax.set_ylabel("one-step RMSE [m]")
        ax.set_title("Online one-step prediction error")
        ax.legend(fontsize=8)
    ax.grid(True, color=GRID_COLOR, lw=0.6)


def plot_results(
    nominal: RunLog,
    adapted: Optional[RunLog],
    history: TrainHistory,
    track: Track,
    target_speed: float,
    outdir: Path,
    *,
    show: bool = True,
) -> List[Path]:
    """Create the evaluation figures and return their paths."""
    import matplotlib

    if not show:
        matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    matplotlib.rcParams.update(
        {
            "figure.dpi": 110,
            "savefig.dpi": 170,
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.titleweight": "bold",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "savefig.bbox": "tight",
        }
    )

    outdir.mkdir(parents=True, exist_ok=True)
    paths: List[Path] = []

    figure, axes = plt.subplots(2, 3, figsize=(16.5, 9.4))
    _plot_track(axes[0, 0], track, nominal, adapted)
    _plot_lateral_error(axes[0, 1], nominal, adapted)
    _plot_speed(axes[0, 2], nominal, adapted, target_speed)
    _plot_lap_times(axes[1, 0], nominal, adapted)
    _plot_training(axes[1, 1], history)
    _plot_adaptation(axes[1, 2], nominal, adapted)

    figure.suptitle(
        "Prior-Informed Adaptation (PIA) - sim-to-real benchmark on a learned ensemble world model",
        fontsize=13,
        fontweight="bold",
    )
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
    path = outdir / "pia_evaluation.png"
    figure.savefig(path)
    paths.append(path)
    log(f"figure written to {path}")

    # ------------------------------------------------------------------ #
    # Second figure: aggregate metrics + table
    # ------------------------------------------------------------------ #
    nominal_metrics = nominal.metrics()
    adapted_metrics = adapted.metrics() if adapted is not None else None

    figure2, (ax_bar, ax_table) = plt.subplots(1, 2, figsize=(13.5, 4.6), gridspec_kw={"width_ratios": [1.1, 1.4]})
    keys = ["lateral_rmse", "lateral_mae", "steady_lateral_rmse"]
    labels = ["RMSE", "MAE", "steady RMSE\n(laps 2+)"]
    x = np.arange(len(keys))
    width = 0.38
    ax_bar.bar(x - width / 2, [nominal_metrics[k] for k in keys], width, color=NOMINAL_COLOR, label=nominal.label)
    if adapted_metrics is not None:
        ax_bar.bar(x + width / 2, [adapted_metrics[k] for k in keys], width, color=PIA_COLOR, label=adapted.label)
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(labels)
    ax_bar.set_ylabel("tracking error [m]")
    ax_bar.set_title("Mean tracking error")
    ax_bar.legend(fontsize=8)
    ax_bar.grid(True, axis="y", color=GRID_COLOR, lw=0.6)

    ax_table.axis("off")
    rows: List[List[str]] = []
    row_labels: List[str] = []
    columns = ["metric", nominal.label] + ([adapted.label] if adapted_metrics else [])
    display = [
        ("laps completed", "laps_completed", "{:.0f}"),
        ("mean lap time [s]", "lap_time_mean", "{:.2f}"),
        ("best lap time [s]", "lap_time_best", "{:.2f}"),
        ("lateral RMSE [m]", "lateral_rmse", "{:.3f}"),
        ("steady RMSE [m]", "steady_lateral_rmse", "{:.3f}"),
        ("max |e| [m]", "lateral_max", "{:.3f}"),
        ("mean speed [m/s]", "mean_speed", "{:.2f}"),
        ("online RMSE [m]", "online_rmse_final", "{:.3f}"),
        ("off-track events", "offtrack_events", "{:.0f}"),
    ]
    for title, key, fmt in display:
        row = [title, fmt.format(nominal_metrics[key])]
        if adapted_metrics is not None:
            row.append(fmt.format(adapted_metrics[key]))
        rows.append(row)
    table = ax_table.table(cellText=rows, colLabels=columns, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.45)
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor("#264653")
            cell.set_text_props(color="white", fontweight="bold")
        elif col == 0:
            cell.set_text_props(ha="left")
    ax_table.set_title("Benchmark summary", fontweight="bold")

    figure2.tight_layout()
    path2 = outdir / "pia_metrics.png"
    figure2.savefig(path2)
    paths.append(path2)
    log(f"figure written to {path2}")

    if show:
        plt.show()
    else:
        plt.close("all")
    return paths


# --------------------------------------------------------------------------- #
# Argument parsing
# --------------------------------------------------------------------------- #


def build_parser() -> argparse.ArgumentParser:
    """Command-line interface of the full pipeline."""
    parser = argparse.ArgumentParser(
        description="Prior-Informed Adaptation (PIA) for autonomous racing - full pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Data / model
    parser.add_argument("--episodes", type=int, default=48, help="offline episodes to collect")
    parser.add_argument("--steps-per-episode", type=int, default=400, help="max steps per offline episode")
    parser.add_argument("--epochs", type=int, default=20, help="world-model training epochs")
    parser.add_argument("--members", type=int, default=4, help="ensemble members")
    parser.add_argument("--hidden", type=int, default=192, help="hidden layer width")
    # Benchmark
    parser.add_argument("--laps-nominal", type=int, default=3, help="laps of the un-adapted run")
    parser.add_argument("--laps-pia", type=int, default=4, help="laps of the PIA run")
    parser.add_argument("--horizon", type=int, default=10, help="MPPI horizon H")
    parser.add_argument("--samples", type=int, default=64, help="MPPI samples per iteration")
    parser.add_argument("--iterations", type=int, default=2, help="MPPI iterations")
    parser.add_argument("--target-speed", type=float, default=13.5, help="reference speed cap [m/s]")
    # Infrastructure
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--outdir", type=str, default="artifacts")
    parser.add_argument("--reuse-dataset", action="store_true", help="reuse cached prior dataset")
    parser.add_argument("--no-show", action="store_true", help="do not open matplotlib windows")
    parser.add_argument("--quick", action="store_true", help="fast smoke-test configuration")
    return parser


def apply_quick_preset(args: argparse.Namespace) -> argparse.Namespace:
    """Shorten the benchmark while keeping the full prior dataset and model.

    The quick preset keeps the expensive *learning* stages intact (a reduced
    model was not representative enough to be useful) and only shortens the
    closed-loop evaluation and the MPPI sample count, which is where most of
    the wall-clock time goes.
    """
    args.laps_nominal = 2
    args.laps_pia = 3
    args.samples = 48
    args.horizon = 10
    return args


def resolve_device(name: str) -> torch.device:
    """Resolve ``auto`` / ``cpu`` / ``cuda`` into a torch device."""
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


# --------------------------------------------------------------------------- #
# Pipeline
# --------------------------------------------------------------------------- #


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Run the complete five-stage PIA pipeline."""
    args = build_parser().parse_args(argv)
    if args.quick:
        args = apply_quick_preset(args)

    device = resolve_device(args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    banner("PRIOR-INFORMED ADAPTATION (PIA) - SIM-TO-REAL BENCHMARK")
    log(f"torch {torch.__version__} | device {device} | seed {args.seed}")
    if args.quick:
        log("quick preset active: reduced dataset, model and lap counts")

    track = build_track()
    log(f"track: {track.length:.1f} m, half width {track.half_width:.2f} m, "
        f"{track.num_points} samples, curvature max {track.curvature.max():.4f} 1/m")

    sim_probe = SimEnv(track, seed=args.seed)
    real_probe = RealEnv(track, seed=args.seed)
    print()
    print(describe_gap(sim_probe.config, real_probe.config))
    print()

    # ------------------------------------------------------------------ #
    # Stage 1
    # ------------------------------------------------------------------ #
    dataset = stage_generate_dataset(args, track, outdir)

    # ------------------------------------------------------------------ #
    # Stage 2
    # ------------------------------------------------------------------ #
    model, history = stage_train_world_model(args, dataset, outdir, device)

    planner_config = MPCConfig(
        horizon=args.horizon,
        num_samples=args.samples,
        num_iterations=args.iterations,
        smoothing=0.0,
        temperature=0.6,
        target_speed=args.target_speed,
        assumed_mu=1.0,
        w_lateral=8.0,
        w_heading=1.5,
        w_speed=0.30,
        w_uncertainty=6.0,
        w_steer_rate=1.5,
        w_accel=0.02,
        w_terminal_lateral=15.0,
        seed=args.seed,
    )

    # ------------------------------------------------------------------ #
    # Stage 3 - nominal deployment (no adaptation)
    # ------------------------------------------------------------------ #
    banner("STAGE 3/5 - Deployment WITHOUT adaptation (sim-to-real gap)")
    nominal_env = RealEnv(track, seed=args.seed + 100)
    nominal_planner = MPCPlanner(model, track, planner_config, device=device, seed=args.seed)
    nominal_log = run_benchmark(
        nominal_env,
        nominal_planner,
        label="Nominal",
        laps=args.laps_nominal,
    )
    print()
    print(nominal_log.summary())

    # ------------------------------------------------------------------ #
    # Stage 4 - PIA deployment
    # ------------------------------------------------------------------ #
    banner("STAGE 4/5 - Deployment WITH Prior-Informed Adaptation")
    adapter = PIAdaptationEngine(
        model,
        PIAConfig(
            capacity=1024,
            warmup=64,
            minibatch=64,
            update_every=4,
            gradient_steps=2,
            learning_rate=1e-3,
            prior_kl_weight=2.0,
            kl_target=0.5,
            max_kl=2.0,
            prior_l2_weight=1e-2,
            train_last_hidden=False,
            max_param_drift=0.02,
            buffer_alpha=0.6,
            eval_every=20,
        ),
        prior_dataset=dataset,
        device=device,
        seed=args.seed,
    )
    log(adapter.describe())

    pia_env = RealEnv(track, seed=args.seed + 100)
    pia_planner = MPCPlanner(adapter.model, track, planner_config, device=device, seed=args.seed)
    pia_log = run_benchmark(
        pia_env,
        pia_planner,
        label="PIA",
        laps=args.laps_pia,
        adapter=adapter,
    )
    metrics = adapter.prediction_metrics()
    log(
        f"live-data fit: prior RMSE {metrics['prior_rmse']:.4f} m -> adapted "
        f"{metrics['buffer_rmse']:.4f} m ({metrics['improvement'] * 100:+.1f}%)"
    )
    print()
    print(pia_log.summary())

    # ------------------------------------------------------------------ #
    # Stage 5 - evaluation
    # ------------------------------------------------------------------ #
    banner("STAGE 5/5 - Evaluation and figures")
    paths = plot_results(
        nominal_log,
        pia_log,
        history,
        track,
        args.target_speed,
        outdir,
        show=not args.no_show,
    )

    nominal_metrics = nominal_log.metrics()
    pia_metrics = pia_log.metrics()

    comparison = {
        "config": {
            "seed": args.seed,
            "device": str(device),
            "track_length": float(track.length),
            "target_speed": float(args.target_speed),
            "horizon": args.horizon,
            "samples": args.samples,
            "iterations": args.iterations,
            "sim_env": sim_probe.config.describe(),
            "real_env": real_probe.config.describe(),
            "model": model.config_dict(),
        },
        "dataset": dataset.statistics(),
        "training": history.to_dict(),
        "nominal": nominal_log.to_dict(),
        "pia": pia_log.to_dict(),
        "adaptation": {
            "observations": adapter.observations,
            "updates": adapter.updates,
            "rejected_steps": adapter.rejected_updates,
            "live_improvement_pct": float(metrics["improvement"] * 100.0),
            "kl_final": float(metrics["kl"]),
            "param_drift": float(metrics["param_drift"]),
        },
        "figures": [str(path) for path in paths],
    }
    report_path = outdir / "metrics.json"
    with open(report_path, "w", encoding="utf-8") as handle:
        json.dump(comparison, handle, indent=2)
    log(f"metrics written to {report_path}")

    # ------------------------------------------------------------------ #
    # Console table
    # ------------------------------------------------------------------ #
    print()
    header = f"{'metric':<26s}{'Nominal':>14s}{'PIA':>14s}"
    print("=" * len(header))
    print(header)
    print("-" * len(header))
    rows = [
        ("laps completed", "laps_completed", "{:.0f}"),
        ("mean lap time [s]", "lap_time_mean", "{:.2f}"),
        ("best lap time [s]", "lap_time_best", "{:.2f}"),
        ("lateral RMSE [m]", "lateral_rmse", "{:.3f}"),
        ("lateral MAE [m]", "lateral_mae", "{:.3f}"),
        ("steady RMSE [m]", "steady_lateral_rmse", "{:.3f}"),
        ("max |e| [m]", "lateral_max", "{:.3f}"),
        ("mean speed [m/s]", "mean_speed", "{:.2f}"),
        ("online pred. RMSE [m]", "online_rmse_final", "{:.3f}"),
        ("off-track events", "offtrack_events", "{:.0f}"),
        ("planning time [ms]", "plan_ms", "{:.1f}"),
        ("wall time [s]", "wall_time_s", "{:.1f}"),
    ]
    for title, key, fmt in rows:
        print(f"{title:<26s}{fmt.format(nominal_metrics[key]):>14s}{fmt.format(pia_metrics[key]):>14s}")
    print("=" * len(header))

    delta_rmse = nominal_metrics["lateral_rmse"] - pia_metrics["lateral_rmse"]
    relative = 100.0 * delta_rmse / max(nominal_metrics["lateral_rmse"], 1e-9)
    log(
        f"PIA reduced the lateral RMSE by {delta_rmse:+.3f} m ({relative:+.1f}%) and the "
        f"one-step prediction error by "
        f"{nominal_metrics['online_rmse_final'] - pia_metrics['online_rmse_final']:+.3f} m"
    )
    log(f"figures: {', '.join(str(path) for path in paths)}")
    log("pipeline complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

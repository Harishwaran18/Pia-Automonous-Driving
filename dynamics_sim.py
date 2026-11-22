"""
dynamics_sim.py
===============
Ground-truth vehicle dynamics and track simulation for the **Prior-Informed
Adaptation (PIA)** autonomous-racing benchmark.

The module provides two concrete environments that share one physics kernel:

``SimEnv``
    The *nominal* simulator.  High grip (``mu = 1.0``), fast steering actuator,
    no wind, no sensor noise.  This is the world in which the offline prior
    dataset is generated, i.e. the world the probabilistic world model is
    trained on.

``RealEnv``
    The *physical* environment used for evaluation.  It keeps the same track
    geometry but changes the true plant: reduced tyre friction (``mu = 0.55``),
    steering / yaw-rate first-order lags, a slowly varying cross-wind
    disturbance, a constant steering bias, one-step actuator delay and noisy
    observations.  None of these parameters are visible to the controller.

Both environments expose a five dimensional state::

    s = [x, y, v, heading, yaw_rate]

and a two dimensional action::

    a = [acceleration, steering_angle]

The continuous-time dynamics are integrated with an explicit Euler step at
``dt`` (default ``20 Hz``).  The model is a *dynamic* bicycle model in the sense
that yaw rate is a state with its own first-order dynamics, the lateral tyre
force is saturated by the friction circle and the longitudinal force is limited
by the remaining friction ellipse budget.

Everything is pure NumPy so that the environment can be stepped thousands of
times per second without a simulator dependency.

Author: PIA reference implementation.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

STATE_DIM: int = 5
ACTION_DIM: int = 2

STATE_LABELS: Tuple[str, ...] = ("x", "y", "v", "heading", "yaw_rate")
ACTION_LABELS: Tuple[str, ...] = ("acceleration", "steering")

IDX_X: int = 0
IDX_Y: int = 1
IDX_V: int = 2
IDX_HEADING: int = 3
IDX_YAW_RATE: int = 4

GRAVITY: float = 9.81


def wrap_angle(angle: Any) -> Any:
    """Wrap an angle (or ``np.ndarray`` of angles) to the interval ``(-pi, pi]``.

    Parameters
    ----------
    angle:
        Scalar or array of angles in radians.

    Returns
    -------
    The wrapped angle with the same shape.  Python floats are returned as
    ``numpy.float64`` scalars.
    """
    return (np.asarray(angle) + np.pi) % (2.0 * np.pi) - np.pi


# --------------------------------------------------------------------------- #
# Track geometry
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class TrackProjection:
    """Result of projecting a 2-D position onto the track centre line.

    Attributes
    ----------
    index:
        Index of the closest centre-line sample.
    point:
        Cartesian coordinates of the projected point (``(2,)``).
    tangent:
        Unit tangent of the centre line at the projection (``(2,)``).
    normal:
        Unit left normal of the centre line at the projection (``(2,)``).
    heading:
        Tangent heading angle in radians.
    signed_lateral_error:
        Signed distance from the centre line.  Positive when the vehicle is to
        the *left* of the racing direction, negative when to the right.
    progress:
        Arc length along the track in ``[0, length)``.
    curvature:
        Local curvature (``1/radius``) in ``1/m``.
    """

    index: int
    point: np.ndarray
    tangent: np.ndarray
    normal: np.ndarray
    heading: float
    signed_lateral_error: float
    progress: float
    curvature: float


class Track:
    """Closed (or open) 2-D racing track built from a dense polyline.

    The polyline is resampled at a uniform arc-length spacing, which makes the
    projection, curvature and progress queries cheap and accurate.  The track
    is only a geometric object: it knows nothing about vehicles.

    Parameters
    ----------
    centerline:
        ``(N, 2)`` array of centre-line waypoints in order of travel.
    half_width:
        Half width of the drivable corridor in metres.
    closed:
        Whether the last waypoint connects back to the first.
    resolution:
        Arc-length spacing used for resampling, in metres.
    name:
        Human readable identifier used in logs and plots.
    """

    def __init__(
        self,
        centerline: np.ndarray,
        half_width: float,
        *,
        closed: bool = True,
        resolution: float = 0.20,
        name: str = "track",
    ) -> None:
        points = np.asarray(centerline, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 2:
            raise ValueError("centerline must have shape (N, 2)")
        if points.shape[0] < 8:
            raise ValueError("centerline needs at least 8 waypoints")

        self.name = name
        self.half_width = float(half_width)
        self.closed = bool(closed)
        self.resolution = float(resolution)

        self.centerline: np.ndarray = self._resample(points, self.resolution, closed)
        self.num_points: int = int(self.centerline.shape[0])

        # --- arc length ---------------------------------------------------- #
        if self.closed:
            deltas = np.roll(self.centerline, -1, axis=0) - self.centerline
        else:
            deltas = np.diff(self.centerline, axis=0)
        self.segment_length: np.ndarray = np.linalg.norm(deltas, axis=1)
        self.segment_length = np.maximum(self.segment_length, 1e-9)

        if self.closed:
            s = np.concatenate([[0.0], np.cumsum(self.segment_length)[:-1]])
        else:
            s = np.concatenate([[0.0], np.cumsum(self.segment_length)])[: self.num_points]
        self.s: np.ndarray = s
        self.length: float = float(self.segment_length.sum())
        self.ds: float = self.length / self.num_points

        # --- tangents / normals / headings --------------------------------- #
        self.tangents: np.ndarray = self._finite_difference_tangents(self.centerline, self.closed)
        self.headings: np.ndarray = np.arctan2(self.tangents[:, 1], self.tangents[:, 0])
        self.normals: np.ndarray = np.stack(
            [-self.tangents[:, 1], self.tangents[:, 0]], axis=1
        )

        # --- curvature ------------------------------------------------------ #
        if self.closed:
            dheading = wrap_angle(np.roll(self.headings, -1) - self.headings)
        else:
            dheading = np.concatenate(
                [wrap_angle(np.diff(self.headings)), np.zeros(1)]
            )[: self.num_points]
        curvature = dheading / self.segment_length
        self.curvature: np.ndarray = self._smooth_circular(curvature, window=7)

    # ------------------------------------------------------------------ #
    # Construction helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _resample(points: np.ndarray, resolution: float, closed: bool) -> np.ndarray:
        """Resample a polyline at uniform arc-length spacing."""
        if closed:
            pts = np.vstack([points, points[:1]])
        else:
            pts = points
        seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        cum = np.concatenate([[0.0], np.cumsum(seg)])
        total = float(cum[-1])
        if total < 10.0 * resolution:
            raise ValueError("track polyline is too short for the requested resolution")
        n = max(16, int(round(total / resolution)))
        targets = np.linspace(0.0, total, n, endpoint=False)
        x = np.interp(targets, cum, pts[:, 0])
        y = np.interp(targets, cum, pts[:, 1])
        return np.stack([x, y], axis=1)

    @staticmethod
    def _finite_difference_tangents(points: np.ndarray, closed: bool) -> np.ndarray:
        if closed:
            nxt = np.roll(points, -1, axis=0)
            prv = np.roll(points, 1, axis=0)
        else:
            nxt = np.vstack([points[1:], points[-1:]])
            prv = np.vstack([points[:1], points[:-1]])
        tangent = nxt - prv
        norm = np.linalg.norm(tangent, axis=1, keepdims=True)
        return tangent / np.maximum(norm, 1e-12)

    @staticmethod
    def _smooth_circular(values: np.ndarray, window: int = 7) -> np.ndarray:
        if window <= 1 or values.size == 0:
            return values
        kernel = np.ones(window, dtype=np.float64) / float(window)
        padded = np.concatenate([values[-window:], values, values[:window]])
        return np.convolve(padded, kernel, mode="same")[window:-window]

    # ------------------------------------------------------------------ #
    # Factory
    # ------------------------------------------------------------------ #
    @classmethod
    def oval(
        cls,
        *,
        radius: float = 32.0,
        straight_length: float = 56.0,
        half_width: float = 3.5,
        resolution: float = 0.20,
        name: str = "stadium-oval",
    ) -> "Track":
        """Build a stadium-shaped oval track (two straights + two semicircles).

        The racing direction is counter-clockwise.  ``straight_length`` is the
        total length of one straight (i.e. ``2 * a = straight_length``).

        Parameters
        ----------
        radius:
            Radius of the two semicircular ends in metres.
        straight_length:
            Total length of each straight section in metres.
        half_width:
            Half width of the drivable corridor in metres.
        resolution:
            Arc-length spacing of the generated centre line in metres.
        name:
            Track identifier.
        """
        a = 0.5 * float(straight_length)
        r = float(radius)
        segment = 0.10

        # Bottom straight: (-a, -r) -> (a, -r), heading +x.
        n_straight = max(2, int(math.ceil(2.0 * a / segment)))
        xs = np.linspace(-a, a, n_straight, endpoint=False)
        bottom = np.stack([xs, np.full_like(xs, -r)], axis=1)

        # Right semicircle: theta from -pi/2 to +pi/2 around (+a, 0).
        n_arc = max(8, int(math.ceil(math.pi * r / segment)))
        theta = np.linspace(-0.5 * np.pi, 0.5 * np.pi, n_arc, endpoint=False)
        right = np.stack([a + r * np.cos(theta), r * np.sin(theta)], axis=1)

        # Top straight: (a, r) -> (-a, r), heading -x.
        xs_top = np.linspace(a, -a, n_straight, endpoint=False)
        top = np.stack([xs_top, np.full_like(xs_top, r)], axis=1)

        # Left semicircle: theta from pi/2 to 3pi/2 around (-a, 0).
        theta_l = np.linspace(0.5 * np.pi, 1.5 * np.pi, n_arc, endpoint=False)
        left = np.stack([-a + r * np.cos(theta_l), r * np.sin(theta_l)], axis=1)

        raw = np.vstack([bottom, right, top, left])
        return cls(raw, half_width, closed=True, resolution=resolution, name=name)

    # ------------------------------------------------------------------ #
    # Geometry queries
    # ------------------------------------------------------------------ #
    def project(self, position: Sequence[float] | np.ndarray) -> TrackProjection:
        """Project a 2-D point onto the centre line.

        The nearest sample is refined against the two adjacent segments, so the
        returned lateral error is exact up to floating point precision and does
        not suffer from the polyline sampling resolution.
        """
        pos = np.asarray(position, dtype=np.float64).reshape(2)
        diff = self.centerline - pos
        idx = int(np.argmin(np.einsum("ij,ij->i", diff, diff)))

        best_dist2 = math.inf
        best_point = self.centerline[idx]
        best_seg = idx
        best_t = 0.0
        for seg in (idx - 1, idx):
            a = self.centerline[seg % self.num_points]
            b = self.centerline[(seg + 1) % self.num_points]
            ab = b - a
            denom = float(ab @ ab)
            t = 0.0 if denom <= 1e-12 else float(np.clip(((pos - a) @ ab) / denom, 0.0, 1.0))
            closest = a + t * ab
            dist2 = float((pos - closest) @ (pos - closest))
            if dist2 < best_dist2:
                best_dist2 = dist2
                best_point = closest
                best_seg = seg % self.num_points
                best_t = t

        a = self.centerline[best_seg]
        b = self.centerline[(best_seg + 1) % self.num_points]
        ab = b - a
        tangent = ab / max(float(np.linalg.norm(ab)), 1e-12)
        normal = np.array([-tangent[1], tangent[0]])
        offset = pos - best_point
        signed = float(tangent[0] * offset[1] - tangent[1] * offset[0])

        progress = float(self.s[best_seg] + best_t * self.segment_length[best_seg])
        progress = progress % self.length if self.closed else min(progress, self.length)

        return TrackProjection(
            index=best_seg,
            point=best_point,
            tangent=tangent,
            normal=normal,
            heading=float(math.atan2(tangent[1], tangent[0])),
            signed_lateral_error=signed,
            progress=progress,
            curvature=float(self.curvature[best_seg]),
        )

    def sample(self, progress: float) -> Tuple[np.ndarray, np.ndarray, float]:
        """Interpolate the centre line at a given arc length.

        Returns
        -------
        point, tangent, heading
        """
        s = float(progress)
        s = s % self.length if self.closed else float(np.clip(s, 0.0, self.length - 1e-9))
        idx = int(np.searchsorted(self.s, s, side="right") - 1)
        idx = int(np.clip(idx, 0, self.num_points - 1))
        t = (s - self.s[idx]) / self.segment_length[idx]
        t = float(np.clip(t, 0.0, 1.0))
        a = self.centerline[idx]
        b = self.centerline[(idx + 1) % self.num_points]
        point = a + t * (b - a)
        tangent = b - a
        tangent = tangent / max(float(np.linalg.norm(tangent)), 1e-12)
        return point, tangent, float(math.atan2(tangent[1], tangent[0]))

    def boundaries(self, margin: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
        """Return ``(left, right)`` boundary polylines, offset by ``margin``."""
        width = self.half_width + float(margin)
        left = self.centerline + self.normals * width
        right = self.centerline - self.normals * width
        return left, right

    def curvature_speed_limit(
        self,
        mu: float,
        *,
        safety_factor: float = 0.92,
        max_speed: float = 40.0,
        min_speed: float = 4.0,
        smooth_window: int = 9,
        brake_decel: float = 4.0,
        use_braking_envelope: bool = True,
    ) -> np.ndarray:
        """Friction-limited reference speed for every centre-line sample.

        First the local limit ``v = sqrt(mu * g * R)`` is computed (with a
        safety factor) and smoothed.  Then a *backward braking envelope* is
        propagated around the track::

            v[i] = min(v[i], sqrt(v[i+1]^2 + 2 * a_brake * ds))

        so that the profile already decelerates *before* a corner instead of
        only at the corner entry.  This gives the MPC a long-range preview of
        the track curvature without extending its planning horizon.

        Parameters
        ----------
        mu:
            Friction coefficient assumed by the profile.
        safety_factor:
            Multiplier applied to the friction-limited speed.
        max_speed, min_speed:
            Clamps of the profile in m/s.
        smooth_window:
            Moving-average window used to smooth the raw curvature limit.
        brake_decel:
            Deceleration used by the backward envelope in m/s^2.
        use_braking_envelope:
            Set to ``False`` to obtain the raw local limit.
        """
        curvature = np.maximum(np.abs(self.curvature), 1e-4)
        v = np.sqrt(safety_factor * mu * GRAVITY / curvature)
        v = np.clip(v, min_speed, max_speed)
        v = self._smooth_circular(v, window=smooth_window)
        if use_braking_envelope and brake_decel > 0.0:
            v = self._apply_braking_envelope(v, brake_decel)
        return v

    def _apply_braking_envelope(self, speeds: np.ndarray, brake_decel: float) -> np.ndarray:
        """Propagate a physically feasible braking envelope backwards in ``s``."""
        out = speeds.astype(np.float64).copy()
        n = int(out.size)
        if n < 2:
            return out
        passes = 2 if self.closed else 1
        for _ in range(passes):
            for i in range(n - 1, -1, -1):
                nxt = out[(i + 1) % n] if self.closed else out[min(i + 1, n - 1)]
                limit = math.sqrt(max(0.0, nxt * nxt + 2.0 * brake_decel * self.segment_length[i]))
                if out[i] > limit:
                    out[i] = limit
        return out

    def contains(self, position: Sequence[float] | np.ndarray, margin: float = 0.0) -> bool:
        """Whether a point lies inside the drivable corridor (+ optional margin)."""
        proj = self.project(position)
        return abs(proj.signed_lateral_error) <= self.half_width + margin

    def render_data(self) -> Dict[str, np.ndarray]:
        """Arrays convenient for plotting the track."""
        left, right = self.boundaries()
        closed_center = np.vstack([self.centerline, self.centerline[:1]])
        return {
            "centerline": closed_center,
            "left": np.vstack([left, left[:1]]),
            "right": np.vstack([right, right[:1]]),
        }

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"Track(name={self.name!r}, length={self.length:.1f} m, "
            f"half_width={self.half_width:.2f} m, samples={self.num_points})"
        )


# --------------------------------------------------------------------------- #
# Vehicle parameters and environment configuration
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class VehicleParams:
    """Physical parameters of the car (shared by both environments).

    Attributes
    ----------
    wheelbase:
        Distance between front and rear axles in metres.
    max_steer:
        Maximum road-wheel steering angle in radians.
    max_steer_rate:
        Maximum steering rate in rad/s (actuator slew limit).
    max_accel:
        Maximum engine acceleration in m/s^2 (before friction limiting).
    max_decel:
        Maximum service braking deceleration in m/s^2 (negative).
    drag_coefficient:
        Aerodynamic drag coefficient ``c_d`` such that ``a_drag = c_d * v^2``.
    rolling_resistance:
        Rolling resistance coefficient (dimensionless, ``a_roll = c_r * g``).
    gravity:
        Gravitational acceleration in m/s^2.
    """

    wheelbase: float = 2.60
    max_steer: float = 0.50
    max_steer_rate: float = 3.50
    max_accel: float = 6.50
    max_decel: float = -10.0
    drag_coefficient: float = 0.0022
    rolling_resistance: float = 0.015
    gravity: float = GRAVITY


@dataclass
class EnvConfig:
    """Configuration of a :class:`TrackedVehicleEnv`.

    Attributes
    ----------
    name:
        Environment identifier used in logs.
    mu:
        Tyre-road friction coefficient (the dominant sim-to-real mismatch).
    tire_lag_tau:
        First-order lag time constant of the steering actuator in seconds.
    yaw_lag_tau:
        First-order lag time constant of the yaw-rate response in seconds.
    wind_amplitude:
        Amplitude of the sinusoidal lateral wind acceleration in m/s^2.
    wind_frequency:
        Frequency of the wind gust in Hz.
    wind_noise_std:
        Standard deviation of the stochastic wind component in m/s^2.
    sensor_noise_std:
        Standard deviation of the position observation noise in metres.  Other
        channels are scaled proportionally.
    steer_bias:
        Constant steering offset (mis-calibrated steering actuator) in radians.
    actuation_delay:
        Number of control steps of actuator delay.
    dt:
        Integration / control time step in seconds.
    initial_speed:
        Default initial speed used by :meth:`TrackedVehicleEnv.reset`.
    max_steps:
        Episode truncation horizon in control steps.
    """

    name: str = "sim"
    mu: float = 1.0
    tire_lag_tau: float = 0.02
    yaw_lag_tau: float = 0.02
    wind_amplitude: float = 0.0
    wind_frequency: float = 0.0
    wind_noise_std: float = 0.0
    sensor_noise_std: float = 0.0
    steer_bias: float = 0.0
    actuation_delay: int = 0
    dt: float = 0.05
    initial_speed: float = 8.0
    max_steps: int = 10_000

    def copy_with(self, **overrides: Any) -> "EnvConfig":
        """Return a copy of the configuration with the given fields replaced."""
        return replace(self, **overrides)

    def describe(self) -> Dict[str, float | int | str]:
        """Serialisable description used for logging / report generation."""
        return {
            "name": self.name,
            "mu": self.mu,
            "tire_lag_tau": self.tire_lag_tau,
            "yaw_lag_tau": self.yaw_lag_tau,
            "wind_amplitude": self.wind_amplitude,
            "wind_frequency": self.wind_frequency,
            "wind_noise_std": self.wind_noise_std,
            "sensor_noise_std": self.sensor_noise_std,
            "steer_bias": self.steer_bias,
            "actuation_delay": self.actuation_delay,
            "dt": self.dt,
        }


# --------------------------------------------------------------------------- #
# Environment
# --------------------------------------------------------------------------- #


class TrackedVehicleEnv:
    """Base class implementing the friction-limited dynamic bicycle model.

    The environment follows the classic ``(next_state, reward, done, info)``
    API.  ``info`` always contains the ground-truth lateral error, the track
    progress, the current lap index and the wind acceleration, which makes the
    class directly usable for evaluation and benchmarking.

    Notes
    -----
    The returned *observation* from :meth:`observe` may be noisy, while
    :attr:`state` and the ``info`` dictionary always contain the ground truth.
    Controllers should call :meth:`observe`; metrics should use the truth.
    """

    def __init__(
        self,
        track: Track,
        config: Optional[EnvConfig] = None,
        vehicle: Optional[VehicleParams] = None,
        *,
        seed: int = 0,
    ) -> None:
        self.track = track
        self.config: EnvConfig = config if config is not None else EnvConfig()
        self.vehicle: VehicleParams = vehicle if vehicle is not None else VehicleParams()
        self.dt: float = float(self.config.dt)
        self._base_seed = int(seed)
        self._rng = np.random.default_rng(seed)

        # Dynamic state.
        self.state: np.ndarray = np.zeros(STATE_DIM, dtype=np.float64)
        self._steer_state: float = 0.0
        self._yaw_state: float = 0.0
        self._actuator_queue: List[Tuple[float, float]] = []

        # Bookkeeping.
        self._time: float = 0.0
        self._steps: int = 0
        self._total_progress: float = 0.0
        self._last_progress: float = 0.0
        self._wind_phase: float = float(self._rng.uniform(0.0, 2.0 * np.pi))
        self._wind_value: float = 0.0
        self._last_action: np.ndarray = np.zeros(ACTION_DIM, dtype=np.float64)

        # Episode statistics.
        self.lap_times: List[float] = []
        self.lap_numbers: List[int] = []
        self.lap_count: int = 0
        self._lap_start_time: float = 0.0
        self.offtrack_events: int = 0

        self.reset()

    # ------------------------------------------------------------------ #
    # Properties
    # ------------------------------------------------------------------ #
    @property
    def time(self) -> float:
        """Elapsed simulated time in seconds since the last full reset."""
        return self._time

    @property
    def steps(self) -> int:
        """Number of control steps since the last full reset."""
        return self._steps

    @property
    def total_progress(self) -> float:
        """Cumulative arc length travelled in metres (monotonically increasing)."""
        return self._total_progress

    @property
    def speed(self) -> float:
        """Current longitudinal speed in m/s."""
        return float(self.state[IDX_V])

    # ------------------------------------------------------------------ #
    # Reset / observation
    # ------------------------------------------------------------------ #
    def reset(
        self,
        *,
        progress: float = 0.0,
        lateral_offset: float = 0.0,
        speed: Optional[float] = None,
        cumulative_progress: Optional[float] = None,
        reset_stats: bool = True,
    ) -> np.ndarray:
        """Reset the vehicle on the track and return the initial state.

        Parameters
        ----------
        progress:
            Arc length at which to spawn the vehicle.
        lateral_offset:
            Signed lateral offset from the centre line in metres (positive is
            left of the racing direction).
        speed:
            Initial longitudinal speed.  Defaults to ``config.initial_speed``.
        cumulative_progress:
            Value used to continue the lap counter after an off-track recovery.
            Defaults to ``progress``.
        reset_stats:
            Whether to clear lap times and counters.
        """
        point, tangent, heading = self.track.sample(progress)
        normal = np.array([-tangent[1], tangent[0]])
        position = point + normal * float(lateral_offset)
        v0 = float(self.config.initial_speed if speed is None else max(0.0, speed))

        self.state = np.array([position[0], position[1], v0, wrap_angle(heading), 0.0], dtype=np.float64)
        self._steer_state = 0.0
        self._yaw_state = 0.0
        self._actuator_queue = [(0.0, 0.0) for _ in range(int(self.config.actuation_delay))]
        self._last_action = np.zeros(ACTION_DIM, dtype=np.float64)
        self._wind_phase = float(self._rng.uniform(0.0, 2.0 * np.pi))
        self._wind_value = 0.0

        self._time = 0.0
        self._steps = 0
        self._last_progress = float(progress % self.track.length)
        self._total_progress = (
            float(cumulative_progress) if cumulative_progress is not None else float(progress)
        )
        self.lap_count = int(self._total_progress // self.track.length)

        if reset_stats:
            self.lap_times = []
            self.lap_numbers = []
            self.offtrack_events = 0
            self._lap_start_time = 0.0
        return self.state.copy()

    def set_state(self, state: Sequence[float]) -> None:
        """Overwrite the dynamic state (useful for tests and warm starts)."""
        arr = np.asarray(state, dtype=np.float64).reshape(STATE_DIM).copy()
        arr[IDX_HEADING] = wrap_angle(arr[IDX_HEADING])
        arr[IDX_V] = max(0.0, arr[IDX_V])
        self.state = arr

    def recover(
        self,
        *,
        progress: Optional[float] = None,
        lateral_offset: float = 0.0,
        speed: Optional[float] = None,
    ) -> np.ndarray:
        """Put the vehicle back on the centre line without resetting the clock.

        This emulates a safety-driver intervention after an off-track
        excursion: the lap timer, the cumulative lap counter and the episode
        statistics keep running, so the recovery remains visible in the lap
        time instead of silently restarting the episode.

        Parameters
        ----------
        progress:
            Arc length at which to place the car.  Defaults to the current
            projection on the centre line.
        lateral_offset:
            Lateral offset from the centre line in metres.
        speed:
            Speed after the recovery.  Defaults to 35% of the current speed.

        Returns
        -------
        The new state vector.
        """
        if progress is None:
            progress = self.track.project(self.state[:2]).progress
        point, tangent, heading = self.track.sample(float(progress))
        normal = np.array([-tangent[1], tangent[0]])
        position = point + normal * float(lateral_offset)
        # Never resume below the low-speed region covered by the prior: a
        # standstill vehicle is outside the offline dataset and the learned
        # model cannot be trusted there.
        target_speed = max(6.0, 0.4 * self.state[IDX_V]) if speed is None else float(max(0.0, speed))

        self.state = np.array(
            [position[0], position[1], target_speed, wrap_angle(heading), 0.0], dtype=np.float64
        )
        self._steer_state = 0.0
        self._yaw_state = 0.0
        self._actuator_queue = [(0.0, 0.0) for _ in range(int(self.config.actuation_delay))]
        self._last_action = np.zeros(ACTION_DIM, dtype=np.float64)
        self._last_progress = float(progress % self.track.length)
        return self.state.copy()

    def observe(self) -> np.ndarray:
        """Return a (possibly noisy) observation of the state.

        Sensor noise is scaled per channel: position in metres, speed in m/s,
        heading in radians and yaw rate in rad/s.  With
        ``config.sensor_noise_std == 0`` this is an exact copy of the state.
        """
        obs = self.state.copy()
        std = float(self.config.sensor_noise_std)
        if std > 0.0:
            scale = np.array([1.0, 1.0, 0.6, 0.08, 0.20])
            obs = obs + self._rng.normal(0.0, std * scale, size=STATE_DIM)
            obs[IDX_HEADING] = wrap_angle(obs[IDX_HEADING])
            obs[IDX_V] = max(0.0, obs[IDX_V])
        return obs

    # ------------------------------------------------------------------ #
    # Disturbances
    # ------------------------------------------------------------------ #
    def _wind_acceleration(self, t: float) -> float:
        """Slowly varying lateral wind acceleration in m/s^2."""
        cfg = self.config
        if cfg.wind_amplitude <= 0.0 and cfg.wind_noise_std <= 0.0:
            return 0.0
        value = cfg.wind_amplitude * math.sin(2.0 * math.pi * cfg.wind_frequency * t + self._wind_phase)
        if cfg.wind_noise_std > 0.0:
            value += float(self._rng.normal(0.0, cfg.wind_noise_std))
        return value

    # ------------------------------------------------------------------ #
    # Core dynamics
    # ------------------------------------------------------------------ #
    def _integrate(
        self,
        state: np.ndarray,
        accel_cmd: float,
        steer_cmd: float,
        dt: float,
        wind_accel: float,
    ) -> Tuple[np.ndarray, float, float]:
        """Advance the plant by one Euler step.

        Parameters
        ----------
        state:
            Current state vector ``(5,)``.
        accel_cmd:
            Requested longitudinal acceleration in m/s^2.
        steer_cmd:
            Requested road-wheel steering angle in radians (before bias/lag).
        dt:
            Integration step in seconds.
        wind_accel:
            Lateral wind acceleration in m/s^2.

        Returns
        -------
        ``(next_state, steer_state, yaw_state)`` where the latter two are the
        hidden actuator / yaw-rate filter states carried between steps.
        """
        vp = self.vehicle
        cfg = self.config
        grip = max(cfg.mu, 1e-3) * vp.gravity

        # --- steering actuator: bias, rate limit, first-order lag ---------- #
        steer_cmd = float(np.clip(steer_cmd + cfg.steer_bias, -vp.max_steer, vp.max_steer))
        max_step = vp.max_steer_rate * dt
        steer_limited = self._steer_state + float(np.clip(steer_cmd - self._steer_state, -max_step, max_step))
        tau_steer = max(cfg.tire_lag_tau, 1e-4)
        alpha_steer = 1.0 - math.exp(-dt / tau_steer)
        steer_state = self._steer_state + (steer_limited - self._steer_state) * alpha_steer

        # --- yaw dynamics: kinematic target saturated by the friction circle #
        v = float(state[IDX_V])
        yaw_target = v * math.tan(steer_state) / vp.wheelbase
        yaw_limit = grip / max(abs(v), 2.0)
        yaw_target = float(np.clip(yaw_target, -yaw_limit, yaw_limit))

        tau_yaw = max(cfg.yaw_lag_tau, 1e-4)
        alpha_yaw = 1.0 - math.exp(-dt / tau_yaw)
        yaw_state = self._yaw_state + (yaw_target - self._yaw_state) * alpha_yaw
        yaw_state = float(np.clip(yaw_state, -yaw_limit, yaw_limit))

        # Cross-wind enters as a curvature bias (lateral acceleration / speed).
        curvature_bias = wind_accel / max(abs(v), 3.0)
        omega_effective = yaw_state + curvature_bias

        # --- longitudinal dynamics with a friction ellipse ------------------ #
        lateral_accel = abs(yaw_state * v)
        usage = min(1.0, lateral_accel / grip) if grip > 1e-6 else 1.0
        accel_budget = grip * math.sqrt(max(0.0, 1.0 - usage * usage))

        accel_cmd = float(np.clip(accel_cmd, vp.max_decel, vp.max_accel))
        accel = float(np.clip(accel_cmd, -accel_budget, accel_budget))

        if v > 0.1:
            accel -= vp.drag_coefficient * v * v + vp.rolling_resistance * vp.gravity
        v_new = max(0.0, v + accel * dt)

        # --- pose integration (midpoint rule) ------------------------------- #
        heading = float(state[IDX_HEADING])
        heading_mid = wrap_angle(heading + 0.5 * omega_effective * dt)
        v_mid = 0.5 * (v + v_new)
        x_new = float(state[IDX_X]) + v_mid * math.cos(heading_mid) * dt
        y_new = float(state[IDX_Y]) + v_mid * math.sin(heading_mid) * dt
        if abs(wind_accel) > 1e-9:
            # Gusts push the car laterally (0.5 * a * dt^2 along the normal).
            lateral_slip = 0.5 * wind_accel * dt * dt
            x_new += -math.sin(heading_mid) * lateral_slip
            y_new += math.cos(heading_mid) * lateral_slip

        heading_new = wrap_angle(heading + omega_effective * dt)
        next_state = np.array([x_new, y_new, v_new, heading_new, yaw_state], dtype=np.float64)
        return next_state, steer_state, yaw_state

    # ------------------------------------------------------------------ #
    # Step
    # ------------------------------------------------------------------ #
    def step(self, action: Sequence[float] | np.ndarray) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """Advance the environment by one control step.

        Parameters
        ----------
        action:
            ``[acceleration (m/s^2), steering angle (rad)]``.

        Returns
        -------
        ``(next_state, reward, done, info)``.  ``done`` is ``True`` when the
        vehicle leaves the track or the episode horizon is reached.
        """
        act = np.asarray(action, dtype=np.float64).reshape(ACTION_DIM)
        accel_cmd = float(act[0])
        steer_cmd = float(act[1])

        # Actuator transport delay.
        delay = int(self.config.actuation_delay)
        if delay > 0:
            self._actuator_queue.append((accel_cmd, steer_cmd))
            accel_cmd, steer_cmd = self._actuator_queue.pop(0)

        self._wind_value = self._wind_acceleration(self._time)
        next_state, steer_state, yaw_state = self._integrate(
            self.state, accel_cmd, steer_cmd, self.dt, self._wind_value
        )

        self.state = next_state
        self._steer_state = steer_state
        self._yaw_state = yaw_state
        self._last_action = act.copy()
        self._time += self.dt
        self._steps += 1

        # --- progress / lap bookkeeping ------------------------------------ #
        proj = self.track.project(self.state[:2])
        delta_s = proj.progress - self._last_progress
        if delta_s < -0.5 * self.track.length:
            delta_s += self.track.length
        elif delta_s > 0.5 * self.track.length:
            delta_s -= self.track.length
        self._total_progress += delta_s
        self._last_progress = proj.progress

        new_lap_count = max(0, int(self._total_progress // self.track.length))
        lap_completed = new_lap_count > self.lap_count
        lap_time: Optional[float] = None
        if lap_completed:
            for _ in range(new_lap_count - self.lap_count):
                lap_time = self._time - self._lap_start_time
                self.lap_times.append(float(lap_time))
                self.lap_numbers.append(len(self.lap_times))
                self._lap_start_time = self._time
            self.lap_count = new_lap_count

        lateral_error = float(proj.signed_lateral_error)
        off_track = abs(lateral_error) > self.track.half_width
        if off_track:
            self.offtrack_events += 1

        done = bool(off_track or self._steps >= self.config.max_steps)

        reward = -0.5 * lateral_error * lateral_error - 0.01 * max(0.0, self.speed - 16.0) ** 2

        info: Dict[str, Any] = {
            "time": self._time,
            "lateral_error": lateral_error,
            "progress": proj.progress,
            "total_progress": self._total_progress,
            "curvature": proj.curvature,
            "lap": self.lap_count,
            "lap_time": lap_time,
            "off_track": off_track,
            "wind_accel": self._wind_value,
            "speed": self.speed,
            "acceleration": accel_cmd,
            "steering": steer_cmd,
            "steering_state": self._steer_state,
            "yaw_rate": float(self.state[IDX_YAW_RATE]),
        }
        return self.state.copy(), float(reward), done, info

    # ------------------------------------------------------------------ #
    # Convenience helpers
    # ------------------------------------------------------------------ #
    def lateral_error(self, state: Optional[Sequence[float]] = None) -> float:
        """Signed lateral error (metres, positive left) for a given state."""
        pos = self.state[:2] if state is None else np.asarray(state, dtype=np.float64)[:2]
        return float(self.track.project(pos).signed_lateral_error)

    def is_off_track(self, state: Optional[Sequence[float]] = None, margin: float = 0.0) -> bool:
        """Whether the vehicle centre is outside the drivable corridor."""
        return abs(self.lateral_error(state)) > (self.track.half_width + margin)

    def rollout(
        self,
        actions: Sequence[Sequence[float]] | np.ndarray,
        state: Optional[Sequence[float]] = None,
    ) -> np.ndarray:
        """Roll the *true* dynamics forward with a fixed action sequence.

        The environment state is not mutated.  Returns an array of shape
        ``(len(actions) + 1, STATE_DIM)`` starting with the initial state.
        """
        actions_arr = np.asarray(actions, dtype=np.float64).reshape(-1, ACTION_DIM)
        saved = (
            self.state.copy(),
            self._steer_state,
            self._yaw_state,
            self._time,
            self._steps,
            self._last_progress,
            self._total_progress,
            list(self._actuator_queue),
            self._wind_value,
        )
        if state is not None:
            self.set_state(state)
        trajectory = [self.state.copy()]
        for action in actions_arr:
            accel_cmd, steer_cmd = float(action[0]), float(action[1])
            delay = int(self.config.actuation_delay)
            if delay > 0:
                self._actuator_queue.append((accel_cmd, steer_cmd))
                accel_cmd, steer_cmd = self._actuator_queue.pop(0)
            wind = self._wind_acceleration(self._time)
            self.state, self._steer_state, self._yaw_state = self._integrate(
                self.state, accel_cmd, steer_cmd, self.dt, wind
            )
            self._time += self.dt
            trajectory.append(self.state.copy())
        (
            self.state,
            self._steer_state,
            self._yaw_state,
            self._time,
            self._steps,
            self._last_progress,
            self._total_progress,
            self._actuator_queue,
            self._wind_value,
        ) = saved
        return np.asarray(trajectory)

    def refresh_seed(self, seed: int) -> None:
        """Re-seed the stochastic components of the environment."""
        self._rng = np.random.default_rng(seed)
        self._base_seed = int(seed)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"{self.__class__.__name__}(mu={self.config.mu}, dt={self.dt}, "
            f"tire_lag={self.config.tire_lag_tau}, wind={self.config.wind_amplitude})"
        )


class SimEnv(TrackedVehicleEnv):
    """Nominal simulation environment used to generate the offline priors.

    Friction is high (``mu = 1.0``), the steering actuator is fast, there is no
    wind, no bias, no actuator delay and observations are noise-free.  This is
    the "synthetic prior" world.
    """

    DEFAULT_CONFIG = EnvConfig(
        name="SimEnv",
        mu=1.0,
        tire_lag_tau=0.02,
        yaw_lag_tau=0.02,
        wind_amplitude=0.0,
        wind_frequency=0.0,
        wind_noise_std=0.0,
        sensor_noise_std=0.0,
        steer_bias=0.0,
        actuation_delay=0,
        dt=0.05,
        initial_speed=10.0,
        max_steps=20_000,
    )

    def __init__(
        self,
        track: Track,
        *,
        seed: int = 0,
        config: Optional[EnvConfig] = None,
        vehicle: Optional[VehicleParams] = None,
    ) -> None:
        super().__init__(track, config or replace(self.DEFAULT_CONFIG), vehicle, seed=seed)


class RealEnv(TrackedVehicleEnv):
    """Physical ground-truth environment with realistic sim-to-real mismatches.

    Differences with respect to :class:`SimEnv`:

    * ``mu = 0.55`` - the tyre can only generate ~55% of the nominal grip.
    * steering actuator lag of 120 ms (vs 20 ms nominal).
    * yaw-rate lag of 60 ms.
    * sinusoidal + stochastic cross-wind of up to ~0.75 m/s^2.
    * constant steering bias of 0.01 rad.
    * one control step of actuator delay.
    * Gaussian observation noise on every channel.

    None of these quantities are exposed to the controller through
    :meth:`TrackedVehicleEnv.observe`; they can only be inferred from the data,
    which is exactly the setting PIA is designed for.
    """

    DEFAULT_CONFIG = EnvConfig(
        name="RealEnv",
        mu=0.55,
        tire_lag_tau=0.12,
        yaw_lag_tau=0.06,
        wind_amplitude=0.60,
        wind_frequency=0.06,
        wind_noise_std=0.15,
        sensor_noise_std=0.04,
        steer_bias=0.010,
        actuation_delay=1,
        dt=0.05,
        initial_speed=10.0,
        max_steps=20_000,
    )

    def __init__(
        self,
        track: Track,
        *,
        seed: int = 0,
        config: Optional[EnvConfig] = None,
        vehicle: Optional[VehicleParams] = None,
    ) -> None:
        super().__init__(track, config or replace(self.DEFAULT_CONFIG), vehicle, seed=seed)


# --------------------------------------------------------------------------- #
# Module-level helpers
# --------------------------------------------------------------------------- #


def build_track(
    radius: float = 32.0,
    straight_length: float = 56.0,
    half_width: float = 3.5,
) -> Track:
    """Convenience factory for the default benchmark oval."""
    return Track.oval(
        radius=radius,
        straight_length=straight_length,
        half_width=half_width,
        resolution=0.20,
        name="stadium-oval",
    )


def state_to_dict(state: Sequence[float]) -> Dict[str, float]:
    """Convert a state vector into a labelled dictionary."""
    arr = np.asarray(state, dtype=np.float64).reshape(STATE_DIM)
    return {label: float(value) for label, value in zip(STATE_LABELS, arr)}


def action_to_dict(action: Sequence[float]) -> Dict[str, float]:
    """Convert an action vector into a labelled dictionary."""
    arr = np.asarray(action, dtype=np.float64).reshape(ACTION_DIM)
    return {label: float(value) for label, value in zip(ACTION_LABELS, arr)}


# --------------------------------------------------------------------------- #
# Self-test
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    _track = build_track()
    print(_track)
    print(f"curvature range: [{_track.curvature.min():+.4f}, {_track.curvature.max():+.4f}] 1/m")

    for _env in (SimEnv(_track, seed=1), RealEnv(_track, seed=1)):
        _env.reset(progress=0.0, speed=10.0)
        _laps_before = len(_env.lap_times)
        _max_speed = 0.0
        _max_error = 0.0
        for _ in range(600):
            # Simple constant-steer probe: hold 20% throttle and a mild left input.
            _state, _reward, _done, _info = _env.step(np.array([1.5, 0.06]))
            _max_speed = max(_max_speed, _env.speed)
            _max_error = max(_max_error, abs(_info["lateral_error"]))
            if _done:
                break
        print(
            f"{_env.config.name:7s} | steps={_env.steps:4d} t={_env.time:6.1f}s "
            f"v_max={_max_speed:5.2f} m/s |e|_max={_max_error:5.2f} m "
            f"laps={len(_env.lap_times) - _laps_before} progress={_env.total_progress:6.1f} m"
        )

    # Verify that the two plants actually diverge under identical actions.
    _actions = np.tile(np.array([[2.0, 0.10]]), (80, 1))
    _sim = SimEnv(_track, seed=3)
    _real = RealEnv(_track, seed=3)
    _traj_sim = _sim.rollout(_actions, state=np.array([0.0, -32.0, 9.0, 0.0, 0.0]))
    _traj_real = _real.rollout(_actions, state=np.array([0.0, -32.0, 9.0, 0.0, 0.0]))
    _gap = float(np.linalg.norm(_traj_sim[-1, :2] - _traj_real[-1, :2]))
    print(f"identical-input 4 s rollout divergence: {_gap:.2f} m (sim-to-real gap)")
    assert _gap > 0.5, "expected a non-trivial sim-to-real gap"
    print("dynamics_sim.py self-test OK")

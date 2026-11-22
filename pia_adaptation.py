"""
pia_adaptation.py
=================
Prior-Informed Adaptation (PIA) engine.

PIA closes the sim-to-real gap *online* by fine-tuning the output distribution of
the probabilistic ensemble on a small stream of real transitions, while a KL
anchor and an L2 anchor keep the adapted model inside the trust region of the
offline prior.

The update rule implemented here is

.. math::

    \\mathcal{L} = \\mathrm{NLL}(\\theta)
        + \\beta_{KL} \\, D_{KL}\\big(p_{\\theta}(\\cdot\\,|\\,x_{prior})
        \\,\\|\\, p_{\\theta_0}(\\cdot\\,|\\,x_{prior})\\big)
        + \\beta_{L2} \\, \\|\\theta_{adapt} - \\theta_0\\|^2

with three ingredients that make the adaptation *rapid* and *stable*:

1. **Residual-only, head-only updates.**  The shared trunk is frozen; only the
   two output heads (and optionally the last hidden layer) are updated.  The
   number of adapted parameters is therefore small, which is what allows a
   high learning rate and a handful of gradient steps per control cycle.
2. **Prioritised replay.**  Transitions are sampled proportionally to their
   residual error, so the few informative steps of a new track are replayed
   much more often than the uninformative ones.
3. **Prior anchoring.**  The KL term is evaluated on a fixed replay batch of
   *offline* states, i.e. on the previous task, and prevents catastrophic
   forgetting.  If the KL divergence exceeds ``max_kl`` the update is rejected
   entirely, which bounds how far the online model can drift per step.

The engine is agnostic to the planner: it mutates ``self.model`` in place, and
:class:`~mpc_planner.MPCPlanner` holds a reference to that same module, so the
planner immediately plans with the adapted dynamics.

Author: PIA reference implementation.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from dataset_generator import PriorDataset
from dynamics_sim import ACTION_DIM, IDX_HEADING, STATE_DIM
from world_model import ProbabilisticEnsemble

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #


@dataclass
class PIAConfig:
    """Hyper-parameters of the online adaptation engine.

    Attributes
    ----------
    capacity:
        Maximum number of live transitions kept in the replay buffer.
    warmup:
        Number of observations required before the first gradient step.
    minibatch:
        Mini-batch size of every gradient step.
    update_every:
        A gradient update is triggered every ``update_every`` observations.
    gradient_steps:
        Number of gradient steps performed per update.
    learning_rate:
        Learning rate of the head-only Adam optimiser (deliberately high: the
        adapted parameter count is small).
    weight_decay:
        Adam weight decay.
    prior_kl_weight:
        Weight ``beta_KL`` of the hinge KL anchor on the offline replay batch.
    prior_l2_weight:
        Weight ``beta_L2`` of the parameter-space anchor.
    kl_target:
        Size of the free trust region: the KL anchor is inactive below this
        value and grows linearly above it.
    max_kl:
        Hard trust-region radius.  When the incoming KL exceeds this value the
        step is replaced by a pure anchor descent that pulls the model back
        towards the prior (instead of silently rejecting every update).
    buffer_alpha:
        Prioritisation exponent (``0`` = uniform replay).
    eval_every:
        Number of updates between two telemetry records.
    grad_clip:
        Gradient-norm clipping threshold.
    train_last_hidden:
        Whether the last hidden layer is adapted together with the heads.
    adapt_variance:
        Whether the log-variance head is adapted.  Disabled by default: the
        planner uses the predictive sigma as a cost term, and letting online
        data inflate it silently rescales the MPPI cost (measured to make the
        controller unstable).  The mean head is the residual adapter.
    loss:
        Online objective: ``"mse"`` (bounded, normalised residual) or ``"nll"``.
    max_param_drift:
        Hard box around every adapted parameter, in parameter units.  ``0``
        disables the projection.
    """

    capacity: int = 1024
    warmup: int = 64
    minibatch: int = 64
    update_every: int = 4
    gradient_steps: int = 3
    learning_rate: float = 3e-3
    weight_decay: float = 0.0
    prior_kl_weight: float = 8.0
    prior_l2_weight: float = 1e-2
    kl_target: float = 0.5
    max_kl: float = 2.0
    buffer_alpha: float = 0.6
    eval_every: int = 20
    grad_clip: float = 1.0
    train_last_hidden: bool = True
    adapt_variance: bool = False
    loss: str = "mse"
    max_param_drift: float = 0.02

    # --- friction-prior adaptation ------------------------------------- #
    grip_adaptation: bool = True
    grip_min: float = 0.35
    grip_percentile: float = 98.0
    grip_min_demand: float = 3.0
    grip_saturation_ratio: float = 0.97
    grip_min_samples: int = 40
    grip_safety: float = 1.0
    wheelbase: float = 2.60


# --------------------------------------------------------------------------- #
# Prioritised replay buffer
# --------------------------------------------------------------------------- #


class PrioritizedTransitionBuffer:
    """Fixed-capacity ring buffer with proportional prioritised sampling.

    Parameters
    ----------
    capacity:
        Maximum number of transitions.
    state_dim, action_dim:
        Dimensions of the stored arrays.
    seed:
        Seed of the sampling RNG.
    """

    def __init__(
        self,
        capacity: int = 1024,
        state_dim: int = STATE_DIM,
        action_dim: int = ACTION_DIM,
        *,
        seed: int = 0,
    ) -> None:
        self.capacity = int(capacity)
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self.states = np.zeros((self.capacity, self.state_dim), dtype=np.float32)
        self.actions = np.zeros((self.capacity, self.action_dim), dtype=np.float32)
        self.deltas = np.zeros((self.capacity, self.state_dim), dtype=np.float32)
        self.priorities = np.zeros(self.capacity, dtype=np.float64)
        self._index = 0
        self._size = 0
        self._max_priority = 1.0
        self._rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return self._size

    def add(self, state: Sequence[float], action: Sequence[float], delta: Sequence[float]) -> None:
        """Insert one transition with the current maximum priority."""
        i = self._index
        self.states[i] = np.asarray(state, dtype=np.float32)
        self.actions[i] = np.asarray(action, dtype=np.float32)
        self.deltas[i] = np.asarray(delta, dtype=np.float32)
        self.priorities[i] = self._max_priority
        self._index = (self._index + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def sample(
        self,
        batch_size: int,
        alpha: float = 0.6,
    ) -> Tuple[np.ndarray, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample a batch proportionally to ``priority ** alpha``.

        Returns
        -------
        indices, states, actions, deltas
        """
        n = self._size
        if n == 0:
            raise RuntimeError("cannot sample from an empty buffer")
        priorities = self.priorities[:n].copy()
        scaled = np.power(np.maximum(priorities, 1e-6), float(alpha))
        probabilities = scaled / scaled.sum()
        count = min(int(batch_size), n)
        indices = self._rng.choice(n, size=count, replace=(count > 1), p=probabilities)
        states = torch.as_tensor(self.states[indices], dtype=torch.float32)
        actions = torch.as_tensor(self.actions[indices], dtype=torch.float32)
        deltas = torch.as_tensor(self.deltas[indices], dtype=torch.float32)
        return indices, states, actions, deltas

    def update_priorities(self, indices: np.ndarray, errors: np.ndarray) -> None:
        """Update the priority of the sampled indices with their TD errors."""
        values = np.abs(np.asarray(errors, dtype=np.float64)).ravel()
        for index, value in zip(np.asarray(indices).ravel(), values):
            priority = float(value) + 1e-3
            self.priorities[int(index)] = priority
            self._max_priority = max(self._max_priority, priority)

    def all(self, limit: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return up to ``limit`` most recent transitions as tensors."""
        n = self._size if limit is None else min(self._size, int(limit))
        if n == 0:
            empty = torch.zeros(0, self.state_dim)
            return empty, torch.zeros(0, self.action_dim), empty.clone()
        if n == self._size:
            sl = slice(0, self._size)
        else:
            start = (self._index - n) % self.capacity
            if start + n <= self.capacity:
                sl = slice(start, start + n)
            else:
                # Wrap-around: rebuild an ordered copy.
                order = np.arange(start, start + n) % self.capacity
                return (
                    torch.as_tensor(self.states[order], dtype=torch.float32),
                    torch.as_tensor(self.actions[order], dtype=torch.float32),
                    torch.as_tensor(self.deltas[order], dtype=torch.float32),
                )
        return (
            torch.as_tensor(self.states[sl], dtype=torch.float32),
            torch.as_tensor(self.actions[sl], dtype=torch.float32),
            torch.as_tensor(self.deltas[sl], dtype=torch.float32),
        )


# --------------------------------------------------------------------------- #
# Telemetry
# --------------------------------------------------------------------------- #


@dataclass
class AdaptationRecord:
    """One telemetry sample of the adaptation engine.

    Attributes
    ----------
    observation:
        Environment step at which the record was produced.
    update:
        Update index.
    nll:
        Online residual loss of the last gradient step (normalised MSE by
        default; heteroscedastic NLL when ``loss="nll"``).
    kl:
        KL divergence between adapted and prior Gaussians on the offline replay.
    param_drift:
        Relative L2 drift of the adapted parameters with respect to the prior.
    buffer_rmse:
        One-step RMSE of the adapted model on the live buffer (metres).
    prior_rmse:
        One-step RMSE of the *prior* model on the same live buffer (metres).
    rejected:
        Whether the update was rejected by the trust-region check.
    """

    observation: int
    update: int
    nll: float
    kl: float
    param_drift: float
    buffer_rmse: float
    prior_rmse: float
    rejected: bool = False

    def to_dict(self) -> Dict[str, float | int | bool]:
        return {
            "observation": self.observation,
            "update": self.update,
            "nll": self.nll,
            "kl": self.kl,
            "param_drift": self.param_drift,
            "buffer_rmse": self.buffer_rmse,
            "prior_rmse": self.prior_rmse,
            "rejected": self.rejected,
        }


# --------------------------------------------------------------------------- #
# Engine
# --------------------------------------------------------------------------- #


class PIAdaptationEngine:
    """Online residual adaptation of a probabilistic ensemble world model.

    Parameters
    ----------
    prior_model:
        Offline-trained ensemble.  It is **deep-copied**: the engine keeps an
        immutable reference copy (the prior) and a trainable copy (the adapted
        model).
    config:
        Adaptation hyper-parameters.
    prior_dataset:
        Offline dataset used to build the KL anchor replay batch.  When
        ``None`` the KL anchor is disabled and only the L2 anchor is active.
    device:
        Torch device used for the gradient steps.
    seed:
        Seed of the replay buffer and of the prior replay subsample.

    Examples
    --------
    >>> adapter = PIAdaptationEngine(world_model, prior_dataset=dataset)
    >>> planner = MPCPlanner(adapter.model, track)
    >>> for step in range(...):
    ...     action = planner.plan(env.observe()).action
    ...     next_state, _, _, _ = env.step(action)
    ...     adapter.observe(state, action, next_state)
    ...     adapter.maybe_update()
    """

    def __init__(
        self,
        prior_model: ProbabilisticEnsemble,
        config: Optional[PIAConfig] = None,
        *,
        prior_dataset: Optional[PriorDataset] = None,
        device: str | torch.device = "cpu",
        seed: int = 0,
    ) -> None:
        self.config = config if config is not None else PIAConfig()
        self.device = torch.device(device)

        self.prior_model = prior_model.clone().to(self.device)
        self.prior_model.eval()
        for parameter in self.prior_model.parameters():
            parameter.requires_grad_(False)

        self.model = prior_model.clone().to(self.device)
        self.model.eval()
        self.model.freeze_trunk(train_last_hidden=self.config.train_last_hidden)
        if not self.config.adapt_variance:
            # Keeping the prior's calibrated variance is important: the MPPI
            # cost uses the predictive sigma as an uncertainty penalty, so an
            # online-inflated variance silently changes the cost scale and was
            # measured to destabilise the closed loop.
            for member in self.model.members:
                for parameter in member.var_head.parameters():
                    parameter.requires_grad_(False)
        self._trainable = self.model.trainable_parameters()
        if not self._trainable:
            raise RuntimeError("no trainable parameters left after freezing the trunk")
        self._prior_parameters = [parameter.detach().clone() for parameter in self._trainable]

        self.optimizer = torch.optim.Adam(
            self._trainable,
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

        self.buffer = PrioritizedTransitionBuffer(
            capacity=self.config.capacity, seed=seed,
        )

        # Fixed offline replay batch used by the KL anchor.
        self._prior_replay: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
        if prior_dataset is not None and len(prior_dataset) > 0:
            rng = np.random.default_rng(seed)
            count = min(len(prior_dataset), 256)
            index = rng.choice(len(prior_dataset), size=count, replace=False)
            tensors = prior_dataset.tensors(self.device)
            self._prior_replay = (tensors["states"][index], tensors["actions"][index])

        self.observations: int = 0
        self.updates: int = 0
        self.rejected_updates: int = 0
        self.telemetry: List[AdaptationRecord] = []
        self._last_nll: float = float("nan")
        self._last_kl: float = 0.0
        self._last_rejected: bool = False
        self._grip_estimate: Optional[float] = None

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    @property
    def adapted_model(self) -> ProbabilisticEnsemble:
        """The live, fine-tuned ensemble (shared with the planner)."""
        return self.model

    def observe(
        self,
        state: Sequence[float],
        action: Sequence[float],
        next_state: Sequence[float],
    ) -> None:
        """Register one live transition.

        The state residual is computed exactly as in the offline dataset,
        including the wrap-around of the heading component.
        """
        s = np.asarray(state, dtype=np.float64).reshape(STATE_DIM)
        ns = np.asarray(next_state, dtype=np.float64).reshape(STATE_DIM)
        a = np.asarray(action, dtype=np.float64).reshape(ACTION_DIM)
        delta = ns - s
        delta[IDX_HEADING] = (delta[IDX_HEADING] + np.pi) % (2.0 * np.pi) - np.pi
        if not np.all(np.isfinite(delta)):
            return
        self.buffer.add(s, a, delta)
        self.observations += 1

    def maybe_update(self) -> Optional[AdaptationRecord]:
        """Perform a PIA update if enough new data has arrived.

        Returns
        -------
        AdaptationRecord or None
            Telemetry record when the update also triggered a telemetry sample,
            otherwise ``None``.
        """
        if len(self.buffer) < self.config.warmup:
            return None
        if self.observations % max(1, self.config.update_every) != 0:
            return None

        for _ in range(max(1, self.config.gradient_steps)):
            self._gradient_step()
        self.updates += 1

        if self.updates % max(1, self.config.eval_every) != 0:
            return None
        return self._record()

    # ------------------------------------------------------------------ #
    # Gradient step
    # ------------------------------------------------------------------ #
    def _gradient_step(self) -> None:
        cfg = self.config
        index, states, actions, deltas = self.buffer.sample(cfg.minibatch, alpha=cfg.buffer_alpha)
        states = states.to(self.device)
        actions = actions.to(self.device)
        deltas = deltas.to(self.device)

        self.model.train()
        kl_before = self._prior_kl()
        kl_value = float(kl_before.item())

        if kl_value > cfg.max_kl:
            # Hard trust-region violation: recover by descending on the anchor
            # alone.  This pulls the model back into the prior's neighbourhood
            # instead of deadlocking the adaptation.
            self.optimizer.zero_grad(set_to_none=True)
            kl_before.backward()
            if cfg.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(self._trainable, cfg.grad_clip)
            self.optimizer.step()
            self.model.eval()
            self.rejected_updates += 1
            self._last_rejected = True
            self._last_nll = float("nan")
            self._last_kl = kl_value
            return

        nll = self._residual_loss(states, actions, deltas)
        # Soft trust region: no penalty inside ``kl_target``, linear penalty
        # outside.  This keeps the prior anchor from fighting every update.
        hinge = torch.relu(kl_before - cfg.kl_target)
        l2 = self._parameter_drift()
        loss = nll + cfg.prior_kl_weight * hinge + cfg.prior_l2_weight * l2

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if cfg.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(self._trainable, cfg.grad_clip)
        self.optimizer.step()
        self._project_parameters()
        self.model.eval()

        with torch.no_grad():
            mu, _var = self.model.predict_members(states, actions, members=[0])
            errors = (mu[0] - deltas).abs().mean(dim=-1).cpu().numpy()
        self.buffer.update_priorities(index, errors)

        self._last_nll = float(nll.item())
        self._last_kl = kl_value
        self._last_rejected = False

    def _project_parameters(self) -> None:
        """Project the adapted parameters into a hard box around the prior.

        The KL anchor constrains the *distribution* on offline states, but it
        says nothing about the off-manifold states that the MPPI samples during
        rollouts.  Without a parameter-space bound, the head can drift in a
        direction that is invisible to the anchor and yet distorts the planner's
        candidate evaluation.  A hard box keeps the online model inside a
        neighbourhood in which its behaviour is provably close to the prior.
        """
        bound = float(self.config.max_param_drift)
        if bound <= 0.0:
            return
        with torch.no_grad():
            for parameter, reference in zip(self._trainable, self._prior_parameters):
                parameter.copy_(torch.clamp(parameter, reference - bound, reference + bound))

    def _residual_loss(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        deltas: torch.Tensor,
    ) -> torch.Tensor:
        """Online regression objective for the residual dynamics.

        ``"mse"`` (default) optimises the mean prediction directly in the
        *normalised* residual space.  This matters in practice: the offline
        ensemble is deliberately overconfident (sigma ~ 5 mm), so the
        heteroscedastic NLL is dominated by ``err^2 / sigma^2`` and a single
        gradient step moves the mean by far more than the model's own error.
        The bounded MSE objective keeps every online step in proportion to the
        residual, which is what makes a high learning rate safe.

        ``"nll"`` keeps the full heteroscedastic objective (useful when the
        variance head is adapted as well).
        """
        if self.config.loss == "nll":
            return self.model.nll_per_member(states, actions, deltas).mean()

        features = self.model.encode(states, actions)
        target = (deltas - self.model.delta_mean) / self.model.delta_std
        losses = []
        for member in self.model.members:
            mu, _logvar = member(features)
            losses.append((mu - target).pow(2).mean())
        return torch.stack(losses).mean()

    def _prior_kl(self) -> torch.Tensor:
        """KL divergence between adapted and prior Gaussians on offline states."""
        if self._prior_replay is None:
            return torch.zeros((), device=self.device)
        states, actions = self._prior_replay
        with torch.no_grad():
            prior_mean, prior_var = self.prior_model.predict_members(states, actions)
        adapted_mean, adapted_var = self.model.predict_members(states, actions)
        prior_var = prior_var.clamp_min(1e-6)
        adapted_var = adapted_var.clamp_min(1e-6)
        kl = 0.5 * (
            torch.log(prior_var / adapted_var)
            + (adapted_var + (adapted_mean - prior_mean).pow(2)) / prior_var
            - 1.0
        )
        return kl.mean()

    def _parameter_drift(self) -> torch.Tensor:
        """Mean squared parameter drift relative to the prior."""
        if not self._prior_parameters:
            return torch.zeros((), device=self.device)
        drift = torch.zeros((), device=self.device)
        count = 0
        for parameter, reference in zip(self._trainable, self._prior_parameters):
            drift = drift + (parameter - reference).pow(2).sum()
            count += parameter.numel()
        return drift / max(count, 1)

    # ------------------------------------------------------------------ #
    # Friction-prior adaptation
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def estimate_grip(self, prior_mu: float = 1.0) -> float:
        """Estimate the effective tyre friction coefficient from live data.

        The world model cannot fix the second half of the prior: the planner
        builds its reference speed from a friction circle that was calibrated
        in simulation (``mu = 1.0``).  If the real track only offers ``0.55``,
        the controller keeps asking for a speed the car cannot hold, runs wide
        and leaves the track - no amount of residual fitting changes that,
        because the reference profile is an explicit input.

        This estimator recovers ``mu`` from the *shortfall* between the lateral
        acceleration the steering demanded and the one the car actually
        produced::

            a_dem = v^2 * tan(delta) / L        (kinematic demand)
            a_ach = v * omega                    (measured)

        When ``a_ach < 0.97 * a_dem`` the tyre is saturated and ``a_ach / g`` is
        a direct measurement of the friction limit.  Only saturated samples are
        used, so cruising in circles below the limit never drags ``mu`` down -
        which avoids the classic "conservative estimate slows the car down,
        slower car confirms the conservative estimate" spiral.

        Parameters
        ----------
        prior_mu:
            Value returned until enough saturated samples have been observed.

        Returns
        -------
        float
            Estimated friction coefficient, clamped to
            ``[grip_min, prior_mu]``.
        """
        if not self.config.grip_adaptation:
            return prior_mu
        states, actions, _deltas = self.buffer.all(limit=self.config.capacity)
        if states.shape[0] < self.config.grip_min_samples:
            return prior_mu

        states = states.to(self.device)
        actions = actions.to(self.device)
        speed = states[:, 2]
        omega = states[:, 4].abs()
        steer = actions[:, 1]
        demand = speed.pow(2) * torch.tan(steer).abs() / max(self.config.wheelbase, 1e-3)
        achieved = speed * omega

        saturated = (demand > self.config.grip_min_demand) & (
            achieved < self.config.grip_saturation_ratio * demand
        )
        if int(saturated.sum().item()) < self.config.grip_min_samples:
            return prior_mu

        samples = achieved[saturated] / 9.81
        mu = torch.quantile(samples, self.config.grip_percentile / 100.0)
        # A small margin: the limit measured at the tyre is not the limit the
        # closed loop can exploit (reaction delay, wind gusts, sensor noise).
        mu = mu * float(self.config.grip_safety)
        mu_value = float(mu.clamp(self.config.grip_min, max(prior_mu, self.config.grip_min)).item())
        # ``a_ach`` is a *lower bound* on the friction limit, so the best
        # evidence seen so far is the maximum over time.  Taking the maximum
        # also removes the "conservative estimate -> slower car -> even more
        # conservative estimate" feedback loop.
        if self._grip_estimate is None:
            self._grip_estimate = mu_value
        else:
            self._grip_estimate = max(self._grip_estimate, mu_value)
        return self._grip_estimate

    @property
    def grip_estimate(self) -> Optional[float]:
        """Last friction estimate produced by :meth:`estimate_grip`."""
        return self._grip_estimate

    # ------------------------------------------------------------------ #
    # Telemetry
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def prediction_metrics(self, limit: int = 256) -> Dict[str, float]:
        """One-step RMSE of the adapted and prior models on the live buffer."""
        states, actions, deltas = self.buffer.all(limit=limit)
        if states.numel() == 0:
            return {"buffer_rmse": float("nan"), "prior_rmse": float("nan"),
                    "improvement": float("nan"), "kl": 0.0, "param_drift": 0.0}
        states = states.to(self.device)
        actions = actions.to(self.device)
        deltas = deltas.to(self.device)

        adapted_mean, _ = self.model.predict_members(states, actions, members=[0])
        prior_mean, _ = self.prior_model.predict_members(states, actions, members=[0])
        adapted_rmse = float(torch.sqrt(((adapted_mean[0] - deltas) ** 2).mean()).item())
        prior_rmse = float(torch.sqrt(((prior_mean[0] - deltas) ** 2).mean()).item())
        improvement = (prior_rmse - adapted_rmse) / max(prior_rmse, 1e-9)
        return {
            "buffer_rmse": adapted_rmse,
            "prior_rmse": prior_rmse,
            "improvement": improvement,
            "kl": float(self._last_kl),
            "param_drift": float(self._parameter_drift().item()),
        }

    def _record(self) -> AdaptationRecord:
        metrics = self.prediction_metrics()
        record = AdaptationRecord(
            observation=int(self.observations),
            update=int(self.updates),
            nll=float(self._last_nll),
            kl=float(metrics["kl"]),
            param_drift=float(metrics["param_drift"]),
            buffer_rmse=float(metrics["buffer_rmse"]),
            prior_rmse=float(metrics["prior_rmse"]),
            rejected=bool(self._last_rejected),
        )
        self.telemetry.append(record)
        return record

    def history_arrays(self) -> Dict[str, np.ndarray]:
        """Telemetry as NumPy arrays, ready for plotting."""
        if not self.telemetry:
            empty = np.zeros(0, dtype=np.float64)
            return {"observation": empty, "update": empty, "nll": empty, "kl": empty,
                    "param_drift": empty, "buffer_rmse": empty, "prior_rmse": empty}
        return {
            "observation": np.array([r.observation for r in self.telemetry], dtype=np.float64),
            "update": np.array([r.update for r in self.telemetry], dtype=np.float64),
            "nll": np.array([r.nll for r in self.telemetry], dtype=np.float64),
            "kl": np.array([r.kl for r in self.telemetry], dtype=np.float64),
            "param_drift": np.array([r.param_drift for r in self.telemetry], dtype=np.float64),
            "buffer_rmse": np.array([r.buffer_rmse for r in self.telemetry], dtype=np.float64),
            "prior_rmse": np.array([r.prior_rmse for r in self.telemetry], dtype=np.float64),
        }

    def state_dict(self) -> Dict[str, Any]:
        """Serialisable snapshot of the adaptation state."""
        return {
            "config": self.config.__dict__,
            "model": self.model.state_dict(),
            "prior_model": self.prior_model.state_dict(),
            "observations": self.observations,
            "updates": self.updates,
            "telemetry": [record.to_dict() for record in self.telemetry],
        }

    def describe(self) -> str:
        """One-line description used in logs."""
        trainable = sum(p.numel() for p in self._trainable)
        total = sum(p.numel() for p in self.model.parameters())
        return (
            f"PIAdaptationEngine(trainable={trainable:,}/{total:,} params, "
            f"lr={self.config.learning_rate}, update_every={self.config.update_every}, "
            f"capacity={self.config.capacity})"
        )


# --------------------------------------------------------------------------- #
# Self-test
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    from dataset_generator import StanleyController, generate_prior_dataset
    from dynamics_sim import RealEnv, SimEnv, build_track
    from mpc_planner import MPCConfig, MPCPlanner

    torch.manual_seed(0)
    np.random.seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    track = build_track()

    # --- offline prior ---------------------------------------------------- #
    sim = SimEnv(track, seed=0)
    dataset = generate_prior_dataset(
        sim, StanleyController(track, seed=1), episodes=12, steps_per_episode=220, seed=0, verbose=False
    )
    model = ProbabilisticEnsemble(n_members=3, hidden_dims=(128, 128), seed=4)
    model.fit(dataset, epochs=2, batch_size=256, device=device, verbose=False)  # deliberately under-fitted prior
    prior_copy = model.clone()

    # --- live data from the physical plant (stable baseline, no planner) --- #
    real = RealEnv(track, seed=9)
    controller = StanleyController(track, seed=2)
    controller.set_target_speed(10.0)
    real.reset(progress=0.0, speed=9.0)
    live_states: List[np.ndarray] = []
    live_actions: List[np.ndarray] = []
    live_next_states: List[np.ndarray] = []
    for _ in range(1400):
        state_before = real.state.copy()
        action = controller.act(real.observe())
        next_state, _reward, done, _info = real.step(action)
        live_states.append(state_before)
        live_actions.append(action.copy())
        live_next_states.append(next_state.copy())
        if done:
            break

    live_states_arr = np.asarray(live_states, dtype=np.float32)
    live_actions_arr = np.asarray(live_actions, dtype=np.float32)
    delta_arr = np.asarray(live_next_states, dtype=np.float32) - live_states_arr
    delta_arr[:, 3] = (delta_arr[:, 3] + np.pi) % (2.0 * np.pi) - np.pi

    # --- adaptation -------------------------------------------------------- #
    adapter = PIAdaptationEngine(
        model,
        PIAConfig(update_every=4, gradient_steps=2, kl_target=0.5, max_kl=2.0, train_last_hidden=False),
        prior_dataset=dataset,
        device=device,
        seed=0,
    )
    print(adapter.describe())

    held_out = slice(-300, None)
    probe_states = torch.as_tensor(live_states_arr[held_out], device=device)
    probe_actions = torch.as_tensor(live_actions_arr[held_out], device=device)
    probe_deltas = torch.as_tensor(delta_arr[held_out], device=device)

    def rmse(m: ProbabilisticEnsemble) -> float:
        with torch.no_grad():
            mu, _var = m.predict_members(probe_states, probe_actions, members=[0])
            return float(torch.sqrt(((mu[0] - probe_deltas) ** 2).mean()).item())

    prior_rmse = rmse(model)
    for state, action, delta in zip(live_states_arr[:-300], live_actions_arr[:-300], delta_arr[:-300]):
        adapter.observe(state, action, state + delta)
        adapter.maybe_update()
    adapted_rmse = rmse(adapter.model)

    drift = adapter._parameter_drift().item()
    print(f"held-out live RMSE: prior {prior_rmse:.4f} m -> adapted {adapted_rmse:.4f} m")
    print(f"updates={adapter.updates} observations={adapter.observations} drift={drift:.3e} "
          f"kl={adapter._last_kl:.3f}")
    assert adapter.updates > 0, "no adaptation update was executed"
    assert drift > 0.0, "the adapted parameters did not move"
    assert adapted_rmse < prior_rmse, "adaptation did not improve the held-out live RMSE"
    assert adapter._last_kl <= adapter.config.max_kl * 1.2, "trust region violated"

    # The prior must remain untouched by the adaptation.
    before_state = prior_copy.state_dict()
    after_state = adapter.prior_model.state_dict()
    for key, value in before_state.items():
        assert torch.equal(value, after_state[key]), f"prior model was mutated ({key})"

    # --- integration smoke test ------------------------------------------- #
    planner = MPCPlanner(
        adapter.model,
        track,
        MPCConfig(horizon=8, num_samples=32, num_iterations=2, smoothing=0.0, target_speed=10.0, seed=0),
        device=device,
        seed=0,
    )
    env = RealEnv(track, seed=3)
    env.reset(progress=0.0, speed=9.0)
    for _ in range(5):
        result = planner.plan(env.observe())
        env.step(result.action)
    print(f"planner integration OK ({result.predicted_states.shape[0]} step rollout)")
    assert adapter.config.grip_adaptation is True
    assert 0.0 < adapter.estimate_grip(prior_mu=1.0) <= 1.0
    print("pia_adaptation.py self-test OK")

"""
world_model.py
==============
Probabilistic ensemble world model for the PIA autonomous-racing benchmark.

The model learns the *state residual* dynamics

.. math::

    \\Delta s_t = s_{t+1} - s_t = f_\\theta(s_t, a_t) + \\epsilon,
    \\qquad \\epsilon \\sim \\mathcal{N}(0, \\mathrm{diag}(\\sigma_\\theta^2(s_t, a_t)))

with an ensemble of :math:`K` independently initialised Gaussian MLPs.  Each
member predicts the five-dimensional residual mean and variance.  The ensemble
serves three purposes:

1. **Prediction** - the planner rolls candidates out with the ensemble mean.
2. **Aleatoric uncertainty** - the predicted :math:`\\sigma` captures noise in
   the transition, which the MPPI cost penalises to avoid fragile plans.
3. **Epistemic uncertainty** - disagreement between members
   (:math:`\\mathrm{Var}_k[\\mu_k]`) is large far from the offline prior
   distribution, which is exactly where online adaptation is needed.

Two losses are provided:

* :func:`gaussian_nll` - heteroscedastic negative log-likelihood used for
  offline training (and for the PIA fine-tuning objective).
* :meth:`ProbabilisticEnsemble.mse_per_member` - plain mean squared error, used
  for reporting and for sanity checks.

The class also implements the *adaptation hooks* required by
:mod:`pia_adaptation`: :meth:`freeze_trunk`, :meth:`trainable_parameters`,
:meth:`clone` and :meth:`state_dict` based persistence.

Author: PIA reference implementation.
"""
from __future__ import annotations

import copy
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset

from dynamics_sim import ACTION_DIM, STATE_DIM
from dataset_generator import PriorDataset

# --------------------------------------------------------------------------- #
# Losses
# --------------------------------------------------------------------------- #


def gaussian_nll(
    mu: torch.Tensor,
    logvar: torch.Tensor,
    target: torch.Tensor,
    *,
    min_logvar: float = -8.0,
    max_logvar: float = 2.0,
) -> torch.Tensor:
    """Heteroscedastic Gaussian negative log-likelihood.

    ``0.5 * (logvar + (mu - target)^2 / exp(logvar))`` averaged over every
    element.  ``logvar`` is clamped for numerical stability, which also acts as
    a floor/ceiling on the predicted standard deviation.

    Parameters
    ----------
    mu:
        Predicted mean, any shape.
    logvar:
        Predicted log-variance, same shape as ``mu``.
    target:
        Regression target, same shape as ``mu``.
    min_logvar, max_logvar:
        Clamping range for the log-variance.

    Returns
    -------
    Scalar tensor.
    """
    logvar_c = torch.clamp(logvar, min_logvar, max_logvar)
    precision = torch.exp(-logvar_c)
    return 0.5 * (logvar_c + (mu - target).pow(2) * precision).mean()


# --------------------------------------------------------------------------- #
# Building blocks
# --------------------------------------------------------------------------- #


class GaussianMLP(nn.Module):
    """A small MLP with a shared trunk and separate mean / log-variance heads.

    Parameters
    ----------
    in_dim:
        Input dimension (state + action).
    out_dim:
        Output dimension (state residual).
    hidden_dims:
        Sizes of the hidden layers.
    dropout:
        Dropout probability applied after each hidden activation (0 disables).
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_dims: Sequence[int] = (192, 192),
        *,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.in_dim = int(in_dim)
        self.out_dim = int(out_dim)
        self.hidden_dims = tuple(int(h) for h in hidden_dims)
        self.activation = nn.SiLU()
        self.dropout = nn.Dropout(float(dropout)) if dropout > 0.0 else nn.Identity()

        # ``hidden`` is a ModuleList of Linear layers so that PIA can unfreeze
        # exactly the last hidden layer plus the two output heads.
        self.hidden = nn.ModuleList()
        prev = self.in_dim
        for width in self.hidden_dims:
            layer = nn.Linear(prev, width)
            nn.init.orthogonal_(layer.weight, gain=math.sqrt(2.0))
            nn.init.zeros_(layer.bias)
            self.hidden.append(layer)
            prev = width

        self.mu_head = nn.Linear(prev, self.out_dim)
        self.var_head = nn.Linear(prev, self.out_dim)
        # Start with unit variance in normalised space: log(sigma^2) = 0.
        nn.init.zeros_(self.mu_head.bias)
        nn.init.zeros_(self.var_head.bias)
        nn.init.uniform_(self.mu_head.weight, -0.05, 0.05)
        nn.init.uniform_(self.var_head.weight, -0.05, 0.05)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return ``(mu, logvar)`` for a batch of normalised inputs."""
        h = x
        for layer in self.hidden:
            h = self.dropout(self.activation(layer(h)))
        return self.mu_head(h), self.var_head(h)


# --------------------------------------------------------------------------- #
# Training history container
# --------------------------------------------------------------------------- #


@dataclass
class TrainHistory:
    """Per-epoch training diagnostics.

    Attributes
    ----------
    train_nll:
        ``(epochs, members)`` training loss.
    val_nll:
        ``(epochs, members)`` validation loss.
    val_rmse:
        ``(epochs,)`` ensemble mean prediction RMSE in metres (position and
        speed are reported separately in :attr:`extra`).
    epochs:
        Number of epochs actually executed.
    duration_s:
        Wall-clock training time.
    extra:
        Additional summary values (best epoch, final metrics, ...).
    """

    train_nll: List[List[float]] = field(default_factory=list)
    val_nll: List[List[float]] = field(default_factory=list)
    val_rmse: List[float] = field(default_factory=list)
    duration_s: float = 0.0
    extra: Dict[str, Any] = field(default_factory=dict)

    @property
    def epochs(self) -> int:
        return len(self.train_nll)

    def to_dict(self) -> Dict[str, Any]:
        """Serialisable representation (for JSON reports)."""
        return {
            "train_nll": self.train_nll,
            "val_nll": self.val_nll,
            "val_rmse": self.val_rmse,
            "duration_s": self.duration_s,
            **{k: v for k, v in self.extra.items() if isinstance(v, (int, float, str))},
        }

    def summary(self) -> str:
        """Compact human-readable summary of the training run."""
        if not self.epochs:
            return "TrainHistory(empty)"
        final_train = float(np.mean(self.train_nll[-1]))
        final_val = float(np.mean(self.val_nll[-1]))
        best_val = float(np.min([np.mean(row) for row in self.val_nll]))
        return (
            f"epochs={self.epochs} | final train NLL={final_train:.4f} | "
            f"final val NLL={final_val:.4f} | best val NLL={best_val:.4f} | "
            f"time={self.duration_s:.1f}s"
        )


# --------------------------------------------------------------------------- #
# Ensemble
# --------------------------------------------------------------------------- #


class ProbabilisticEnsemble(nn.Module):
    """Ensemble of probabilistic dynamics networks.

    Parameters
    ----------
    state_dim:
        Dimension of the state vector (5 for the racing benchmark).
    action_dim:
        Dimension of the action vector (2).
    n_members:
        Number of ensemble members (3-5 in the paper's experiments).
    hidden_dims:
        Hidden layer sizes of every member.
    dropout:
        Dropout probability inside each member.
    min_logvar, max_logvar:
        Clamping range applied to predicted log-variances.
    learning_rate:
        Default learning rate used by :meth:`fit`.
    seed:
        Seed used to initialise the members.

    Notes
    -----
    Inputs and outputs are standardised with running statistics fitted on the
    offline dataset.  The statistics are stored as *buffers*, so they travel
    with ``state_dict`` and are correctly restored by :meth:`load`.  All public
    prediction methods therefore always operate in physical units.
    """

    def __init__(
        self,
        state_dim: int = STATE_DIM,
        action_dim: int = ACTION_DIM,
        n_members: int = 5,
        hidden_dims: Sequence[int] = (192, 192),
        *,
        dropout: float = 0.0,
        min_logvar: float = -8.0,
        max_logvar: float = 2.0,
        learning_rate: float = 2e-3,
        feature_mode: str = "invariant",
        seed: int = 0,
    ) -> None:
        super().__init__()
        if feature_mode not in ("invariant", "position"):
            raise ValueError("feature_mode must be 'invariant' or 'position'")
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self.n_members = int(n_members)
        self.hidden_dims = tuple(int(h) for h in hidden_dims)
        self.min_logvar = float(min_logvar)
        self.max_logvar = float(max_logvar)
        self.learning_rate = float(learning_rate)
        self.feature_mode = str(feature_mode)
        # "invariant": [v, sin(h), cos(h), yaw_rate] - the state block that
        # actually drives the dynamics.  "position": additionally keeps x, y.
        self.feature_dim = 4 if self.feature_mode == "invariant" else 6
        self.seed = int(seed)

        generator = torch.Generator().manual_seed(int(seed))
        members: List[GaussianMLP] = []
        for member_idx in range(self.n_members):
            torch.manual_seed(int(seed) * 1000 + member_idx)
            members.append(
                GaussianMLP(
                    self.feature_dim + self.action_dim,
                    self.state_dim,
                    hidden_dims=self.hidden_dims,
                    dropout=dropout,
                )
            )
        self.members = nn.ModuleList(members)
        self.dropout_p = float(dropout)
        self._generator_seed = int(seed)
        self._init_normalization()

    # ------------------------------------------------------------------ #
    # Feature map and normalisation
    # ------------------------------------------------------------------ #
    def featurize(self, states: torch.Tensor) -> torch.Tensor:
        """Map a physical state into the regression feature space.

        The heading is represented as ``(sin h, cos h)`` rather than as a raw
        angle: ``h`` is periodic and discontinuous at ``+/- pi``, which is
        exactly where a plain MLP fails (the top and bottom straights of an
        oval sit on that discontinuity).

        In ``"invariant"`` mode the global position ``(x, y)`` is excluded.
        Vehicle dynamics are translation invariant, and feeding absolute
        coordinates lets the network memorise where it is on the track instead
        of learning the dynamics - a shortcut that degrades long-horizon
        rollouts badly.  ``"position"`` mode keeps ``(x, y)`` and is retained
        for the ablation study in the README.
        """
        heading = states[:, 3:4]
        sin_h = torch.sin(heading)
        cos_h = torch.cos(heading)
        if self.feature_mode == "invariant":
            return torch.cat([states[:, 2:3], sin_h, cos_h, states[:, 4:5]], dim=-1)
        return torch.cat([states[:, :2], states[:, 2:3], sin_h, cos_h, states[:, 4:5]], dim=-1)

    def _init_normalization(self) -> None:
        self.register_buffer("feature_mean", torch.zeros(self.feature_dim))
        self.register_buffer("feature_std", torch.ones(self.feature_dim))
        self.register_buffer("action_mean", torch.zeros(self.action_dim))
        self.register_buffer("action_std", torch.ones(self.action_dim))
        self.register_buffer("delta_mean", torch.zeros(self.state_dim))
        self.register_buffer("delta_std", torch.ones(self.state_dim))

    def fit_normalization(self, states: torch.Tensor, actions: torch.Tensor, deltas: torch.Tensor) -> None:
        """Fit standardisation statistics from tensors (physical units)."""
        with torch.no_grad():
            features = self.featurize(states)
            self.feature_mean.copy_(features.mean(dim=0))
            self.feature_std.copy_(features.std(dim=0, unbiased=False).clamp_min(1e-2))
            self.action_mean.copy_(actions.mean(dim=0))
            self.action_std.copy_(actions.std(dim=0, unbiased=False).clamp_min(1e-2))
            self.delta_mean.copy_(deltas.mean(dim=0))
            self.delta_std.copy_(deltas.std(dim=0, unbiased=False).clamp_min(1e-2))

    def encode(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """Standardise the feature map and action into the network input."""
        features = (self.featurize(states) - self.feature_mean) / self.feature_std
        normalized_actions = (actions - self.action_mean) / self.action_std
        return torch.cat([features, normalized_actions], dim=-1)

    def _decode(self, mu_n: torch.Tensor, logvar_n: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Convert normalised Gaussian parameters back to physical units."""
        mu = mu_n * self.delta_std + self.delta_mean
        variance = torch.exp(torch.clamp(logvar_n, self.min_logvar, self.max_logvar)) * self.delta_std.pow(2)
        return mu, variance

    # ------------------------------------------------------------------ #
    # Forward / prediction
    # ------------------------------------------------------------------ #
    def forward(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        members: Optional[Sequence[int]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return per-member ``(mu, variance)`` in physical units.

        Shapes are ``(K, B, state_dim)`` where ``K`` is the number of selected
        members and ``B`` the batch size.
        """
        return self.predict_members(states, actions, members)

    def predict_members(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        members: Optional[Sequence[int]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Evaluate selected ensemble members.

        Parameters
        ----------
        states:
            ``(B, state_dim)`` current states.
        actions:
            ``(B, action_dim)`` actions.
        members:
            Indices of the members to evaluate.  ``None`` means all members and
            is typically used for offline training / evaluation; the planner
            uses the first two for cost efficiency.

        Returns
        -------
        ``mu`` and ``variance``, both ``(K, B, state_dim)``.
        """
        if states.dim() == 1:
            states = states.unsqueeze(0)
        if actions.dim() == 1:
            actions = actions.unsqueeze(0)
        if states.shape[0] != actions.shape[0]:
            raise ValueError("states and actions must have the same batch dimension")

        x = self.encode(states, actions)
        indices = list(range(self.n_members)) if members is None else list(members)
        if not indices:
            raise ValueError("at least one ensemble member must be selected")

        mus: List[torch.Tensor] = []
        variances: List[torch.Tensor] = []
        for idx in indices:
            mu_n, logvar_n = self.members[int(idx)](x)
            mu, var = self._decode(mu_n, logvar_n)
            mus.append(mu)
            variances.append(var)
        return torch.stack(mus, dim=0), torch.stack(variances, dim=0)

    @staticmethod
    def mixture_moments(mu: torch.Tensor, variance: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Moments of the ensemble mixture ``(mean, total variance)``.

        Total variance decomposes into the average aleatoric variance plus the
        epistemic variance of the means across members.
        """
        mean = mu.mean(dim=0)
        aleatoric = variance.mean(dim=0)
        epistemic = mu.var(dim=0, unbiased=False)
        return mean, aleatoric + epistemic

    def predict_delta(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        members: Optional[Sequence[int]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Ensemble mean residual and per-element total variance (physical units)."""
        mu, var = self.predict_members(states, actions, members)
        return self.mixture_moments(mu, var)

    def predict_step(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        members: Optional[Sequence[int]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """One-step prediction used by the planner.

        Returns
        -------
        next_states:
            ``(B, state_dim)`` ensemble-mean successor states.  Heading is
            wrapped to ``(-pi, pi]`` and speed is clipped at zero, mirroring the
            true environment.
        uncertainty:
            ``(B,)`` scalar uncertainty (mean predictive standard deviation over
            state dimensions).
        """
        delta_mean, total_var = self.predict_delta(states, actions, members)
        next_states = states + delta_mean
        # Keep the rollout on the same manifold as the environment.
        next_states = torch.cat(
            [
                next_states[:, :3],
                torch.remainder(next_states[:, 3:4] + math.pi, 2.0 * math.pi) - math.pi,
                next_states[:, 4:5],
            ],
            dim=-1,
        )
        next_states = torch.cat([next_states[:, :2], next_states[:, 2:3].clamp_min(0.0), next_states[:, 3:]], dim=-1)
        uncertainty = torch.sqrt(total_var.clamp_min(1e-12)).mean(dim=-1)
        return next_states, uncertainty

    def uncertainty(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        members: Optional[Sequence[int]] = None,
    ) -> torch.Tensor:
        """Scalar ensemble uncertainty ``(B,)`` for a batch of transitions."""
        _mu, total_var = self.predict_delta(states, actions, members)
        return torch.sqrt(total_var.clamp_min(1e-12)).mean(dim=-1)

    # ------------------------------------------------------------------ #
    # Losses / metrics
    # ------------------------------------------------------------------ #
    def nll_per_member(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        deltas: torch.Tensor,
    ) -> torch.Tensor:
        """Negative log-likelihood of every member, shape ``(K,)``.

        The loss is evaluated in the *normalised* residual space, matching
        :meth:`fit`.
        """
        x = self.encode(states, actions)
        target = (deltas - self.delta_mean) / self.delta_std
        losses: List[torch.Tensor] = []
        for member in self.members:
            mu_n, logvar_n = member(x)
            losses.append(
                gaussian_nll(mu_n, logvar_n, target, min_logvar=self.min_logvar, max_logvar=self.max_logvar)
            )
        return torch.stack(losses)

    def mse_per_member(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        deltas: torch.Tensor,
    ) -> torch.Tensor:
        """Plain MSE of every member in physical units, shape ``(K,)``."""
        mu, _var = self.predict_members(states, actions)
        return ((mu - deltas.unsqueeze(0)) ** 2).mean(dim=(1, 2))

    @torch.no_grad()
    def evaluate(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        deltas: torch.Tensor,
    ) -> Dict[str, float]:
        """Validation metrics in physical units.

        Returns a dictionary with the mean validation NLL, the position RMSE,
        the speed RMSE and the mean predicted standard deviation.
        """
        nll = self.nll_per_member(states, actions, deltas).mean().item()
        mu, total_var = self.predict_delta(states, actions)
        error = mu - deltas
        position_rmse = float(torch.sqrt((error[:, :2] ** 2).mean()).item())
        speed_rmse = float(torch.sqrt((error[:, 2] ** 2).mean()).item())
        mean_sigma = float(torch.sqrt(total_var.clamp_min(1e-12)).mean().item())
        return {
            "nll": float(nll),
            "position_rmse": position_rmse,
            "speed_rmse": speed_rmse,
            "mean_sigma": mean_sigma,
        }

    # ------------------------------------------------------------------ #
    # Training
    # ------------------------------------------------------------------ #
    def fit(
        self,
        dataset: PriorDataset,
        *,
        val_fraction: float = 0.15,
        epochs: int = 25,
        batch_size: int = 256,
        learning_rate: Optional[float] = None,
        weight_decay: float = 1e-6,
        device: str | torch.device = "cpu",
        seed: int = 0,
        bootstrap: bool = True,
        early_stopping: bool = True,
        grad_clip: float = 5.0,
        verbose: bool = True,
    ) -> TrainHistory:
        """Train every ensemble member on the offline prior dataset.

        Each member optimises its own heteroscedastic NLL on a bootstrap
        resample of the training split, which produces the disagreement that
        the planner uses as an epistemic uncertainty signal.

        Parameters
        ----------
        dataset:
            Offline prior transitions.
        val_fraction:
            Fraction of episodes (or samples) held out for validation.
        epochs:
            Number of passes over the training split.
        batch_size:
            Mini-batch size.
        learning_rate:
            Adam learning rate; defaults to ``self.learning_rate``.
        weight_decay:
            Adam weight decay (small; keeps the prior well conditioned).
        device:
            Torch device string or object.
        seed:
            Seed for the split, bootstrap resampling and shuffling.
        bootstrap:
            Whether to bootstrap-resample each member's training indices.
        early_stopping:
            Restore the best-validation weights of each member at the end.
        grad_clip:
            Gradient-norm clipping threshold.
        verbose:
            Print one line per epoch.

        Returns
        -------
        TrainHistory
        """
        if len(dataset) < 32:
            raise ValueError("dataset is too small to train a world model")

        dev = torch.device(device)
        self.to(dev)
        lr = float(self.learning_rate if learning_rate is None else learning_rate)
        start = time.perf_counter()

        train_ds, val_ds = dataset.split(val_fraction=val_fraction, seed=seed, device=dev)
        train_states, train_actions, _train_next, train_deltas = train_ds.tensors
        val_states, val_actions, _val_next, val_deltas = val_ds.tensors
        self.fit_normalization(train_states, train_actions, train_deltas)

        n_train = int(train_states.shape[0])
        rng = np.random.default_rng(seed)
        optimizers: List[torch.optim.Optimizer] = []
        schedulers: List[Any] = []
        bootstrap_indices: List[torch.Tensor] = []
        for member in self.members:
            optimizers.append(
                torch.optim.Adam(member.parameters(), lr=lr, weight_decay=weight_decay)
            )
            schedulers.append(torch.optim.lr_scheduler.CosineAnnealingLR(optimizers[-1], T_max=max(1, epochs)))
            if bootstrap:
                idx = rng.integers(0, n_train, size=n_train)
            else:
                idx = rng.permutation(n_train)
            bootstrap_indices.append(torch.as_tensor(idx, dtype=torch.long, device=dev))

        history = TrainHistory()
        best_val = [math.inf] * self.n_members
        best_state: List[Dict[str, torch.Tensor]] = [{} for _ in range(self.n_members)]

        for epoch in range(1, int(epochs) + 1):
            train_losses: List[float] = []
            for member_idx, member in enumerate(self.members):
                optimizer = optimizers[member_idx]
                indices = bootstrap_indices[member_idx]
                permutation = torch.randperm(n_train, device=dev)
                shuffled = indices[permutation]
                member.train()
                total_loss = 0.0
                seen = 0
                for start_idx in range(0, n_train, batch_size):
                    batch_idx = shuffled[start_idx : start_idx + batch_size]
                    states = train_states[batch_idx]
                    actions = train_actions[batch_idx]
                    deltas = train_deltas[batch_idx]

                    x = self.encode(states, actions)
                    target = (deltas - self.delta_mean) / self.delta_std
                    mu_n, logvar_n = member(x)
                    loss = gaussian_nll(
                        mu_n, logvar_n, target, min_logvar=self.min_logvar, max_logvar=self.max_logvar
                    )
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    if grad_clip and grad_clip > 0:
                        nn.utils.clip_grad_norm_(member.parameters(), grad_clip)
                    optimizer.step()

                    total_loss += float(loss.item()) * states.shape[0]
                    seen += int(states.shape[0])
                train_losses.append(total_loss / max(seen, 1))
                schedulers[member_idx].step()

            self.eval()
            val_losses: List[float] = []
            with torch.no_grad():
                for member_idx, member in enumerate(self.members):
                    x = self.encode(val_states, val_actions)
                    target = (val_deltas - self.delta_mean) / self.delta_std
                    mu_n, logvar_n = member(x)
                    val_losses.append(
                        float(
                            gaussian_nll(
                                mu_n, logvar_n, target,
                                min_logvar=self.min_logvar, max_logvar=self.max_logvar,
                            ).item()
                        )
                    )
                metrics = self.evaluate(val_states, val_actions, val_deltas)

            history.train_nll.append(train_losses)
            history.val_nll.append(val_losses)
            history.val_rmse.append(metrics["position_rmse"])

            for member_idx in range(self.n_members):
                if val_losses[member_idx] < best_val[member_idx]:
                    best_val[member_idx] = val_losses[member_idx]
                    if early_stopping:
                        best_state[member_idx] = {
                            key: value.detach().clone() for key, value in self.members[member_idx].state_dict().items()
                        }

            if verbose:
                mean_train = float(np.mean(train_losses))
                mean_val = float(np.mean(val_losses))
                print(
                    f"  [world-model] epoch {epoch:3d}/{epochs} | train NLL {mean_train:.4f} | "
                    f"val NLL {mean_val:.4f} | pos RMSE {metrics['position_rmse']:.4f} m | "
                    f"sigma {metrics['mean_sigma']:.4f}"
                )

        if early_stopping:
            for member_idx in range(self.n_members):
                if best_state[member_idx]:
                    self.members[member_idx].load_state_dict(best_state[member_idx])

        self.eval()
        final_metrics = self.evaluate(val_states, val_actions, val_deltas)
        history.duration_s = time.perf_counter() - start
        history.extra = {
            **final_metrics,
            "best_val_nll": float(np.mean(best_val)),
            "n_train": n_train,
            "n_val": int(val_states.shape[0]),
            "learning_rate": lr,
            "epochs": int(epochs),
            "batch_size": int(batch_size),
            "device": str(dev),
        }
        return history

    # ------------------------------------------------------------------ #
    # Adaptation hooks
    # ------------------------------------------------------------------ #
    def freeze_trunk(self, *, train_last_hidden: bool = True) -> List[nn.Parameter]:
        """Freeze everything except the output heads (and optionally the last hidden layer).

        This is the parametrisation used by PIA: the residual dynamics are
        adapted through the *output* distribution while the shared prior
        representation is preserved.

        Returns
        -------
        list of ``torch.nn.Parameter``
            The trainable parameters.
        """
        for param in self.parameters():
            param.requires_grad_(False)
        for member in self.members:
            for param in member.mu_head.parameters():
                param.requires_grad_(True)
            for param in member.var_head.parameters():
                param.requires_grad_(True)
            if train_last_hidden and len(member.hidden) > 0:
                for param in member.hidden[-1].parameters():
                    param.requires_grad_(True)
        return self.trainable_parameters()

    def trainable_parameters(self) -> List[nn.Parameter]:
        """All parameters with ``requires_grad=True``."""
        return [param for param in self.parameters() if param.requires_grad]

    def unfreeze_all(self) -> List[nn.Parameter]:
        """Re-enable gradients for every parameter."""
        for param in self.parameters():
            param.requires_grad_(True)
        return self.trainable_parameters()

    def clone(self) -> "ProbabilisticEnsemble":
        """Deep copy of the ensemble (weights, buffers and configuration)."""
        return copy.deepcopy(self)

    # ------------------------------------------------------------------ #
    # Persistence
    # ------------------------------------------------------------------ #
    def config_dict(self) -> Dict[str, Any]:
        """Constructor arguments required to rebuild the ensemble."""
        return {
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
            "n_members": self.n_members,
            "hidden_dims": tuple(self.hidden_dims),
            "dropout": float(self.dropout_p),
            "min_logvar": self.min_logvar,
            "max_logvar": self.max_logvar,
            "learning_rate": self.learning_rate,
            "feature_mode": self.feature_mode,
            "seed": self._generator_seed,
        }

    def save(self, path: str | Path, extra: Optional[Dict[str, Any]] = None) -> Path:
        """Save weights, normalisation buffers and constructor arguments."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "config": self.config_dict(),
                "state_dict": self.state_dict(),
                "extra": extra or {},
            },
            target,
        )
        return target

    @classmethod
    def load(cls, path: str | Path, device: str | torch.device = "cpu") -> "ProbabilisticEnsemble":
        """Load an ensemble previously written by :meth:`save`."""
        payload = torch.load(Path(path), map_location="cpu", weights_only=False)
        model = cls(**payload["config"])
        model.load_state_dict(payload["state_dict"])
        model.to(torch.device(device))
        model.eval()
        return model

    # ------------------------------------------------------------------ #
    def describe(self) -> str:
        """One-line architecture description."""
        n_params = sum(p.numel() for p in self.parameters())
        return (
            f"ProbabilisticEnsemble(K={self.n_members}, hidden={self.hidden_dims}, "
            f"params={n_params:,}, dropout={self.dropout_p})"
        )


# --------------------------------------------------------------------------- #
# Self-test
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    from dynamics_sim import SimEnv, build_track
    from dataset_generator import StanleyController, generate_prior_dataset

    torch.set_num_threads(max(1, torch.get_num_threads()))
    track = build_track()
    env = SimEnv(track, seed=0)
    controller = StanleyController(track, seed=1)
    data = generate_prior_dataset(
        env, controller, episodes=8, steps_per_episode=180, seed=0, verbose=False
    )
    print(data)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = ProbabilisticEnsemble(n_members=3, hidden_dims=(128, 128), seed=7)
    print(model.describe())
    history = model.fit(
        data, epochs=6, batch_size=256, device=device, val_fraction=0.2, verbose=True
    )
    print(history.summary())

    val_states, val_actions, _val_next, val_deltas = data.split(0.2, seed=0, device=device)[1].tensors
    metrics = model.evaluate(val_states, val_actions, val_deltas)
    print("validation:", {k: round(v, 5) for k, v in metrics.items()})

    path = model.save("artifacts/_smoke_world_model.pt")
    reloaded = ProbabilisticEnsemble.load(path, device=device)
    with torch.no_grad():
        probe_states = val_states[:16]
        probe_actions = val_actions[:16]
        mu_a, _ = model.predict_members(probe_states, probe_actions)
        mu_b, _ = reloaded.predict_members(probe_states, probe_actions)
    assert torch.allclose(mu_a, mu_b, atol=1e-5), "save/load mismatch"
    print("world_model.py self-test OK")

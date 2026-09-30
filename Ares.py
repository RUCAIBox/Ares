# mypy: allow-untyped-defs
r"""Ares: a spectral-driven, adaptive-rank optimizer for RLVR.

Ares adapts the *rank* of each parameter update to the observed gradient
spectrum. For every 2D weight matrix, at each step it:

  1. Factorizes the gradient with a randomized block power iteration (RBPI)
     to expose its leading singular subspace ``(U, S, V)``.
  2. Estimates an effective rank from the Shannon entropy of the spectral
     energy distribution, ``r_hat = exp(-sum_i p_i log p_i)`` with
     ``p_i = sigma_i^2 / sum_j sigma_j^2``, then smooths it across steps with a
     bias-corrected exponential moving average and clips it to ``[1, k]``.
  3. Reconstructs a rank-``r`` truncated-SVD gradient ``g~ = U[:, :r] diag(S[:r]) V[:r]``
     and applies an AdamW-style update (bias-corrected moments + decoupled
     weight decay) to the parameter.

One-dimensional parameters (biases, norm gains, and other vectors) fall back to
a standard AdamW step.

See the accompanying paper for the derivation and analysis; Algorithm 1 gives
the per-step pseudocode this file implements.
"""

from typing import Optional, Union

import torch
from torch import Tensor

from torch.optim.optimizer import (
    _default_to_fused_or_foreach,
    _get_capturable_supported_devices,
    _get_scalar_dtype,
    _to_scalar,
    _use_grad_for_differentiable,
    Optimizer,
    ParamsT,
)


__all__ = ["Ares"]


def _adam_update(grad, exp_avg, exp_avg_sq, step, betas, eps):
    """Bias-corrected Adam update, returning ``m_hat / (sqrt(v_hat) + eps)``."""
    exp_avg.lerp_(grad, 1 - betas[0])
    exp_avg_sq.lerp_(grad.square(), 1 - betas[1])
    m_hat = exp_avg / (1 - betas[0] ** step)
    v_hat = exp_avg_sq / (1 - betas[1] ** step)
    return m_hat / (v_hat.sqrt() + eps)


@torch.no_grad()
def randomized_block_power_iteration(
    A: Tensor,
    target_rank: int,
    last_rank: Tensor,
    rank_ema_coef: float,
    step: Tensor,
    n_iter: int = 20,
    oversample: int = 10,
):
    r"""Randomized block power iteration with spectral-energy rank selection.

    Approximates the leading singular subspace of ``A`` (Halko et al., 2011),
    then chooses an effective rank from the entropy of the singular-value energy
    distribution and smooths it across steps.

    Args:
        A: gradient matrix of shape ``(m, n)``.
        target_rank: maximum target rank ``k``.
        last_rank: previous step's tracked rank, for EMA smoothing.
        rank_ema_coef: EMA smoothing coefficient ``c`` for the rank estimate.
        step: current step tensor, used for EMA bias correction.
        n_iter: number of power iterations ``q``.
        oversample: oversampling parameter ``p`` (block size is ``k + p``).

    Returns:
        ``(U, S, V, r)`` where ``A ~= U diag(S) V`` restricted to the selected
        rank ``r``: ``U`` is ``(m, r)``, ``S`` is ``(r,)``, ``V`` is ``(r, n)``.
    """
    m, n = A.size()
    block_size = target_rank + oversample

    # Random projection sketch: Y ~ A Omega.
    omega = torch.randn(n, block_size, device=A.device, dtype=A.dtype)
    Y = A @ omega

    # Power iterations, re-orthonormalizing in float32 for stability, so that Y
    # behaves like (A A^T)^q A Omega and the leading spectrum is amplified.
    for _ in range(n_iter):
        Y, _ = torch.linalg.qr(Y.to(torch.float32), mode="reduced")
        Y = A.T @ Y.to(A.dtype)
        Y, _ = torch.linalg.qr(Y.to(torch.float32), mode="reduced")
        Y = A @ Y.to(A.dtype)

    Q, _ = torch.linalg.qr(Y.to(torch.float32), mode="reduced")
    Q = Q.to(A.dtype)

    # Project A onto the basis and take the SVD of the small matrix B = Q^T A.
    B = Q.T @ A
    Ub, S, Vh = torch.linalg.svd(B.to(torch.float32), full_matrices=False)
    U = Q @ Ub.to(Q.dtype)

    # Effective rank from the entropy of the spectral energy distribution.
    energy = S ** 2
    p = energy / energy.sum()
    entropy = -(p * torch.log(p + 1e-12)).sum()
    r_hat = int(torch.exp(entropy))
    r_hat = max(1, min(r_hat, target_rank))

    # Bias-corrected EMA smoothing of the rank estimate across steps.
    r_smoothed = rank_ema_coef * last_rank + (1 - rank_ema_coef) * r_hat
    r = int(r_smoothed / (1 - rank_ema_coef ** step))
    r = max(1, min(r, target_rank))

    return U[:, :r].to(A.dtype), S[:r].to(A.dtype), Vh[:r].to(A.dtype), r


def _single_tensor_lowrank(
    params: list[Tensor],
    grads: list[Tensor],
    square_avgs: list[Tensor],
    momentum_buffer_list: list[Tensor],
    state_steps: list[Tensor],
    rank_list: list[Tensor],
    *,
    lr: float,
    eps: float,
    weight_decay: float,
    maximize: bool,
    capturable: bool,
    scale: float,
    target_rank: int,
    rank_ema_coef: float,
    betas: tuple,
):
    """Spectral low-rank update for 2D weight matrices."""
    if not torch.jit.is_scripting():
        lr = _to_scalar(lr)

    for i, param in enumerate(params):
        step = state_steps[i]
        rank = rank_list[i]

        if not torch.compiler.is_compiling() and capturable:
            supported = _get_capturable_supported_devices()
            assert (
                param.device.type == step.device.type and param.device.type in supported
            ), f"If capturable=True, params and state_steps must be on supported devices: {supported}."

        step += 1

        grad = grads[i]
        grad = grad if not maximize else -grad

        # Factorize, select the effective rank, and reconstruct the low-rank
        # (truncated-SVD) gradient with its singular-value weighting retained.
        u, s, v, cur_rank = randomized_block_power_iteration(
            grad, target_rank, rank, rank_ema_coef, step
        )
        rank.add_(cur_rank - rank)
        grad = u @ torch.diag(s) @ v

        update = _adam_update(
            grad, momentum_buffer_list[i], square_avgs[i], step=step, betas=betas, eps=eps
        )
        param.mul_(1 - lr * weight_decay)
        param.add_(update.reshape(param.shape), alpha=-lr * scale)


def _single_tensor_adam(
    params: list[Tensor],
    grads: list[Tensor],
    square_avgs: list[Tensor],
    momentum_buffer_list: list[Tensor],
    state_steps: list[Tensor],
    *,
    lr: float,
    eps: float,
    weight_decay: float,
    maximize: bool,
    capturable: bool,
    betas: tuple,
):
    """Standard AdamW update for 1D parameters."""
    if not torch.jit.is_scripting():
        lr = _to_scalar(lr)

    for i, param in enumerate(params):
        step = state_steps[i]

        if not torch.compiler.is_compiling() and capturable:
            supported = _get_capturable_supported_devices()
            assert (
                param.device.type == step.device.type and param.device.type in supported
            ), f"If capturable=True, params and state_steps must be on supported devices: {supported}."

        step += 1

        grad = grads[i]
        grad = grad if not maximize else -grad

        update = _adam_update(
            grad, momentum_buffer_list[i], square_avgs[i], step=step, betas=betas, eps=eps
        )
        param.mul_(1 - lr * weight_decay)
        param.add_(update.reshape(param.shape), alpha=-lr)


class Ares(Optimizer):
    r"""Spectral-driven adaptive-rank optimizer.

    Args:
        params: iterable of parameters or named parameter groups.
        lr: learning rate (default: 1e-2).
        eps: term added to the denominator for numerical stability (default: 1e-8).
        weight_decay: decoupled weight-decay coefficient (default: 0).
        betas: Adam moment coefficients ``(beta1, beta2)`` (default: (0.9, 0.95)).
        capturable: whether the step can be captured in a CUDA graph (default: False).
        foreach: whether to use a foreach implementation (default: None).
        maximize: maximize the objective instead of minimizing (default: False).
        differentiable: whether the step is differentiable (default: False).
        scale: scale factor ``s`` applied to the low-rank update (default: 1.0).
        init_rank: target-rank budget ``k``; also the initial tracked rank (default: 32).
        rank_ema_coef: EMA smoothing coefficient ``c`` for the tracked rank (default: 0.0).
    """

    def __init__(
        self,
        params: ParamsT,
        lr: Union[float, Tensor] = 1e-2,
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        betas: tuple = (0.9, 0.95),
        capturable: bool = False,
        foreach: Optional[bool] = None,
        maximize: bool = False,
        differentiable: bool = False,
        scale: float = 1.0,
        init_rank: int = 32,
        rank_ema_coef: float = 0.0,
    ):
        if isinstance(lr, Tensor) and lr.numel() != 1:
            raise ValueError("Tensor lr must be 1-element")
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta1: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta2: {betas[1]}")
        if not 1 <= init_rank:
            raise ValueError(f"Invalid init_rank: {init_rank}")
        if not 0.0 <= rank_ema_coef < 1.0:
            raise ValueError(f"Invalid rank_ema_coef: {rank_ema_coef}")

        defaults = {
            "lr": lr,
            "eps": eps,
            "weight_decay": weight_decay,
            "betas": betas,
            "capturable": capturable,
            "foreach": foreach,
            "maximize": maximize,
            "differentiable": differentiable,
        }
        super().__init__(params, defaults)

        self.scale = scale
        self.init_rank = init_rank
        self.rank_ema_coef = rank_ema_coef

    def __setstate__(self, state):
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault("foreach", None)
            group.setdefault("maximize", False)
            group.setdefault("differentiable", False)
            group.setdefault("capturable", False)
            group.setdefault("betas", (0.9, 0.95))
            for p in group["params"]:
                p_state = self.state.get(p, {})
                if len(p_state) != 0 and not torch.is_tensor(p_state["step"]):
                    step_val = float(p_state["step"])
                    p_state["step"] = torch.tensor(
                        step_val,
                        dtype=_get_scalar_dtype(),
                        device=p.device if group["capturable"] else None,
                    )

    def _init_group(self, group, matrix_params: bool):
        """Collect parameters and their state for one update mode.

        ``matrix_params=True`` selects 2D weight matrices (spectral low-rank
        update); ``matrix_params=False`` selects 1D parameters (AdamW).
        """
        params_with_grad: list[Tensor] = []
        grads: list[Tensor] = []
        square_avgs: list[Tensor] = []
        momentum_buffer_list: list[Tensor] = []
        state_steps: list[Tensor] = []
        rank_list: list[Tensor] = []

        for p in group["params"]:
            if p.grad is None:
                continue
            if matrix_params and p.ndim < 2:
                continue
            if not matrix_params and p.ndim >= 2:
                continue
            if p.grad.is_sparse:
                raise RuntimeError("Ares does not support sparse gradients")

            params_with_grad.append(p)
            grads.append(p.grad)

            state = self.state[p]
            if len(state) == 0:
                state["step"] = torch.zeros(
                    (),
                    dtype=_get_scalar_dtype(),
                    device=p.device if group["capturable"] else None,
                )
                state["square_avg"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                state["momentum_buffer"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                state["rank"] = torch.tensor(
                    self.init_rank,
                    dtype=_get_scalar_dtype(),
                    device=p.device if group["capturable"] else None,
                )

            square_avgs.append(state["square_avg"])
            momentum_buffer_list.append(state["momentum_buffer"])
            state_steps.append(state["step"])
            rank_list.append(state["rank"])

        return params_with_grad, grads, square_avgs, momentum_buffer_list, state_steps, rank_list

    @_use_grad_for_differentiable
    def step(self, closure=None):
        """Perform a single optimization step.

        Args:
            closure (Callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        self._cuda_graph_capture_health_check()

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            foreach = group["foreach"]
            if foreach is None:
                _, foreach = _default_to_fused_or_foreach(
                    group["params"], group["differentiable"], use_fused=False
                )
            if foreach and torch.jit.is_scripting():
                raise RuntimeError("torch.jit.script not supported with foreach optimizers")

            # 2D weight matrices: spectral low-rank update.
            params, grads, square_avgs, momentum, steps, ranks = self._init_group(
                group, matrix_params=True
            )
            _single_tensor_lowrank(
                params,
                grads,
                square_avgs,
                momentum,
                steps,
                ranks,
                lr=group["lr"],
                eps=group["eps"],
                weight_decay=group["weight_decay"],
                maximize=group["maximize"],
                capturable=group["capturable"],
                scale=self.scale,
                target_rank=self.init_rank,
                rank_ema_coef=self.rank_ema_coef,
                betas=group["betas"],
            )

            # 1D parameters: standard AdamW.
            params, grads, square_avgs, momentum, steps, _ = self._init_group(
                group, matrix_params=False
            )
            _single_tensor_adam(
                params,
                grads,
                square_avgs,
                momentum,
                steps,
                lr=group["lr"],
                eps=group["eps"],
                weight_decay=group["weight_decay"],
                maximize=group["maximize"],
                capturable=group["capturable"],
                betas=group["betas"],
            )

        return loss

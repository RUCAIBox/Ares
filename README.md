# Ares: A Spectral-Driven Adaptive-Rank Optimizer for RLVR

Ares is an optimizer for **Reinforcement Learning with Verifiable Rewards (RLVR)** on large language models. It adapts the *rank* of each parameter update to the gradient it actually sees: at every step it reads an effective rank off the gradient's spectral-energy distribution and confines the update to that low-rank subspace.

RLVR gradients progressively concentrate their energy in a few leading singular directions, so their effective update dimensionality declines throughout training. Fixed-rank methods (e.g. GaLore) and rank schedules choose this dimensionality independently of the observed gradient. Ares instead **infers the rank online from the spectrum** and tracks it as it evolves, matching the optimizer to the gradient geometry rather than to a preset rank.

---

## Method

For each **2D weight matrix**, one Ares step does the following (1D parameters — biases, norm gains, scalars — fall back to a standard AdamW step):

1. **Efficient gradient factorization.** A **randomized block power iteration (RBPI)** ([Halko et al., 2011](https://arxiv.org/abs/0909.4061)) extracts the leading singular subspace `(U, Σ, V)` of the gradient. Its cost is governed by the target rank `k` rather than the ambient dimension `n`, replacing the `O(mn²)` cost of a dense SVD with `O(q·mn·(k+p))`.

2. **Spectral energy-based rank estimation.** From the singular values it forms the normalized energy `pᵢ = σᵢ² / Σⱼ σⱼ²`, takes the Shannon entropy `H = −Σᵢ pᵢ log pᵢ`, and converts it to an effective rank `r̂ = exp(H)`. This estimate is smoothed across steps with a bias-corrected exponential moving average and clipped to `[1, k]`:

   ```
   z_t = c·z_{t-1} + (1−c)·r̂_t
   r_t = clip( round( z_t / (1 − c^t) ), 1, k )
   ```

3. **Adaptive low-rank optimization.** It reconstructs the rank-`r` truncated-SVD gradient

   ```
   g̃_t = U[:, :r] · diag(Σ[:r]) · Vᵀ[:r]
   ```

   (retaining the singular-value weighting — *spectral normalization*) and applies an AdamW-style update with bias-corrected first/second moments and decoupled weight decay:

   ```
   θ_t = (1 − ηλ)·θ_{t-1} − η·s · m̂_t / (√v̂_t + ε)
   ```

The full algorithm and analysis are in the accompanying paper (Sec. 4, Algorithm 1, Appendix on RBPI).

---

## Installation

Ares is a single-file `torch.optim.Optimizer` with no dependencies beyond PyTorch.

```bash
pip install torch
```

Copy `Ares.py` into your project (or your training framework's optimizer directory) and import it.

---

## Usage

```python
import torch
from Ares import Ares

model = ...  # your model

optimizer = Ares(
    model.parameters(),
    lr=5e-6,           # learning rate η
    weight_decay=0.0,  # decoupled weight decay λ
    betas=(0.9, 0.95), # Adam moment coefficients
    scale=2.0,         # low-rank update scale s
    init_rank=64,      # target-rank budget k (also the initial tracked rank)
    rank_ema_coef=0.9, # rank-EMA smoothing c
)

for batch in dataloader:
    optimizer.zero_grad()
    loss = compute_loss(model, batch)
    loss.backward()
    optimizer.step()
```

Ares routes by parameter shape automatically: 2D weight matrices get the spectral low-rank update; everything else gets AdamW. No per-layer configuration is required.

### Hyperparameters

| Argument | Symbol | Default | Description |
| --- | --- | --- | --- |
| `lr` | η | `1e-2` | Learning rate. |
| `eps` | ε | `1e-8` | Denominator term for numerical stability. |
| `weight_decay` | λ | `0.0` | Decoupled (AdamW-style) weight decay. |
| `betas` | (β₁, β₂) | `(0.9, 0.95)` | Adam first/second-moment coefficients. |
| `scale` | s | `1.0` | Scale applied to the low-rank update. |
| `init_rank` | k | `32` | Target-rank budget; also the initial tracked rank. |
| `rank_ema_coef` | c | `0.0` | EMA smoothing coefficient for the tracked rank (`0` disables smoothing). |
| `maximize` | | `False` | Maximize instead of minimize the objective. |
| `foreach`, `capturable`, `differentiable` | | | Standard `torch.optim` flags. |

**Recommended settings** (from the paper's RLVR experiments on Qwen3-Base with GRPO): `lr=5e-6`, `init_rank=64`, `rank_ema_coef=0.9`, `scale=2.0`, RBPI oversampling `p=10` and power iterations `q=20` (set inside `randomized_block_power_iteration`). `init_rank` is a *loose upper bound*, not a sensitive knob: once it covers the leading gradient energy, enlarging it only adds low-energy directions that Ares truncates away.

---

## Results

Average accuracy (GRPO on Qwen3-Base, mathematical reasoning benchmarks; higher is better):

| Model scale | AdamW | GaLore | Muon | **Ares** |
| --- | --- | --- | --- | --- |
| Qwen3-1.7B-Base | 22.2 | 22.9 | 2.3 | **27.2** |
| Qwen3-4B-Base | 35.8 | 32.8 | 5.8 | **38.5** |
| Qwen3-8B-Base | 39.8 | 38.8 | 8.9 | **42.1** |

Ares reaches AdamW's accuracy in substantially fewer optimizer steps and converges to a higher final accuracy. The gains also transfer beyond mathematics (MMLU-Pro, GPQA) despite training only on mathematical data. See the paper for full tables, ablations, and analysis.

---

## How it maps to the code

| Paper | `Ares.py` |
| --- | --- |
| RBPI factorization + rank estimation | `randomized_block_power_iteration` |
| Effective rank `r̂ = exp(H)` and EMA smoothing | inside `randomized_block_power_iteration` |
| Rank-`r` truncated-SVD reconstruction `g̃ = UΣVᵀ` | `_single_tensor_lowrank` |
| AdamW moments + decoupled weight decay | `_adam_update`, `_single_tensor_adam` |
| Shape routing (2D vs 1D) | `Ares.step` / `Ares._init_group` |

---

## Citation

If you use Ares, please cite the accompanying paper:

```bibtex
@inproceedings{chen2026ares,
  title     = {Spectral-Driven Adaptive-Rank Optimizer for Large Language Model Reinforcement Learning with Verifiable Rewards},
  author    = {Chen, Zhipeng and Hao, Chuan and Zhao, Wayne Xin and Tao, Ran and Dai, Bryan and Yang, Jian and Wen, Ji-Rong},
  journal   = {arXiv preprint arXiv:2610.xxxxxx},
  year      = {2026}
}
```

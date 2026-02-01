# General Solution (End-to-end: Data → Optimal Labels → Training → RL Fine-tuning)

This document provides a reproducible end-to-end pipeline: starting from real-world data collection, we build a small-scale dataset that can be solved exactly, then train an interpretable neural policy, and finally fine-tune it with reinforcement learning.

> One-sentence summary:
> We use deterministic search/DP to solve **small instances optimally**, then train a neural network to learn a **scalable near-optimal policy**, and finally use RL to directly align the policy with the original objective.

---

## 1. Real-data collection: building a WNBA player feature pool

### 1.1 Data source and goal

- Source: Basketball-Reference WNBA season “advanced” pages (one page per season).
- Goal: provide a realistic statistical feature distribution for the candidate player pool (ability vectors, playing time/availability, etc.).

### 1.2 Processing pipeline

1) Download HTML (one page per season)
2) Parse table fields → unify column names / handle missing values
3) Produce a season-level “player pool” (pool), which will be used for sampling instances

---

## 2. Data augmentation: generate controllable small instances from real distributions

Exact algorithms are reliable only at small scale, but purely synthetic data can drift away from real distributions. We therefore use a **real + synthetic mix** augmentation scheme.

### 2.1 Why augmentation is needed

- Real season decision samples are too few (and we cannot directly observe optimal strategy sequences).
- Purely synthetic data often has unrealistic marginal distributions (abilities/salaries), harming generalization.
- We need many labeled training pairs: `instance → DP-opt solution`.

### 2.2 Augmentation strategy (core idea)

Given a player pool, we generate a multi-season instance by sampling:

- Sample a candidate set (e.g., `n_players=15`)
- For each season, generate an ability matrix and salaries (with optional small perturbations / year-to-year drift)
- Sample environment parameters (salary cap, number of games, opponent strength, etc.)
- Fix small-scale parameters (e.g., `T=3, K=6, L=11, U=12`) so the instance remains exactly solvable

We also support a mixing ratio (`real-frac`):

- Part of the samples closely follow the original pool distribution
- The rest inject perturbations and recombinations to increase coverage and robustness

---

## 3. Deterministic search + dynamic programming: generate global optima for small instances

### 3.1 Action-space compression: enumerate feasible roster masks

Treating roster selection as subset selection over $2^n$ is infeasible.

Our approach:

- Enumerate all rosters that satisfy the size constraint (bitmasks)
- Filter by salary-cap constraint to obtain the feasible action set per season

This turns the per-step action space from exponential to a finite discrete set (e.g., about 1820 masks when `n=15, L=11, U=12`).

### 3.2 DP structure

For a short planning horizon (few seasons), we can define state using “previous-season roster + current season environment/player features”.

The typical DP form is:

$$
V_t(s_t) = \max_{a_t \in \mathcal{A}(s_t)} \left[ r(s_t,a_t) + \gamma V_{t+1}(s_{t+1}) \right]
$$

The transition includes:

- How the new roster affects “continuity/churn” into the next season
- The trade-off between wins and profit in the objective (see [model_general.md](model_general.md))

Outputs:

- Per instance: optimal objective $J^*(s)$
- Per season: optimal action sequence (optimal roster mask sequence)

These are exported as supervised training pairs.

---

## 4. Train a policy network on small-scale optimal labels (BC / imitation learning)

### 4.1 What the supervision signal is

For each training sample, DP provides the optimal sequence:

- Input: `instance` (multi-season player features + environment + initial roster)
- Label: `solution.masks[t]` (the optimal mask per season)

### 4.2 Training objective

We cast action selection as classification:

- The model outputs logits over feasible masks
- Cross-entropy encourages high probability on the optimal mask

BC serves as:

- A fast way to learn a strong initial policy “that behaves like DP”
- A stable initialization for RL (avoids exploring from a random policy)

---

## 5. Reinforcement-learning fine-tuning: directly optimize the original objective

BC imitates DP behavior on the training distribution; under distribution shift and when the decoder/model capacity changes, pure imitation can plateau. We therefore fine-tune with RL on the original reward.

### 5.1 How the environment provides reward

Each episode corresponds to one multi-season instance:

- State $s_t$: current-season environment + player features + previous roster
- Action $a_t$: choose one feasible roster mask
- Reward $r_t$: decomposed from the problem objective (wins/profit/churn penalty, etc.)

### 5.2 Where Actor–Critic gradients come from

- The Actor increases the probability of high-return actions (policy gradient)
- The Critic estimates $V(s_t)$ as a baseline to reduce variance
- The reward signal propagates back via $\nabla_\theta \log \pi_\theta(a_t\mid s_t)$

### 5.3 Why AWBC is also used

RL exploration introduces noise and can drift away from the “feasible and sensible” DP behavior.

AWBC (advantage-weighted behavior cloning) intuition:

- If rollout advantage $A_t$ is positive, the behavior is better than the current value estimate
- We then increase the probability of that behavior, while still keeping a preference for teacher-like actions (with strength scaled by advantage)

---

## 6. Final deliverables (reproducible artifacts)

This repository produces model checkpoints, learning curves, evaluation caches, etc. These are “reproducible experiment artifacts” and are typically not submitted with the final write-up (they are already in `.gitignore`; you can keep them locally).

Key points:

- Training outputs are controlled by the training command’s `--out-dir`
- Evaluation outputs are controlled by the evaluation command’s `--out-dir`
- If you need figures for the report, export/copy them into a stable docs directory (see README and [plots.md](plots.md))

More details:

- Training/RL details and interpretability notes: [train_strategy.md](train_strategy.md)
- How to read the metric plots: [plots.md](plots.md)

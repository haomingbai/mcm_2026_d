# Plot Guide (Validation Set + Different U Constraints)

This document explains the meaning of the evaluation curves exported by the **current training run**, and what “good/expected” levels and trends should look like.

Figures for this run are copied into a more “paper-friendly” directory (so they can be cited in a report without mixing with checkpoints):

- Training curves (loss / greedy rollout, etc.): `docs/figures/nn/ab3b_constraint_shadow_cost_critic_adam_lrsmall_es8_e100_bs32/train/`
- Per-checkpoint evaluation curves (recommended for reporting; Top-K rerank decoding): `docs/figures/nn/ab3b_constraint_shadow_cost_critic_adam_lrsmall_es8_e100_bs32/eval/`

> Note: the training run directory (`datasets/nn_runs/...`) is ignored by `.gitignore` and intermediate weights may be cleaned.
> The documentation figure directory (`docs/figures/...`) is suitable to keep/commit.

> These plots come from `metrics.jsonl` written during training. Each epoch is evaluated on a **random validation split**, and for multiple upper bounds `U` we compute the DP optimum as a reference. They are mainly used to check:
>
> 1) Whether the model improves/converges on validation.
> 2) Whether the policy remains strong under **different roster-size upper bounds U**.

> Important: we prioritize objective-based metrics (objective / gap / regret / ratio).
> `accuracy` is only a supplementary reference, because the DP optimum can have **multiple optimal solutions**, so exact action-match can underestimate quality.

> If you can “only see one curve” (e.g., just U=13): it is usually not a plotting bug—values can be very close and curves overlap, so the last plotted curve covers others.
> To avoid this, we also export **faceted plots (one subplot per U)**.

---

## 1) `val_obj_greedy_by_u.png` (validation objective under different U)

![val_obj_greedy_by_u](docs/figures/nn/ab3b_constraint_shadow_cost_critic_adam_lrsmall_es8_e100_bs32/eval/val_obj_greedy_by_u.png)

(Faceted by U, recommended)

![val_obj_greedy_facets](docs/figures/nn/ab3b_constraint_shadow_cost_critic_adam_lrsmall_es8_e100_bs32/eval/val_obj_greedy_facets.png)

- Ideal level/trend
  - For each curve (each U): increases with epoch or stabilizes.
  - Relative ordering across U: larger U provides a larger action space, so the theoretical optimal objective upper bound is higher;
    whether the model can exploit it depends on training sufficiency.
- What it means
  - Measures robustness/generalization w.r.t. the roster-size upper bound.
  - If it is strong only at the training U but collapses for other U, conditioning on constraints is likely insufficient (or the scorer does not generalize cost/churn well).

> Remark: if U=11/12/13 curves are extremely close (even overlapping), that is not unusual—on some subsets, relaxing U does not meaningfully increase the optimum.

---

## 2) `val_gap_by_u.png` (gap to DP optimum under different U)

(We recommend looking at the best-so-far version first; it is less visually noisy.)

![val_gap_best_by_u](docs/figures/nn/ab3b_constraint_shadow_cost_critic_adam_lrsmall_es8_e100_bs32/eval/val_gap_best_by_u.png)

(Faceted best-so-far, recommended)

![val_gap_best_facets](docs/figures/nn/ab3b_constraint_shadow_cost_critic_adam_lrsmall_es8_e100_bs32/eval/val_gap_best_facets.png)

![val_gap_by_u](docs/figures/nn/ab3b_constraint_shadow_cost_critic_adam_lrsmall_es8_e100_bs32/eval/val_gap_by_u.png)

(Faceted)

![val_gap_facets](docs/figures/nn/ab3b_constraint_shadow_cost_critic_adam_lrsmall_es8_e100_bs32/eval/val_gap_facets.png)

- Definition (as implemented in this repo)
  - $\text{gap} = \frac{J^*(s) - J^{\pi}(s)}{|J^*(s)| + \epsilon}$, where $J^*$ is the DP-optimal objective and $J^{\pi}$ is the model’s greedy-rollout objective.
  - Smaller is better; positive means the model is worse than optimal.
- Ideal level/trend
  - Decreases with epoch and approaches 0.
  - Rule-of-thumb (intuition only):
    - `gap < 0.05`: extremely close to optimal
    - `gap ~ 0.05–0.15`: usable, still room for improvement
    - `gap > 0.2`: often indicates the policy is not stable or does not generalize
- What it means
  - Core alignment plot: visualizes how closely the policy matches the DP optimum.

---

## 3) `val_obj_ratio_by_u.png` (objective ratio model/opt under different U)

(Best-so-far is recommended; it shows “how good you can get so far” more directly.)

![val_obj_ratio_best_by_u](docs/figures/nn/ab3b_constraint_shadow_cost_critic_adam_lrsmall_es8_e100_bs32/eval/val_obj_ratio_best_by_u.png)

(Faceted best-so-far)

![val_obj_ratio_best_facets](docs/figures/nn/ab3b_constraint_shadow_cost_critic_adam_lrsmall_es8_e100_bs32/eval/val_obj_ratio_best_facets.png)

![val_obj_ratio_by_u](docs/figures/nn/ab3b_constraint_shadow_cost_critic_adam_lrsmall_es8_e100_bs32/eval/val_obj_ratio_by_u.png)

(Faceted)

![val_obj_ratio_facets](docs/figures/nn/ab3b_constraint_shadow_cost_critic_adam_lrsmall_es8_e100_bs32/eval/val_obj_ratio_facets.png)

- Definition
  - $\text{ratio} = \frac{J^{\pi}(s)}{J^*(s) + \epsilon}$
- Ideal level/trend
  - Approaches 1 and increases with epoch.
  - Rule-of-thumb (intuition only):
    - `ratio >= 0.90`: very close to the exact DP solver (your target)
    - `ratio ~ 0.80–0.90`: clearly useful but likely needs another training/tuning iteration
    - `ratio < 0.75`: usually indicates the policy is unstable or the value head is not learning well
- What it means
  - A direct, objective-based “percent of optimal” metric.

> Note: if $J^*$ is very small (near 0) for some instances, ratio can be unstable; therefore we also inspect regret/gap.

---

## 4) `val_regret_by_u.png` (regret opt − model under different U)

![val_regret_by_u](docs/figures/nn/ab3b_constraint_shadow_cost_critic_adam_lrsmall_es8_e100_bs32/eval/val_regret_by_u.png)

(Faceted)

![val_regret_facets](docs/figures/nn/ab3b_constraint_shadow_cost_critic_adam_lrsmall_es8_e100_bs32/eval/val_regret_facets.png)

- Definition
  - $\text{regret} = J^*(s) - J^{\pi}(s)$
- Ideal level/trend
  - Approaches 0 and decreases with epoch.
- What it means
  - Absolute distance to the optimal objective; useful for comparing experiments.

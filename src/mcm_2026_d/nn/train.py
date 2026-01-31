from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

from contextlib import nullcontext
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.utils.data
from tqdm import tqdm

from ..schemas import Instance
from ..solver import solve_exact
from .data import PairDataset, collate_pairs
from .env import compute_season_reward_all_actions
from .masks import build_mask_set, build_popcount_table
from .model import TFTPolicy
from .plots import plot_metrics, write_metric


@dataclass(frozen=True)
class TrainConfig:
    data_path: str
    out_dir: str

    batch_size: int = 16
    num_workers: int = 0

    d_model: int = 96
    d_hidden: int = 192
    lstm_hidden: int = 128
    dropout: float = 0.1

    # Optimizer
    optimizer: str = "adamw"  # "adam" or "adamw"

    # Separate learning rates often helps: RL is noisier, so smaller lr is safer.
    lr_bc: float = 2e-4
    lr_rl: float = 1e-4
    weight_decay: float = 1e-4

    # Allow more epochs; early-stopping will usually cut this down.
    bc_epochs: int = 20
    rl_epochs: int = 20

    bc_coef: float = 1.0
    value_coef: float = 0.2
    entropy_coef: float = 0.01

    # Advantage-weighted behavior cloning during RL
    awbc_coef: float = 0.5
    awbc_tau: float = 0.5

    # Early stopping / scheduling (use objective-based metric by default)
    # We maximize ratio (model/opt) and minimize regret/gap.
    monitor_name: str = "val_obj_ratio_mean"
    monitor_u: int | None = None  # None -> use dataset U
    early_stop_min_delta: float = 1e-4
    bc_patience: int = 3
    rl_patience: int = 5

    # ReduceLROnPlateau (optional; works well with early stopping)
    lr_plateau_factor: float = 0.5
    lr_plateau_patience: int = 2
    lr_plateau_min: float = 1e-6

    # RL params
    gamma_override: float | None = None
    max_grad_norm: float = 1.0

    # Memory/compute
    amp: bool = True
    grad_accum: int = 1

    seed: int = 0

    # train/val split
    val_frac: float = 0.1
    val_seed: int = 0

    # Evaluate generalization under different roster upper bounds U
    eval_us: tuple[int, ...] = (11, 12, 13)

    # Precompute DP-opt on validation for each U (can be slow if val is large)
    val_opt_max_instances: int | None = 256

    # ===== Ablations / architecture tweaks (enable one at a time) =====
    use_constraint_env: bool = False
    use_shadow_price: bool = False
    use_cost_modulation: bool = False
    critic_decompose: bool = False


def _set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@torch.no_grad()
def evaluate_greedy_objective(
    *,
    model: TFTPolicy,
    batch,
    mask_set,
    popcount_table,
    device: torch.device,
) -> dict[str, float]:
    model.eval()

    abilities = batch.abilities.to(device)
    salaries = batch.salaries.to(device)
    env = batch.env.to(device)
    w = batch.w.to(device)

    # Denormalize env inputs for reward computation
    # env: [G/50, C/2e6, R_base/2.5e7, rho/8e5, Q_opp/3, t]
    G = env[..., 0] * 50.0
    C = env[..., 1] * 2.0e6
    R_base = env[..., 2] * 2.5e7
    rho = env[..., 3] * 8.0e5
    Q_opp = env[..., 4] * 3.0

    gamma = batch.gamma.to(device)
    lambda_win = batch.lambda_win.to(device)
    beta = batch.beta.to(device)
    churn_penalty = batch.churn_penalty.to(device)

    if batch.x_prev_mask is None:
        return {}

    B, T, n, K = abilities.shape

    prev_mask = batch.x_prev_mask.to(device)
    total = torch.zeros((B,), dtype=torch.float32, device=device)
    disc = torch.ones((B,), dtype=torch.float32, device=device)

    for t in range(T):
        prev_in = ((prev_mask[:, None] >> torch.arange(n, device=device)) & 1).float()

        reward_all, feasible, cost_all, churn_all = compute_season_reward_all_actions(
            mask_matrix=mask_set.mask_matrix,
            sizes=mask_set.sizes,
            masks=mask_set.masks,
            popcount_table=popcount_table,
            abilities_t=abilities[:, t],
            salaries_t=salaries[:, t],
            w=w,
            G_t=G[:, t],
            C_t=C[:, t],
            R_base_t=R_base[:, t],
            rho_t=rho[:, t],
            Q_opp_t=Q_opp[:, t],
            beta=beta,
            lambda_win=lambda_win,
            churn_penalty=churn_penalty,
            prev_mask=prev_mask,
        )

        out = model.forward_season(
            abilities_t=abilities[:, t],
            salaries_t=salaries[:, t],
            prev_in=prev_in,
            env_t=env[:, t],
            lambda_win=lambda_win,
            mask_matrix=mask_set.mask_matrix,
            sizes=mask_set.sizes,
            cost_all=cost_all,
            churn_all=churn_all,
            C_t=C[:, t],
        )

        logits = out.logits.masked_fill(~feasible, -1e9)
        a_idx = torch.argmax(logits, dim=-1)
        a_mask = mask_set.masks[a_idx]

        r = reward_all[torch.arange(B, device=device), a_idx]
        total = total + disc * r
        disc = disc * gamma
        prev_mask = a_mask

    return {"obj_greedy_mean": float(total.mean().item())}


def train(cfg: TrainConfig) -> Path:
    _set_seed(cfg.seed)
    device = _device()

    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ds = PairDataset(cfg.data_path)

    # Split train/val
    rng = np.random.default_rng(cfg.val_seed)
    idx = np.arange(len(ds))
    rng.shuffle(idx)
    n_val = max(1, int(round(len(ds) * float(cfg.val_frac))))
    val_idx = idx[:n_val]
    train_idx = idx[n_val:]

    train_ds = torch.utils.data.Subset(ds, train_idx.tolist())
    val_ds = torch.utils.data.Subset(ds, val_idx.tolist())

    dl = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        collate_fn=collate_pairs,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
    )
    val_dl = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        collate_fn=collate_pairs,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )

    mask_set_np = build_mask_set(n_players=ds.n_players, L=ds.L, U=ds.U)
    mask_set = mask_set_np.to_torch(device)
    popcount_np = build_popcount_table(n_players=ds.n_players)
    popcount_table = torch.as_tensor(popcount_np, dtype=torch.int16, device=device)

    env_dim = 6 + (3 if bool(cfg.use_constraint_env) else 0)
    model = TFTPolicy(
        n_players=ds.n_players,
        K=ds.K,
        env_dim=env_dim,
        use_constraint_env=bool(cfg.use_constraint_env),
        use_shadow_price=bool(cfg.use_shadow_price),
        use_cost_modulation=bool(cfg.use_cost_modulation),
        critic_decompose=bool(cfg.critic_decompose),
        d_model=cfg.d_model,
        d_hidden=cfg.d_hidden,
        lstm_hidden=cfg.lstm_hidden,
        dropout=cfg.dropout,
    ).to(device)

    opt_name = str(cfg.optimizer).lower()
    if opt_name == "adam":
        opt = torch.optim.Adam(model.parameters(), lr=cfg.lr_bc, weight_decay=cfg.weight_decay)
    elif opt_name == "adamw":
        opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr_bc, weight_decay=cfg.weight_decay)
    else:
        raise ValueError(f"Unsupported optimizer: {cfg.optimizer!r} (expected 'adam' or 'adamw')")
    scaler = torch.cuda.amp.GradScaler(enabled=(cfg.amp and device.type == "cuda"))

    def _set_opt_lr(lr: float) -> None:
        for pg in opt.param_groups:
            pg["lr"] = float(lr)

    def _make_sched() -> torch.optim.lr_scheduler.ReduceLROnPlateau:
        # Metric-driven scheduler; we always maximize the monitored scalar.
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt,
            mode="max",
            factor=float(cfg.lr_plateau_factor),
            patience=int(cfg.lr_plateau_patience),
            min_lr=float(cfg.lr_plateau_min),
            threshold=float(cfg.early_stop_min_delta),
            threshold_mode="rel",
        )

    # Save config
    (out_dir / "config.json").write_text(json.dumps(asdict(cfg), ensure_ascii=False, indent=2), encoding="utf-8")

    metrics_path = out_dir / "metrics.jsonl"
    if metrics_path.exists():
        metrics_path.unlink()

    # Precompute DP-opt on a subset of val instances for each U, so we can track
    # how greedy decisions relate to optimal solutions over epochs.
    eval_us = tuple(int(u) for u in cfg.eval_us)
    val_opt_by_u: dict[int, dict[str, dict[str, Any]]] = {}
    val_idx_list = val_idx.tolist()
    if cfg.val_opt_max_instances is not None:
        val_idx_list = val_idx_list[: int(cfg.val_opt_max_instances)]

    for u in eval_us:
        if u < ds.L or u > ds.n_players:
            continue
        cache_path = out_dir / f"val_opt_u{u}.jsonl"
        if cache_path.exists():
            mapping: dict[str, dict[str, Any]] = {}
            with cache_path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    d = json.loads(line)
                    mapping[str(d["id"])] = {"objective": float(d["objective"]), "masks": [int(x) for x in d["masks"]]}
            val_opt_by_u[u] = mapping
            continue

        mapping = {}
        with cache_path.open("w", encoding="utf-8") as f:
            for orig_idx in tqdm(val_idx_list, desc=f"DP-opt val U={u}"):
                row = ds[int(orig_idx)]
                inst_d = row["instance"]
                inst = Instance(**inst_d)
                inst_u = Instance(
                    id=inst.id,
                    n_players=inst.n_players,
                    T=inst.T,
                    K=inst.K,
                    L=inst.L,
                    U=int(u),
                    gamma=inst.gamma,
                    lambda_win=inst.lambda_win,
                    beta=inst.beta,
                    churn_penalty=inst.churn_penalty,
                    G=inst.G,
                    C=inst.C,
                    R_base=inst.R_base,
                    rho=inst.rho,
                    Q_opp=inst.Q_opp,
                    w=inst.w,
                    abilities=inst.abilities,
                    salaries=inst.salaries,
                    x_prev_mask=inst.x_prev_mask,
                )
                sol = solve_exact(inst_u)
                rec = {"id": sol.id, "objective": float(sol.objective), "masks": [int(m) for m in sol.masks]}
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                mapping[sol.id] = {"objective": float(sol.objective), "masks": [int(m) for m in sol.masks]}
        val_opt_by_u[u] = mapping

    def denorm_env(env: torch.Tensor) -> tuple[torch.Tensor, ...]:
        G = env[..., 0] * 50.0
        C = env[..., 1] * 2.0e6
        R_base = env[..., 2] * 2.5e7
        rho = env[..., 3] * 8.0e5
        Q_opp = env[..., 4] * 3.0
        return G, C, R_base, rho, Q_opp

    def mask_invalid_logits(logits: torch.Tensor, feasible: torch.Tensor) -> torch.Tensor:
        # In fp16, using -1e9 can overflow to -inf and cause downstream NaNs in some ops.
        return logits.masked_fill(~feasible, -1e4)

    def _maximize_monitored_from_u(
        *,
        obj_mean: float,
        ratio_mean: float | None,
        regret_mean: float | None,
        gap_mean: float | None,
    ) -> float:
        # Convert the user-facing monitor metric into a scalar that we always *maximize*.
        # - ratio/objective: maximize
        # - regret/gap: minimize -> maximize negative
        name = str(cfg.monitor_name)
        if name == "val_obj_ratio_mean" and ratio_mean is not None:
            return float(ratio_mean)
        if name == "val_regret_mean" and regret_mean is not None:
            return float(-regret_mean)
        if name == "val_gap_mean" and gap_mean is not None:
            return float(-gap_mean)
        return float(obj_mean)

    @torch.no_grad()
    def eval_on_val(epoch: int, phase: str) -> float | None:
        # Evaluate greedy objective on val split for baseline U and for additional U values.
        # For U != ds.U we can't compare to provided teacher, but we can still track objective.
        model.eval()
        device0 = device
        pop = popcount_table

        # Prepare U mask sets
        mask_sets = {}
        for u in eval_us:
            if u < ds.L or u > ds.n_players:
                continue
            mask_sets[u] = build_mask_set(n_players=ds.n_players, L=ds.L, U=u).to_torch(device0)

        totals_by_u = {u: [] for u in mask_sets.keys()}
        gaps_by_u: dict[int, list[float]] = {u: [] for u in mask_sets.keys()}
        ratios_by_u: dict[int, list[float]] = {u: [] for u in mask_sets.keys()}
        regrets_by_u: dict[int, list[float]] = {u: [] for u in mask_sets.keys()}
        season_correct: dict[int, int] = {u: 0 for u in mask_sets.keys()}
        season_total: dict[int, int] = {u: 0 for u in mask_sets.keys()}
        allseason_correct: dict[int, int] = {u: 0 for u in mask_sets.keys()}
        allseason_total: dict[int, int] = {u: 0 for u in mask_sets.keys()}

        for b in val_dl:
            # denorm env
            abilities = b.abilities.to(device0)
            salaries = b.salaries.to(device0)
            env = b.env.to(device0)
            w0 = b.w.to(device0)
            G, Cc, R_base, rho0, Q_opp0 = denorm_env(env)
            gamma0 = b.gamma.to(device0).float()
            lambda0 = b.lambda_win.to(device0).float()
            beta0 = b.beta.to(device0).float()
            churn0 = b.churn_penalty.to(device0).float()

            B, T, n, K = abilities.shape

            for u, ms in mask_sets.items():
                prev = b.x_prev_mask.to(device0)
                disc = torch.ones((B,), dtype=torch.float32, device=device0)
                total = torch.zeros((B,), dtype=torch.float32, device=device0)
                chosen = torch.zeros((B, T), dtype=torch.int64, device=device0)
                for t in range(T):
                    prev_in = ((prev[:, None] >> torch.arange(n, device=device0)) & 1).float()
                    reward_all, feasible, cost_all, churn_all = compute_season_reward_all_actions(
                        mask_matrix=ms.mask_matrix,
                        sizes=ms.sizes,
                        masks=ms.masks,
                        popcount_table=pop,
                        abilities_t=abilities[:, t].float(),
                        salaries_t=salaries[:, t].float(),
                        w=w0.float(),
                        G_t=G[:, t].float(),
                        C_t=Cc[:, t].float(),
                        R_base_t=R_base[:, t].float(),
                        rho_t=rho0[:, t].float(),
                        Q_opp_t=Q_opp0[:, t].float(),
                        beta=beta0,
                        lambda_win=lambda0,
                        churn_penalty=churn0,
                        prev_mask=prev,
                    )
                    outp = model.forward_season(
                        abilities_t=abilities[:, t],
                        salaries_t=salaries[:, t],
                        prev_in=prev_in,
                        env_t=env[:, t],
                        lambda_win=lambda0,
                        mask_matrix=ms.mask_matrix,
                        sizes=ms.sizes,
                        cost_all=cost_all,
                        churn_all=churn_all,
                        C_t=Cc[:, t],
                    )
                    logits = mask_invalid_logits(outp.logits, feasible)
                    a_idx = torch.argmax(logits, dim=-1)
                    a_mask = ms.masks[a_idx]
                    chosen[:, t] = a_mask
                    r = reward_all[torch.arange(B, device=device0), a_idx]
                    total = total + disc * r
                    disc = disc * gamma0
                    prev = a_mask
                totals_by_u[u].append(total.detach().cpu())

                # Compare to DP-opt cache if available
                opt_map = val_opt_by_u.get(u)
                if opt_map:
                    for i in range(B):
                        rid = b.ids[i]
                        rec = opt_map.get(rid)
                        if rec is None:
                            continue
                        opt_obj = float(rec["objective"])
                        model_obj = float(total[i].detach().cpu().item())
                        gaps_by_u[u].append((opt_obj - model_obj) / (abs(opt_obj) + 1e-9))
                        ratios_by_u[u].append(model_obj / (opt_obj + 1e-9))
                        regrets_by_u[u].append(opt_obj - model_obj)

                        opt_masks = rec["masks"]
                        ok_all = True
                        for t in range(T):
                            season_total[u] += 1
                            if int(chosen[i, t].item()) == int(opt_masks[t]):
                                season_correct[u] += 1
                            else:
                                ok_all = False
                        allseason_total[u] += 1
                        if ok_all:
                            allseason_correct[u] += 1

        # Track a single monitored metric for early stopping.
        monitor_u = int(ds.U if cfg.monitor_u is None else cfg.monitor_u)
        monitored: float | None = None

        for u, chunks in totals_by_u.items():
            if not chunks:
                continue
            val_mean = float(torch.cat(chunks).mean().item())
            if u == ds.U:
                # Keep legacy aggregate metric (u=None) for the default-U.
                write_metric(
                    path=metrics_path,
                    phase=phase,
                    epoch=epoch,
                    split="val",
                    name="obj_greedy_mean",
                    value=val_mean,
                    u=None,
                )
                # Also write the per-U version so facets are complete.
                write_metric(
                    path=metrics_path,
                    phase=phase,
                    epoch=epoch,
                    split="val",
                    name="val_obj_greedy_mean",
                    value=val_mean,
                    u=int(u),
                )
            else:
                write_metric(
                    path=metrics_path,
                    phase=phase,
                    epoch=epoch,
                    split="val",
                    name="val_obj_greedy_mean",
                    value=val_mean,
                    u=int(u),
                )

            # objective-based comparison to DP ("几成水平")
            ratio_mean: float | None = None
            regret_mean: float | None = None
            gap_mean: float | None = None

            if ratios_by_u.get(u):
                ratio_mean = float(np.mean(ratios_by_u[u]))
                write_metric(
                    path=metrics_path,
                    phase=phase,
                    epoch=epoch,
                    split="val",
                    name="val_obj_ratio_mean",
                    value=float(ratio_mean),
                    u=int(u),
                )
            if regrets_by_u.get(u):
                regret_mean = float(np.mean(regrets_by_u[u]))
                write_metric(
                    path=metrics_path,
                    phase=phase,
                    epoch=epoch,
                    split="val",
                    name="val_regret_mean",
                    value=float(regret_mean),
                    u=int(u),
                )

            if gaps_by_u.get(u):
                gap_mean = float(np.mean(gaps_by_u[u]))
                write_metric(
                    path=metrics_path,
                    phase=phase,
                    epoch=epoch,
                    split="val",
                    name="val_gap_mean",
                    value=float(gap_mean),
                    u=int(u),
                )
            if season_total.get(u, 0) > 0:
                write_metric(
                    path=metrics_path,
                    phase=phase,
                    epoch=epoch,
                    split="val",
                    name="val_season_acc",
                    value=float(season_correct[u] / max(1, season_total[u])),
                    u=int(u),
                )
            if allseason_total.get(u, 0) > 0:
                write_metric(
                    path=metrics_path,
                    phase=phase,
                    epoch=epoch,
                    split="val",
                    name="val_allseason_acc",
                    value=float(allseason_correct[u] / max(1, allseason_total[u])),
                    u=int(u),
                )

            if u == monitor_u:
                monitored = _maximize_monitored_from_u(
                    obj_mean=float(val_mean),
                    ratio_mean=ratio_mean,
                    regret_mean=regret_mean,
                    gap_mean=gap_mean,
                )

        # Fixed reporting: only ratio/gap/regret, grouped by U.
        try:
            rows = []
            for u in sorted(mask_sets.keys()):
                ratio_m = float(np.mean(ratios_by_u[u])) if ratios_by_u.get(u) else float("nan")
                gap_m = float(np.mean(gaps_by_u[u])) if gaps_by_u.get(u) else float("nan")
                regret_m = float(np.mean(regrets_by_u[u])) if regrets_by_u.get(u) else float("nan")
                rows.append((u, ratio_m, gap_m, regret_m))
            msg = " | ".join(
                [
                    f"U={u}: ratio={ratio_m:.4f} gap={gap_m:.4f} regret={regret_m:.4f}"
                    for (u, ratio_m, gap_m, regret_m) in rows
                ]
            )
            tqdm.write(f"[val/{phase} epoch={epoch}] {msg}")
        except Exception:
            pass

        model.train()
        return monitored

    # ======= Phase A: Behavior cloning =======
    model.train()
    _set_opt_lr(cfg.lr_bc)
    sched_bc = _make_sched()
    global_epoch = 0
    best_bc = -float("inf")
    bad_bc = 0

    step = 0
    for epoch in range(cfg.bc_epochs):
        epoch_g = int(global_epoch)
        pbar = tqdm(dl, desc=f"BC {epoch+1}/{cfg.bc_epochs}")
        loss_sum = 0.0
        n_steps = 0
        for batch in pbar:
            abilities = batch.abilities.to(device)
            salaries = batch.salaries.to(device)
            env = batch.env.to(device)
            w = batch.w.to(device)
            teacher_masks = batch.teacher_masks.to(device)

            gamma = batch.gamma.to(device)
            lambda_win = batch.lambda_win.to(device)
            beta = batch.beta.to(device)
            churn_penalty = batch.churn_penalty.to(device)
            x_prev_mask = batch.x_prev_mask.to(device)

            if cfg.gamma_override is not None:
                gamma = torch.full_like(gamma, float(cfg.gamma_override))

            G, C, R_base, rho, Q_opp = denorm_env(env)

            B, T, n, K = abilities.shape

            loss_ce = torch.tensor(0.0, device=device)
            loss_v = torch.tensor(0.0, device=device)
            loss_vw = torch.tensor(0.0, device=device)
            loss_vpi = torch.tensor(0.0, device=device)
            loss_ent = torch.tensor(0.0, device=device)
            valid_count = 0

            # Precompute teacher returns in float32 (avoid fp16 overflow: salaries/cost ~ 1e6)
            with torch.no_grad():
                rewards_teacher: list[torch.Tensor] = []
                w_terms_teacher: list[torch.Tensor] = []
                pi_terms_teacher: list[torch.Tensor] = []
                prev_tmp = x_prev_mask
                for t in range(T):
                    reward_all_t, _, _, _ = compute_season_reward_all_actions(
                        mask_matrix=mask_set.mask_matrix,
                        sizes=mask_set.sizes,
                        masks=mask_set.masks,
                        popcount_table=popcount_table,
                        abilities_t=abilities[:, t].float(),
                        salaries_t=salaries[:, t].float(),
                        w=w.float(),
                        G_t=G[:, t].float(),
                        C_t=C[:, t].float(),
                        R_base_t=R_base[:, t].float(),
                        rho_t=rho[:, t].float(),
                        Q_opp_t=Q_opp[:, t].float(),
                        beta=beta.float(),
                        lambda_win=lambda_win.float(),
                        churn_penalty=churn_penalty.float(),
                        prev_mask=prev_tmp,
                    )
                    eq = mask_set.masks[None, :] == teacher_masks[:, t][:, None]
                    idx = torch.argmax(eq.to(torch.int64), dim=-1)
                    ok = eq.any(dim=-1)
                    r_t = reward_all_t[torch.arange(B, device=device), idx]
                    rewards_teacher.append(torch.where(ok, r_t, torch.zeros_like(r_t)))

                    if bool(cfg.critic_decompose):
                        m_t = teacher_masks[:, t]
                        sel = ((m_t[:, None] >> torch.arange(n, device=device)) & 1).float()
                        sum_abil = torch.einsum("bn,bnk->bk", sel, abilities[:, t].float())
                        q = torch.einsum("bk,bk->b", sum_abil, w.float())
                        p = torch.sigmoid(beta.float() * (q - Q_opp[:, t].float()))
                        W = G[:, t].float() * p
                        cost = torch.einsum("bn,bn->b", sel, salaries[:, t].float())
                        profit = (R_base[:, t].float() + rho[:, t].float() * W) - cost
                        w_terms_teacher.append(W / (G[:, t].float() + 1e-9))
                        pi_terms_teacher.append(profit / (R_base[:, t].float() + 1e-9))
                    prev_tmp = teacher_masks[:, t]

                Gt = torch.zeros((B,), dtype=torch.float32, device=device)
                teacher_returns: list[torch.Tensor] = []
                for t in reversed(range(T)):
                    Gt = rewards_teacher[t] + gamma.float() * Gt
                    teacher_returns.append(Gt)
                teacher_returns = list(reversed(teacher_returns))

                teacher_returns_w: list[torch.Tensor] | None = None
                teacher_returns_pi: list[torch.Tensor] | None = None
                if bool(cfg.critic_decompose):
                    Gt_w = torch.zeros((B,), dtype=torch.float32, device=device)
                    Gt_pi = torch.zeros((B,), dtype=torch.float32, device=device)
                    tmp_w: list[torch.Tensor] = []
                    tmp_pi: list[torch.Tensor] = []
                    for t in reversed(range(T)):
                        Gt_w = w_terms_teacher[t] + gamma.float() * Gt_w
                        Gt_pi = pi_terms_teacher[t] + gamma.float() * Gt_pi
                        tmp_w.append(Gt_w)
                        tmp_pi.append(Gt_pi)
                    teacher_returns_w = list(reversed(tmp_w))
                    teacher_returns_pi = list(reversed(tmp_pi))

            prev_mask = x_prev_mask
            amp_ctx = torch.cuda.amp.autocast(enabled=scaler.is_enabled()) if device.type == "cuda" else nullcontext()
            with amp_ctx:
                for t in range(T):
                    prev_in = ((prev_mask[:, None] >> torch.arange(n, device=device)) & 1).float()

                    with torch.no_grad():
                        reward_all, feasible, cost_all, churn_all = compute_season_reward_all_actions(
                            mask_matrix=mask_set.mask_matrix,
                            sizes=mask_set.sizes,
                            masks=mask_set.masks,
                            popcount_table=popcount_table,
                            abilities_t=abilities[:, t].float(),
                            salaries_t=salaries[:, t].float(),
                            w=w.float(),
                            G_t=G[:, t].float(),
                            C_t=C[:, t].float(),
                            R_base_t=R_base[:, t].float(),
                            rho_t=rho[:, t].float(),
                            Q_opp_t=Q_opp[:, t].float(),
                            beta=beta.float(),
                            lambda_win=lambda_win.float(),
                            churn_penalty=churn_penalty.float(),
                            prev_mask=prev_mask,
                        )

                    out = model.forward_season(
                        abilities_t=abilities[:, t],
                        salaries_t=salaries[:, t],
                        prev_in=prev_in,
                        env_t=env[:, t],
                        lambda_win=lambda_win,
                        mask_matrix=mask_set.mask_matrix,
                        sizes=mask_set.sizes,
                        cost_all=cost_all,
                        churn_all=churn_all,
                        C_t=C[:, t],
                    )

                    logits = mask_invalid_logits(out.logits, feasible)

                    eq = mask_set.masks[None, :] == teacher_masks[:, t][:, None]
                    idx = torch.argmax(eq.to(torch.int64), dim=-1)
                    ok = eq.any(dim=-1)

                    ce_t = F.cross_entropy(logits, idx, reduction="none")
                    ce_t = torch.where(ok, ce_t, torch.zeros_like(ce_t))
                    loss_ce = loss_ce + ce_t.mean()

                    v_t = out.value
                    loss_v = loss_v + F.mse_loss(v_t, teacher_returns[t].detach())

                    if bool(cfg.critic_decompose) and out.value_w is not None and out.value_pi is not None:
                        assert teacher_returns_w is not None and teacher_returns_pi is not None
                        loss_vw = loss_vw + F.mse_loss(out.value_w, teacher_returns_w[t].detach())
                        loss_vpi = loss_vpi + F.mse_loss(out.value_pi, teacher_returns_pi[t].detach())

                    probs = F.softmax(logits, dim=-1)
                    ent = -(probs * torch.log(probs + 1e-9)).sum(dim=-1).mean()
                    loss_ent = loss_ent + ent

                    prev_mask = teacher_masks[:, t]
                    valid_count += 1

                if bool(cfg.critic_decompose):
                    loss_value = (loss_v + loss_vw + loss_vpi) / (3.0 * valid_count)
                else:
                    loss_value = loss_v / valid_count

                loss = cfg.bc_coef * (loss_ce / valid_count) + cfg.value_coef * loss_value - cfg.entropy_coef * (loss_ent / valid_count)

            loss = loss / max(1, cfg.grad_accum)
            scaler.scale(loss).backward()

            if (step + 1) % cfg.grad_accum == 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)

            step += 1
            loss_sum += float(loss.item())
            n_steps += 1
            pbar.set_postfix({"loss": float(loss.item())})

        stop_bc = False
        write_metric(path=metrics_path, phase="bc", epoch=epoch_g, split="train", name="loss_bc", value=loss_sum / max(1, n_steps))
        write_metric(path=metrics_path, phase="bc", epoch=epoch_g, split="train", name="lr", value=float(opt.param_groups[0]["lr"]))
        # evaluate on val + early stop
        monitored = eval_on_val(epoch=epoch_g, phase="bc")
        if monitored is not None:
            sched_bc.step(float(monitored))
            if float(monitored) > float(best_bc + cfg.early_stop_min_delta):
                best_bc = float(monitored)
                bad_bc = 0
                torch.save(
                    {"model": model.state_dict(), "epoch": epoch, "step": step, "monitored": float(monitored)},
                    out_dir / "ckpt_bc_best.pt",
                )
            else:
                bad_bc += 1
                if cfg.bc_patience > 0 and bad_bc >= int(cfg.bc_patience):
                    stop_bc = True

        ckpt_path = out_dir / f"ckpt_bc_epoch{epoch+1}.pt"
        torch.save({"model": model.state_dict(), "epoch": epoch, "step": step}, ckpt_path)

        global_epoch += 1
        if stop_bc:
            break

    # ======= Phase B: Actor-Critic fine-tune =======
    _set_opt_lr(cfg.lr_rl)
    sched_rl = _make_sched()
    best_rl = -float("inf")
    bad_rl = 0
    for epoch in range(cfg.rl_epochs):
        epoch_g = int(global_epoch)
        pbar = tqdm(dl, desc=f"RL {epoch+1}/{cfg.rl_epochs}")
        loss_sum = 0.0
        pg_sum = 0.0
        v_sum = 0.0
        ent_sum = 0.0
        awbc_sum = 0.0
        n_steps = 0
        for batch in pbar:
            abilities = batch.abilities.to(device)
            salaries = batch.salaries.to(device)
            env = batch.env.to(device)
            w = batch.w.to(device)
            teacher_masks = batch.teacher_masks.to(device)

            gamma = batch.gamma.to(device)
            lambda_win = batch.lambda_win.to(device)
            beta = batch.beta.to(device)
            churn_penalty = batch.churn_penalty.to(device)
            x_prev_mask = batch.x_prev_mask.to(device)

            if cfg.gamma_override is not None:
                gamma = torch.full_like(gamma, float(cfg.gamma_override))

            G, C, R_base, rho, Q_opp = denorm_env(env)

            B, T, n, K = abilities.shape

            # RL is more numerically sensitive (policy gradients + sampling). To avoid fp16 NaNs,
            # run RL in fp32 even when AMP is enabled.
            # RL phase: keep fp32 to avoid fp16 overflows.
            with nullcontext():
                prev_mask = x_prev_mask
                logps = []
                values = []
                values_w = []
                values_pi = []
                rewards = []
                w_terms = []
                pi_terms = []
                entropies = []
                actions_idx = []

                # rollout
                for t in range(T):
                    prev_in = ((prev_mask[:, None] >> torch.arange(n, device=device)) & 1).float()

                    with torch.no_grad():
                        reward_all, feasible, cost_all, churn_all = compute_season_reward_all_actions(
                            mask_matrix=mask_set.mask_matrix,
                            sizes=mask_set.sizes,
                            masks=mask_set.masks,
                            popcount_table=popcount_table,
                            abilities_t=abilities[:, t].float(),
                            salaries_t=salaries[:, t].float(),
                            w=w.float(),
                            G_t=G[:, t].float(),
                            C_t=C[:, t].float(),
                            R_base_t=R_base[:, t].float(),
                            rho_t=rho[:, t].float(),
                            Q_opp_t=Q_opp[:, t].float(),
                            beta=beta.float(),
                            lambda_win=lambda_win.float(),
                            churn_penalty=churn_penalty.float(),
                            prev_mask=prev_mask,
                        )

                    out = model.forward_season(
                        abilities_t=abilities[:, t],
                        salaries_t=salaries[:, t],
                        prev_in=prev_in,
                        env_t=env[:, t],
                        lambda_win=lambda_win,
                        mask_matrix=mask_set.mask_matrix,
                        sizes=mask_set.sizes,
                        cost_all=cost_all,
                        churn_all=churn_all,
                        C_t=C[:, t],
                    )

                    logits = mask_invalid_logits(out.logits, feasible)
                    dist = torch.distributions.Categorical(logits=logits)
                    a_idx = dist.sample()
                    a_mask = mask_set.masks[a_idx]

                    r = reward_all[torch.arange(B, device=device), a_idx]

                    logps.append(dist.log_prob(a_idx))
                    values.append(out.value)
                    if bool(cfg.critic_decompose):
                        values_w.append(out.value_w)
                        values_pi.append(out.value_pi)
                    rewards.append(r)
                    entropies.append(dist.entropy())
                    actions_idx.append(a_idx)

                    if bool(cfg.critic_decompose):
                        sel = ((a_mask[:, None] >> torch.arange(n, device=device)) & 1).float()
                        sum_abil = torch.einsum("bn,bnk->bk", sel, abilities[:, t].float())
                        q = torch.einsum("bk,bk->b", sum_abil, w.float())
                        p = torch.sigmoid(beta.float() * (q - Q_opp[:, t].float()))
                        W = G[:, t].float() * p
                        cost = torch.einsum("bn,bn->b", sel, salaries[:, t].float())
                        profit = (R_base[:, t].float() + rho[:, t].float() * W) - cost
                        w_terms.append(W / (G[:, t].float() + 1e-9))
                        pi_terms.append(profit / (R_base[:, t].float() + 1e-9))

                    prev_mask = a_mask

                # returns + advantages
                returns = []
                Gt = torch.zeros((B,), dtype=torch.float32, device=device)
                for t in reversed(range(T)):
                    Gt = rewards[t] + gamma * Gt
                    returns.append(Gt)
                returns = list(reversed(returns))

                returns_w: list[torch.Tensor] | None = None
                returns_pi: list[torch.Tensor] | None = None
                if bool(cfg.critic_decompose):
                    Gt_w = torch.zeros((B,), dtype=torch.float32, device=device)
                    Gt_pi = torch.zeros((B,), dtype=torch.float32, device=device)
                    tmp_w: list[torch.Tensor] = []
                    tmp_pi: list[torch.Tensor] = []
                    for t in reversed(range(T)):
                        Gt_w = w_terms[t] + gamma * Gt_w
                        Gt_pi = pi_terms[t] + gamma * Gt_pi
                        tmp_w.append(Gt_w)
                        tmp_pi.append(Gt_pi)
                    returns_w = list(reversed(tmp_w))
                    returns_pi = list(reversed(tmp_pi))

                losses_pg = []
                losses_v = []
                losses_vw = []
                losses_vpi = []
                losses_ent = []
                losses_awbc = []

                prev_mask = x_prev_mask
                for t in range(T):
                    adv = (returns[t] - values[t]).detach()
                    losses_pg.append(-(logps[t] * adv).mean())
                    losses_v.append(F.mse_loss(values[t], returns[t].detach()))

                    if bool(cfg.critic_decompose):
                        assert returns_w is not None and returns_pi is not None
                        vw_t = values_w[t]
                        vpi_t = values_pi[t]
                        if vw_t is None or vpi_t is None:
                            raise RuntimeError("critic_decompose=True but model did not return value_w/value_pi")
                        losses_vw.append(F.mse_loss(vw_t, returns_w[t].detach()))
                        losses_vpi.append(F.mse_loss(vpi_t, returns_pi[t].detach()))
                    losses_ent.append(entropies[t].mean())

                    # Advantage-weighted BC (teacher guidance)
                    eq = (mask_set.masks[None, :] == teacher_masks[:, t][:, None])
                    idx_teacher = torch.argmax(eq.to(torch.int64), dim=-1)
                    ok = eq.any(dim=-1)

                    # Recompute logits for CE using the same forward as rollout? (cheap enough for T<=3)
                    prev_in = ((prev_mask[:, None] >> torch.arange(n, device=device)) & 1).float()
                    with torch.no_grad():
                        reward_all, feasible, cost_all, churn_all = compute_season_reward_all_actions(
                            mask_matrix=mask_set.mask_matrix,
                            sizes=mask_set.sizes,
                            masks=mask_set.masks,
                            popcount_table=popcount_table,
                            abilities_t=abilities[:, t].float(),
                            salaries_t=salaries[:, t].float(),
                            w=w.float(),
                            G_t=G[:, t].float(),
                            C_t=C[:, t].float(),
                            R_base_t=R_base[:, t].float(),
                            rho_t=rho[:, t].float(),
                            Q_opp_t=Q_opp[:, t].float(),
                            beta=beta.float(),
                            lambda_win=lambda_win.float(),
                            churn_penalty=churn_penalty.float(),
                            prev_mask=prev_mask,
                        )
                    out2 = model.forward_season(
                        abilities_t=abilities[:, t],
                        salaries_t=salaries[:, t],
                        prev_in=prev_in,
                        env_t=env[:, t],
                        lambda_win=lambda_win,
                        mask_matrix=mask_set.mask_matrix,
                        sizes=mask_set.sizes,
                        cost_all=cost_all,
                        churn_all=churn_all,
                        C_t=C[:, t],
                    )
                    logits2 = mask_invalid_logits(out2.logits, feasible)
                    ce = F.cross_entropy(logits2, idx_teacher, reduction="none")
                    ce = torch.where(ok, ce, torch.zeros_like(ce))
                    weight = torch.exp((adv / max(1e-6, cfg.awbc_tau)).clamp(-10.0, 10.0)).detach()
                    losses_awbc.append((weight * ce).mean())

                    prev_mask = teacher_masks[:, t]

                loss_pg = torch.stack(losses_pg).mean()
                loss_v_main = torch.stack(losses_v).mean()
                if bool(cfg.critic_decompose):
                    loss_vw = torch.stack(losses_vw).mean()
                    loss_vpi = torch.stack(losses_vpi).mean()
                    loss_value = (loss_v_main + loss_vw + loss_vpi) / 3.0
                else:
                    loss_value = loss_v_main
                loss_ent = torch.stack(losses_ent).mean()
                loss_awbc = torch.stack(losses_awbc).mean()

                loss = loss_pg + cfg.value_coef * loss_value - cfg.entropy_coef * loss_ent + cfg.awbc_coef * loss_awbc

            if not torch.isfinite(loss):
                opt.zero_grad(set_to_none=True)
                pbar.set_postfix({"loss": "nan/inf-skip"})
                continue

            loss = loss / max(1, cfg.grad_accum)

            # fp32 step (no scaler) because autocast is disabled
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
            opt.step()
            opt.zero_grad(set_to_none=True)

            pbar.set_postfix({"loss": float(loss.item()), "pg": float(loss_pg.item())})
            loss_sum += float(loss.item())
            pg_sum += float(loss_pg.item())
            v_sum += float(loss_value.item())
            ent_sum += float(loss_ent.item())
            awbc_sum += float(loss_awbc.item())
            n_steps += 1

        ckpt_path = out_dir / f"ckpt_rl_epoch{epoch+1}.pt"
        torch.save({"model": model.state_dict(), "epoch": epoch, "step": step}, ckpt_path)

        write_metric(path=metrics_path, phase="rl", epoch=epoch_g, split="train", name="loss_rl", value=loss_sum / max(1, n_steps))
        write_metric(path=metrics_path, phase="rl", epoch=epoch_g, split="train", name="loss_pg", value=pg_sum / max(1, n_steps))
        write_metric(path=metrics_path, phase="rl", epoch=epoch_g, split="train", name="loss_value", value=v_sum / max(1, n_steps))
        write_metric(path=metrics_path, phase="rl", epoch=epoch_g, split="train", name="entropy", value=ent_sum / max(1, n_steps))
        write_metric(path=metrics_path, phase="rl", epoch=epoch_g, split="train", name="loss_awbc", value=awbc_sum / max(1, n_steps))

        stop_rl = False
        write_metric(path=metrics_path, phase="rl", epoch=epoch_g, split="train", name="lr", value=float(opt.param_groups[0]["lr"]))
        monitored = eval_on_val(epoch=epoch_g, phase="rl")
        if monitored is not None:
            sched_rl.step(float(monitored))
            if float(monitored) > float(best_rl + cfg.early_stop_min_delta):
                best_rl = float(monitored)
                bad_rl = 0
                torch.save(
                    {"model": model.state_dict(), "epoch": epoch, "step": step, "monitored": float(monitored)},
                    out_dir / "ckpt_rl_best.pt",
                )
            else:
                bad_rl += 1
                if cfg.rl_patience > 0 and bad_rl >= int(cfg.rl_patience):
                    stop_rl = True

        global_epoch += 1
        if stop_rl:
            break

    final_path = out_dir / "ckpt_final.pt"
    torch.save({"model": model.state_dict(), "step": step}, final_path)

    # plot
    try:
        plot_metrics(metrics_path, out_dir, subdir="plots")
    except Exception:
        pass
    return final_path

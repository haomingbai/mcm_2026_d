from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import json
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from ..schemas import Instance
from ..solver import solve_exact
from .data import PairDataset, collate_pairs
from .env import compute_season_reward_all_actions
from .masks import build_mask_set, build_popcount_table
from .model import TFTPolicy
from .plots import write_metric


@dataclass(frozen=True)
class EvalConfig:
    data_path: str
    ckpt_path: str
    out_dir: str

    val_frac: float = 0.1
    seed: int = 0
    batch_size: int = 32
    num_workers: int = 0

    # Evaluate across different U constraints
    eval_us: tuple[int, ...] = (11, 12, 13)

    # How many val instances to solve optimally for each U (None means all)
    val_max_instances: int | None = None

    # Optional: top-k re-ranking decode (k=1 means pure argmax logits)
    topk: int = 1


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _split_indices(n: int, val_frac: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    n_val = max(1, int(round(n * val_frac)))
    val_idx = idx[:n_val]
    train_idx = idx[n_val:]
    return train_idx, val_idx


@torch.no_grad()
def _greedy_rollout_objective(
    *,
    model: TFTPolicy,
    batch,
    mask_set,
    popcount_table,
    device: torch.device,
    topk: int = 1,
) -> tuple[torch.Tensor, list[list[int]]]:
    model.eval()

    abilities = batch.abilities.to(device)
    salaries = batch.salaries.to(device)
    env = batch.env.to(device)
    w = batch.w.to(device)

    # denorm env
    G = env[..., 0] * 50.0
    C = env[..., 1] * 2.0e6
    R_base = env[..., 2] * 2.5e7
    rho = env[..., 3] * 8.0e5
    Q_opp = env[..., 4] * 3.0

    gamma = batch.gamma.to(device).float()
    lambda_win = batch.lambda_win.to(device).float()
    beta = batch.beta.to(device).float()
    churn_penalty = batch.churn_penalty.to(device).float()

    B, T, n, K = abilities.shape

    prev_mask = batch.x_prev_mask.to(device)
    total = torch.zeros((B,), dtype=torch.float32, device=device)
    disc = torch.ones((B,), dtype=torch.float32, device=device)

    chosen: list[list[int]] = [[] for _ in range(B)]

    for t in range(T):
        prev_in = ((prev_mask[:, None] >> torch.arange(n, device=device)) & 1).float()

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
            lambda_win=batch.lambda_win.to(device),
            mask_matrix=mask_set.mask_matrix,
            sizes=mask_set.sizes,
            cost_all=cost_all,
            churn_all=churn_all,
            C_t=C[:, t],
        )

        logits = out.logits.masked_fill(~feasible, -1e4)

        k = int(topk)
        if k <= 1:
            a_idx = torch.argmax(logits, dim=-1)
        else:
            k = min(k, int(logits.shape[1]))
            top_idx = torch.topk(logits, k=k, dim=-1).indices  # (B,k)
            cand_r = reward_all.gather(1, top_idx)  # (B,k)
            best_j = torch.argmax(cand_r, dim=-1)
            a_idx = top_idx[torch.arange(B, device=device), best_j]
        a_mask = mask_set.masks[a_idx]

        r = reward_all[torch.arange(B, device=device), a_idx]
        total = total + disc * r
        disc = disc * gamma

        for i in range(B):
            chosen[i].append(int(a_mask[i].item()))

        prev_mask = a_mask

    return total, chosen


def _precompute_opt_solutions(
    *,
    ds: PairDataset,
    val_indices: Iterable[int],
    us: tuple[int, ...],
    out_dir: Path,
    max_instances: int | None,
) -> dict[int, dict[str, dict]]:
    """Compute DP-optimal solutions for validation subset under different U.

    Returns mapping: u -> id -> {"objective": float, "masks": [int,..]}
    Also caches to out_dir/val_opt_u{u}.jsonl.
    """

    out: dict[int, dict[str, dict]] = {}
    val_idx_list = list(val_indices)
    if max_instances is not None:
        val_idx_list = val_idx_list[: int(max_instances)]

    for u in us:
        cache_path = out_dir / f"val_opt_u{u}.jsonl"
        if cache_path.exists():
            mapping: dict[str, dict] = {}
            with cache_path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    d = json.loads(line)
                    mapping[str(d["id"])] = {"objective": float(d["objective"]), "masks": [int(x) for x in d["masks"]]}
            out[u] = mapping
            continue

        mapping = {}
        with cache_path.open("w", encoding="utf-8") as f:
            for idx in tqdm(val_idx_list, desc=f"DP-opt val U={u}"):
                inst_d = ds[idx]["instance"]
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
                row = {"id": sol.id, "objective": float(sol.objective), "masks": [int(m) for m in sol.masks]}
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                mapping[sol.id] = {"objective": float(sol.objective), "masks": [int(m) for m in sol.masks]}

        out[u] = mapping

    return out


def evaluate(cfg: EvalConfig) -> Path:
    device = _device()
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ds = PairDataset(cfg.data_path)
    _, val_idx = _split_indices(len(ds), cfg.val_frac, cfg.seed)

    val_subset = Subset(ds, val_idx.tolist())
    val_dl = DataLoader(
        val_subset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        collate_fn=collate_pairs,
        pin_memory=torch.cuda.is_available(),
    )

    # Load model config implicitly from dataset shape; use safe defaults and load weights.
    # d_model is stored in checkpoint's run config; if unavailable, assume 96.
    ckpt = torch.load(cfg.ckpt_path, map_location="cpu")

    sd = ckpt["model"]
    d_model = int(sd["player_proj.weight"].shape[0])
    lstm_hidden = int(sd["temporal.weight_ih_l0"].shape[0] // 4)
    env_dim = int(sd["env_proj.weight"].shape[1])

    use_shadow_price = ("lambda_head.weight" in sd) or any(k.startswith("lambda_head.") for k in sd.keys())
    use_cost_modulation = ("cost_proj.weight" in sd) or any(k.startswith("cost_proj.") for k in sd.keys())
    critic_decompose = any(k.startswith("value_head_w.") for k in sd.keys()) and any(k.startswith("value_head_pi.") for k in sd.keys())
    use_constraint_env = env_dim > 6

    model = TFTPolicy(
        n_players=ds.n_players,
        K=ds.K,
        env_dim=env_dim,
        use_constraint_env=use_constraint_env,
        use_shadow_price=use_shadow_price,
        use_cost_modulation=use_cost_modulation,
        critic_decompose=critic_decompose,
        d_model=d_model,
        d_hidden=max(192, d_model * 2),
        lstm_hidden=lstm_hidden,
        dropout=0.1,
    ).to(device)
    model.load_state_dict(sd, strict=True)

    popcount = torch.as_tensor(build_popcount_table(n_players=ds.n_players), dtype=torch.int16, device=device)

    # Precompute DP-opt for each U on val subset
    opt = _precompute_opt_solutions(
        ds=ds,
        val_indices=val_idx,
        us=cfg.eval_us,
        out_dir=out_dir,
        max_instances=cfg.val_max_instances,
    )

    metrics_path = out_dir / "metrics.jsonl"
    if metrics_path.exists():
        metrics_path.unlink()

    for u in cfg.eval_us:
        if u < ds.L:
            continue
        if u > ds.n_players:
            continue

        mask_set = build_mask_set(n_players=ds.n_players, L=ds.L, U=u).to_torch(device)

        all_obj = []
        gaps: list[float] = []
        ratios: list[float] = []
        regrets: list[float] = []
        season_correct = 0
        season_total = 0
        allseason_correct = 0
        allseason_total = 0

        for batch in tqdm(val_dl, desc=f"Eval greedy U={u}"):
            obj, chosen_masks = _greedy_rollout_objective(
                model=model,
                batch=batch,
                mask_set=mask_set,
                popcount_table=popcount,
                device=device,
                topk=int(cfg.topk),
            )

            all_obj.append(obj.detach().cpu())

            opt_map = opt.get(u, {})
            if opt_map:
                for i in range(len(batch.ids)):
                    rid = batch.ids[i]
                    rec = opt_map.get(rid)
                    if rec is None:
                        continue
                    opt_obj = float(rec["objective"])
                    model_obj = float(obj[i].detach().cpu().item())
                    gaps.append((opt_obj - model_obj) / (abs(opt_obj) + 1e-9))
                    # objective-based metrics (ratio closer to 1 is better; regret closer to 0 is better)
                    ratios.append(model_obj / (opt_obj + 1e-9))
                    regrets.append(opt_obj - model_obj)

                    opt_masks = rec["masks"]
                    ok_all = True
                    for t, m in enumerate(opt_masks):
                        season_total += 1
                        if int(chosen_masks[i][t]) == int(m):
                            season_correct += 1
                        else:
                            ok_all = False
                    allseason_total += 1
                    if ok_all:
                        allseason_correct += 1

        obj_cat = torch.cat(all_obj)
        obj_mean = float(obj_cat.mean().item())

        write_metric(path=metrics_path, phase="eval", epoch=0, split="val", name="val_obj_greedy_mean", value=obj_mean, u=int(u))
        if ratios:
            write_metric(path=metrics_path, phase="eval", epoch=0, split="val", name="val_obj_ratio_mean", value=float(np.mean(ratios)), u=int(u))
        if regrets:
            write_metric(path=metrics_path, phase="eval", epoch=0, split="val", name="val_regret_mean", value=float(np.mean(regrets)), u=int(u))
        if gaps:
            write_metric(
                path=metrics_path,
                phase="eval",
                epoch=0,
                split="val",
                name="val_gap_mean",
                value=float(np.mean(gaps)),
                u=int(u),
            )
        if season_total > 0:
            write_metric(
                path=metrics_path,
                phase="eval",
                epoch=0,
                split="val",
                name="val_season_acc",
                value=float(season_correct / max(1, season_total)),
                u=int(u),
            )
        if allseason_total > 0:
            write_metric(
                path=metrics_path,
                phase="eval",
                epoch=0,
                split="val",
                name="val_allseason_acc",
                value=float(allseason_correct / max(1, allseason_total)),
                u=int(u),
            )

    return metrics_path


def evaluate_run_dir(
    *,
    data_path: str,
    run_dir: str,
    out_dir: str,
    val_frac: float = 0.1,
    seed: int = 0,
    batch_size: int = 32,
    eval_us: tuple[int, ...] = (11, 12, 13),
    val_max_instances: int | None = 256,
    topk: int = 1,
) -> Path:
    """Evaluate every checkpoint in a run directory and export curves."""

    device = _device()
    run_p = Path(run_dir)
    out_p = Path(out_dir)
    out_p.mkdir(parents=True, exist_ok=True)

    ds = PairDataset(data_path)
    _, val_idx = _split_indices(len(ds), val_frac, seed)
    val_subset = Subset(ds, val_idx.tolist())
    val_dl = DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_pairs,
        pin_memory=torch.cuda.is_available(),
    )

    popcount = torch.as_tensor(build_popcount_table(n_players=ds.n_players), dtype=torch.int16, device=device)

    # Precompute DP-opt for val subset
    opt = _precompute_opt_solutions(
        ds=ds,
        val_indices=val_idx,
        us=eval_us,
        out_dir=out_p,
        max_instances=val_max_instances,
    )

    def _epoch_from_name(p: Path) -> int:
        s = p.stem
        # ckpt_bc_epoch6
        return int(s.split("epoch")[-1])

    # Discover checkpoints (sort numerically; lexicographic sort would be epoch1, epoch10, epoch2 ...)
    bc_ckpts = sorted(run_p.glob("ckpt_bc_epoch*.pt"), key=_epoch_from_name)
    rl_ckpts = sorted(run_p.glob("ckpt_rl_epoch*.pt"), key=_epoch_from_name)
    bc_n = len(bc_ckpts)

    checkpoints: list[tuple[str, int, Path]] = []
    for p in bc_ckpts:
        e = _epoch_from_name(p)
        checkpoints.append(("bc", e - 1, p))
    for p in rl_ckpts:
        e = _epoch_from_name(p)
        checkpoints.append(("rl", bc_n + (e - 1), p))
    if (run_p / "ckpt_final.pt").exists():
        checkpoints.append(("final", bc_n + len(rl_ckpts), run_p / "ckpt_final.pt"))

    metrics_path = out_p / "run_eval_metrics.jsonl"
    if metrics_path.exists():
        metrics_path.unlink()

    for phase, epoch_g, ckpt_path in checkpoints:
        ckpt = torch.load(ckpt_path, map_location="cpu")
        sd = ckpt["model"]
        d_model = int(sd["player_proj.weight"].shape[0])
        lstm_hidden = int(sd["temporal.weight_ih_l0"].shape[0] // 4)
        env_dim = int(sd["env_proj.weight"].shape[1])
        use_shadow_price = ("lambda_head.weight" in sd) or any(k.startswith("lambda_head.") for k in sd.keys())
        use_cost_modulation = ("cost_proj.weight" in sd) or any(k.startswith("cost_proj.") for k in sd.keys())
        critic_decompose = any(k.startswith("value_head_w.") for k in sd.keys()) and any(k.startswith("value_head_pi.") for k in sd.keys())
        use_constraint_env = env_dim > 6

        model = TFTPolicy(
            n_players=ds.n_players,
            K=ds.K,
            env_dim=env_dim,
            use_constraint_env=use_constraint_env,
            use_shadow_price=use_shadow_price,
            use_cost_modulation=use_cost_modulation,
            critic_decompose=critic_decompose,
            d_model=d_model,
            d_hidden=max(192, d_model * 2),
            lstm_hidden=lstm_hidden,
            dropout=0.1,
        ).to(device)
        model.load_state_dict(sd, strict=True)

        for u in eval_us:
            if u < ds.L or u > ds.n_players:
                continue
            mask_set = build_mask_set(n_players=ds.n_players, L=ds.L, U=u).to_torch(device)
            obj_chunks = []
            gaps: list[float] = []
            ratios: list[float] = []
            regrets: list[float] = []
            season_correct = 0
            season_total = 0
            allseason_correct = 0
            allseason_total = 0

            opt_map = opt.get(u, {})

            for batch in val_dl:
                obj, chosen_masks = _greedy_rollout_objective(
                    model=model,
                    batch=batch,
                    mask_set=mask_set,
                    popcount_table=popcount,
                    device=device,
                    topk=int(topk),
                )
                obj_chunks.append(obj.detach().cpu())
                if opt_map:
                    for i in range(len(batch.ids)):
                        rid = batch.ids[i]
                        rec = opt_map.get(rid)
                        if rec is None:
                            continue
                        opt_obj = float(rec["objective"])
                        model_obj = float(obj[i].detach().cpu().item())
                        gaps.append((opt_obj - model_obj) / (abs(opt_obj) + 1e-9))
                        ratios.append(model_obj / (opt_obj + 1e-9))
                        regrets.append(opt_obj - model_obj)
                        opt_masks = rec["masks"]
                        ok_all = True
                        for t, m in enumerate(opt_masks):
                            season_total += 1
                            if int(chosen_masks[i][t]) == int(m):
                                season_correct += 1
                            else:
                                ok_all = False
                        allseason_total += 1
                        if ok_all:
                            allseason_correct += 1

            obj_mean = float(torch.cat(obj_chunks).mean().item())
            write_metric(path=metrics_path, phase=phase, epoch=epoch_g, split="val", name="val_obj_greedy_mean", value=obj_mean, u=int(u))
            if ratios:
                write_metric(path=metrics_path, phase=phase, epoch=epoch_g, split="val", name="val_obj_ratio_mean", value=float(np.mean(ratios)), u=int(u))
            if regrets:
                write_metric(path=metrics_path, phase=phase, epoch=epoch_g, split="val", name="val_regret_mean", value=float(np.mean(regrets)), u=int(u))
            if gaps:
                write_metric(path=metrics_path, phase=phase, epoch=epoch_g, split="val", name="val_gap_mean", value=float(np.mean(gaps)), u=int(u))
            if season_total > 0:
                write_metric(path=metrics_path, phase=phase, epoch=epoch_g, split="val", name="val_season_acc", value=float(season_correct / max(1, season_total)), u=int(u))
            if allseason_total > 0:
                write_metric(path=metrics_path, phase=phase, epoch=epoch_g, split="val", name="val_allseason_acc", value=float(allseason_correct / max(1, allseason_total)), u=int(u))

    return metrics_path

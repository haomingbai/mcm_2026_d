from __future__ import annotations

from pathlib import Path
import shutil

import typer
import pandas as pd
from tqdm import tqdm

from .dataset import make_pair_row, write_jsonl
from .generate import generate_many
from .mixed_generate import (
    generate_mixed_instances,
    generate_real_augmented_instances,
    load_cached_pool,
)
from .realdata.bref_wnba import (
    BRefFetchConfig,
    build_feature_pool,
    default_cache_dir,
    fetch_year_html,
    load_year_table,
)
from .solver import solve_exact

app = typer.Typer(no_args_is_help=True)


@app.callback()
def _root() -> None:
    """MCM 2026 D: instance generation + exact solve + dataset export."""
    return


@app.command("generate-and-solve")
def generate_and_solve(
    out: Path = typer.Option(..., help="输出JSONL文件路径（每行=instance+solution）"),
    out_instances: Path | None = typer.Option(None, help="可选：仅输出instance的JSONL"),
    out_solutions: Path | None = typer.Option(None, help="可选：仅输出solution的JSONL"),
    out_summary: Path | None = typer.Option(None, help="可选：输出汇总CSV（每行一个样本）"),
    n: int = typer.Option(100, help="生成样本数"),
    seed: int = typer.Option(0, help="随机种子"),
    n_players: int = typer.Option(15, help="候选球员数（<=15建议）"),
    T: int = typer.Option(3, help="赛季数（<=3建议）"),
    K: int = typer.Option(6, help="能力维度数"),
    L: int = typer.Option(11, help="阵容下限"),
    U: int = typer.Option(12, help="阵容上限"),
) -> None:
    instances = generate_many(n=n, seed=seed, n_players=n_players, T=T, K=K, L=L, U=U)

    rows = []
    inst_rows = []
    sol_rows = []
    summary_rows = []
    for inst in tqdm(instances, desc="Solving"):
        sol = solve_exact(inst)
        rows.append(make_pair_row(inst, sol))

        if out_instances is not None:
            inst_rows.append(inst.to_dict())
        if out_solutions is not None:
            sol_rows.append(sol.to_dict())
        if out_summary is not None:
            summary_rows.append(
                {
                    "id": inst.id,
                    "objective": sol.objective,
                    "T": inst.T,
                    "n_players": inst.n_players,
                    "L": inst.L,
                    "U": inst.U,
                    "x_prev_mask": inst.x_prev_mask,
                    "masks": ",".join(str(m) for m in sol.masks),
                    "avg_profit": float(sum(m.profit for m in sol.per_season) / len(sol.per_season)),
                    "avg_W": float(sum(m.W for m in sol.per_season) / len(sol.per_season)),
                }
            )

    write_jsonl(out, rows)
    typer.echo(f"Wrote {len(rows)} rows -> {out}")

    if out_instances is not None:
        write_jsonl(out_instances, inst_rows)
        typer.echo(f"Wrote {len(inst_rows)} rows -> {out_instances}")
    if out_solutions is not None:
        write_jsonl(out_solutions, sol_rows)
        typer.echo(f"Wrote {len(sol_rows)} rows -> {out_solutions}")
    if out_summary is not None:
        out_summary.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(summary_rows).to_csv(out_summary, index=False)
        typer.echo(f"Wrote {len(summary_rows)} rows -> {out_summary}")


@app.command("scrape-bref")
def scrape_bref(
    years: list[int] = typer.Option(..., help="要抓取的年份列表，例如：--years 2022 --years 2023"),
    cache_dir: Path = typer.Option(default_cache_dir(), help="缓存目录（HTML与pool CSV）"),
    force: bool = typer.Option(False, help="强制重新下载HTML"),
    min_mp: float = typer.Option(200.0, help="最小上场分钟过滤阈值"),
) -> None:
    """抓取 Basketball-Reference WNBA 年度 advanced 数据，并生成可复用的球员特征池 CSV。"""

    cfg = BRefFetchConfig(cache_dir=cache_dir)
    pool_dir = cache_dir / "pool"
    pool_dir.mkdir(parents=True, exist_ok=True)

    ok = 0
    for y in years:
        try:
            html_path = fetch_year_html(year=y, table="advanced", cfg=cfg, force=force)
            last_err: Exception | None = None
            df = None
            for table_id in ["advanced", "all_advanced", "advanced_stats"]:
                try:
                    df = load_year_table(html_path=html_path, table_id=table_id)
                    break
                except Exception as e:
                    last_err = e
                    df = None
            if df is None:
                raise last_err or RuntimeError("failed to locate advanced table")
            pool = build_feature_pool(advanced_df=df, year=y, min_mp=min_mp)
            out_csv = pool_dir / f"pool_{y}.csv"
            pool.to_csv(out_csv, index=False)
            typer.echo(f"OK year={y}: {len(pool)} players -> {out_csv}")
            ok += 1
        except Exception as e:
            typer.echo(f"FAIL year={y}: {e}")

    if ok == 0:
        raise typer.Exit(code=1)


@app.command("generate-and-solve-mixed")
def generate_and_solve_mixed(
    out: Path = typer.Option(..., help="输出pairs.jsonl（每行=instance+solution）"),
    pool_dir: Path = typer.Option(default_cache_dir() / "pool", help="scrape-bref 生成的pool目录"),
    years: list[int] = typer.Option(..., help="使用哪些真实年份做抽样，例如：--years 2022 --years 2023"),
    real_frac: float = typer.Option(0.7, help="样本中使用真实球员池的比例(0~1)"),
    n: int = typer.Option(200, help="样本数"),
    seed: int = typer.Option(0, help="随机种子"),
    n_players: int = typer.Option(15, help="候选球员数"),
    T: int = typer.Option(3, help="赛季数"),
    K: int = typer.Option(6, help="能力维度数"),
    L: int = typer.Option(11, help="阵容下限"),
    U: int = typer.Option(12, help="阵容上限"),
) -> None:
    """混合真实+合成生成小规模实例，并用精确DP求最优解导出训练数据。"""

    pool = None
    try:
        pool = load_cached_pool(pool_dir)
    except Exception as e:
        typer.echo(f"WARN: cannot load real pool ({e}); will fallback to synthetic only")

    instances = generate_mixed_instances(
        n=n,
        seed=seed,
        pool=pool,
        real_frac=real_frac,
        years=years,
        n_players=n_players,
        T=T,
        K=K,
        L=L,
        U=U,
    )

    rows = []
    for inst in tqdm(instances, desc="Solving"):
        sol = solve_exact(inst)
        rows.append(make_pair_row(inst, sol))

    write_jsonl(out, rows)
    typer.echo(f"Wrote {len(rows)} rows -> {out}")


@app.command("generate-and-solve-real-aug")
def generate_and_solve_real_aug(
    out: Path = typer.Option(..., help="输出pairs.jsonl（每行=instance+solution）"),
    pool_dir: Path = typer.Option(default_cache_dir() / "pool", help="scrape-bref 生成的pool目录"),
    years: list[int] = typer.Option(..., help="使用哪些真实年份做抽样，例如：--years 2022 --years 2023"),
    n: int = typer.Option(4000, help="样本数（最终输出条数）"),
    seed: int = typer.Option(0, help="随机种子"),
    aug_per_base: int = typer.Option(2, help="每条base样本生成多少条增强样本（约 1:aug_per_base）"),
    abilities_sigma: float = typer.Option(0.10, help="能力向量增强噪声强度"),
    salary_sigma: float = typer.Option(0.05, help="薪资增强乘性噪声强度"),
    n_players: int = typer.Option(15, help="候选球员数"),
    T: int = typer.Option(3, help="赛季数"),
    K: int = typer.Option(6, help="能力维度数"),
    L: int = typer.Option(11, help="阵容下限"),
    U: int = typer.Option(12, help="阵容上限"),
) -> None:
    """仅使用真实球员池生成数据，并按指定比例做数据增强后求精确最优解导出。"""

    pool = load_cached_pool(pool_dir)

    inst_with_meta = generate_real_augmented_instances(
        n=n,
        seed=seed,
        pool=pool,
        years=years,
        aug_per_base=aug_per_base,
        abilities_sigma=abilities_sigma,
        salary_sigma=salary_sigma,
        n_players=n_players,
        T=T,
        K=K,
        L=L,
        U=U,
    )

    rows = []
    n_base = 0
    n_aug = 0
    for inst, meta in tqdm(inst_with_meta, desc="Solving"):
        sol = solve_exact(inst)
        rows.append(make_pair_row(inst, sol, meta=meta))
        if bool(meta.get("augmented")):
            n_aug += 1
        else:
            n_base += 1

    write_jsonl(out, rows)
    typer.echo(f"Wrote {len(rows)} rows -> {out}")
    typer.echo(f"Breakdown: base={n_base}, augmented={n_aug} (approx 1:{aug_per_base})")


@app.command("version")
def version() -> None:
    from . import __version__

    typer.echo(__version__)


@app.command("train-nn")
def train_nn(
    data: Path = typer.Option(..., help="训练数据 JSONL（如 datasets/pairs_real_aug_4000.jsonl）"),
    out_dir: Path = typer.Option(Path("datasets/nn_runs/run1"), help="输出目录（ckpt/metrics/config）"),
    batch_size: int = typer.Option(16, help="batch size（8GB 显存建议 8~32）"),
    lr: float | None = typer.Option(None, help="(兼容旧参数) 同时设置 BC/RL 学习率"),
    lr_bc: float = typer.Option(2e-4, help="BC 学习率（监督预训练）"),
    lr_rl: float = typer.Option(1e-4, help="RL 学习率（强化学习微调）"),
    bc_epochs: int = typer.Option(20, help="监督预训练轮数（配合早停可设大一些）"),
    rl_epochs: int = typer.Option(20, help="强化学习微调轮数（配合早停可设大一些）"),
    bc_patience: int = typer.Option(3, help="BC 早停 patience（<=0 禁用）"),
    rl_patience: int = typer.Option(5, help="RL 早停 patience（<=0 禁用）"),
    early_stop_min_delta: float = typer.Option(1e-4, help="早停最小提升阈值"),
    monitor_name: str = typer.Option(
        "val_obj_ratio_mean",
        help="早停监控指标：val_obj_ratio_mean(越大越好) / val_regret_mean(越小越好) / val_gap_mean(越小越好) / obj_greedy_mean",
    ),
    monitor_u: int = typer.Option(0, help="监控用的 U（0 表示用数据集 U）"),
    lr_plateau_factor: float = typer.Option(0.5, help="Plateau 降学习率倍率"),
    lr_plateau_patience: int = typer.Option(2, help="Plateau patience"),
    lr_plateau_min: float = typer.Option(1e-6, help="学习率下限"),
    optimizer: str = typer.Option("adam", help="优化器：adam 或 adamw"),
    d_model: int = typer.Option(96, help="模型宽度（越大越占显存）"),
    lstm_hidden: int = typer.Option(128, help="LSTM hidden"),
    dropout: float = typer.Option(0.1, help="dropout"),
    amp: bool = typer.Option(True, help="启用混合精度（推荐）"),
    seed: int = typer.Option(0, help="随机种子"),
    val_frac: float = typer.Option(0.1, help="随机划分验证集比例"),
    val_seed: int = typer.Option(0, help="验证集划分随机种子"),
    eval_us: str = typer.Option("11,12,13", help="评估用的 U 列表（逗号分隔）"),
    val_opt_max_instances: int = typer.Option(256, help="每个 U 预计算DP最优的验证样本数（0表示全部）"),

    # ===== Ablations (enable one at a time) =====
    use_constraint_env: bool = typer.Option(False, help="在 env embedding 追加 cap_slack/cap_tightness/roster_slack"),
    use_shadow_price: bool = typer.Option(False, help="新增影子价格头 lambda_t=softplus(w·h+b)"),
    use_cost_modulation: bool = typer.Option(False, help="成本显式调制：a~=a-lambda*c（需要 use_shadow_price）"),
    critic_decompose: bool = typer.Option(False, help="critic 分项预测并按 lambda_win 合成 V"),
) -> None:
    """监督(BC) + Actor-Critic(RL) 训练一个可解释策略网络。

    依赖：需要安装 ml extra：
      uv sync --extra ml
    """

    try:
        from .nn.train import TrainConfig, train
    except Exception as e:  # pragma: no cover
        raise typer.BadParameter(
            f"无法导入训练模块（可能未安装 ml 依赖）。请先运行 `uv sync --extra ml`。详细错误：{e}"
        )

    if lr is not None:
        lr_bc = float(lr)
        lr_rl = float(lr)

    cfg = TrainConfig(
        data_path=str(data),
        out_dir=str(out_dir),
        batch_size=batch_size,
        optimizer=str(optimizer),
        lr_bc=float(lr_bc),
        lr_rl=float(lr_rl),
        bc_epochs=bc_epochs,
        rl_epochs=rl_epochs,
        bc_patience=int(bc_patience),
        rl_patience=int(rl_patience),
        early_stop_min_delta=float(early_stop_min_delta),
        monitor_name=str(monitor_name),
        monitor_u=(None if int(monitor_u) == 0 else int(monitor_u)),
        lr_plateau_factor=float(lr_plateau_factor),
        lr_plateau_patience=int(lr_plateau_patience),
        lr_plateau_min=float(lr_plateau_min),
        d_model=d_model,
        lstm_hidden=lstm_hidden,
        dropout=dropout,
        amp=amp,
        seed=seed,
        val_frac=float(val_frac),
        val_seed=int(val_seed),
        eval_us=tuple(int(x.strip()) for x in eval_us.split(",") if x.strip()),
        val_opt_max_instances=(None if int(val_opt_max_instances) == 0 else int(val_opt_max_instances)),

        use_constraint_env=bool(use_constraint_env),
        use_shadow_price=bool(use_shadow_price),
        use_cost_modulation=bool(use_cost_modulation),
        critic_decompose=bool(critic_decompose),
    )

    ckpt = train(cfg)
    typer.echo(f"Training done. Final checkpoint: {ckpt}")


@app.command("cleanup-nn-run")
def cleanup_nn_run(
    run_dir: Path = typer.Option(..., help="训练输出目录（包含 ckpt_*.pt / metrics.jsonl / plots）"),
    keep_best: bool = typer.Option(True, help="保留 ckpt_bc_best.pt / ckpt_rl_best.pt"),
    keep_final: bool = typer.Option(True, help="保留 ckpt_final.pt"),
    dry_run: bool = typer.Option(False, help="只打印要删除的文件，不实际删除"),
) -> None:
    """删除中间过程权重（ckpt_*_epoch*.pt 等），只保留 best/final。"""

    rd = Path(run_dir)
    if not rd.exists():
        raise typer.BadParameter(f"run_dir 不存在：{run_dir}")

    keep = set()
    if keep_best:
        keep.update({"ckpt_bc_best.pt", "ckpt_rl_best.pt"})
    if keep_final:
        keep.add("ckpt_final.pt")

    to_delete: list[Path] = []
    for p in sorted(rd.glob("ckpt_*.pt")):
        if p.name in keep:
            continue
        # delete all intermediate checkpoints
        to_delete.append(p)

    if not to_delete:
        typer.echo("No intermediate checkpoints to delete.")
        return

    for p in to_delete:
        typer.echo(f"DEL {p}")
        if not dry_run:
            try:
                p.unlink()
            except FileNotFoundError:
                pass


@app.command("export-nn-run")
def export_nn_run(
    run_dir: Path = typer.Option(..., help="训练输出目录（datasets/nn_runs/...）"),
    out_dir: Path = typer.Option(..., help="导出目录（建议 docs/figures/nn/<run_name>）"),
) -> None:
    """把本次运行的图片/曲线数据复制到一个“正常”的文档目录，便于在 plots.md 引用。"""

    rd = Path(run_dir)
    od = Path(out_dir)
    od.mkdir(parents=True, exist_ok=True)
    (od / "train").mkdir(parents=True, exist_ok=True)
    (od / "eval").mkdir(parents=True, exist_ok=True)

    # copy plots
    for p in sorted((rd / "plots").glob("*.png")):
        shutil.copy2(p, od / "train" / p.name)

    if (rd / "plots_eval").exists():
        for p in sorted((rd / "plots_eval").glob("*.png")):
            shutil.copy2(p, od / "eval" / p.name)

    # copy metrics jsonl (small, useful)
    for name in ["metrics.jsonl", "run_eval_metrics.jsonl", "config.json"]:
        src = rd / name
        if src.exists():
            shutil.copy2(src, od / name)

    typer.echo(f"Exported assets: {rd} -> {od}")


@app.command("eval-nn")
def eval_nn(
    data: Path = typer.Option(..., help="数据 JSONL（用于随机划分 train/val）"),
    ckpt: Path = typer.Option(..., help="checkpoint 路径（例如 datasets/nn_runs/real_aug_full/ckpt_final.pt）"),
    out_dir: Path = typer.Option(Path("datasets/nn_eval"), help="输出目录（eval_metrics/plots/opt_cache）"),
    val_frac: float = typer.Option(0.1, help="验证集比例"),
    seed: int = typer.Option(0, help="随机种子（划分用）"),
    batch_size: int = typer.Option(32, help="评估 batch size"),
    eval_us: str = typer.Option("11,12,13", help="要评估的 U 列表（逗号分隔）"),
    val_max_instances: int = typer.Option(256, help="每个 U 预计算DP最优的验证样本数（0表示全部）"),
    topk: int = typer.Option(1, help="Top-K 复评解码：从 logits top-k 候选中按 reward 复评选最优（1=关闭）"),
) -> None:
    """在随机验证集上评估 checkpoint，并导出折线图。

    - 会为每个 U 预计算 DP 最优（缓存到 out_dir/val_opt_u{U}.jsonl）
    - 输出 out_dir/metrics.jsonl（统一格式，便于画图）
    """

    from .nn.eval import EvalConfig, evaluate
    from .nn.plots import plot_metrics

    us = tuple(int(x.strip()) for x in eval_us.split(",") if x.strip())
    max_n = None if val_max_instances == 0 else int(val_max_instances)
    cfg = EvalConfig(
        data_path=str(data),
        ckpt_path=str(ckpt),
        out_dir=str(out_dir),
        val_frac=float(val_frac),
        seed=int(seed),
        batch_size=int(batch_size),
        eval_us=us,
        val_max_instances=max_n,
        topk=int(topk),
    )
    metrics_path = evaluate(cfg)

    # Also plot training-style metrics if present (optional)
    try:
        plot_metrics(metrics_path, out_dir, subdir="plots_eval")
    except Exception:
        pass

    typer.echo(f"Wrote eval metrics -> {metrics_path}")


@app.command("eval-run-nn")
def eval_run_nn(
    data: Path = typer.Option(..., help="数据 JSONL（用于随机划分 train/val）"),
    run_dir: Path = typer.Option(..., help="训练输出目录（包含 ckpt_bc_epoch*/ckpt_rl_epoch*/ckpt_final.pt）"),
    out_dir: Path | None = typer.Option(None, help="输出目录（默认与 run_dir 相同）"),
    val_frac: float = typer.Option(0.1, help="验证集比例"),
    seed: int = typer.Option(0, help="随机种子（划分用）"),
    batch_size: int = typer.Option(32, help="评估 batch size"),
    eval_us: str = typer.Option("11,12,13", help="要评估的 U 列表（逗号分隔）"),
    val_max_instances: int = typer.Option(256, help="每个 U 预计算DP最优的验证样本数（0表示全部）"),
    topk: int = typer.Option(1, help="Top-K 复评解码：从 logits top-k 候选中按 reward 复评选最优（1=关闭）"),
) -> None:
    """评估一个 run_dir 下的所有 checkpoint，并导出随训练进度变化的折线图。"""

    from .nn.eval import evaluate_run_dir
    from .nn.plots import plot_metrics

    out_dir2 = run_dir if out_dir is None else out_dir
    us = tuple(int(x.strip()) for x in eval_us.split(",") if x.strip())
    max_n = None if val_max_instances == 0 else int(val_max_instances)

    metrics_path = evaluate_run_dir(
        data_path=str(data),
        run_dir=str(run_dir),
        out_dir=str(out_dir2),
        val_frac=float(val_frac),
        seed=int(seed),
        batch_size=int(batch_size),
        eval_us=us,
        val_max_instances=max_n,
        topk=int(topk),
    )
    try:
        plot_metrics(metrics_path, out_dir2, subdir="plots_eval")
    except Exception:
        pass

    typer.echo(f"Wrote run eval metrics -> {metrics_path}")


if __name__ == "__main__":
    app()

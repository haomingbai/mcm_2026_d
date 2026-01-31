from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class MetricRow:
    phase: str
    epoch: int
    split: str
    name: str
    value: float
    u: int | None = None


def read_metrics(path: str | Path) -> list[MetricRow]:
    rows: list[MetricRow] = []
    p = Path(path)
    if not p.exists():
        return rows
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            # backward-compatible: accept older rows with direct keys
            if "name" in d:
                rows.append(
                    MetricRow(
                        phase=str(d.get("phase", "?")),
                        epoch=int(d.get("epoch", 0)),
                        split=str(d.get("split", "train")),
                        name=str(d["name"]),
                        value=float(d["value"]),
                        u=(int(d["u"]) if d.get("u") is not None else None),
                    )
                )
            else:
                # Try to lift known legacy format: {phase, epoch, obj_greedy_mean}
                if "obj_greedy_mean" in d:
                    rows.append(
                        MetricRow(
                            phase=str(d.get("phase", "?")),
                            epoch=int(d.get("epoch", 0)),
                            split=str(d.get("split", "train")),
                            name="obj_greedy_mean",
                            value=float(d["obj_greedy_mean"]),
                            u=(int(d["u"]) if d.get("u") is not None else None),
                        )
                    )
    return rows


def write_metric(
    *,
    path: Path,
    phase: str,
    epoch: int,
    split: str,
    name: str,
    value: float,
    u: int | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {"phase": phase, "epoch": int(epoch), "split": split, "name": name, "value": float(value)}
    if u is not None:
        row["u"] = int(u)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def plot_metrics(metrics_path: str | Path, out_dir: str | Path, *, subdir: str = "plots") -> list[Path]:
    """Render line charts from metrics.jsonl.

    Outputs several PNGs into out_dir/plots.
    """

    # Import lazily to keep core training usable without plotting deps.
    import matplotlib.pyplot as plt

    # Prettier defaults (safe fallbacks if a style is unavailable)
    for style in ["seaborn-v0_8-whitegrid", "seaborn-whitegrid", "ggplot"]:
        try:
            plt.style.use(style)
            break
        except Exception:
            pass

    rows = read_metrics(metrics_path)
    if not rows:
        return []

    # Some runs record the dataset default-U greedy objective as:
    #   name="obj_greedy_mean", u=None
    # while other U values use:
    #   name="val_obj_greedy_mean", u=<U>
    # This makes the U==default panel empty in val_obj_greedy facets.
    # We infer the default-U by looking for the single U that has ratio data
    # but is missing val_obj_greedy_mean.
    u_with_ratio = {
        int(r.u)
        for r in rows
        if r.u is not None and r.split == "val" and r.name == "val_obj_ratio_mean"
    }
    u_with_val_obj = {
        int(r.u)
        for r in rows
        if r.u is not None and r.split == "val" and r.name == "val_obj_greedy_mean"
    }
    missing_val_obj = sorted(u_with_ratio - u_with_val_obj)
    obj_greedy_val_xs, obj_greedy_val_ys = [], []
    try:
        obj_greedy_val_xs, obj_greedy_val_ys = [], []
        pts: list[tuple[int, float]] = []
        for r in rows:
            if r.name == "obj_greedy_mean" and r.split == "val" and r.u is None:
                pts.append((int(r.epoch), float(r.value)))
        pts.sort(key=lambda t: t[0])
        obj_greedy_val_xs = [p[0] for p in pts]
        obj_greedy_val_ys = [p[1] for p in pts]
    except Exception:
        obj_greedy_val_xs, obj_greedy_val_ys = [], []

    default_u_for_obj: int | None = None
    if obj_greedy_val_xs and len(missing_val_obj) == 1:
        default_u_for_obj = int(missing_val_obj[0])

    out = Path(out_dir) / subdir
    out.mkdir(parents=True, exist_ok=True)

    def _series(filter_fn):
        pts: list[tuple[int, float]] = []
        for r in rows:
            if filter_fn(r):
                pts.append((int(r.epoch), float(r.value)))
        # Always sort by epoch so lines don't zig-zag if the jsonl is out of order.
        pts.sort(key=lambda t: t[0])
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        return xs, ys

    def _best_so_far(xs: list[int], ys: list[float], *, mode: str = "max") -> tuple[list[int], list[float]]:
        if not xs:
            return xs, ys
        best = None
        out_y: list[float] = []
        for y in ys:
            if best is None:
                best = y
            else:
                best = max(best, y) if mode == "max" else min(best, y)
            out_y.append(float(best))
        return xs, out_y

    saved: list[Path] = []

    # 1) BC loss / RL loss
    for name in ["loss_bc", "loss_rl", "loss_pg", "loss_value", "loss_awbc", "entropy"]:
        plt.figure()
        has_line = False
        for split in ["train", "val"]:
            xs, ys = _series(lambda r, n=name, s=split: r.name == n and r.split == s)
            if xs:
                plt.plot(xs, ys, label=split, linewidth=2)
                has_line = True
        plt.title(name)
        plt.xlabel("epoch")
        plt.ylabel(name)
        if not has_line:
            plt.close()
            continue
        plt.legend()
        p = out / f"{name}.png"
        plt.grid(True, alpha=0.25)
        plt.tight_layout()
        plt.savefig(p, dpi=160)
        plt.close()
        saved.append(p)

    # 2) Objective curves for train/val
    plt.figure()
    has_line = False
    for split in ["train", "val"]:
        xs, ys = _series(lambda r, s=split: r.name == "obj_greedy_mean" and r.split == s and r.u is None)
        if xs:
            plt.plot(xs, ys, label=split, linewidth=2)
            has_line = True
    plt.title("obj_greedy_mean")
    plt.xlabel("epoch")
    plt.ylabel("objective")
    if not has_line:
        plt.close()
    else:
        plt.legend()
        p = out / "obj_greedy_mean.png"
        plt.grid(True, alpha=0.25)
        plt.tight_layout()
        plt.savefig(p, dpi=160)
        plt.close()
        saved.append(p)

    # 3) Val gap / accuracy by U
    # group by U
    us = sorted({r.u for r in rows if r.u is not None})
    if us:
        # Facets (one subplot per U) to avoid overplotting when lines overlap.
        def _facet(metric_name: str, title: str, ylabel: str, filename: str) -> None:
            import matplotlib.pyplot as plt

            n = len(us)
            fig, axes = plt.subplots(n, 1, figsize=(7.2, max(2.0, 1.8 * n)), sharex=True)
            if n == 1:
                axes = [axes]
            any_line = False
            for ax, u in zip(axes, us):
                xs, ys = _series(lambda r, uu=u: r.name == metric_name and r.split == "val" and r.u == uu)
                if (
                    not xs
                    and metric_name == "val_obj_greedy_mean"
                    and default_u_for_obj is not None
                    and int(u) == int(default_u_for_obj)
                    and obj_greedy_val_xs
                ):
                    xs, ys = obj_greedy_val_xs, obj_greedy_val_ys
                if xs:
                    ax.plot(xs, ys, marker="o", markersize=3, linewidth=2)
                    any_line = True
                ax.set_title(f"U={u}")
                ax.set_ylabel(ylabel)
                ax.grid(True, alpha=0.25)
            axes[-1].set_xlabel("epoch")
            fig.suptitle(title)
            fig.tight_layout()
            if not any_line:
                plt.close(fig)
                return
            p = out / filename
            fig.savefig(p, dpi=170)
            plt.close(fig)
            saved.append(p)

        def _facet_best(metric_name: str, title: str, ylabel: str, filename: str, *, mode: str = "max") -> None:
            import matplotlib.pyplot as plt

            n = len(us)
            fig, axes = plt.subplots(n, 1, figsize=(7.2, max(2.0, 1.8 * n)), sharex=True)
            if n == 1:
                axes = [axes]
            any_line = False
            for ax, u in zip(axes, us):
                xs, ys = _series(lambda r, uu=u: r.name == metric_name and r.split == "val" and r.u == uu)
                if xs:
                    xs2, ys2 = _best_so_far(xs, ys, mode=mode)
                    ax.plot(xs2, ys2, linewidth=2.2)
                    any_line = True
                ax.set_title(f"U={u}")
                ax.set_ylabel(ylabel)
                ax.grid(True, alpha=0.25)
            axes[-1].set_xlabel("epoch")
            fig.suptitle(title)
            fig.tight_layout()
            if not any_line:
                plt.close(fig)
                return
            p = out / filename
            fig.savefig(p, dpi=180)
            plt.close(fig)
            saved.append(p)

        _facet("val_obj_greedy_mean", "val_obj_greedy_mean (facets)", "objective", "val_obj_greedy_facets.png")
        _facet("val_obj_ratio_mean", "val_obj_ratio_mean (facets)", "model/opt", "val_obj_ratio_facets.png")
        _facet("val_regret_mean", "val_regret_mean (facets)", "opt - model", "val_regret_facets.png")
        _facet("val_gap_mean", "val_gap_mean (facets)", "(opt-model)/|opt|", "val_gap_facets.png")

        # Best-so-far versions (more readable progress signal)
        _facet_best("val_obj_ratio_mean", "val_obj_ratio_mean best-so-far (facets)", "best model/opt", "val_obj_ratio_best_facets.png", mode="max")
        _facet_best("val_gap_mean", "val_gap_mean best-so-far (facets)", "best (opt-model)/|opt|", "val_gap_best_facets.png", mode="min")

        plt.figure()
        has_line = False
        for u in us:
            xs, ys = _series(lambda r, uu=u: r.name == "val_obj_greedy_mean" and r.split == "val" and r.u == uu)
            if (
                not xs
                and default_u_for_obj is not None
                and int(u) == int(default_u_for_obj)
                and obj_greedy_val_xs
            ):
                xs, ys = obj_greedy_val_xs, obj_greedy_val_ys
            if xs:
                plt.plot(xs, ys, label=f"U={u}", linewidth=2.2, alpha=0.85)
                has_line = True
        plt.title("val_obj_greedy_mean by U")
        plt.xlabel("epoch")
        plt.ylabel("objective")
        if not has_line:
            plt.close()
        else:
            plt.legend()
            p = out / "val_obj_greedy_by_u.png"
            plt.grid(True, alpha=0.25)
            plt.tight_layout()
            plt.savefig(p, dpi=160)
            plt.close()
            saved.append(p)

        # Objective-based: ratio and regret
        plt.figure()
        has_line = False
        for u in us:
            xs, ys = _series(lambda r, uu=u: r.name == "val_obj_ratio_mean" and r.split == "val" and r.u == uu)
            if xs:
                plt.plot(xs, ys, label=f"U={u}", linewidth=2.2, alpha=0.85)
                has_line = True
        plt.title("val_obj_ratio_mean by U")
        plt.xlabel("epoch")
        plt.ylabel("model/opt")
        if not has_line:
            plt.close()
        else:
            plt.legend()
            p = out / "val_obj_ratio_by_u.png"
            plt.grid(True, alpha=0.25)
            plt.tight_layout()
            plt.savefig(p, dpi=160)
            plt.close()
            saved.append(p)

        # Best-so-far (single panel)
        plt.figure()
        has_line = False
        for u in us:
            xs, ys = _series(lambda r, uu=u: r.name == "val_obj_ratio_mean" and r.split == "val" and r.u == uu)
            if xs:
                xs2, ys2 = _best_so_far(xs, ys, mode="max")
                plt.plot(xs2, ys2, label=f"U={u}", linewidth=2.4, alpha=0.9)
                has_line = True
        plt.title("val_obj_ratio_mean best-so-far by U")
        plt.xlabel("epoch")
        plt.ylabel("best model/opt")
        if not has_line:
            plt.close()
        else:
            plt.legend()
            plt.grid(True, alpha=0.25)
            p = out / "val_obj_ratio_best_by_u.png"
            plt.tight_layout()
            plt.savefig(p, dpi=170)
            plt.close()
            saved.append(p)

        plt.figure()
        has_line = False
        for u in us:
            xs, ys = _series(lambda r, uu=u: r.name == "val_regret_mean" and r.split == "val" and r.u == uu)
            if xs:
                plt.plot(xs, ys, label=f"U={u}", linewidth=2.2, alpha=0.85)
                has_line = True
        plt.title("val_regret_mean by U")
        plt.xlabel("epoch")
        plt.ylabel("opt - model")
        if not has_line:
            plt.close()
        else:
            plt.legend()
            p = out / "val_regret_by_u.png"
            plt.grid(True, alpha=0.25)
            plt.tight_layout()
            plt.savefig(p, dpi=160)
            plt.close()
            saved.append(p)

        plt.figure()
        has_line = False
        for u in us:
            xs, ys = _series(lambda r, uu=u: r.name == "val_gap_mean" and r.split == "val" and r.u == uu)
            if xs:
                plt.plot(xs, ys, label=f"U={u}", linewidth=2.2, alpha=0.85)
                has_line = True
        plt.title("val_gap_mean by U")
        plt.xlabel("epoch")
        plt.ylabel("(opt - model)/|opt|")
        if not has_line:
            plt.close()
        else:
            plt.legend()
            p = out / "val_gap_by_u.png"
            plt.grid(True, alpha=0.25)
            plt.tight_layout()
            plt.savefig(p, dpi=160)
            plt.close()
            saved.append(p)

        # Best-so-far gap (single panel; lower is better)
        plt.figure()
        has_line = False
        for u in us:
            xs, ys = _series(lambda r, uu=u: r.name == "val_gap_mean" and r.split == "val" and r.u == uu)
            if xs:
                xs2, ys2 = _best_so_far(xs, ys, mode="min")
                plt.plot(xs2, ys2, label=f"U={u}", linewidth=2.4, alpha=0.9)
                has_line = True
        plt.title("val_gap_mean best-so-far by U")
        plt.xlabel("epoch")
        plt.ylabel("best (opt - model)/|opt|")
        if not has_line:
            plt.close()
        else:
            plt.legend()
            plt.grid(True, alpha=0.25)
            p = out / "val_gap_best_by_u.png"
            plt.tight_layout()
            plt.savefig(p, dpi=170)
            plt.close()
            saved.append(p)

        plt.figure()
        has_line = False
        for u in us:
            xs, ys = _series(lambda r, uu=u: r.name == "val_season_acc" and r.split == "val" and r.u == uu)
            if xs:
                plt.plot(xs, ys, label=f"U={u}")
                has_line = True
        plt.title("val_season_acc by U")
        plt.xlabel("epoch")
        plt.ylabel("accuracy")
        if not has_line:
            plt.close()
        else:
            plt.legend()
            p = out / "val_season_acc_by_u.png"
            plt.tight_layout()
            plt.savefig(p, dpi=160)
            plt.close()
            saved.append(p)

        plt.figure()
        has_line = False
        for u in us:
            xs, ys = _series(lambda r, uu=u: r.name == "val_allseason_acc" and r.split == "val" and r.u == uu)
            if xs:
                plt.plot(xs, ys, label=f"U={u}")
                has_line = True
        plt.title("val_allseason_acc by U")
        plt.xlabel("epoch")
        plt.ylabel("accuracy")
        if not has_line:
            plt.close()
        else:
            plt.legend()
            p = out / "val_allseason_acc_by_u.png"
            plt.tight_layout()
            plt.savefig(p, dpi=160)
            plt.close()
            saved.append(p)

    return saved

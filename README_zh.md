# mcm_2026_d

本仓库用于 **MCM 2026 Problem D** 的“训练数据集构建”：

- 生成小规模实例（合成 or 真实+合成混合）
- 用传统精确算法（穷举可行阵容 + 动态规划）求每条实例的全局最优解
- 导出可用于监督学习/模仿学习的训练对（`instance` → `solution`）

核心目标：在问题规模可控（例如候选球员 `n_players <= 15`、赛季数 `T <= 3`）时，稳定产出高质量标签数据。

> 说明：真实数据抓取目前使用 Basketball-Reference 的 WNBA 年度 advanced 数据页，仅用于构建“球员特征池”。

## 快速开始

在仓库根目录：

```bash
uv sync
mkdir -p datasets

# 生成 50 条合成实例 + 精确最优解
mcm2026d generate-and-solve \
  --out datasets/pairs.jsonl \
  --n 50 --seed 0
```

输出：

- `datasets/pairs.jsonl`：每行一个样本，包含 `instance` 与其 `solution`（最优阵容序列）

可选安装（后续训练神经网络时）：

```bash
uv sync --extra ml
```

---

## 端到端流程（数据 → 训练 → 推理）

下面给出一条从“真实数据爬取/数据生成”到“训练”再到“加载已训练模型做策略推理”的完整链路。

建议先通读 [general_solution.md](general_solution.md)（思路），再按本 README 跑通命令（工程复现）。

---

## 神经网络训练（BC + 强化学习微调）

本项目提供一个可解释的策略网络（TFT-inspired）来近似传统 DP 的最优策略，并用 objective-based 指标（ratio/gap/regret）评估“达到传统方法几成水平”。

更详细、面向非 ML 读者的解释见：

- [general_solution.md](general_solution.md)
- [train_strategy.md](train_strategy.md)
- [plots.md](plots.md)

### 0) 安装（训练/推理需要 ml 依赖）

```bash
uv sync --extra ml
```

### 1) 训练（BC → RL，支持早停）

示例（低学习率 + 大 epoch 上限 + patience=8 早停，batch_size=32）：

```bash
./.venv/bin/python -m mcm_2026_d train-nn \
  --data datasets/pairs_real_aug_4000.jsonl \
  --out-dir datasets/nn_runs/<your_run_name> \
  --use-constraint-env \
  --use-shadow-price \
  --use-cost-modulation \
  --critic-decompose \
  --optimizer adam \
  --lr-bc 1e-4 --lr-rl 5e-5 \
  --bc-epochs 100 --rl-epochs 100 \
  --bc-patience 8 --rl-patience 8 \
  --monitor-name val_obj_ratio_mean \
  --val-frac 0.1 --val-seed 0 \
  --val-opt-max-instances 256 \
  --batch-size 32
```

> 输出目录会包含：`metrics.jsonl`、`plots/`、`ckpt_*.pt`、以及验证集 DP 最优缓存 `val_opt_u*.jsonl`。

你通常会使用其中之一做推理：

- `ckpt_rl_best.pt`：按早停指标选出的 RL 最佳
- `ckpt_final.pt`：最后一个阶段结束时的权重

### 2) 评估一个 run（对多个 checkpoint 画 eval 曲线）

```bash
./.venv/bin/python -m mcm_2026_d eval-run-nn \
  --data datasets/pairs_real_aug_4000.jsonl \
  --run-dir datasets/nn_runs/<your_run_name> \
  --val-frac 0.1 --seed 0 \
  --eval-us 11,12,13 \
  --val-max-instances 256 \
  --topk 10
```

> 说明：`--topk 10` 会在推理时对 logits Top-K 的候选动作做一次“基于 reward 的复评”，通常能显著提升最终 ratio（但不会改变训练过程本身）。

### 3) 推理（加载已训练模型输出策略动作）

如果你希望“真正拿模型做策略推理”（输出每赛季选择的 roster mask 序列），推荐两种方式：

1) **用 CLI 在验证集上推理并打分（最省事）**：

```bash
./.venv/bin/python -m mcm_2026_d eval-nn \
  --data datasets/pairs_real_aug_4000.jsonl \
  --ckpt datasets/nn_runs/<your_run_name>/ckpt_rl_best.pt \
  --out-dir datasets/nn_eval/<your_eval_name> \
  --val-frac 0.1 --seed 0 \
  --eval-us 11,12,13 \
  --val-max-instances 256 \
  --topk 10
```

它会自动：

- 从 `--data` 随机划分验证集
- 加载 `--ckpt`
- 用策略进行 rollout（Top-1 或 Top-K 复评）
- 计算并输出 `val_obj_ratio_mean / gap / regret`（按 U 分组汇总）

2) **用 Python 代码对单条样本做推理（拿到 masks 序列）**：

下面示例从 JSONL 里取第 1 条样本，加载 checkpoint，并输出推理得到的 `masks`：

```python
import torch
from torch.utils.data import DataLoader

from mcm_2026_d.nn.data import PairDataset, collate_pairs
from mcm_2026_d.nn.eval import _device, _greedy_rollout_objective
from mcm_2026_d.nn.masks import build_mask_set, build_popcount_table
from mcm_2026_d.nn.model import TFTPolicy

data_path = "datasets/pairs_real_aug_4000.jsonl"
ckpt_path = "datasets/nn_runs/<your_run_name>/ckpt_rl_best.pt"  # 或 ckpt_final.pt
topk = 10

ds = PairDataset(data_path)
batch = next(iter(DataLoader([ds[0]], batch_size=1, collate_fn=collate_pairs)))

ckpt = torch.load(ckpt_path, map_location="cpu")
sd = ckpt["model"]

# 从 checkpoint 反推结构开关（与 eval.py 的做法一致）
d_model = int(sd["player_proj.weight"].shape[0])
lstm_hidden = int(sd["temporal.weight_ih_l0"].shape[0] // 4)
env_dim = int(sd["env_proj.weight"].shape[1])
use_shadow_price = ("lambda_head.weight" in sd) or any(k.startswith("lambda_head.") for k in sd)
use_cost_modulation = ("cost_proj.weight" in sd) or any(k.startswith("cost_proj.") for k in sd)
critic_decompose = any(k.startswith("value_head_w.") for k in sd) and any(k.startswith("value_head_pi.") for k in sd)
use_constraint_env = env_dim > 6

device = _device()
model = TFTPolicy(
    n_players=ds.n_players,
    K=ds.K,
    env_dim=env_dim,
    d_model=d_model,
    lstm_hidden=lstm_hidden,
    use_constraint_env=use_constraint_env,
    use_shadow_price=use_shadow_price,
    use_cost_modulation=use_cost_modulation,
    critic_decompose=critic_decompose,
)
model.load_state_dict(sd)
model.to(device)

mask_set = build_mask_set(n_players=ds.n_players, L=ds.L, U=ds.U).to_torch(device)
popcount_table = build_popcount_table(n_players=ds.n_players)

obj, chosen_masks = _greedy_rollout_objective(
    model=model,
    batch=batch,
    mask_set=mask_set,
    popcount_table=popcount_table,
    device=device,
    topk=topk,
)

print("chosen_masks:", chosen_masks[0])
print("objective:", float(obj.item()))
```

> 提示：`chosen_masks[0]` 是长度为 `T` 的整数列表，每个整数是一个 bitmask，表示该赛季入选的球员集合。

### 4) 清理中间权重（只保留 best/final）

```bash
./.venv/bin/python -m mcm_2026_d cleanup-nn-run \
  --run-dir datasets/nn_runs/<your_run_name>
```

### 5) 导出图片到文档目录（推荐）

把训练/评估的图片复制到 `docs/figures/...`，便于在 `plots.md` 引用并纳入报告：

```bash
./.venv/bin/python -m mcm_2026_d export-nn-run \
  --run-dir datasets/nn_runs/<your_run_name> \
  --out-dir docs/figures/nn/<your_run_name>
```

如果你的 shell 没有把 `.venv/bin` 加进 PATH，可用以下任一方式运行：

- `./.venv/bin/mcm2026d ...`
- `uv run mcm2026d ...`

---

## 环境与依赖

- Python: `>=3.11`
- 依赖管理：`uv`

常用命令：

```bash
# 安装依赖并创建 .venv
uv sync

# 查看 CLI 帮助
mcm2026d --help
mcm2026d generate-and-solve --help
mcm2026d scrape-bref --help
mcm2026d generate-and-solve-mixed --help
```

---

## 主要命令

### 1) 合成数据集：`generate-and-solve`

用途：纯合成实例 → 精确DP最优解 → 导出 JSONL。

```bash
mcm2026d generate-and-solve \
  --out datasets/pairs.jsonl \
  --out-instances datasets/instances.jsonl \
  --out-solutions datasets/solutions.jsonl \
  --out-summary datasets/summary.csv \
  --n 200 --seed 0 \
  --n-players 15 --t 3 --k 6 --l 11 --u 12
```

关键参数：

- `--out`: 必填，训练对输出路径（JSONL）
- `--n`: 样本条数
- `--seed`: 随机种子（保证可复现）
- `--n-players`: 候选球员数（建议 `<=15`，否则精确解枚举会爆炸）
- `--t`: 赛季数（建议 `<=3`）
- `--k`: 能力维度数（向量维度）
- `--l / --u`: 阵容人数下限/上限（例如 11~12）

可选输出：

- `--out-instances`: 仅输入
- `--out-solutions`: 仅标签
- `--out-summary`: 每条样本的扁平汇总（便于快速 sanity check）

### 2) 真实数据抓取：`scrape-bref`

用途：下载并缓存 Basketball-Reference WNBA 年度 advanced 页面，解析成“球员特征池” CSV。

```bash
# 可选：使用本机代理（示例：Clash 7890）
export http_proxy=http://127.0.0.1:7890
export https_proxy=http://127.0.0.1:7890

mcm2026d scrape-bref \
  --years 2022 --years 2023 --years 2024 \
  --min-mp 200
```

输出位置：

- HTML 缓存：`data/raw/bref/bref_wnba_<year>_advanced.html`
- 球员池 CSV：`data/raw/bref/pool/pool_<year>.csv`

参数说明：

- `--years`: 可重复传入多次
- `--min-mp`: 最小上场分钟过滤阈值（分钟过少会被剔除）
- `--cache-dir`: 缓存目录（默认 `data/raw/bref`）
- `--force`: 强制重新下载 HTML

### 3) 真实+合成混合数据集：`generate-and-solve-mixed`

用途：从“真实球员池”抽样生成球员能力/薪资，再结合合成环境参数生成实例，最终仍用精确DP求解并导出训练对。

```bash
mcm2026d generate-and-solve-mixed \
  --out datasets/pairs_mixed_2000.jsonl \
  --pool-dir data/raw/bref/pool \
  --years 2022 --years 2023 --years 2024 \
  --real-frac 0.7 \
  --n 2000 --seed 1 \
  --n-players 15 --t 3 --k 6 --l 11 --u 12
```

参数说明：

- `--pool-dir`: `scrape-bref` 生成的 pool 目录
- `--years`: 从哪些年份的 pool 里抽样
- `--real-frac`: 样本中使用真实球员池的比例（0~1）；其余样本使用纯合成生成

---

## 输出数据格式（JSONL）

### `pairs.jsonl` / `pairs_mixed_*.jsonl`

每行都是一个 JSON 对象，结构为：

```json
{
  "instance": { ... },
  "solution": { ... }
}
```

### `instance` 主要字段（概览）

- `id`: 样本唯一标识
- `n_players`: 候选球员数
- `T`: 赛季数
- `K`: 能力维度
- `L`, `U`: 阵容人数上下限
- `gamma`, `lambda_win`, `beta`, `churn_penalty`: 奖励/折扣相关超参（见 `model_general.md` 的建模定义）
- `G[t]`: 每赛季场次
- `C[t]`: 每赛季工资帽
- `R_base[t]`, `rho[t]`: 收入/利润相关参数
- `Q_opp[t]`: 对手强度（或外部强度项）
- `w[k]`: 由能力向量映射到球队实力的权重
- `abilities[t][i][k]`: 第 t 赛季第 i 名球员的 K 维能力向量
- `salaries[t][i]`: 第 t 赛季第 i 名球员薪资
- `x_prev_mask`: 上一赛季阵容（bitmask，训练时可作为条件输入）

### `instance` 的最小示例（形状示意）

下面是一个“结构正确”的示意（数组内容用省略号代替）：

```json
{
  "id": "demo_0001",
  "n_players": 15,
  "T": 3,
  "K": 6,
  "L": 11,
  "U": 12,

  "gamma": 0.95,
  "lambda_win": 0.7,
  "beta": 0.6,
  "churn_penalty": 0.1,

  "G": [40, 40, 40],
  "C": [1600000.0, 1650000.0, 1700000.0],
  "R_base": [2.0e7, 2.0e7, 2.0e7],
  "rho": [6.0e5, 6.0e5, 6.0e5],
  "Q_opp": [1.0, 1.0, 1.0],

  "w": [0.2, 0.1, 0.15, 0.05, 0.25, 0.25],
  "abilities": [
    [[0.1, 0.2, 0.0, 0.3, 0.1, 0.2], "... 14 more players ..."],
    ["... season 2 ..."],
    ["... season 3 ..."]
  ],
  "salaries": [
    [120000.0, "... 14 more ..."],
    ["... season 2 ..."],
    ["... season 3 ..."]
  ],
  "x_prev_mask": 0
}
```

### `solution` 主要字段（概览）

- `objective`: 最优折扣总奖励
- `masks[t]`: 每赛季最优阵容（bitmask）
- `per_season[t]`: 每赛季关键指标（例如 `reward/W/profit/cost` 等）

> 具体字段以代码中的序列化输出为准：`src/mcm_2026_d/schemas.py`。

---

## 规模建议与性能提示

传统精确解的瓶颈在于“枚举可行阵容”，复杂度随 `n_players` 指数增长。

经验建议：

- `n_players <= 15`：通常可在秒级~分钟级生成上千条样本
- `T <= 3`：建议保持小（多赛季会放大 DP 状态数）
- 若要更大规模数据集：优先增加样本条数 `--n`，不要增加 `n_players/T`

---

## 常见问题（Troubleshooting）

### 1) `mcm2026d: command not found`

你的 PATH 没包含 `.venv/bin`：

- 直接用 `./.venv/bin/mcm2026d ...`
- 或用 `uv run mcm2026d ...`

### 2) 访问 BRef 超时 / 被拒绝

- 设置代理（如果你本地有可用代理）：

```bash
export http_proxy=http://127.0.0.1:7890
export https_proxy=http://127.0.0.1:7890
```

- 或稍后重试；程序会把 HTML 缓存到 `data/raw/bref/`，避免重复下载。

### 3) `Not enough players for year=YYYY`

表示 `pool_YYYY.csv` 在 `--min-mp` 过滤后可用球员太少。

- 降低 `--min-mp`（例如 50~100）
- 或减少 `--n-players`

---

## 代码结构（入口索引）

- CLI：`src/mcm_2026_d/cli.py`
- 合成实例：`src/mcm_2026_d/generate.py`
- 混合实例：`src/mcm_2026_d/mixed_generate.py`
- 精确求解：`src/mcm_2026_d/solver.py`
- 真实数据抓取/解析：`src/mcm_2026_d/realdata/bref_wnba.py`
- JSONL 导出：`src/mcm_2026_d/dataset.py`

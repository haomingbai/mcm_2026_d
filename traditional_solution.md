# 小规模阵容优化的精确求解与性能优化

## 问题回顾

给定一支 WNBA 球队在规划期 $t=1,\dots,T$ 内的阵容选择，需在满足工资帽 $C_t$ 及阵容规模上下限 $L,U$ 的约束下，最大化折扣总奖励：
$$
\max_{\{\mathbf{x}_t\}_{t=1}^{T}} \sum_{t=1}^{T} \gamma^{t-1}\, r_t(\mathbf{x}_t, \mathbf{x}_{t-1}).
$$
其中 $\mathbf{x}_t \in \{0,1\}^{|\mathcal{I}|}$ 表示第 $t$ 赛季的阵容决策，$\mathbf{x}_{t-1}$ 为上一赛季阵容；即时奖励 $r_t$ 由第 5–8 章给出的公式确定（聚合球员能力得到球队实力 $Q_t$，计算胜率 $p_t$，进而得到胜场 $W_t$ 和利润 $\Pi_t$）。

约束包括：

* **阵容规模约束**：$L \le \sum_i x_{i,t} \le U$；
* **工资帽约束**：$\sum_i c_{i,t} \, x_{i,t} \le C_t$。

对于球员总数较小（例如可选球员不超过 15 人），赛季数较短（$T\le 3$）的情况，可以通过枚举所有可行阵容并使用动态规划获得全局最优解。

## 精确算法：穷举 + 备忘录

1. **生成可行阵容**：对每个赛季 $t$，预先生成满足 $\sum_i x_{i,t}\in[L,U]$ 且 $\sum_i c_{i,t} x_{i,t}\le C_t$ 的所有阵容集合 $\mathcal{X}_t$。通过遍历球员组合并检查成本即可得到。

2. **状态定义**：使用函数 $V(t,\mathbf{p})$ 表示从赛季 $t$ 开始，在上一赛季阵容为 $\mathbf{p}$ 的情况下可以获得的最大累计折扣奖励。终止条件为 $t>T$ 时 $V=0$。

3. **递归转移**：对于每个可行阵容 $\mathbf{x}_t\in\mathcal{X}_t$，计算即时奖励 $r_t(\mathbf{x}_t,\mathbf{p})$，并递归调用下一赛季：
   $$
   V(t,\mathbf{p}) = \max_{\mathbf{x}_t\in\mathcal{X}_t}\left[r_t(\mathbf{x}_t,\mathbf{p})+\gamma\,V(t+1,\mathbf{x}_t)\right].
   $$
   利用备忘录（memoization）缓存 $(t,\mathbf{p})$ 的结果，避免重复计算。

4. **恢复策略**：记录每个状态下实现最大值的阵容 $\mathbf{x}_t^\star$，回溯得到整个最优序列 $\{\mathbf{x}_t^\star\}_{t=1}^T$。

该方法在小规模问题上能得到严格最优解，但枚举数量增长较快。以下几种优化可在保证最优性的前提下提升效率。

## 性能优化（保持精确性）

### 1. 剪枝：支配关系筛选

在生成可行阵容集合时，可依据**支配关系**过滤劣势阵容。如果存在两份阵容 $\mathbf{x}$ 和 $\mathbf{y}$，满足：

* $\sum_i c_{i,t} x_i \ge \sum_i c_{i,t} y_i$（$\mathbf{y}$ 成本更低或相等），且
* $\mathbf{u}_t(\mathbf{x}) \preceq \mathbf{u}_t(\mathbf{y})$（能力向量每个维度均不超过 $\mathbf{y}$），

则 $\mathbf{x}$ 在成本和能力上均被 $\mathbf{y}$ 支配，不可能产生更好回报，可将 $\mathbf{x}$ 从 $\mathcal{X}_t$ 中剔除。通过保留 **帕累托前沿**（Pareto frontier）的阵容，可大幅缩减搜索空间，同时不影响最优解。

### 2. 上界估计与分支限界

在递归过程中，对每个部分解计算一个**乐观上界** $U(t,\mathbf{p})$，代表在当前已选 $\mathbf{p}$ 且接下来赛季都能获得最大可能奖励的假设下，理论上能达到的最大折扣收益。若某条分支的当前累计奖励加上上界 $U(t,\mathbf{p})$ 仍不及已知最佳解，则该分支不再深入。这一“分支限界”（branch‑and‑bound）策略可以有效裁剪大量不可能的组合。

构造上界的方法可以简单地假设未来每个赛季都选择理论上的完美阵容（例如选择能力向量各维度最大值之和）获得的最大可能奖励。

### 3. 预排序与启发式初始化

虽然优化过程最终仍要穷举所有可行阵容，但可以提前对 $\mathcal{X}_t$ 按潜在价值排序（例如按照 $\mathbf{u}_t$ 的线性评分或单位能力/成本比排序），使得算法更快发现高价值方案并更新当前最佳值，从而结合上界剪枝起到更强的效果。

此外，可以先使用贪婪算法或简单启发式（如按能力/成本比率挑选阵容）得到一个较好的初始解，将其作为初始最佳值。这有助于剪枝算法尽早排除价值较低的组合。

### 4. 位运算和缓存优化

用位掩码表示阵容（比如一个整数的每一位代表一名球员是否入选），可以大幅减少内存占用并加速集合操作。预先计算每个掩码对应的成本、聚合能力以及在各赛季的即时奖励，有助于在递归时快速检索。利用高效的哈希表或 `lru_cache` 缓存递归结果也能降低重复计算。

## 小结

上述精确算法通过对每赛季可行阵容的枚举结合动态规划求解，可在小规模条件下找到全局最优解。通过支配关系筛选、分支限界剪枝、初始启发式和位运算表示等性能优化措施，可以在保证**最优性不变**的前提下显著降低搜索空间和运行时间。这些优化使该算法更适合用于生成高质量的训练数据，帮助数据驱动模型学习逼近精确解的策略。

---

## 代码实现与数据导出（已跑通）

本仓库已给出可直接运行的“合成实例生成 + 精确DP最优解 + 数据导出”流水线：

- 精确求解器实现： [src/mcm_2026_d/solver.py](src/mcm_2026_d/solver.py)
- 合成数据生成： [src/mcm_2026_d/generate.py](src/mcm_2026_d/generate.py)
- 数据集导出（JSONL）： [src/mcm_2026_d/dataset.py](src/mcm_2026_d/dataset.py)
- 命令行入口： [src/mcm_2026_d/cli.py](src/mcm_2026_d/cli.py)

### 环境与运行

在仓库根目录：

```bash
uv sync

# 生成 N 个样本，逐个用精确DP求最优解，并导出：
mcm2026d generate-and-solve \
   --out datasets/pairs.jsonl \
   --out-instances datasets/instances.jsonl \
   --out-solutions datasets/solutions.jsonl \
   --out-summary datasets/summary.csv \
   --n 200 --seed 0
```

> 如果你的 shell 没有把 `.venv/bin` 加入 PATH，也可以用 `./.venv/bin/mcm2026d ...` 或 `uv run mcm2026d ...`。

导出文件说明：

- `datasets/pairs.jsonl`：每行包含 `instance` 与其对应的 `solution`（推荐用于训练，输入/标签同存一行）
- `datasets/instances.jsonl`：仅输入
- `datasets/solutions.jsonl`：仅最优解
- `datasets/summary.csv`：快速检查用的汇总表

### 真实数据抓取（Basketball-Reference WNBA）

抓取/缓存 BRef WNBA 年度 advanced 页面，并生成可复用的球员特征池：

```bash
# 如需代理（示例：本机 Clash 7890）
export http_proxy=http://127.0.0.1:7890
export https_proxy=http://127.0.0.1:7890

mcm2026d scrape-bref \
   --years 2022 --years 2023 --years 2024 \
   --min-mp 200
```

输出示例：`data/raw/bref/pool/pool_2022.csv`（每行一个球员的特征与分钟）。

### 真实+合成混合数据集

在“真实球员池”基础上，混合生成小规模实例，并用精确DP求最优解导出训练对：

```bash
mcm2026d generate-and-solve-mixed \
   --out datasets/pairs_mixed_2000.jsonl \
   --pool-dir data/raw/bref/pool \
   --years 2022 --years 2023 --years 2024 \
   --real-frac 0.7 \
   --n 2000 --seed 1 \
   --n-players 15 --t 3 --k 6 --l 11 --u 12
```

`--real-frac` 控制“使用真实球员池”的样本比例；其余参数与合成版一致。

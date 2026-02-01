# Dynamic Roster Decision Optimization Model for a WNBA Team

## 0. Modeling goal and overall framework

We study a WNBA team’s **roster decision problem** over a multi-season planning horizon. The goal is to optimize the trade-off between **competitive performance** and **financial returns**.

We formulate the problem as a **Markov decision process (MDP)** with the standard tuple

$$
\langle \mathcal{S}, \mathcal{A}, P, R, \gamma \rangle.
$$

States describe the team context and history, actions correspond to season roster choices, and rewards capture both wins and profit.

---

## 1. Notation and dimension conventions (must be consistent)

### 1.1 Basic notation

| Type | Notation | Example |
| ---- | -------- | ------- |
| Scalar | italic lowercase | $x_{i,t},\ c_{i,t}$ |
| Vector | bold lowercase | $\mathbf{x}_t,\ \mathbf{a}_{i,t}$ |
| Matrix | bold uppercase | $\mathbf{A}_t,\ \mathbf{X}$ |
| Tensor | calligraphic uppercase | $\mathcal{A}$ |

### 1.2 Index sets

- Players: $i \in \mathcal{I}$
- Seasons: $t \in \{1,\dots,T\}$
- Ability dimensions: $k \in \{1,\dots,K\}$

---

## 2. Inputs and decision variables

### 2.1 Player ability (multi-dimensional, time-varying)

#### 2.1.1 Single-player ability vector

$$
\mathbf{a}_{i,t}=
\begin{bmatrix}
a_{i,t}^{(1)}\\
\vdots\\
a_{i,t}^{(K)}
\end{bmatrix}
\in \mathbb{R}^K.
$$

Each component represents an observable/estimable skill indicator, for example:

- Offensive efficiency (TS%)
- Usage (USG%)
- Assist rate (AST%)
- Defensive impact (STL%, BLK%)
- Rebounding (ORB%, DRB%)
- On-court impact (On/Off, BPM)
- Availability/load (MP, games played)

All components are assumed to be normalized (z-score or quantile normalization).

#### 2.1.2 Season ability matrix

$$
\mathbf{A}_t=
\begin{bmatrix}
\mathbf{a}_{1,t}^\top\\
\mathbf{a}_{2,t}^\top\\
\vdots\\
\mathbf{a}_{|\mathcal{I}|,t}^\top
\end{bmatrix}
\in \mathbb{R}^{|\mathcal{I}|\times K}.
$$

#### 2.1.3 Multi-season ability tensor

$$
\mathcal{A}\in \mathbb{R}^{T\times|\mathcal{I}|\times K},
\qquad
\mathcal{A}[t,i,:]=\mathbf{a}_{i,t}.
$$

---

### 2.2 Roster decision variables

#### 2.2.1 Per-player binary decision

$$
x_{i,t}=
\begin{cases}
1, & \text{player } i \text{ is on the roster in season } t\\
0, & \text{otherwise}
\end{cases}
$$

#### 2.2.2 Season roster vector

$$
\mathbf{x}_t=(x_{1,t},\dots,x_{|\mathcal{I}|,t})^\top
\in \{0,1\}^{|\mathcal{I}|}.
$$

#### 2.2.3 Multi-season roster matrix

$$
\mathbf{X}=[\mathbf{x}_1,\dots,\mathbf{x}_T]
\in \{0,1\}^{|\mathcal{I}|\times T}.
$$

---

## 3. State space

At the beginning of season $t$, define the system state as

$$
s_t=(\mathbf{x}_{t-1},\ \mathbf{A}_t,\ \mathbf{e}_t)\in \mathcal{S},
$$

where

- $\mathbf{x}_{t-1}$: previous-season roster (the **only source of temporal dependence**)
- $\mathbf{A}_t$: current-season player ability inputs
- $\mathbf{e}_t=(N_t, C_t, G_t)$: external environment
  - $N_t$: number of teams in the league
  - $C_t$: salary cap
  - $G_t$: number of games in the season

Note: abilities are treated as exogenous inputs (given or predicted), not as decision outcomes.

---

## 4. Action space

Given state $s_t$, the team chooses a season roster:

$$
a_t \equiv \mathbf{x}_t \in \mathcal{A}.
$$

Actions must satisfy hard constraints:

- Salary cap:
  $$
  \sum_i c_{i,t}x_{i,t} \le C_t
  $$
- Roster size:
  $$
  L \le \sum_i x_{i,t} \le U
  $$

---

## 5. Team strength modeling (core structure)

### 5.1 Aggregate roster ability (vector)

$$
\mathbf{u}_t=\sum_{i\in\mathcal{I}} x_{i,t}\,\mathbf{a}_{i,t}
\in \mathbb{R}^K.
$$

### 5.2 Map to a scalar team quality

**Linear version (interpretable):**

$$
Q_t = \mathbf{w}^\top \mathbf{u}_t,
$$

where $\mathbf{w}\in\mathbb{R}^K$ are ability weights.

**Nonlinear version (recommended):**

$$
Q_t = g(\mathbf{u}_t;\theta),
$$

where $g$ can be a neural network or a tree-based model.

---

## 6. Win probability and season wins

### 6.1 Per-game win probability

$$
p_t = \sigma\bigl(Q_t - Q_t^{\text{opp}}\bigr),
\quad
\sigma(z)=\frac{1}{1+e^{-z}}.
$$

### 6.2 Season wins

$$
W_t = G_t \cdot p_t.
$$

---

## 7. Financial model

### 7.1 Cost

$$
Cost_t = \sum_i c_{i,t}x_{i,t}.
$$

### 7.2 Revenue (linear approximation)

$$
R_t = R_t^{\text{base}} + \rho W_t.
$$

### 7.3 Profit

$$
\Pi_t = R_t - Cost_t.
$$

---

## 8. Reward function and objective

### 8.1 Single-season reward

$$
r_t
=\lambda_{\text{win}} \frac{W_t}{W^\ast}
+(1-\lambda_{\text{win}})\frac{\Pi_t}{\Pi^\ast}.
$$

Notation:

- $\lambda_{\text{win}}\in[0,1]$ weights “wins vs. profit” (corresponds to `lambda_win` in code).
- The neural network implementation also introduces a time-varying “salary-cap shadow price” (field `lambda_t`). In the write-up, we recommend denoting it as $\lambda^{\text{cap}}_t$ to avoid confusion with $\lambda_{\text{win}}$.

### 8.2 Multi-season objective

$$
\max_{\pi}\ \mathbb{E}_{\pi}\left[\sum_{t=1}^T \gamma^{t-1} r_t\right],
$$

where the policy $\pi : s_t \mapsto \mathbf{x}_t$.

---

## 9. Data sources and extensibility

This section summarizes the external data required by the model and possible extensions. Since data quality directly impacts decision quality, prefer official/authoritative sources and perform thorough cleaning/standardization.

### 9.1 Player ability data

The ability vector $\mathbf{a}_{i,t}\in\mathbb{R}^K$ can include offensive efficiency, usage, assist rate, defensive contribution, etc. Potential sources:

- **Official WNBA Stats** (<https://stats.wnba.com/>): basic stats and some advanced stats.
- **Basketball-Reference (WNBA)** (<https://www.basketball-reference.com/wnba/>): season/player advanced summaries (TS%, USG%, On/Off, BPM, etc.).
- **Her Hoop Stats** (<https://herhoopstats.com/>): salary/cap info and additional analytics (some features may require subscription).

### 9.2 Environment and financial parameters

#### 9.2.1 League size and schedule length

The environment vector $\mathbf{e}_t=(N_t, C_t, G_t)$ includes number of teams $N_t$, salary cap $C_t$, and number of games $G_t$.

- **$G_t$ (games):** may change by season; updating $G_t$ is sufficient when it does.
- **$C_t$ (cap) and roster size:** use CBA/salary databases for the cap; set $(L,U)$ based on league roster rules.

#### 9.2.2 Ticketing and media revenue

Because team revenues are not fully public, estimate baseline revenue $R_t^{\text{base}}$ and per-win incremental revenue $\rho$ from market statistics.

- Ticket price and attendance can be used to estimate ticket revenue scale, then fit $R_t^{\text{base}}$ and $\rho$.
- If star players cause large ticket bumps, capture this via a larger $\rho$ or an explicit “star effect” variable.
- Media rights and revenue-sharing can inform a stable baseline $R_t^{\text{base}}$ (then add tickets/sponsorships).

#### 9.2.3 Data collection recommendations

- **Standardize ability features:** use z-score or quantile normalization across metrics.
- **Timing and forecasting:** cap/schedule/ticketing are typically known before the season; abilities can be estimated from prior seasons plus predictive models.
- **Automation:** use scripts to periodically pull WNBA Stats and salary/cap sources; update business variables from public announcements/news as needed.

### 9.3 Possible extensions

Although this model focuses on a single-team roster optimization, it can be extended:

- **Aging decay and injury uncertainty:** incorporate age decline and injury probabilities into the transition $P$.
- **Position/chemistry constraints:** add position quotas and/or “chemistry” terms to discourage imbalanced rosters.
- **Multi-team interactions:** extend to a stochastic game where other teams’ decisions are included.
- **Richer revenue streams:** add merchandising, sponsorships, etc.

---

## 10. Recommended constants and their sources

This section lists recommended default values for key constants (update them for the latest season). Values are rough public-data-based estimates and should be recalibrated when applying the model.

### 10.1 Recommended constants (defaults)

| Constant | Recommended value (update by season) | Source / rationale (examples) |
| --- | --- | --- |
| Roster lower bound $L$ | 11 | WNBA rules/CBA generally require at least 11 players; see salary/rules summaries at <https://herhoopstats.com/>. |
| Roster upper bound $U$ | 12 | Typical roster cap is 12; adjust for special exemptions if needed; see <https://herhoopstats.com/>. |
| Salary cap $C_t$ | $\$1{,}500{,}000$–$\$1{,}600{,}000$ | Example: 2025 cap is about $\$1{,}507{,}100$ and grows by year; use a range and increase annually; see <https://herhoopstats.com/>. |
| Games per season $G_t$ | 40–44 | Example: 2024 had 40 games and later seasons increased (e.g., 44); confirm via schedule announcements/news. |
| Baseline revenue $R_t^{\text{base}}$ | $\$20{,}000{,}000$ / season | Order-of-magnitude estimate from media rights sharing + ticket revenue; calibrate with available reporting. |
| Incremental revenue per win $\rho$ | $\$500{,}000$–$\$800{,}000$ / win | Ticket bumps vary widely (including star effects); model as a range and recalibrate. |
| Win normalization $W^*$ | $G_t$ | Maximum wins equals total games, so set $W^*=G_t$. |
| Profit normalization $\Pi^*$ | $R_t^{\text{base}}$ | Normalize profit by baseline revenue to match scales: $\Pi^*=R_t^{\text{base}}$. |
| Reward weight $\lambda_{\text{win}}$ | 0.5–0.7 | Controls wins vs. profit; choose 0.6–0.7 for performance-first, 0.4–0.5 for financial balance. |
| Discount factor $\gamma$ | 0.9 | Common RL choice reflecting present value of future rewards. |

### 10.2 Practical tips

- For a specific season, update $C_t$ and $G_t$ from official announcements.
- If the market changes, re-fit $R_t^{\text{base}}$ and $\rho$ from historical/regression analysis.
- The table provides magnitudes/ranges mainly to help you quickly run simulations and policy evaluation.

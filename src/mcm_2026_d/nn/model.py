from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


class GRN(nn.Module):
    """Gated Residual Network (simplified).

    x -> LayerNorm -> (Linear -> ELU -> Linear) -> GLU gate -> residual.
    """

    def __init__(self, d_in: int, d_hidden: int, d_out: int | None = None, dropout: float = 0.1):
        super().__init__()
        d_out = d_in if d_out is None else d_out
        self.norm = nn.LayerNorm(d_in)
        self.fc1 = nn.Linear(d_in, d_hidden)
        self.fc2 = nn.Linear(d_hidden, d_out)
        self.gate = nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout(dropout)
        self.proj = nn.Identity() if d_in == d_out else nn.Linear(d_in, d_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x0 = self.proj(x)
        h = self.norm(x)
        h = self.fc1(h)
        h = F.elu(h)
        h = self.dropout(h)
        h = self.fc2(h)
        g = torch.sigmoid(self.gate(h))
        h = h * g
        return x0 + self.dropout(h)


class VariableSelection(nn.Module):
    def __init__(self, d_model: int, d_hidden: int, n_vars: int, dropout: float = 0.1):
        super().__init__()
        self.n_vars = n_vars
        self.weight_grn = GRN(d_model * n_vars, d_hidden, d_out=n_vars, dropout=dropout)

    def forward(self, x_vars: torch.Tensor) -> torch.Tensor:
        """x_vars: (B, n_vars, d_model) -> weights (B, n_vars)"""
        B, n, d = x_vars.shape
        flat = x_vars.reshape(B, n * d)
        logits = self.weight_grn(flat)
        return F.softmax(logits, dim=-1)


@dataclass(frozen=True)
class PolicyOutput:
    logits: torch.Tensor  # (B,M)
    value: torch.Tensor  # (B,) value estimate for season t
    player_importance: torch.Tensor  # (B,n_players)
    value_w: torch.Tensor | None = None
    value_pi: torch.Tensor | None = None
    lambda_t: torch.Tensor | None = None


class TFTPolicy(nn.Module):
    """TFT-inspired policy/value model for roster selection.

    Memory goal: keep B*M*d manageable by using small d_model.
    """

    def __init__(
        self,
        *,
        n_players: int,
        K: int,
        env_dim: int,
        use_constraint_env: bool = False,
        use_shadow_price: bool = False,
        use_cost_modulation: bool = False,
        critic_decompose: bool = False,
        d_model: int = 96,
        d_hidden: int = 192,
        lstm_hidden: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.n_players = n_players
        self.K = K
        self.env_dim = env_dim
        self.d_model = d_model

        self.use_constraint_env = bool(use_constraint_env)
        self.use_shadow_price = bool(use_shadow_price)
        self.use_cost_modulation = bool(use_cost_modulation)
        self.critic_decompose = bool(critic_decompose)

        if self.use_cost_modulation and (not self.use_shadow_price):
            raise ValueError("use_cost_modulation=True requires use_shadow_price=True")

        player_in = K + 2  # abilities + salary + prev_in
        self.player_proj = nn.Linear(player_in, d_model)
        self.player_grn = GRN(d_model, d_hidden, d_out=d_model, dropout=dropout)
        self.var_sel = VariableSelection(d_model, d_hidden, n_vars=n_players, dropout=dropout)

        self.env_proj = nn.Linear(env_dim, d_model)
        self.env_grn = GRN(d_model, d_hidden, d_out=d_model, dropout=dropout)

        if self.use_shadow_price:
            self.lambda_head = nn.Linear(d_model, 1)
        else:
            self.lambda_head = None

        if self.use_cost_modulation:
            # Project scalar cost ratio c_{i,t} into d_model, then a~=a - lambda*c
            self.cost_proj = nn.Linear(1, d_model)
        else:
            self.cost_proj = None

        self.temporal = nn.LSTM(
            input_size=d_model,
            hidden_size=lstm_hidden,
            num_layers=1,
            batch_first=True,
            dropout=0.0,
            bidirectional=False,
        )
        self.temporal_out = nn.Linear(lstm_hidden, d_model)

        # Candidate scorer: [roster_emb, season_ctx, cost_ratio, churn_ratio]
        scorer_in = d_model + d_model + 2
        self.scorer_grn1 = GRN(scorer_in, d_hidden, d_out=d_hidden, dropout=dropout)
        self.scorer_grn2 = GRN(d_hidden, d_hidden, d_out=d_hidden, dropout=dropout)
        self.logit_head = nn.Linear(d_hidden, 1)

        if not self.critic_decompose:
            self.value_head = nn.Sequential(
                GRN(d_model, d_hidden, d_out=d_hidden, dropout=dropout),
                nn.Linear(d_hidden, 1),
            )
            self.value_head_w = None
            self.value_head_pi = None
        else:
            # Decomposed critic: predict two components and combine using lambda_win.
            self.value_head = None
            self.value_head_w = nn.Sequential(
                GRN(d_model, d_hidden, d_out=d_hidden, dropout=dropout),
                nn.Linear(d_hidden, 1),
            )
            self.value_head_pi = nn.Sequential(
                GRN(d_model, d_hidden, d_out=d_hidden, dropout=dropout),
                nn.Linear(d_hidden, 1),
            )

    def forward_season(
        self,
        *,
        abilities_t: torch.Tensor,  # (B,n,K)
        salaries_t: torch.Tensor,  # (B,n)
        prev_in: torch.Tensor,  # (B,n) float
        env_t: torch.Tensor,  # (B,env)
        lambda_win: torch.Tensor | None = None,  # (B,)
        mask_matrix: torch.Tensor,  # (M,n)
        sizes: torch.Tensor,  # (M,)
        cost_all: torch.Tensor,  # (B,M)
        churn_all: torch.Tensor,  # (B,M)
        C_t: torch.Tensor,  # (B,)
        churn_norm: float = 15.0,
    ) -> PolicyOutput:
        B, n, K = abilities_t.shape
        assert n == self.n_players

        # Build env embedding (optionally append constraint-derived features)
        env_in = env_t
        if self.use_constraint_env:
            # cap_tightness := cost(prev roster)/cap, cap_slack := 1 - tightness
            prev_cost = (prev_in * salaries_t).sum(dim=-1)
            cap_tightness = (prev_cost / (C_t + 1e-9)).clamp(0.0, 5.0)
            cap_slack = (1.0 - cap_tightness).clamp(-4.0, 1.0)

            # roster_slack := (U - |prev|)/U, where U is inferred from candidate mask sizes
            u_max = sizes.max().to(prev_in.dtype).clamp(min=1.0)
            prev_size = prev_in.sum(dim=-1)
            roster_slack = ((u_max - prev_size) / u_max).clamp(-1.0, 1.0)

            extra = torch.stack([cap_slack, cap_tightness, roster_slack], dim=-1)
            env_in = torch.cat([env_t, extra], dim=-1)

        env_e = self.env_grn(self.env_proj(env_in))

        # Player embeddings
        salary_scaled = salaries_t / (C_t[:, None] + 1e-9)
        x = torch.cat([abilities_t, salary_scaled.unsqueeze(-1), prev_in.unsqueeze(-1)], dim=-1)
        p_base = self.player_grn(self.player_proj(x))

        # Variable selection weights (player importance)
        alpha_base = self.var_sel(p_base)  # (B,n)
        pooled_base = torch.einsum("bn,bnd->bd", alpha_base, p_base)
        season_ctx_base = pooled_base + env_e

        lambda_t: torch.Tensor | None = None
        if self.use_shadow_price:
            assert self.lambda_head is not None
            lambda_t = F.softplus(self.lambda_head(season_ctx_base)).squeeze(-1)

        p = p_base
        if self.use_cost_modulation:
            assert lambda_t is not None
            assert self.cost_proj is not None
            c_it = salary_scaled.clamp(0.0, 5.0).unsqueeze(-1)  # (B,n,1)
            c_vec = self.cost_proj(c_it)
            p = p - lambda_t[:, None, None] * c_vec

        alpha = self.var_sel(p)  # (B,n)
        pooled = torch.einsum("bn,bnd->bd", alpha, p)
        season_ctx = pooled + env_e

        # Roster embedding for all candidate masks: (B,M,d)
        roster = torch.einsum("mn,bnd->bmd", mask_matrix, p)
        roster = roster / (sizes[None, :, None] + 1e-9)

        # Extra features
        cost_ratio = (cost_all / (C_t[:, None] + 1e-9)).clamp(0.0, 5.0)
        churn_ratio = (churn_all.to(cost_ratio.dtype) / churn_norm).clamp(0.0, 2.0)

        ctx = season_ctx[:, None, :].expand(roster.shape[0], roster.shape[1], season_ctx.shape[-1])
        feats = torch.cat([roster, ctx, cost_ratio.unsqueeze(-1), churn_ratio.unsqueeze(-1)], dim=-1)

        h = self.scorer_grn1(feats)
        h = self.scorer_grn2(h)
        logits = self.logit_head(h).squeeze(-1)

        if not self.critic_decompose:
            assert self.value_head is not None
            v = self.value_head(season_ctx).squeeze(-1)
            v_w = None
            v_pi = None
        else:
            if lambda_win is None:
                raise ValueError("lambda_win must be provided when critic_decompose=True")
            lw = lambda_win.to(season_ctx.dtype).view(-1)
            assert self.value_head_w is not None and self.value_head_pi is not None
            v_w = self.value_head_w(season_ctx).squeeze(-1)
            v_pi = self.value_head_pi(season_ctx).squeeze(-1)
            v = lw * v_w + (1.0 - lw) * v_pi

        return PolicyOutput(
            logits=logits,
            value=v,
            value_w=v_w,
            value_pi=v_pi,
            player_importance=alpha,
            lambda_t=lambda_t,
        )

    def forward(
        self,
        *,
        abilities: torch.Tensor,  # (B,T,n,K)
        salaries: torch.Tensor,  # (B,T,n)
        env: torch.Tensor,  # (B,T,E)
        prev_in: torch.Tensor,  # (B,T,n)
    ) -> torch.Tensor:
        """Encode temporal season context.

        Returns:
            season_ctx: (B,T,d_model)
        """
        B, T, n, K = abilities.shape

        salary_scaled = salaries / (salaries.mean(dim=(2), keepdim=True) + 1e-9)
        x = torch.cat([abilities, salary_scaled.unsqueeze(-1), prev_in.unsqueeze(-1)], dim=-1)
        p = self.player_grn(self.player_proj(x))  # (B,T,n,d)

        # pool per season
        alpha = self.var_sel(p.reshape(B * T, n, self.d_model)).reshape(B, T, n)
        pooled = torch.einsum("btn,btnd->btd", alpha, p)

        env_e = self.env_grn(self.env_proj(env))
        season_in = pooled + env_e

        out, _ = self.temporal(season_in)
        season_ctx = self.temporal_out(out)
        return season_ctx

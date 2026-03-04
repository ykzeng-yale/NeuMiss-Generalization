"""Mixture-of-Experts NeuMiss for non-Gaussian distributions.

Theoretical motivation:
For mixture of Gaussians X ~ sum_k pi_k N(mu_k, Sigma_k), the Bayes predictor is:
    E[Y|X_obs, M] = sum_k w_k(X_obs, M) * f_k*(X_obs, M)
where w_k are posterior mixture weights and f_k* is the Bayes predictor
assuming component k. Each f_k* has the standard NeuMiss form but with
component-specific parameters (mu_k, Sigma_k).

This suggests a Mixture-of-Experts architecture:
- Multiple NeuMiss "experts" handle different mixture components
- A gating network assigns weights based on (X_obs, M)

Three MoE architectures:
1. MoE_NeuMiss: K NeuMiss experts + gating -> direct prediction
2. MoE_NeuMiss_Plus: K NeuMiss+C experts + gating -> direct prediction
3. MoE_NeuMiss_MLP: K NeuMiss experts (imputation) + gating + shared MLP head
"""
import sys
sys.path.insert(0, '/Users/yukang/Desktop/NeuroMiss/src')

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from copy import deepcopy
from sklearn.base import BaseEstimator, RegressorMixin

from data_generation import DataScenario
from neumiss_plus import (NeuMissPlus, ImputeMLP, PretrainEncoder,
                          EarlyStopping)


# ============================================================================
# Expert Module: Standard NeuMiss expert
# ============================================================================
class NeuMissExpert(nn.Module):
    """Single NeuMiss expert with its own parameters (mu_k, W_k, Wc_k, beta_k).

    Implements the standard NeuMiss forward pass:
        h0 = x + m * mu_k
        h = x - (1-m) * mu_k
        for each layer: h = (h @ W_k^(l)) * (1-m)
        h = (h @ Wc_k) * m + h0
        y = h @ beta_k + b_k
    """
    def __init__(self, n_features, depth):
        super().__init__()
        self.depth = depth
        self.n_features = n_features

        self.l_W = nn.ParameterList([
            nn.Parameter(torch.empty(n_features, n_features))
            for _ in range(depth)
        ])
        self.Wc = nn.Parameter(torch.empty(n_features, n_features))
        self.mu = nn.Parameter(torch.empty(n_features))
        self.beta = nn.Parameter(torch.empty(n_features))
        self.b = nn.Parameter(torch.empty(1))

        self._init_weights()

    def _init_weights(self):
        for W in self.l_W:
            nn.init.xavier_normal_(W)
        nn.init.xavier_normal_(self.Wc)
        nn.init.normal_(self.mu)
        nn.init.normal_(self.beta)
        nn.init.zeros_(self.b)

    def forward(self, x, m):
        obs = 1 - m
        h0 = x + m * self.mu
        h = x - obs * self.mu

        if len(self.l_W) > 0:
            h = torch.matmul(h, self.l_W[0]) * obs

        for W in self.l_W[1:]:
            h = torch.matmul(h, W) * obs

        h = torch.matmul(h, self.Wc) * m + h0
        y = torch.matmul(h, self.beta) + self.b
        return y


# ============================================================================
# Expert Module: NeuMiss+C expert (expand-activate-compress-mask)
# ============================================================================
class NeuMissPlusCExpert(nn.Module):
    """Single NeuMiss+C expert with expand-activate-compress-mask layers.

    Each Neumann layer:
        h -> W_expand -> bias -> activation -> W_compress -> bias -> mask
    """
    def __init__(self, n_features, depth, expansion_factor=2, activation='gelu'):
        super().__init__()
        self.depth = depth
        self.n_features = n_features
        self.hidden_dim = n_features * expansion_factor

        act_map = {'relu': nn.ReLU, 'gelu': nn.GELU, 'tanh': nn.Tanh,
                   'silu': nn.SiLU, 'elu': nn.ELU}
        act_cls = act_map.get(activation, nn.GELU)

        self.l_W_expand = nn.ParameterList([
            nn.Parameter(torch.empty(n_features, self.hidden_dim))
            for _ in range(depth)
        ])
        self.l_W_compress = nn.ParameterList([
            nn.Parameter(torch.empty(self.hidden_dim, n_features))
            for _ in range(depth)
        ])
        self.l_bias1 = nn.ParameterList([
            nn.Parameter(torch.zeros(self.hidden_dim)) for _ in range(depth)
        ])
        self.l_bias2 = nn.ParameterList([
            nn.Parameter(torch.zeros(n_features)) for _ in range(depth)
        ])
        self.acts = nn.ModuleList([act_cls() for _ in range(depth)])

        self.Wc = nn.Parameter(torch.empty(n_features, n_features))
        self.mu = nn.Parameter(torch.empty(n_features))
        self.beta = nn.Parameter(torch.empty(n_features))
        self.b = nn.Parameter(torch.empty(1))

        self._init_weights()

    def _init_weights(self):
        for W in list(self.l_W_expand) + list(self.l_W_compress) + [self.Wc]:
            nn.init.xavier_normal_(W)
        nn.init.normal_(self.mu)
        nn.init.normal_(self.beta)
        nn.init.zeros_(self.b)

    def forward(self, x, m):
        obs = 1 - m
        h0 = x + m * self.mu
        h = x - obs * self.mu

        for i in range(self.depth):
            h = torch.matmul(h, self.l_W_expand[i]) + self.l_bias1[i]
            h = self.acts[i](h)
            h = torch.matmul(h, self.l_W_compress[i]) + self.l_bias2[i]
            h = h * obs

        h = torch.matmul(h, self.Wc) * m + h0
        y = torch.matmul(h, self.beta) + self.b
        return y


# ============================================================================
# Expert Module: NeuMiss imputation expert (returns imputed representation)
# ============================================================================
class NeuMissImputeExpert(nn.Module):
    """NeuMiss expert that returns imputed d-dimensional representation.

    Used in MoE_NeuMiss_MLP where experts impute and a shared MLP predicts.
    """
    def __init__(self, n_features, depth):
        super().__init__()
        self.depth = depth
        self.n_features = n_features

        self.l_W = nn.ParameterList([
            nn.Parameter(torch.empty(n_features, n_features))
            for _ in range(depth)
        ])
        self.Wc = nn.Parameter(torch.empty(n_features, n_features))
        self.mu = nn.Parameter(torch.empty(n_features))

        self._init_weights()

    def _init_weights(self):
        for W in self.l_W:
            nn.init.xavier_normal_(W)
        nn.init.xavier_normal_(self.Wc)
        nn.init.normal_(self.mu)

    def forward(self, x, m):
        obs = 1 - m
        h0 = x + m * self.mu
        h = x - obs * self.mu

        if len(self.l_W) > 0:
            h = torch.matmul(h, self.l_W[0]) * obs

        for W in self.l_W[1:]:
            h = torch.matmul(h, W) * obs

        h = torch.matmul(h, self.Wc) * m + h0
        return h  # (batch, d) imputed representation


# ============================================================================
# Gating Network
# ============================================================================
class GatingNetwork(nn.Module):
    """Gating network: g(X_obs, M) = softmax(W_g @ [X_obs * (1-M), M]).

    Input: concatenation of observed features (zeros where missing) and mask.
    Output: K-dimensional softmax weights.
    """
    def __init__(self, n_features, n_experts, hidden_dim=None):
        super().__init__()
        input_dim = 2 * n_features  # [X_obs * (1-M), M]
        if hidden_dim is None:
            hidden_dim = max(n_features, n_experts * 2)

        self.gate = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, n_experts),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.gate.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x, m):
        obs = 1 - m
        gate_input = torch.cat([x * obs, m], dim=1)  # (batch, 2d)
        return torch.softmax(self.gate(gate_input), dim=1)  # (batch, K)


# ============================================================================
# Architecture 1: MoE_NeuMiss (K NeuMiss experts + gating -> prediction)
# ============================================================================
class MoE_NeuMiss_Net(nn.Module):
    """Mixture-of-Experts with K standard NeuMiss experts.

    Output: y = sum_k g_k(X_obs, M) * expert_k(X, M)
    where g_k is the gating weight and expert_k is a full NeuMiss predictor.
    """
    def __init__(self, n_features, n_experts, depth):
        super().__init__()
        self.n_experts = n_experts
        self.experts = nn.ModuleList([
            NeuMissExpert(n_features, depth) for _ in range(n_experts)
        ])
        self.gate = GatingNetwork(n_features, n_experts)

    def forward(self, x, m):
        weights = self.gate(x, m)  # (batch, K)
        expert_outputs = torch.stack(
            [expert(x, m) for expert in self.experts], dim=1
        )  # (batch, K)
        y = (weights * expert_outputs).sum(dim=1)  # (batch,)
        return y


# ============================================================================
# Architecture 2: MoE_NeuMiss_Plus (K NeuMiss+C experts + gating)
# ============================================================================
class MoE_NeuMiss_Plus_Net(nn.Module):
    """Mixture-of-Experts with K NeuMiss+C experts (expand-activate-compress)."""
    def __init__(self, n_features, n_experts, depth, expansion_factor=2,
                 activation='gelu'):
        super().__init__()
        self.n_experts = n_experts
        self.experts = nn.ModuleList([
            NeuMissPlusCExpert(n_features, depth, expansion_factor, activation)
            for _ in range(n_experts)
        ])
        self.gate = GatingNetwork(n_features, n_experts)

    def forward(self, x, m):
        weights = self.gate(x, m)  # (batch, K)
        expert_outputs = torch.stack(
            [expert(x, m) for expert in self.experts], dim=1
        )  # (batch, K)
        y = (weights * expert_outputs).sum(dim=1)
        return y


# ============================================================================
# Architecture 3: MoE_NeuMiss_MLP (K NeuMiss imputers + gating + shared MLP)
# ============================================================================
class MoE_NeuMiss_MLP_Net(nn.Module):
    """MoE with K NeuMiss imputation experts + shared MLP head.

    Each expert produces an imputed d-dim representation.
    Gating weights the representations.
    Weighted representation -> MLP -> prediction.

    This separates expert imputation from nonlinear prediction.
    """
    def __init__(self, n_features, n_experts, depth,
                 mlp_layers=(64, 32), activation='gelu', dropout=0.1):
        super().__init__()
        self.n_experts = n_experts
        self.n_features = n_features

        self.experts = nn.ModuleList([
            NeuMissImputeExpert(n_features, depth) for _ in range(n_experts)
        ])
        self.gate = GatingNetwork(n_features, n_experts)

        # Shared MLP head
        act_map = {'relu': nn.ReLU, 'gelu': nn.GELU, 'tanh': nn.Tanh,
                   'silu': nn.SiLU}
        act_cls = act_map.get(activation, nn.GELU)

        layers = []
        in_dim = n_features
        for h in mlp_layers:
            layers.append(nn.Linear(in_dim, h))
            layers.append(act_cls())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        self.mlp_head = nn.Sequential(*layers)

        self._init_mlp()

    def _init_mlp(self):
        for m in self.mlp_head.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x, m):
        weights = self.gate(x, m)  # (batch, K)

        # Stack expert imputed representations: (batch, K, d)
        expert_reps = torch.stack(
            [expert(x, m) for expert in self.experts], dim=1
        )

        # Weighted combination: (batch, K, 1) * (batch, K, d) -> sum -> (batch, d)
        weighted_rep = (weights.unsqueeze(-1) * expert_reps).sum(dim=1)

        # MLP prediction
        y = self.mlp_head(weighted_rep).squeeze(-1)
        return y


# ============================================================================
# Unified sklearn-compatible MoE Estimator
# ============================================================================
class MoE_NeuMiss_Estimator(BaseEstimator, RegressorMixin):
    """sklearn-compatible wrapper for all MoE-NeuMiss variants.

    Parameters
    ----------
    variant : str
        'moe_neumiss', 'moe_neumiss_plus', or 'moe_neumiss_mlp'
    n_experts : int
        Number of experts (K)
    depth : int
        NeuMiss depth per expert
    expansion_factor : int
        For 'moe_neumiss_plus' variant
    mlp_layers : tuple
        For 'moe_neumiss_mlp' variant
    activation : str
    dropout : float
    n_epochs, batch_size, lr, weight_decay : training parameters
    early_stopping : bool
    verbose : bool
    """
    def __init__(self, variant='moe_neumiss', n_experts=3, depth=3,
                 expansion_factor=2, mlp_layers=(64, 32),
                 activation='gelu', dropout=0.1,
                 n_epochs=200, batch_size=64, lr=0.001, weight_decay=1e-5,
                 early_stopping=True, verbose=False):
        self.variant = variant
        self.n_experts = n_experts
        self.depth = depth
        self.expansion_factor = expansion_factor
        self.mlp_layers = mlp_layers
        self.activation = activation
        self.dropout = dropout
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.early_stopping = early_stopping
        self.verbose = verbose

    def _build_net(self, n_features):
        if self.variant == 'moe_neumiss':
            return MoE_NeuMiss_Net(n_features, self.n_experts, self.depth)
        elif self.variant == 'moe_neumiss_plus':
            return MoE_NeuMiss_Plus_Net(
                n_features, self.n_experts, self.depth,
                self.expansion_factor, self.activation)
        elif self.variant == 'moe_neumiss_mlp':
            return MoE_NeuMiss_MLP_Net(
                n_features, self.n_experts, self.depth,
                self.mlp_layers, self.activation, self.dropout)
        else:
            raise ValueError(f"Unknown variant: {self.variant}")

    def fit(self, X, y, X_val=None, y_val=None):
        M = np.isnan(X).astype(np.float32)
        X_clean = np.nan_to_num(X).astype(np.float32)
        n, d = X.shape

        X_t = torch.tensor(X_clean, dtype=torch.float32)
        M_t = torch.tensor(M, dtype=torch.float32)
        y_t = torch.tensor(y, dtype=torch.float32)

        has_val = X_val is not None
        if has_val:
            Mv = np.isnan(X_val).astype(np.float32)
            Xv = torch.tensor(np.nan_to_num(X_val).astype(np.float32))
            Mv_t = torch.tensor(Mv, dtype=torch.float32)
            yv = torch.tensor(y_val, dtype=torch.float32)

        self.net = self._build_net(d)
        opt = optim.Adam(self.net.parameters(), lr=self.lr,
                         weight_decay=self.weight_decay)
        sched = ReduceLROnPlateau(opt, factor=0.2, patience=7)
        crit = nn.MSELoss()
        es = EarlyStopping(patience=25)

        for epoch in range(self.n_epochs):
            self.net.train()
            idx = torch.randperm(n)
            X_t, M_t, y_t = X_t[idx], M_t[idx], y_t[idx]

            bx = torch.split(X_t, self.batch_size)
            bm = torch.split(M_t, self.batch_size)
            by = torch.split(y_t, self.batch_size)

            loss_sum = 0
            for x_, m_, y_ in zip(bx, bm, by):
                opt.zero_grad()
                pred = self.net(x_, m_)
                loss = crit(pred, y_)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
                opt.step()
                loss_sum += loss.item()

            avg = loss_sum / len(bx)
            sched.step(avg)

            if self.verbose and epoch % 50 == 0:
                print(f"  Epoch {epoch}: loss={avg:.6f}")

            if has_val:
                self.net.eval()
                with torch.no_grad():
                    vl = crit(self.net(Xv, Mv_t), yv).item()
                if self.early_stopping:
                    es(vl, self.net)
                    if es.early_stop:
                        break

            if opt.param_groups[0]['lr'] < 1e-7:
                break

        if self.early_stopping and es.checkpoint:
            self.net.load_state_dict(es.checkpoint)
        return self

    def predict(self, X):
        M = torch.tensor(np.isnan(X).astype(np.float32))
        X_t = torch.tensor(np.nan_to_num(X).astype(np.float32))
        self.net.eval()
        with torch.no_grad():
            return self.net(X_t, M).numpy()

    def score(self, X, y):
        pred = self.predict(X)
        ss_res = np.mean((y - pred) ** 2)
        ss_tot = np.mean((y - np.mean(y)) ** 2)
        if ss_tot < 1e-12:
            return 0.0
        return 1 - ss_res / ss_tot


# ============================================================================
# Experiment
# ============================================================================
def run_experiment():
    n_train, n_val, n_test = 10000, 2500, 3000
    seeds = [42, 123, 456]
    d = 10

    # -- Scenarios --
    scenarios = [
        ("Gauss+Linear",
         DataScenario('gaussian', 'linear', 'MCAR', d, 0.5, 10)),
        ("Gauss+Quadratic",
         DataScenario('gaussian', 'quadratic', 'MCAR', d, 0.5, 10)),
        ("MixGauss3+Linear",
         DataScenario('mixture_gaussian', 'linear', 'MCAR', d, 0.5, 10,
                      distribution_params={'n_components': 3})),
        ("MixGauss3+Quadratic",
         DataScenario('mixture_gaussian', 'quadratic', 'MCAR', d, 0.5, 10,
                      distribution_params={'n_components': 3})),
        ("StudentT+Quadratic",
         DataScenario('student_t', 'quadratic', 'MCAR', d, 0.5, 10,
                      distribution_params={'df': 5})),
    ]

    # -- Models --
    # Baselines
    baseline_models = [
        ("NeuMiss(d=3)", lambda: NeuMissPlus(
            variant='original', depth=3, n_epochs=200, batch_size=64,
            lr=0.001, early_stopping=True, verbose=False)),
        ("NeuMiss+C", lambda: NeuMissPlus(
            variant='C', depth=3, activation='gelu', expansion_factor=2,
            n_epochs=200, batch_size=64, lr=0.001, early_stopping=True,
            verbose=False)),
        ("PretrainEnc", lambda: PretrainEncoder(
            depth=3, mlp_layers=(128,), activation='gelu', dropout=0.1,
            pretrain_epochs=50, train_epochs=200, freeze_encoder=False,
            mask_rate=0.3, batch_size=64, lr=0.001, weight_decay=1e-5,
            early_stopping=True, verbose=False)),
        ("ImputeMLP", lambda: ImputeMLP(
            hidden_layers=(128, 64, 32), activation='gelu',
            n_epochs=200, batch_size=64, lr=0.001, dropout=0.1,
            early_stopping=True, verbose=False)),
    ]

    # MoE models
    moe_models = []
    for K in [2, 3, 5]:
        moe_models.append(
            (f"MoE_NeuMiss(K={K})", lambda K=K: MoE_NeuMiss_Estimator(
                variant='moe_neumiss', n_experts=K, depth=3,
                n_epochs=200, batch_size=64, lr=0.001, weight_decay=1e-5,
                early_stopping=True, verbose=False)))
        moe_models.append(
            (f"MoE_NM+C(K={K})", lambda K=K: MoE_NeuMiss_Estimator(
                variant='moe_neumiss_plus', n_experts=K, depth=3,
                expansion_factor=2, activation='gelu',
                n_epochs=200, batch_size=64, lr=0.001, weight_decay=1e-5,
                early_stopping=True, verbose=False)))
        moe_models.append(
            (f"MoE_NM_MLP(K={K})", lambda K=K: MoE_NeuMiss_Estimator(
                variant='moe_neumiss_mlp', n_experts=K, depth=3,
                mlp_layers=(64, 32), activation='gelu', dropout=0.1,
                n_epochs=200, batch_size=64, lr=0.001, weight_decay=1e-5,
                early_stopping=True, verbose=False)))

    all_models = baseline_models + moe_models

    total = len(scenarios) * len(all_models) * len(seeds)
    done = 0
    results = []

    print(f"MoE-NeuMiss Experiment")
    print(f"  {len(scenarios)} scenarios x {len(all_models)} models x "
          f"{len(seeds)} seeds = {total} runs")
    print(f"  d={d}, n_train={n_train}, p=0.5, MCAR")
    print("=" * 90)

    for sc_name, scenario in scenarios:
        print(f"\n{'='*90}")
        print(f"SCENARIO: {sc_name}")
        print(f"{'='*90}")

        for seed in seeds:
            data = scenario.generate(n_train, n_val, n_test, random_state=seed)
            # X_complete needed for PretrainEncoder
            Xc_train = data['X_complete'][:n_train]
            Xc_val = data['X_complete'][n_train:n_train + n_val]

            for model_name, model_fn in all_models:
                done += 1
                model = model_fn()

                try:
                    if isinstance(model, PretrainEncoder):
                        # PretrainEncoder from neumiss_plus.py needs special handling
                        # It does denoising autoencoder pretraining internally
                        model.fit(
                            data['X_train'], data['y_train'],
                            data['X_val'], data['y_val'],
                        )
                    else:
                        model.fit(
                            data['X_train'], data['y_train'],
                            data['X_val'], data['y_val'],
                        )

                    r2 = model.score(data['X_test'], data['y_test'])
                    pred = model.predict(data['X_test'])
                    mse = float(np.mean((data['y_test'] - pred) ** 2))
                except Exception as e:
                    r2 = float('nan')
                    mse = float('nan')
                    print(f"  ERROR {model_name}: {e}")

                results.append({
                    'scenario': sc_name,
                    'model': model_name,
                    'seed': seed,
                    'r2': r2,
                    'mse': mse,
                })
                print(f"  [{done:3d}/{total}] {model_name:22s} seed={seed} "
                      f"R2={r2:+.4f}  MSE={mse:.4f}")

    # ---- Summary ----
    print("\n" + "=" * 90)
    print("SUMMARY: Mean R2 +/- std across 3 seeds")
    print("=" * 90)

    # Build summary manually (no pandas dependency)
    from collections import defaultdict
    grouped = defaultdict(list)
    for r in results:
        grouped[(r['scenario'], r['model'])].append((r['r2'], r['mse']))

    # Organize by scenario
    scenario_names = [s[0] for s in scenarios]
    model_names = [m[0] for m in all_models]

    for sc_name in scenario_names:
        print(f"\n--- {sc_name} ---")
        print(f"  {'Model':24s}  {'R2 mean':>8s} +/- {'std':>6s}  "
              f"{'MSE mean':>9s} +/- {'std':>7s}")
        print(f"  {'-'*72}")

        # Collect and sort by R2
        rows = []
        for mn in model_names:
            vals = grouped.get((sc_name, mn), [])
            if vals:
                r2s = [v[0] for v in vals]
                mses = [v[1] for v in vals]
                r2_mean = np.nanmean(r2s)
                r2_std = np.nanstd(r2s)
                mse_mean = np.nanmean(mses)
                mse_std = np.nanstd(mses)
                rows.append((mn, r2_mean, r2_std, mse_mean, mse_std))

        rows.sort(key=lambda x: -x[1])  # descending R2

        for mn, r2m, r2s, msem, mses in rows:
            print(f"  {mn:24s}  {r2m:+8.4f} +/- {r2s:6.4f}  "
                  f"{msem:9.4f} +/- {mses:7.4f}")

    # ---- Analysis ----
    print("\n" + "=" * 90)
    print("ANALYSIS: Where does MoE help?")
    print("=" * 90)

    for sc_name in scenario_names:
        print(f"\n--- {sc_name} ---")

        def get_r2(mn):
            vals = grouped.get((sc_name, mn), [])
            if not vals:
                return float('nan'), float('nan')
            r2s = [v[0] for v in vals]
            return np.nanmean(r2s), np.nanstd(r2s)

        baseline_r2, _ = get_r2("NeuMiss(d=3)")
        neumiss_c_r2, _ = get_r2("NeuMiss+C")
        pretrain_r2, _ = get_r2("PretrainEnc")
        impute_r2, _ = get_r2("ImputeMLP")

        print(f"  Baselines: NeuMiss={baseline_r2:+.4f}  "
              f"NeuMiss+C={neumiss_c_r2:+.4f}  "
              f"PretrainEnc={pretrain_r2:+.4f}  "
              f"ImputeMLP={impute_r2:+.4f}")

        # Best MoE per architecture type
        best_per_type = {}
        for variant_prefix in ['MoE_NeuMiss(K=', 'MoE_NM+C(K=', 'MoE_NM_MLP(K=']:
            best_name = None
            best_r2 = -999
            for mn in model_names:
                if mn.startswith(variant_prefix):
                    r2m, _ = get_r2(mn)
                    if r2m > best_r2:
                        best_r2 = r2m
                        best_name = mn
            if best_name:
                best_per_type[variant_prefix.rstrip('(')] = (best_name, best_r2)

        for vtype, (bname, br2) in best_per_type.items():
            delta_base = br2 - baseline_r2
            delta_impute = br2 - impute_r2
            print(f"  Best {vtype}: {bname} R2={br2:+.4f} "
                  f"(vs NeuMiss: {delta_base:+.4f}, vs ImputeMLP: {delta_impute:+.4f})")

        # Overall best
        all_r2s = [(mn, get_r2(mn)[0]) for mn in model_names]
        all_r2s.sort(key=lambda x: -x[1])
        winner = all_r2s[0]
        print(f"  >> Winner: {winner[0]} with R2={winner[1]:+.4f}")

    # ---- MoE K sensitivity ----
    print("\n" + "=" * 90)
    print("K SENSITIVITY: How does number of experts affect performance?")
    print("=" * 90)

    for sc_name in scenario_names:
        print(f"\n--- {sc_name} ---")
        for variant_label, prefix in [
            ("MoE_NeuMiss", "MoE_NeuMiss(K="),
            ("MoE_NM+C", "MoE_NM+C(K="),
            ("MoE_NM_MLP", "MoE_NM_MLP(K="),
        ]:
            line = f"  {variant_label:14s}: "
            for K in [2, 3, 5]:
                mn = f"{prefix}{K})"
                r2m, r2s = get_r2(mn)
                line += f"K={K}: {r2m:+.4f}+/-{r2s:.4f}  "
            print(line)

    return results


if __name__ == '__main__':
    results = run_experiment()

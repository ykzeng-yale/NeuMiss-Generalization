"""
Low-Rank NeuMiss Experiment.

Hypothesis: Using low-rank weight matrices W = U*V (U: d->r, V: r->d, r << d)
in NeuMiss Neumann layers should reduce parameters and improve generalization
at higher dimensions (d=20-50) where full d*d matrices overfit.

Two architectures:
1. LowRankNeuMiss  -- Variant C with low-rank weights (expand-activate-compress)
2. LowRankEncoder  -- NeuMiss-Encoder with low-rank weights on both pathways

Compared against:
- NeuMiss (original, depth=5)
- NM+C_gelu (Variant C, depth=3)
- ImputeMLP (mean impute + MLP baseline)
"""

import sys
import time
import warnings
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.base import BaseEstimator, RegressorMixin
from copy import deepcopy

sys.path.insert(0, '/Users/yukang/Desktop/NeuroMiss/src')
from data_generation import DataScenario
from neumiss_plus import (
    NeuMissPlus, ImputeMLP, EarlyStopping
)

warnings.filterwarnings('ignore')

# ============================================================================
# Low-Rank NeuMiss (Variant C structure with low-rank weight matrices)
# ============================================================================
class LowRankNeuMissNet(nn.Module):
    """NeuMiss Variant C with low-rank Neumann weight matrices.

    Each Neumann layer uses:
        h -> U (d->r) -> V (r->hidden) -> activate -> W_compress (hidden->d) -> mask

    The key d*d matrix is factored as U*V giving rank-r, reducing params
    from d*d to d*r + r*d = 2*d*r per layer.

    Retains the expand-activate-compress structure from Variant C:
        d -> r -> (expand to hidden) -> activate -> compress to d -> mask
    """
    def __init__(self, n_features, depth, rank, activation='gelu',
                 expansion_factor=2, residual_connection=False):
        super().__init__()
        self.depth = depth
        self.n_features = n_features
        self.rank = rank
        hidden = n_features * expansion_factor

        self.activation = self._get_activation(activation)

        # Low-rank Neumann layers: W_expand = U_expand @ V_expand
        # U_expand: d -> rank, V_expand: rank -> hidden
        self.l_U_expand = nn.ParameterList([
            nn.Parameter(torch.empty(n_features, rank))
            for _ in range(depth)
        ])
        self.l_V_expand = nn.ParameterList([
            nn.Parameter(torch.empty(rank, hidden))
            for _ in range(depth)
        ])
        # Compress: hidden -> d (kept full rank since hidden != d)
        self.l_W_compress = nn.ParameterList([
            nn.Parameter(torch.empty(hidden, n_features))
            for _ in range(depth)
        ])
        self.l_bias1 = nn.ParameterList([
            nn.Parameter(torch.zeros(hidden)) for _ in range(depth)
        ])
        self.l_bias2 = nn.ParameterList([
            nn.Parameter(torch.zeros(n_features)) for _ in range(depth)
        ])

        # Mixing weight Wc: also low-rank
        self.Uc = nn.Parameter(torch.empty(n_features, rank))
        self.Vc = nn.Parameter(torch.empty(rank, n_features))
        self.mu = nn.Parameter(torch.empty(n_features))
        self.beta = nn.Parameter(torch.empty(n_features))
        self.b = nn.Parameter(torch.empty(1))
        self.residual_connection = residual_connection

        self._init_weights()

    def _get_activation(self, name):
        return {'relu': nn.ReLU(), 'gelu': nn.GELU(), 'tanh': nn.Tanh(),
                'silu': nn.SiLU(), 'elu': nn.ELU()}.get(name, nn.GELU())

    def _init_weights(self):
        for U, V, Wc in zip(self.l_U_expand, self.l_V_expand,
                             self.l_W_compress):
            nn.init.xavier_normal_(U)
            nn.init.xavier_normal_(V)
            nn.init.xavier_normal_(Wc)
        nn.init.xavier_normal_(self.Uc)
        nn.init.xavier_normal_(self.Vc)
        nn.init.normal_(self.mu)
        nn.init.normal_(self.beta)
        nn.init.zeros_(self.b)

    def forward(self, x, m):
        obs = 1 - m
        h0 = x + m * self.mu
        h = x - obs * self.mu
        h_res = h.clone()

        for i in range(self.depth):
            # Low-rank expand: h @ U @ V + bias -> activate -> compress -> mask
            h = torch.matmul(h, self.l_U_expand[i])  # (batch, rank)
            h = torch.matmul(h, self.l_V_expand[i])  # (batch, hidden)
            h = h + self.l_bias1[i]
            h = self.activation(h)
            h = torch.matmul(h, self.l_W_compress[i]) + self.l_bias2[i]
            h = h * obs
            if self.residual_connection and i > 0:
                h = h + h_res
                h_res = h.clone()

        # Low-rank Wc
        Wc = torch.matmul(self.Uc, self.Vc)
        h = torch.matmul(h, Wc) * m + h0
        y = torch.matmul(h, self.beta) + self.b
        return y


class LowRankNeuMiss(BaseEstimator, RegressorMixin):
    """sklearn-compatible wrapper for LowRankNeuMissNet."""
    def __init__(self, depth=3, rank=None, activation='gelu',
                 expansion_factor=2, residual_connection=False,
                 n_epochs=200, batch_size=64, lr=0.001,
                 early_stopping=True, verbose=False):
        self.depth = depth
        self.rank = rank
        self.activation = activation
        self.expansion_factor = expansion_factor
        self.residual_connection = residual_connection
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.early_stopping = early_stopping
        self.verbose = verbose

    def fit(self, X, y, X_val=None, y_val=None):
        M = np.isnan(X).astype(np.float32)
        X_clean = np.nan_to_num(X).astype(np.float32)
        n, d = X.shape

        rank = self.rank if self.rank is not None else max(1, d // 2)

        X_t = torch.tensor(X_clean)
        M_t = torch.tensor(M)
        y_t = torch.tensor(y, dtype=torch.float32)

        if X_val is not None:
            Mv = np.isnan(X_val).astype(np.float32)
            Xv = torch.tensor(np.nan_to_num(X_val).astype(np.float32))
            Mv_t = torch.tensor(Mv)
            yv = torch.tensor(y_val, dtype=torch.float32)

        self.net = LowRankNeuMissNet(
            d, self.depth, rank, self.activation,
            self.expansion_factor, self.residual_connection
        )
        opt = optim.Adam(self.net.parameters(), lr=self.lr, weight_decay=1e-5)
        sched = ReduceLROnPlateau(opt, factor=0.2, patience=5)
        crit = nn.MSELoss()
        es = EarlyStopping(patience=20)

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
                loss = crit(self.net(x_, m_), y_)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
                opt.step()
                loss_sum += loss.item()

            sched.step(loss_sum / len(bx))

            if X_val is not None:
                self.net.eval()
                with torch.no_grad():
                    vl = crit(self.net(Xv, Mv_t), yv).item()
                es(vl, self.net)
                if es.early_stop:
                    break

            if opt.param_groups[0]['lr'] < 1e-7:
                break

        if es.checkpoint:
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
        return 1 - np.mean((y - pred) ** 2) / np.var(y)


# ============================================================================
# Low-Rank NeuMiss-Encoder (both mean and variance pathways low-rank)
# ============================================================================
class LowRankEncoderNet(nn.Module):
    """NeuMiss-Encoder with low-rank weight matrices in both pathways.

    Mean pathway: low-rank Neumann layers -> imputed features
    Variance pathway: low-rank Neumann layers -> uncertainty representation
    MLP head: [imputed, var_repr, obs, mask] -> prediction
    """
    def __init__(self, n_features, depth, rank, activation='gelu',
                 mlp_layers=(128, 64), dropout=0.1):
        super().__init__()
        self.n_features = n_features
        self.depth = depth
        self.rank = rank

        # --- Mean pathway (low-rank) ---
        self.mu = nn.Parameter(torch.empty(n_features))
        self.l_U_mean = nn.ParameterList([
            nn.Parameter(torch.empty(n_features, rank))
            for _ in range(depth)
        ])
        self.l_V_mean = nn.ParameterList([
            nn.Parameter(torch.empty(rank, n_features))
            for _ in range(depth)
        ])
        self.l_bias_mean = nn.ParameterList([
            nn.Parameter(torch.zeros(n_features)) for _ in range(depth)
        ])
        self.Uc_mean = nn.Parameter(torch.empty(n_features, rank))
        self.Vc_mean = nn.Parameter(torch.empty(rank, n_features))
        self.mean_acts = nn.ModuleList([
            self._get_activation(activation) for _ in range(depth)
        ])

        # --- Variance pathway (low-rank) ---
        self.l_U_var = nn.ParameterList([
            nn.Parameter(torch.empty(n_features, rank))
            for _ in range(depth)
        ])
        self.l_V_var = nn.ParameterList([
            nn.Parameter(torch.empty(rank, n_features))
            for _ in range(depth)
        ])
        self.l_bias_var = nn.ParameterList([
            nn.Parameter(torch.zeros(n_features)) for _ in range(depth)
        ])
        self.Uc_var = nn.Parameter(torch.empty(n_features, rank))
        self.Vc_var = nn.Parameter(torch.empty(rank, n_features))
        self.var_acts = nn.ModuleList([
            self._get_activation(activation) for _ in range(depth)
        ])

        # --- MLP head: [imputed(d) + var(d) + obs(d) + mask(d)] = 4d ---
        mlp_in = 4 * n_features
        mlp_modules = []
        for h_dim in mlp_layers:
            mlp_modules.append(nn.Linear(mlp_in, h_dim))
            mlp_modules.append(self._get_activation(activation))
            if dropout > 0:
                mlp_modules.append(nn.Dropout(dropout))
            mlp_in = h_dim
        mlp_modules.append(nn.Linear(mlp_in, 1))
        self.mlp_head = nn.Sequential(*mlp_modules)

        self._init_weights()

    def _get_activation(self, name):
        return {'relu': nn.ReLU(), 'gelu': nn.GELU(), 'tanh': nn.Tanh(),
                'silu': nn.SiLU(), 'elu': nn.ELU()}.get(name, nn.GELU())

    def _init_weights(self):
        for params in [self.l_U_mean, self.l_V_mean,
                       self.l_U_var, self.l_V_var]:
            for W in params:
                nn.init.xavier_normal_(W)
        for W in [self.Uc_mean, self.Vc_mean, self.Uc_var, self.Vc_var]:
            nn.init.xavier_normal_(W)
        nn.init.normal_(self.mu)
        for m in self.mlp_head.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x, m):
        obs = 1 - m
        h0 = x + m * self.mu

        # Mean pathway (low-rank)
        h_mean = x - obs * self.mu
        for i in range(self.depth):
            h_in = h_mean
            # Low-rank: W = U @ V
            h_mean = torch.matmul(h_mean, self.l_U_mean[i])
            h_mean = torch.matmul(h_mean, self.l_V_mean[i])
            h_mean = h_mean * obs + self.l_bias_mean[i]
            h_mean = self.mean_acts[i](h_mean)
            if i > 0:
                h_mean = h_mean + h_in
        Wc_mean = torch.matmul(self.Uc_mean, self.Vc_mean)
        imputed = torch.matmul(h_mean, Wc_mean) * m + h0

        # Variance pathway (low-rank)
        h_var = x - obs * self.mu
        for i in range(self.depth):
            h_in = h_var
            h_var = torch.matmul(h_var, self.l_U_var[i])
            h_var = torch.matmul(h_var, self.l_V_var[i])
            h_var = h_var * obs + self.l_bias_var[i]
            h_var = self.var_acts[i](h_var)
            if i > 0:
                h_var = h_var + h_in
        Wc_var = torch.matmul(self.Uc_var, self.Vc_var)
        var_repr = torch.matmul(h_var, Wc_var) * m

        features = torch.cat([imputed, var_repr, x * obs, m], dim=1)
        return self.mlp_head(features).squeeze(-1)


class LowRankEncoder(BaseEstimator, RegressorMixin):
    """sklearn-compatible wrapper for LowRankEncoderNet."""
    def __init__(self, depth=3, rank=None, activation='gelu',
                 mlp_layers=(128, 64), dropout=0.1,
                 n_epochs=150, batch_size=64, lr=0.001,
                 weight_decay=1e-5, early_stopping=True, verbose=False):
        self.depth = depth
        self.rank = rank
        self.activation = activation
        self.mlp_layers = mlp_layers
        self.dropout = dropout
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.early_stopping = early_stopping
        self.verbose = verbose

    def fit(self, X, y, X_val=None, y_val=None):
        M = np.isnan(X).astype(np.float32)
        X_clean = np.nan_to_num(X).astype(np.float32)
        n, d = X.shape

        rank = self.rank if self.rank is not None else max(1, d // 2)

        X_t = torch.tensor(X_clean)
        M_t = torch.tensor(M)
        y_t = torch.tensor(y, dtype=torch.float32)

        if X_val is not None:
            Mv = np.isnan(X_val).astype(np.float32)
            Xv = torch.tensor(np.nan_to_num(X_val).astype(np.float32))
            Mv_t = torch.tensor(Mv)
            yv = torch.tensor(y_val, dtype=torch.float32)

        self.net = LowRankEncoderNet(
            d, self.depth, rank, self.activation,
            self.mlp_layers, self.dropout
        )
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
                loss = crit(self.net(x_, m_), y_)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
                opt.step()
                loss_sum += loss.item()

            sched.step(loss_sum / len(bx))

            if X_val is not None:
                self.net.eval()
                with torch.no_grad():
                    vl = crit(self.net(Xv, Mv_t), yv).item()
                es(vl, self.net)
                if es.early_stop:
                    break
            if opt.param_groups[0]['lr'] < 1e-7:
                break

        if es.checkpoint:
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
        return 1 - np.mean((y - pred) ** 2) / np.var(y)


# ============================================================================
# Parameter counting utility
# ============================================================================
def count_params(model):
    """Count trainable parameters after fitting."""
    if hasattr(model, 'net'):
        return sum(p.numel() for p in model.net.parameters() if p.requires_grad)
    return 0


# ============================================================================
# Experiment runner
# ============================================================================
def run_experiment():
    print("=" * 80)
    print("LOW-RANK NEUMISS EXPERIMENT")
    print("Hypothesis: Low-rank W = U*V reduces overfitting at high d")
    print("=" * 80)

    # Experiment grid
    dims = [10, 20, 50]
    responses = ['linear', 'quadratic']
    missing_mech = 'MCAR'
    p = 0.5
    n_train = 10000
    n_val = 2000
    n_test = 2000
    seeds = [42, 123, 456]

    # Collect all results
    all_results = []

    for d in dims:
        for resp in responses:
            print(f"\n{'=' * 70}")
            print(f"d={d}, response={resp}, missing={missing_mech}, p={p}")
            print(f"{'=' * 70}")

            # Rank choices
            ranks = [max(1, d // 4), max(1, d // 2), d]

            seed_results = {name: [] for name in (
                ['NeuMiss_d5', 'NM+C_gelu_d3', 'ImputeMLP'] +
                [f'LR-NeuMiss_r{r}' for r in ranks] +
                [f'LR-Encoder_r{r}' for r in ranks]
            )}
            param_counts = {}

            for seed in seeds:
                print(f"\n--- Seed {seed} ---")

                scenario = DataScenario(
                    distribution='gaussian',
                    response=resp,
                    missing_mechanism=missing_mech,
                    n_features=d,
                    missing_rate=p,
                    snr=10.0,
                )
                data = scenario.generate(n_train, n_val, n_test,
                                         random_state=seed)

                X_tr = data['X_train']
                y_tr = data['y_train']
                X_va = data['X_val']
                y_va = data['y_val']
                X_te = data['X_test']
                y_te = data['y_test']

                models = {}

                # --- Baselines ---
                # 1. NeuMiss original (depth=5)
                models['NeuMiss_d5'] = NeuMissPlus(
                    variant='original', depth=5,
                    n_epochs=200, batch_size=128, lr=0.001,
                    early_stopping=True, verbose=False
                )

                # 2. NM+C gelu depth=3
                models['NM+C_gelu_d3'] = NeuMissPlus(
                    variant='C', depth=3, activation='gelu',
                    expansion_factor=2,
                    n_epochs=200, batch_size=128, lr=0.001,
                    early_stopping=True, verbose=False
                )

                # 3. ImputeMLP
                models['ImputeMLP'] = ImputeMLP(
                    hidden_layers=(128, 64, 32), activation='gelu',
                    n_epochs=150, batch_size=128, lr=0.001,
                    early_stopping=True, verbose=False
                )

                # --- Low-rank NeuMiss (variant C structure) ---
                for r in ranks:
                    name = f'LR-NeuMiss_r{r}'
                    models[name] = LowRankNeuMiss(
                        depth=3, rank=r, activation='gelu',
                        expansion_factor=2, residual_connection=False,
                        n_epochs=200, batch_size=128, lr=0.001,
                        early_stopping=True, verbose=False
                    )

                # --- Low-rank Encoder ---
                for r in ranks:
                    name = f'LR-Encoder_r{r}'
                    models[name] = LowRankEncoder(
                        depth=3, rank=r, activation='gelu',
                        mlp_layers=(128, 64), dropout=0.1,
                        n_epochs=150, batch_size=128, lr=0.001,
                        weight_decay=1e-5,
                        early_stopping=True, verbose=False
                    )

                # Train and evaluate all models
                for name, model in models.items():
                    t0 = time.time()
                    try:
                        model.fit(X_tr, y_tr, X_va, y_va)
                        r2 = model.score(X_te, y_te)
                        elapsed = time.time() - t0
                        seed_results[name].append(r2)

                        if seed == seeds[0]:
                            param_counts[name] = count_params(model)

                        print(f"  {name:25s}  R2={r2:.4f}  "
                              f"({elapsed:.1f}s, "
                              f"params={param_counts.get(name, '?')})")
                    except Exception as e:
                        print(f"  {name:25s}  FAILED: {e}")
                        seed_results[name].append(float('nan'))

            # Summarize across seeds
            print(f"\n--- Summary: d={d}, {resp}, {missing_mech} ---")
            print(f"{'Model':30s} {'Mean R2':>10s} {'Std R2':>10s} {'Params':>10s}")
            print("-" * 65)
            for name in seed_results:
                vals = [v for v in seed_results[name] if not np.isnan(v)]
                if vals:
                    mean_r2 = np.mean(vals)
                    std_r2 = np.std(vals)
                else:
                    mean_r2 = float('nan')
                    std_r2 = float('nan')
                pc = param_counts.get(name, '?')
                print(f"{name:30s} {mean_r2:10.4f} {std_r2:10.4f} {str(pc):>10s}")
                all_results.append({
                    'd': d, 'response': resp,
                    'model': name,
                    'mean_r2': mean_r2,
                    'std_r2': std_r2,
                    'params': pc,
                })

    # ====================================================================
    # Final summary table
    # ====================================================================
    print("\n\n" + "=" * 90)
    print("FINAL SUMMARY: Low-Rank NeuMiss Experiment")
    print("=" * 90)

    # Print by dimension
    for d in dims:
        print(f"\n{'=' * 80}")
        print(f"  d = {d}")
        print(f"{'=' * 80}")
        print(f"{'Model':30s}", end="")
        for resp in responses:
            print(f" {resp:>20s}", end="")
        print(f" {'Params':>10s}")
        print("-" * 90)

        # Get all model names for this d
        model_names = []
        for r in all_results:
            if r['d'] == d and r['model'] not in model_names:
                model_names.append(r['model'])

        for mname in model_names:
            print(f"{mname:30s}", end="")
            params_str = '?'
            for resp in responses:
                match = [r for r in all_results
                         if r['d'] == d and r['response'] == resp
                         and r['model'] == mname]
                if match:
                    entry = match[0]
                    mr2 = entry['mean_r2']
                    sr2 = entry['std_r2']
                    params_str = str(entry['params'])
                    if np.isnan(mr2):
                        print(f" {'FAILED':>20s}", end="")
                    else:
                        print(f" {mr2:7.4f} +/- {sr2:.4f}", end="")
                else:
                    print(f" {'N/A':>20s}", end="")
            print(f" {params_str:>10s}")

    # Highlight key findings
    print("\n\n" + "=" * 80)
    print("KEY FINDINGS")
    print("=" * 80)
    for d in dims:
        print(f"\n  d={d}:")
        d_results = [r for r in all_results if r['d'] == d]
        for resp in responses:
            resp_results = [r for r in d_results if r['response'] == resp]
            if resp_results:
                best = max(resp_results,
                           key=lambda r: r['mean_r2']
                           if not np.isnan(r['mean_r2']) else -999)
                print(f"    {resp:12s}: best = {best['model']} "
                      f"(R2={best['mean_r2']:.4f}, params={best['params']})")


if __name__ == '__main__':
    run_experiment()

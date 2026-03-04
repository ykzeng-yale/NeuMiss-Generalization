"""
Ralph Iteration 3: NeuMissAdaptive - Gated mixture of NeuMiss + mean imputation.

Hypothesis: ImputeMLP beats NeuMiss-Encoder by 1.5-6.4% R2 on nonlinear scenarios
because mean imputation provides a strong baseline that the MLP can leverage directly.
The NeuMiss-Encoder's dual Neumann pathways add complexity without enough benefit.

NeuMissAdaptive addresses this by:
1. Gating mechanism: learns to blend NeuMiss imputed features with mean-imputed features
   gate = sigmoid(W_g * [obs_features, mask])
   features = gate * neumiss_features + (1-gate) * mean_imputed_features
2. Feature cross-interactions: cross = imputed * obs (element-wise products)
   This captures quadratic-like interactions that the MLP head would otherwise
   need to discover from scratch.
3. Still uses the dual-pathway NeuMiss-Encoder as the base.

Tests against NeuMiss-Encoder and ImputeMLP on 4 key nonlinear scenarios.
"""
import sys
sys.path.insert(0, '/Users/yukang/Desktop/NeuroMiss/src')

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.base import BaseEstimator, RegressorMixin
from copy import deepcopy
import time

from data_generation import DataScenario
from neumiss_plus import (
    NeuMissEncoderEstimator,
    ImputeMLP,
    EarlyStopping,
)


# ============================================================================
# NeuMissAdaptive: Gated mixture of NeuMiss + mean imputation
# ============================================================================
class NeuMissAdaptiveNet(nn.Module):
    """NeuMiss with adaptive gating between NeuMiss features and mean-imputed features.

    Architecture:
        1. NeuMiss dual pathway -> neumiss_features (mean + variance pathways)
        2. Mean imputation -> impute_features (simple but effective)
        3. Gating network: gate = sigmoid(W_g @ [obs_features, mask])
        4. Blended: features = gate * neumiss_features + (1-gate) * impute_features
        5. Cross-interactions: cross = imputed * obs_features (element-wise)
        6. MLP head over [blended, cross, obs, mask]
    """

    def __init__(self, n_features, depth=3, activation='gelu',
                 mlp_layers=(128, 64), dropout=0.1):
        super().__init__()
        self.n_features = n_features
        self.depth = depth

        # --- Mean pathway (standard NeuMiss) ---
        self.mu = nn.Parameter(torch.empty(n_features))
        self.l_W_mean = nn.ParameterList([
            nn.Parameter(torch.empty(n_features, n_features))
            for _ in range(depth)
        ])
        self.l_bias_mean = nn.ParameterList([
            nn.Parameter(torch.zeros(n_features)) for _ in range(depth)
        ])
        self.Wc_mean = nn.Parameter(torch.empty(n_features, n_features))
        self.mean_acts = nn.ModuleList([
            self._get_activation(activation) for _ in range(depth)
        ])

        # --- Variance pathway (learns uncertainty representation) ---
        self.l_W_var = nn.ParameterList([
            nn.Parameter(torch.empty(n_features, n_features))
            for _ in range(depth)
        ])
        self.l_bias_var = nn.ParameterList([
            nn.Parameter(torch.zeros(n_features)) for _ in range(depth)
        ])
        self.Wc_var = nn.Parameter(torch.empty(n_features, n_features))
        self.var_acts = nn.ModuleList([
            self._get_activation(activation) for _ in range(depth)
        ])

        # --- Gating network ---
        # Input: [obs_features (d), mask (d)] -> gate per imputed feature (d)
        self.gate_net = nn.Sequential(
            nn.Linear(2 * n_features, n_features),
            self._get_activation(activation),
            nn.Linear(n_features, n_features),
            nn.Sigmoid()
        )

        # --- Learnable imputation bias (per-feature) ---
        # Instead of fixed global mean, learn a bias to add to mean imputation
        self.impute_bias = nn.Parameter(torch.zeros(n_features))

        # --- MLP head ---
        # Input: blended(d) + var_repr(d) + cross(d) + obs(d) + mask(d) = 5d
        mlp_in = 5 * n_features
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
        for W in list(self.l_W_mean) + list(self.l_W_var) + [self.Wc_mean, self.Wc_var]:
            nn.init.xavier_normal_(W)
        nn.init.normal_(self.mu)
        for m in self.mlp_head.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                nn.init.zeros_(m.bias)
        # Initialize gate network to produce ~0.5 (balanced start)
        for m in self.gate_net.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x, m):
        obs = 1 - m
        h0 = x + m * self.mu

        # --- Mean pathway ---
        h_mean = x - obs * self.mu
        for i in range(self.depth):
            h_in = h_mean
            h_mean = torch.matmul(h_mean, self.l_W_mean[i]) * obs + self.l_bias_mean[i]
            h_mean = self.mean_acts[i](h_mean)
            if i > 0:
                h_mean = h_mean + h_in
        neumiss_imputed = torch.matmul(h_mean, self.Wc_mean) * m + h0

        # --- Variance pathway ---
        h_var = x - obs * self.mu
        for i in range(self.depth):
            h_in = h_var
            h_var = torch.matmul(h_var, self.l_W_var[i]) * obs + self.l_bias_var[i]
            h_var = self.var_acts[i](h_var)
            if i > 0:
                h_var = h_var + h_in
        var_repr = torch.matmul(h_var, self.Wc_var) * m

        # --- Mean imputation pathway ---
        # x already has NaN replaced with 0; mu acts as learned imputation center
        # impute_features = observed values where present, mu + bias where missing
        impute_features = x * obs + m * (self.mu + self.impute_bias)

        # --- Gating: blend NeuMiss imputed with mean-imputed ---
        gate_input = torch.cat([x * obs, m], dim=1)
        gate = self.gate_net(gate_input)  # (batch, d), values in [0, 1]
        blended = gate * neumiss_imputed + (1 - gate) * impute_features

        # --- Cross-interactions: element-wise product of blended and obs ---
        cross = blended * (x * obs)

        # --- Assemble features for MLP ---
        features = torch.cat([blended, var_repr, cross, x * obs, m], dim=1)
        return self.mlp_head(features).squeeze(-1)


class NeuMissAdaptiveEstimator(BaseEstimator, RegressorMixin):
    """sklearn-compatible wrapper for NeuMissAdaptive."""

    def __init__(self, depth=3, mlp_layers=(128, 64), activation='gelu',
                 dropout=0.1, n_epochs=200, batch_size=64, lr=0.001,
                 weight_decay=1e-5, early_stopping=True, verbose=False):
        self.depth = depth
        self.mlp_layers = mlp_layers
        self.activation = activation
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

        X_t = torch.tensor(X_clean)
        M_t = torch.tensor(M)
        y_t = torch.tensor(y, dtype=torch.float32)

        if X_val is not None:
            Mv = np.isnan(X_val).astype(np.float32)
            Xv = torch.tensor(np.nan_to_num(X_val).astype(np.float32))
            Mv_t = torch.tensor(Mv)
            yv = torch.tensor(y_val, dtype=torch.float32)

        self.net = NeuMissAdaptiveNet(
            d, self.depth, self.activation, self.mlp_layers, self.dropout
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

            avg_loss = loss_sum / len(bx)
            sched.step(avg_loss)

            if self.verbose and epoch % 20 == 0:
                print(f"  Epoch {epoch}: loss={avg_loss:.6f}")

            if X_val is not None:
                self.net.eval()
                with torch.no_grad():
                    vl = crit(self.net(Xv, Mv_t), yv).item()
                es(vl, self.net)
                if es.early_stop:
                    if self.verbose:
                        print(f"  Early stop at epoch {epoch}")
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
# Experiment configuration
# ============================================================================
SCENARIOS = [
    DataScenario('gaussian', 'quadratic', 'MCAR', n_features=10,
                 missing_rate=0.5, snr=10.0),
    DataScenario('gaussian', 'interaction', 'MCAR', n_features=10,
                 missing_rate=0.5, snr=10.0),
    DataScenario('mixture_gaussian', 'quadratic', 'MCAR', n_features=10,
                 missing_rate=0.5, snr=10.0,
                 distribution_params={'n_components': 3}),
    DataScenario('student_t', 'quadratic', 'MCAR', n_features=10,
                 missing_rate=0.5, snr=10.0,
                 distribution_params={'df': 5}),
]

N_TRAIN = 10000
N_VAL = 2500
N_TEST = 3000
SEEDS = [42, 123, 456]
N_EPOCHS = 200


def build_models():
    """Build the three models to compare."""
    return {
        'NeuMiss-Encoder': NeuMissEncoderEstimator(
            variant='encoder', depth=3, mlp_layers=(128, 64),
            activation='gelu', dropout=0.1,
            n_epochs=N_EPOCHS, batch_size=64, lr=0.001,
            weight_decay=1e-5, verbose=False
        ),
        'ImputeMLP': ImputeMLP(
            hidden_layers=(128, 64, 32), activation='gelu',
            n_epochs=N_EPOCHS, batch_size=64, lr=0.001,
            dropout=0.1, verbose=False
        ),
        'NeuMissAdaptive': NeuMissAdaptiveEstimator(
            depth=3, mlp_layers=(128, 64), activation='gelu',
            dropout=0.1, n_epochs=N_EPOCHS, batch_size=64, lr=0.001,
            weight_decay=1e-5, verbose=False
        ),
    }


def run_experiment():
    """Run full experiment: 4 scenarios x 3 models x 3 seeds."""
    print("=" * 80)
    print("RALPH ITERATION 3: NeuMissAdaptive (Gated NeuMiss + Mean Imputation)")
    print("=" * 80)
    print(f"Config: n_train={N_TRAIN}, n_val={N_VAL}, n_test={N_TEST}, "
          f"seeds={SEEDS}, epochs={N_EPOCHS}")
    print()

    # Results storage: {scenario_name: {model_name: [r2_seed1, r2_seed2, ...]}}
    all_results = {}

    for scenario in SCENARIOS:
        scenario_name = scenario.name
        print(f"\n{'='*70}")
        print(f"Scenario: {scenario_name}")
        print(f"{'='*70}")

        all_results[scenario_name] = {}

        for seed in SEEDS:
            print(f"\n  --- Seed {seed} ---")
            torch.manual_seed(seed)
            np.random.seed(seed)

            data = scenario.generate(N_TRAIN, N_VAL, N_TEST, random_state=seed)

            models = build_models()
            for model_name, model in models.items():
                t0 = time.time()
                model.fit(data['X_train'], data['y_train'],
                          data['X_val'], data['y_val'])
                r2 = model.score(data['X_test'], data['y_test'])
                elapsed = time.time() - t0

                if model_name not in all_results[scenario_name]:
                    all_results[scenario_name][model_name] = []
                all_results[scenario_name][model_name].append(r2)

                print(f"    {model_name:25s}  R2={r2:.4f}  ({elapsed:.1f}s)")

    # --- Summary table ---
    print("\n\n" + "=" * 80)
    print("SUMMARY: Mean R2 (std) across 3 seeds")
    print("=" * 80)

    header = f"{'Scenario':45s}"
    model_names = ['NeuMiss-Encoder', 'ImputeMLP', 'NeuMissAdaptive']
    for mn in model_names:
        header += f"  {mn:>18s}"
    print(header)
    print("-" * len(header))

    for scenario_name, model_results in all_results.items():
        row = f"{scenario_name:45s}"
        scores = {}
        for mn in model_names:
            vals = model_results[mn]
            mean_r2 = np.mean(vals)
            std_r2 = np.std(vals)
            scores[mn] = mean_r2
            row += f"  {mean_r2:>7.4f}({std_r2:.4f})"
        print(row)

    # --- Gap analysis ---
    print("\n\n" + "=" * 80)
    print("GAP ANALYSIS: NeuMissAdaptive vs baselines (positive = Adaptive wins)")
    print("=" * 80)
    print(f"{'Scenario':45s}  {'vs Encoder':>12s}  {'vs ImputeMLP':>12s}  {'Winner':>15s}")
    print("-" * 90)

    for scenario_name, model_results in all_results.items():
        enc_mean = np.mean(model_results['NeuMiss-Encoder'])
        imp_mean = np.mean(model_results['ImputeMLP'])
        adp_mean = np.mean(model_results['NeuMissAdaptive'])

        gap_enc = (adp_mean - enc_mean) * 100
        gap_imp = (adp_mean - imp_mean) * 100

        best = max([('NeuMiss-Encoder', enc_mean),
                    ('ImputeMLP', imp_mean),
                    ('NeuMissAdaptive', adp_mean)],
                   key=lambda x: x[1])

        print(f"{scenario_name:45s}  {gap_enc:>+11.2f}%  {gap_imp:>+11.2f}%  {best[0]:>15s}")

    # --- Gate analysis for NeuMissAdaptive ---
    print("\n\n" + "=" * 80)
    print("GATE ANALYSIS: How much does NeuMissAdaptive rely on NeuMiss vs mean imputation?")
    print("(gate ~1 = prefer NeuMiss, gate ~0 = prefer mean imputation)")
    print("=" * 80)

    for scenario in SCENARIOS:
        scenario_name = scenario.name
        seed = SEEDS[0]
        torch.manual_seed(seed)
        np.random.seed(seed)
        data = scenario.generate(N_TRAIN, N_VAL, N_TEST, random_state=seed)

        model = NeuMissAdaptiveEstimator(
            depth=3, mlp_layers=(128, 64), activation='gelu',
            dropout=0.1, n_epochs=N_EPOCHS, batch_size=64, lr=0.001,
            weight_decay=1e-5, verbose=False
        )
        model.fit(data['X_train'], data['y_train'],
                  data['X_val'], data['y_val'])

        # Extract gate values on test set
        M_test = torch.tensor(np.isnan(data['X_test']).astype(np.float32))
        X_test = torch.tensor(np.nan_to_num(data['X_test']).astype(np.float32))
        obs = 1 - M_test
        model.net.eval()
        with torch.no_grad():
            gate_input = torch.cat([X_test * obs, M_test], dim=1)
            gate_vals = model.net.gate_net(gate_input)

        gate_mean = gate_vals.mean().item()
        gate_std = gate_vals.std().item()
        gate_min = gate_vals.min().item()
        gate_max = gate_vals.max().item()

        # Per-feature gate means for missing vs observed
        gate_missing = (gate_vals * M_test).sum(0) / M_test.sum(0).clamp(min=1)
        gate_observed = (gate_vals * obs).sum(0) / obs.sum(0).clamp(min=1)

        print(f"\n  {scenario_name}:")
        print(f"    Overall gate: mean={gate_mean:.4f}, std={gate_std:.4f}, "
              f"range=[{gate_min:.4f}, {gate_max:.4f}]")
        print(f"    Gate for missing features:  mean={gate_missing.mean().item():.4f}")
        print(f"    Gate for observed features: mean={gate_observed.mean().item():.4f}")

    print("\n\nDone.")


if __name__ == '__main__':
    run_experiment()

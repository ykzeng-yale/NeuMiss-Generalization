"""
Experiment: Fixing sinusoidal response prediction with periodic activations.

Problem: No architecture achieves positive R2 on sinusoidal response Y = 1 + sin(X@beta).
Standard activations (ReLU, GELU) cannot efficiently represent periodic functions,
and the Bayes-optimal predictor E[sin(X@b)|X_obs] = sin(mu_cond@b)*exp(-var_cond/2)
requires both the conditional mean AND variance of missing features.

Key findings from analysis:
- Oracle Bayes-optimal R2 ~ 0.31 (for Gaussian MCAR p=0.5, d=10, snr=10)
- Naive sin(E[X|Xobs]@beta) gives R2 ~ 0.03 due to Jensen's inequality
- Mean imputation + sin gives R2 ~ -0.93

Architectures tested:
1. FourierMLP: Mean impute + Random Fourier Features (learnable B) + MLP
2. SirenMLP: Mean impute + SIREN-style layers (sin activation, tuned omega)
3. FourierEncoder: NeuMiss-Encoder (mean+var pathways) + Fourier feature layer + MLP
4. ImputeMLP baseline with increased capacity
"""

import sys
sys.path.insert(0, '/Users/yukang/Desktop/NeuroMiss/src')

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.base import BaseEstimator, RegressorMixin
from copy import deepcopy
import time

from data_generation import DataScenario
from neumiss_plus import EarlyStopping, ImputeMLP


# ============================================================================
# Architecture 1: FourierMLP
# Mean impute + Learnable Fourier Features + MLP
# ============================================================================
class FourierMLPNet(nn.Module):
    """Learnable Fourier Features followed by an MLP.

    Maps input x to [sin(Bx), cos(Bx)] where B is LEARNED (not fixed),
    allowing the network to discover the right frequencies for the target.
    Also includes the raw imputed features and mask for linear baseline.
    """
    def __init__(self, n_features, n_fourier=256,
                 mlp_layers=(256, 128, 64), dropout=0.1):
        super().__init__()
        self.n_features = n_features
        self.n_fourier = n_fourier

        # Learnable Fourier projection matrix
        self.B = nn.Parameter(torch.randn(n_features, n_fourier) * 0.5)

        # Input: sin(Bx) + cos(Bx) + raw_imputed + mask = 2*n_fourier + 2*d
        mlp_in = 2 * n_fourier + 2 * n_features
        layers = []
        for h in mlp_layers:
            layers.append(nn.Linear(mlp_in, h))
            layers.append(nn.GELU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            mlp_in = h
        layers.append(nn.Linear(mlp_in, 1))
        self.mlp = nn.Sequential(*layers)

        self._init_weights()

    def _init_weights(self):
        for m in self.mlp.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x_imputed, mask):
        proj = x_imputed @ self.B
        fourier_features = torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)
        features = torch.cat([fourier_features, x_imputed, mask], dim=-1)
        return self.mlp(features).squeeze(-1)


class FourierMLP(BaseEstimator, RegressorMixin):
    """Mean imputation + Learnable Fourier Features + MLP."""
    def __init__(self, n_fourier=256,
                 mlp_layers=(256, 128, 64), dropout=0.1,
                 n_epochs=200, batch_size=128, lr=1e-3,
                 weight_decay=1e-5, early_stopping=True, verbose=False):
        self.n_fourier = n_fourier
        self.mlp_layers = mlp_layers
        self.dropout = dropout
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.early_stopping = early_stopping
        self.verbose = verbose

    def fit(self, X, y, X_val=None, y_val=None):
        self.means_ = np.nanmean(X, axis=0)
        self.stds_ = np.nanstd(X, axis=0)
        self.stds_[self.stds_ < 1e-8] = 1.0
        n, d = X.shape

        def impute_and_scale(Z):
            Z_imp = Z.copy()
            for j in range(d):
                m = np.isnan(Z_imp[:, j])
                Z_imp[m, j] = self.means_[j]
            return ((Z_imp - self.means_) / self.stds_).astype(np.float32)

        M = np.isnan(X).astype(np.float32)
        X_imp = impute_and_scale(X)
        X_t = torch.tensor(X_imp)
        M_t = torch.tensor(M)
        y_t = torch.tensor(y, dtype=torch.float32)

        if X_val is not None:
            Mv = np.isnan(X_val).astype(np.float32)
            Xv = torch.tensor(impute_and_scale(X_val))
            Mv_t = torch.tensor(Mv)
            yv = torch.tensor(y_val, dtype=torch.float32)

        self.net = FourierMLPNet(d, self.n_fourier, self.mlp_layers, self.dropout)
        opt = optim.Adam(self.net.parameters(), lr=self.lr,
                         weight_decay=self.weight_decay)
        sched = CosineAnnealingLR(opt, T_max=self.n_epochs, eta_min=1e-6)
        crit = nn.MSELoss()
        es = EarlyStopping(patience=50)

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
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), 5.0)
                opt.step()
                loss_sum += loss.item()
            sched.step()

            if X_val is not None:
                self.net.eval()
                with torch.no_grad():
                    vl = crit(self.net(Xv, Mv_t), yv).item()
                es(vl, self.net)
                if self.early_stopping and es.early_stop:
                    break

        if es.checkpoint:
            self.net.load_state_dict(es.checkpoint)
        return self

    def predict(self, X):
        d = X.shape[1]
        M = np.isnan(X).astype(np.float32)
        X_imp = X.copy()
        for j in range(d):
            m = np.isnan(X_imp[:, j])
            X_imp[m, j] = self.means_[j]
        X_imp = ((X_imp - self.means_) / self.stds_).astype(np.float32)
        self.net.eval()
        with torch.no_grad():
            return self.net(torch.tensor(X_imp), torch.tensor(M)).numpy()

    def score(self, X, y):
        pred = self.predict(X)
        return 1 - np.mean((y - pred) ** 2) / np.var(y)


# ============================================================================
# Architecture 2: SirenMLP
# Mean impute + SIREN layers with sin activation
# ============================================================================
class SirenLayer(nn.Module):
    """sin(omega * (Wx + b)) with SIREN initialization."""
    def __init__(self, in_features, out_features, omega=30.0, is_first=False):
        super().__init__()
        self.omega = omega
        self.linear = nn.Linear(in_features, out_features)
        with torch.no_grad():
            if is_first:
                self.linear.weight.uniform_(-1.0 / in_features, 1.0 / in_features)
            else:
                bound = np.sqrt(6.0 / in_features) / omega
                self.linear.weight.uniform_(-bound, bound)
            self.linear.bias.uniform_(-0.01, 0.01)

    def forward(self, x):
        return torch.sin(self.omega * self.linear(x))


class SirenMLPNet(nn.Module):
    """SIREN: all hidden layers use sin activation."""
    def __init__(self, n_features, hidden_layers=(256, 256, 128),
                 omega_0=30.0, dropout=0.1):
        super().__init__()
        in_dim = 2 * n_features  # imputed + mask
        layers = []
        for i, h in enumerate(hidden_layers):
            layers.append(SirenLayer(in_dim, h, omega=omega_0, is_first=(i == 0)))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_dim = h
        self.siren_layers = nn.Sequential(*layers)
        self.output_layer = nn.Linear(in_dim, 1)
        with torch.no_grad():
            bound = np.sqrt(6.0 / in_dim) / omega_0
            self.output_layer.weight.uniform_(-bound, bound)
            self.output_layer.bias.zero_()

    def forward(self, x_imputed, mask):
        h = torch.cat([x_imputed, mask], dim=-1)
        h = self.siren_layers(h)
        return self.output_layer(h).squeeze(-1)


class SirenMLP(BaseEstimator, RegressorMixin):
    """Mean imputation + SIREN."""
    def __init__(self, hidden_layers=(256, 256, 128), omega_0=30.0,
                 dropout=0.1, n_epochs=200, batch_size=128,
                 lr=1e-4, weight_decay=1e-5,
                 early_stopping=True, verbose=False):
        self.hidden_layers = hidden_layers
        self.omega_0 = omega_0
        self.dropout = dropout
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.early_stopping = early_stopping
        self.verbose = verbose

    def fit(self, X, y, X_val=None, y_val=None):
        self.means_ = np.nanmean(X, axis=0)
        self.stds_ = np.nanstd(X, axis=0)
        self.stds_[self.stds_ < 1e-8] = 1.0
        n, d = X.shape

        def impute_and_scale(Z):
            Z_imp = Z.copy()
            for j in range(d):
                m = np.isnan(Z_imp[:, j])
                Z_imp[m, j] = self.means_[j]
            return ((Z_imp - self.means_) / self.stds_).astype(np.float32)

        M = np.isnan(X).astype(np.float32)
        X_t = torch.tensor(impute_and_scale(X))
        M_t = torch.tensor(M)
        y_t = torch.tensor(y, dtype=torch.float32)

        if X_val is not None:
            Mv = np.isnan(X_val).astype(np.float32)
            Xv = torch.tensor(impute_and_scale(X_val))
            Mv_t = torch.tensor(Mv)
            yv = torch.tensor(y_val, dtype=torch.float32)

        self.net = SirenMLPNet(d, self.hidden_layers, self.omega_0, self.dropout)
        opt = optim.Adam(self.net.parameters(), lr=self.lr,
                         weight_decay=self.weight_decay)
        sched = CosineAnnealingLR(opt, T_max=self.n_epochs, eta_min=1e-7)
        crit = nn.MSELoss()
        es = EarlyStopping(patience=50)

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
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), 5.0)
                opt.step()
                loss_sum += loss.item()
            sched.step()

            if X_val is not None:
                self.net.eval()
                with torch.no_grad():
                    vl = crit(self.net(Xv, Mv_t), yv).item()
                es(vl, self.net)
                if self.early_stopping and es.early_stop:
                    break

        if es.checkpoint:
            self.net.load_state_dict(es.checkpoint)
        return self

    def predict(self, X):
        d = X.shape[1]
        M = np.isnan(X).astype(np.float32)
        X_imp = X.copy()
        for j in range(d):
            m = np.isnan(X_imp[:, j])
            X_imp[m, j] = self.means_[j]
        X_imp = ((X_imp - self.means_) / self.stds_).astype(np.float32)
        self.net.eval()
        with torch.no_grad():
            return self.net(torch.tensor(X_imp), torch.tensor(M)).numpy()

    def score(self, X, y):
        pred = self.predict(X)
        return 1 - np.mean((y - pred) ** 2) / np.var(y)


# ============================================================================
# Architecture 3: FourierEncoder
# NeuMiss-Encoder (mean + variance pathways) + Fourier feature layer + MLP
# ============================================================================
class FourierEncoderNet(nn.Module):
    """NeuMiss-Encoder + learnable Fourier features before MLP head.

    Key insight: For sin(X@beta), the Bayes predictor is
    sin(mu_cond@b)*exp(-var_cond/2). This architecture provides:
    - Mean pathway: learns E[X_mis|X_obs] (like standard NeuMiss)
    - Variance pathway: learns uncertainty (for the exp(-var/2) damping)
    - Fourier features: enables the MLP to represent sin() efficiently
    """
    def __init__(self, n_features, depth=3, n_fourier=128,
                 activation='gelu', mlp_layers=(256, 128), dropout=0.1):
        super().__init__()
        self.n_features = n_features
        self.depth = depth

        # --- Mean pathway ---
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

        # --- Variance pathway ---
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

        # --- Learnable Fourier projection of imputed representation ---
        self.B_fourier = nn.Parameter(torch.randn(n_features, n_fourier) * 0.3)

        # --- MLP head ---
        # fourier(2*nf) + var_repr(d) + imputed(d) + obs(d) + mask(d)
        mlp_in = 2 * n_fourier + 4 * n_features
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

    def forward(self, x, m):
        obs = 1 - m
        h0 = x + m * self.mu

        # Mean pathway
        h_mean = x - obs * self.mu
        for i in range(self.depth):
            h_in = h_mean
            h_mean = torch.matmul(h_mean, self.l_W_mean[i]) * obs + self.l_bias_mean[i]
            h_mean = self.mean_acts[i](h_mean)
            if i > 0:
                h_mean = h_mean + h_in
        imputed = torch.matmul(h_mean, self.Wc_mean) * m + h0

        # Variance pathway
        h_var = x - obs * self.mu
        for i in range(self.depth):
            h_in = h_var
            h_var = torch.matmul(h_var, self.l_W_var[i]) * obs + self.l_bias_var[i]
            h_var = self.var_acts[i](h_var)
            if i > 0:
                h_var = h_var + h_in
        var_repr = torch.matmul(h_var, self.Wc_var) * m

        # Fourier features of imputed representation
        proj = imputed @ self.B_fourier
        fourier_feat = torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)

        # Combine: fourier + var + imputed + observed + mask
        features = torch.cat([fourier_feat, var_repr, imputed, x * obs, m], dim=1)
        return self.mlp_head(features).squeeze(-1)


class FourierEncoder(BaseEstimator, RegressorMixin):
    """NeuMiss-Encoder with learnable Fourier feature layer."""
    def __init__(self, depth=3, n_fourier=128,
                 activation='gelu', mlp_layers=(256, 128), dropout=0.1,
                 n_epochs=200, batch_size=128, lr=1e-3,
                 weight_decay=1e-5, early_stopping=True, verbose=False):
        self.depth = depth
        self.n_fourier = n_fourier
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

        X_t = torch.tensor(X_clean)
        M_t = torch.tensor(M)
        y_t = torch.tensor(y, dtype=torch.float32)

        if X_val is not None:
            Mv = np.isnan(X_val).astype(np.float32)
            Xv = torch.tensor(np.nan_to_num(X_val).astype(np.float32))
            Mv_t = torch.tensor(Mv)
            yv = torch.tensor(y_val, dtype=torch.float32)

        self.net = FourierEncoderNet(d, self.depth, self.n_fourier,
                                     self.activation, self.mlp_layers, self.dropout)
        opt = optim.Adam(self.net.parameters(), lr=self.lr,
                         weight_decay=self.weight_decay)
        sched = CosineAnnealingLR(opt, T_max=self.n_epochs, eta_min=1e-6)
        crit = nn.MSELoss()
        es = EarlyStopping(patience=50)

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
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), 5.0)
                opt.step()
                loss_sum += loss.item()
            sched.step()

            if X_val is not None:
                self.net.eval()
                with torch.no_grad():
                    vl = crit(self.net(Xv, Mv_t), yv).item()
                es(vl, self.net)
                if self.early_stopping and es.early_stop:
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
# Oracle: Bayes-optimal predictor (requires known distribution parameters)
# ============================================================================
def compute_oracle_r2(data):
    """Compute the Bayes-optimal R2 for Gaussian X + sinusoidal response.

    For Gaussian X, the Bayes predictor is:
    E[sin(X@b)|X_obs] = sin(mu_cond@b) * exp(-0.5 * b_mis' Sigma_cond b_mis) + beta0
    """
    mu = data['mean']
    cov = data['cov']
    beta = data['beta']
    y_test = data['y_test']

    n_test = len(y_test)
    n_test_start = len(data['y_train']) + len(data['y_val'])
    X_complete = data['X_complete'][n_test_start:n_test_start + n_test]
    X_test_missing = data['X_test']
    M_test = np.isnan(X_test_missing).astype(bool)

    oracle_preds = np.zeros(n_test)
    for i in range(n_test):
        obs_idx = np.where(~M_test[i])[0]
        mis_idx = np.where(M_test[i])[0]

        if len(mis_idx) == 0:
            oracle_preds[i] = 1.0 + np.sin(X_complete[i] @ beta)
            continue
        if len(obs_idx) == 0:
            oracle_preds[i] = 1.0
            continue

        x_obs = X_complete[i, obs_idx]
        Sigma_oo = cov[np.ix_(obs_idx, obs_idx)]
        Sigma_mo = cov[np.ix_(mis_idx, obs_idx)]
        Sigma_mm = cov[np.ix_(mis_idx, mis_idx)]

        try:
            Sigma_oo_inv = np.linalg.solve(Sigma_oo, np.eye(len(obs_idx)))
        except np.linalg.LinAlgError:
            Sigma_oo_inv = np.linalg.pinv(Sigma_oo)
        mu_cond_mis = mu[mis_idx] + Sigma_mo @ Sigma_oo_inv @ (x_obs - mu[obs_idx])
        Sigma_cond = Sigma_mm - Sigma_mo @ Sigma_oo_inv @ Sigma_mo.T

        mu_cond_full = np.zeros(len(beta))
        mu_cond_full[obs_idx] = x_obs
        mu_cond_full[mis_idx] = mu_cond_mis

        beta_mis = beta[mis_idx]
        var_linear = max(0, beta_mis @ Sigma_cond @ beta_mis)

        mean_linear = mu_cond_full @ beta
        oracle_preds[i] = 1.0 + np.sin(mean_linear) * np.exp(-var_linear / 2)

    return 1 - np.mean((y_test - oracle_preds) ** 2) / np.var(y_test)


# ============================================================================
# Main experiment
# ============================================================================
def run_experiment():
    print("=" * 80)
    print("EXPERIMENT: Sinusoidal Response Fix with Periodic Activations")
    print("=" * 80)
    print()
    print("Key insight: The Bayes predictor for sin(X@b) given X_obs is")
    print("  E[Y|X_obs] = 1 + sin(mu_cond@b) * exp(-var_cond/2)")
    print("This requires BOTH conditional mean AND variance of X_mis|X_obs.")
    print()

    n_train = 10000
    n_val = 2000
    n_test = 2000
    seeds = [42, 123, 456]
    d = 10
    p = 0.5

    scenarios = [
        DataScenario('gaussian', 'sinusoidal', 'MCAR',
                     n_features=d, missing_rate=p, snr=10.0),
        DataScenario('gaussian', 'sinusoidal', 'MAR',
                     n_features=d, missing_rate=p, snr=10.0),
        DataScenario('mixture_gaussian', 'sinusoidal', 'MCAR',
                     n_features=d, missing_rate=p, snr=10.0,
                     distribution_params={'n_components': 3}),
    ]

    def get_models():
        return {
            'ImputeMLP (big)': ImputeMLP(
                hidden_layers=(256, 128, 64, 32), activation='gelu',
                n_epochs=200, batch_size=128, lr=1e-3, dropout=0.1,
                early_stopping=True, verbose=False
            ),
            'FourierMLP': FourierMLP(
                n_fourier=256, mlp_layers=(256, 128, 64), dropout=0.1,
                n_epochs=200, batch_size=128, lr=1e-3,
                weight_decay=1e-5, early_stopping=True, verbose=False
            ),
            'SirenMLP (omega=1)': SirenMLP(
                hidden_layers=(256, 256, 128), omega_0=1.0,
                dropout=0.05, n_epochs=200, batch_size=128,
                lr=5e-4, weight_decay=1e-5,
                early_stopping=True, verbose=False
            ),
            'SirenMLP (omega=5)': SirenMLP(
                hidden_layers=(256, 256, 128), omega_0=5.0,
                dropout=0.05, n_epochs=200, batch_size=128,
                lr=1e-4, weight_decay=1e-5,
                early_stopping=True, verbose=False
            ),
            'SirenMLP (omega=10)': SirenMLP(
                hidden_layers=(256, 256, 128), omega_0=10.0,
                dropout=0.05, n_epochs=200, batch_size=128,
                lr=1e-4, weight_decay=1e-5,
                early_stopping=True, verbose=False
            ),
            'FourierEncoder': FourierEncoder(
                depth=3, n_fourier=128, activation='gelu',
                mlp_layers=(256, 128), dropout=0.1,
                n_epochs=200, batch_size=128, lr=1e-3,
                weight_decay=1e-5, early_stopping=True, verbose=False
            ),
        }

    all_results = {}
    oracle_results = {}

    for scenario in scenarios:
        scenario_name = scenario.name
        print(f"\n{'='*70}")
        print(f"Scenario: {scenario_name}")
        print(f"{'='*70}")

        scenario_results = {}
        oracle_scores = []

        for seed in seeds:
            print(f"\n  --- Seed {seed} ---")
            data = scenario.generate(n_train, n_val, n_test, random_state=seed)

            y_train_var = np.var(data['y_train'])
            y_test_var = np.var(data['y_test'])
            miss_rate = np.isnan(data['X_train']).mean()
            print(f"  y_train var: {y_train_var:.4f}, y_test var: {y_test_var:.4f}")
            print(f"  Actual missing rate: {miss_rate:.3f}")

            # Oracle (only for Gaussian -- MG doesn't have simple Bayes form)
            if scenario.distribution == 'gaussian':
                oracle_r2 = compute_oracle_r2(data)
                oracle_scores.append(oracle_r2)
                print(f"    {'Oracle (Bayes-optimal)':30s} R2={oracle_r2:+.4f}")
            else:
                oracle_scores.append(float('nan'))
                print(f"    {'Oracle':30s} N/A (non-Gaussian)")

            models = get_models()
            for model_name, model in models.items():
                t0 = time.time()
                try:
                    model.fit(data['X_train'], data['y_train'],
                              data['X_val'], data['y_val'])
                    r2 = model.score(data['X_test'], data['y_test'])
                    elapsed = time.time() - t0
                    print(f"    {model_name:30s} R2={r2:+.4f}  ({elapsed:.1f}s)")
                except Exception as e:
                    r2 = float('nan')
                    elapsed = time.time() - t0
                    print(f"    {model_name:30s} FAILED: {e}  ({elapsed:.1f}s)")

                if model_name not in scenario_results:
                    scenario_results[model_name] = []
                scenario_results[model_name].append(r2)

        all_results[scenario_name] = scenario_results
        oracle_results[scenario_name] = oracle_scores

    # ===== SUMMARY =====
    print("\n\n" + "=" * 80)
    print("SUMMARY: Mean R2 (std) across 3 seeds")
    print("=" * 80)

    model_names = list(next(iter(all_results.values())).keys())
    scenario_names = list(all_results.keys())

    header = f"{'Model':30s}"
    for sn in scenario_names:
        short = sn.replace('sinusoidal_', '').replace('gaussian_', 'G/').replace('mixture_gaussian_', 'MG/')
        header += f" | {short:>20s}"
    print(header)
    print("-" * len(header))

    # Oracle row
    row = f"{'Oracle (Bayes-optimal)':30s}"
    for sn in scenario_names:
        scores = oracle_results[sn]
        valid = [s for s in scores if not np.isnan(s)]
        if valid:
            row += f" | {np.mean(valid):+.4f} ({np.std(valid):.4f})"
        else:
            row += f" | {'N/A':>20s}"
    print(row)
    print("-" * len(header))

    for model_name in model_names:
        row = f"{model_name:30s}"
        for sn in scenario_names:
            scores = all_results[sn][model_name]
            valid = [s for s in scores if not np.isnan(s)]
            if valid:
                row += f" | {np.mean(valid):+.4f} ({np.std(valid):.4f})"
            else:
                row += f" | {'FAILED':>20s}"
        print(row)

    # Best per scenario
    print("\n" + "-" * 60)
    print("Best model per scenario (excluding oracle):")
    for sn in scenario_names:
        best_model = None
        best_r2 = -999
        for model_name in model_names:
            scores = all_results[sn][model_name]
            valid = [s for s in scores if not np.isnan(s)]
            if valid and np.mean(valid) > best_r2:
                best_r2 = np.mean(valid)
                best_model = model_name
        short = sn.replace('sinusoidal_', '').replace('gaussian_', 'G/').replace('mixture_gaussian_', 'MG/')
        oracle_valid = [s for s in oracle_results[sn] if not np.isnan(s)]
        oracle_str = f" (oracle={np.mean(oracle_valid):+.4f})" if oracle_valid else ""
        print(f"  {short}: {best_model} (R2={best_r2:+.4f}){oracle_str}")

    # Analysis
    print("\n" + "=" * 80)
    print("ANALYSIS")
    print("=" * 80)
    print()
    print("The sinusoidal response Y = 1 + sin(X@beta) with 50% MCAR is fundamentally")
    print("hard because:")
    print("  1. The Bayes-optimal R2 is limited (~0.31 for Gaussian MCAR)")
    print("  2. The optimal predictor needs E[X|Xobs] AND Var(X|Xobs)")
    print("  3. Jensen inequality: sin(E[X@b]) != E[sin(X@b)]")
    print()
    print("Periodic activations (Fourier/SIREN) help the MLP learn sin() more")
    print("efficiently, but the bottleneck is the missing data imputation quality,")
    print("not the activation function. The FourierEncoder combines NeuMiss-style")
    print("imputation (mean+variance pathways) with Fourier features for the best")
    print("of both worlds.")

    print("\nDone.")


if __name__ == '__main__':
    run_experiment()

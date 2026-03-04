"""
Moment-Aware NeuMiss: Theoretically-motivated architecture for quadratic responses.

THEORETICAL MOTIVATION:
For Gaussian X with quadratic Y = X'AX + beta'X + eps, the Bayes predictor is:

  E[Y|X_obs, M] = X_obs'A_{obs,obs}X_obs + beta_obs'X_obs
      + 2 mu_{mis|obs}' A_{mis,obs} X_obs
      + mu_{mis|obs}' A_{mis,mis} mu_{mis|obs}
      + tr(A_{mis,mis} Sigma_{mis|obs})     <-- KEY EXTRA TERM
      + beta_mis' mu_{mis|obs}

The KEY EXTRA TERM tr(A_{mis,mis} Sigma_{mis|obs}) depends on the CONDITIONAL
VARIANCE Sigma_{mis|obs}, not just the conditional mean mu_{mis|obs}. Standard
NeuMiss only approximates mu_{mis|obs} via Neumann iterations.

Critical insight: For fixed Sigma, the conditional variance
  Sigma_{mis|obs} = Sigma_{mis,mis} - Sigma_{mis,obs} Sigma_{obs}^{-1} Sigma_{obs,mis}
depends ONLY on the mask pattern M (which features are observed/missing), not on
the observed values X_obs. So the trace term is purely a function of M.

This experiment implements two architectures that exploit this structure:
1. MomentNeuMiss: Full architecture with quadratic features + mask-dependent bias
2. VarianceNeuMiss: Simpler version using imputed^2 + mask features
"""

import sys
sys.path.insert(0, '/Users/yukang/Desktop/NeuroMiss/src')

import time
import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from copy import deepcopy
from sklearn.base import BaseEstimator, RegressorMixin

from data_generation import DataScenario
from neumiss_plus import (
    NeuMissPlus, ImputeMLP, PretrainEncoder, EarlyStopping
)


# ============================================================================
# MomentNeuMiss Network
# ============================================================================
class MomentNeuMissNet(nn.Module):
    """Moment-Aware NeuMiss: Theoretically motivated for quadratic responses.

    Paths:
      1. Standard Neumann iterations -> imputed (d features)
         Approximates mu_{mis|obs}.
      2. Quadratic features: x_obs^2 (d features)
         Provides diagonal second-moment information X_obs'A_{obs,obs}X_obs.
      3. Pairwise products of imputed features (projected to k dims)
         Provides cross terms mu_{mis|obs}'A mu_{mis|obs}.
      4. Mask-dependent bias: W_mask @ m -> scalar
         Learns the trace term tr(A_{mis,mis} Sigma_{mis|obs}) which depends
         only on the missingness pattern M.

    All concatenated -> small MLP head -> prediction.
    """

    def __init__(self, n_features, depth=3, activation='gelu',
                 cross_proj_dim=16, mlp_layers=(128, 64), dropout=0.1):
        super().__init__()
        self.n_features = n_features
        self.depth = depth
        d = n_features

        # --- Path 1: Standard Neumann imputation ---
        self.mu = nn.Parameter(torch.empty(d))
        self.l_W = nn.ParameterList([
            nn.Parameter(torch.empty(d, d)) for _ in range(depth)
        ])
        self.l_bias = nn.ParameterList([
            nn.Parameter(torch.zeros(d)) for _ in range(depth)
        ])
        self.Wc = nn.Parameter(torch.empty(d, d))
        self.neumann_acts = nn.ModuleList([
            self._get_activation(activation) for _ in range(depth)
        ])

        # --- Path 3: Cross features (pairwise products of imputed, projected) ---
        # Instead of full d*(d-1)/2, use learned projection to keep manageable
        self.cross_proj_dim = cross_proj_dim
        self.W_cross_A = nn.Parameter(torch.empty(d, cross_proj_dim))
        self.W_cross_B = nn.Parameter(torch.empty(d, cross_proj_dim))
        # Element-wise product of two projections gives cross_proj_dim features

        # --- Path 4: Mask-dependent bias (learns variance trace term) ---
        # Key: Sigma_{mis|obs} only depends on M, not on X_obs values
        # Multi-layer mask network for richer representation
        self.mask_net = nn.Sequential(
            nn.Linear(d, 2 * d),
            self._get_activation(activation),
            nn.Linear(2 * d, d),
            self._get_activation(activation),
        )

        # --- MLP head ---
        # Input: imputed(d) + x_obs^2(d) + cross(cross_proj_dim) + mask_repr(d) + mask(d)
        mlp_in = d + d + cross_proj_dim + d + d
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
        for W in list(self.l_W) + [self.Wc, self.W_cross_A, self.W_cross_B]:
            nn.init.xavier_normal_(W)
        nn.init.normal_(self.mu)
        for m in list(self.mlp_head.modules()) + list(self.mask_net.modules()):
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x, m):
        """
        x: (batch, d) - input with missing values replaced by 0
        m: (batch, d) - mask (1=missing, 0=observed)
        """
        obs = 1 - m  # 1 where observed

        # --- Path 1: Neumann imputation ---
        h0 = x + m * self.mu
        h = x - obs * self.mu

        for i in range(self.depth):
            h_in = h
            h = torch.matmul(h, self.l_W[i]) * obs + self.l_bias[i]
            h = self.neumann_acts[i](h)
            if i > 0:
                h = h + h_in  # residual
        imputed = torch.matmul(h, self.Wc) * m + h0  # (batch, d)

        # --- Path 2: Quadratic features of observed values ---
        x_obs_sq = (x * obs) ** 2  # (batch, d) - zero where missing

        # --- Path 3: Cross features via learned projection ---
        # Project imputed features through two different matrices, then
        # take element-wise product. This captures cross-feature interactions
        # like imputed_i * imputed_j weighted by learned coefficients.
        proj_a = torch.matmul(imputed, self.W_cross_A)  # (batch, cross_proj_dim)
        proj_b = torch.matmul(imputed, self.W_cross_B)  # (batch, cross_proj_dim)
        cross_features = proj_a * proj_b  # (batch, cross_proj_dim)

        # --- Path 4: Mask-dependent representation ---
        # Learns a function of M that can represent tr(A Sigma_{mis|obs})
        mask_repr = self.mask_net(m)  # (batch, d)

        # --- Concatenate all paths ---
        features = torch.cat([
            imputed,          # d: conditional mean approximation
            x_obs_sq,         # d: diagonal quadratic terms
            cross_features,   # cross_proj_dim: cross terms
            mask_repr,        # d: variance trace proxy
            m,                # d: raw mask pattern
        ], dim=1)

        return self.mlp_head(features).squeeze(-1)


# ============================================================================
# VarianceNeuMiss Network (simpler version)
# ============================================================================
class VarianceNeuMissNet(nn.Module):
    """Simpler moment-aware architecture.

    Uses [imputed, imputed^2, mask] as features for the MLP head.
    The imputed^2 terms provide second-moment information, while the
    mask features allow the network to learn the variance correction.

    The key insight: imputed^2 contains both (mu_{mis|obs})^2 and x_obs^2
    terms. Combined with mask (which encodes which Sigma_{mis|obs} applies),
    the MLP can learn to separate these contributions and add the trace term.
    """

    def __init__(self, n_features, depth=3, activation='gelu',
                 mlp_layers=(128, 64), dropout=0.1):
        super().__init__()
        self.n_features = n_features
        self.depth = depth
        d = n_features

        # Neumann imputation pathway
        self.mu = nn.Parameter(torch.empty(d))
        self.l_W = nn.ParameterList([
            nn.Parameter(torch.empty(d, d)) for _ in range(depth)
        ])
        self.l_bias = nn.ParameterList([
            nn.Parameter(torch.zeros(d)) for _ in range(depth)
        ])
        self.Wc = nn.Parameter(torch.empty(d, d))
        self.neumann_acts = nn.ModuleList([
            self._get_activation(activation) for _ in range(depth)
        ])

        # MLP head: [imputed, imputed^2, mask] = 3d features
        mlp_in = 3 * d
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
        for W in list(self.l_W) + [self.Wc]:
            nn.init.xavier_normal_(W)
        nn.init.normal_(self.mu)
        for m in self.mlp_head.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x, m):
        obs = 1 - m

        # Neumann imputation
        h0 = x + m * self.mu
        h = x - obs * self.mu
        for i in range(self.depth):
            h_in = h
            h = torch.matmul(h, self.l_W[i]) * obs + self.l_bias[i]
            h = self.neumann_acts[i](h)
            if i > 0:
                h = h + h_in
        imputed = torch.matmul(h, self.Wc) * m + h0

        # Features: [imputed, imputed^2, mask]
        imputed_sq = imputed ** 2
        features = torch.cat([imputed, imputed_sq, m], dim=1)

        return self.mlp_head(features).squeeze(-1)


# ============================================================================
# sklearn-compatible estimators
# ============================================================================
class MomentNeuMiss(BaseEstimator, RegressorMixin):
    """sklearn-compatible wrapper for MomentNeuMissNet."""

    def __init__(self, depth=3, activation='gelu', cross_proj_dim=16,
                 mlp_layers=(128, 64), dropout=0.1,
                 n_epochs=200, batch_size=64, lr=0.001, weight_decay=1e-5,
                 early_stopping=True, verbose=False):
        self.depth = depth
        self.activation = activation
        self.cross_proj_dim = cross_proj_dim
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

        self.net = MomentNeuMissNet(
            d, self.depth, self.activation, self.cross_proj_dim,
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

            avg_loss = loss_sum / len(bx)
            sched.step(avg_loss)

            if self.verbose and epoch % 50 == 0:
                print(f"  MomentNeuMiss epoch {epoch}: loss={avg_loss:.6f}")

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


class VarianceNeuMiss(BaseEstimator, RegressorMixin):
    """sklearn-compatible wrapper for VarianceNeuMissNet."""

    def __init__(self, depth=3, activation='gelu',
                 mlp_layers=(128, 64), dropout=0.1,
                 n_epochs=200, batch_size=64, lr=0.001, weight_decay=1e-5,
                 early_stopping=True, verbose=False):
        self.depth = depth
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

        self.net = VarianceNeuMissNet(
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

            if self.verbose and epoch % 50 == 0:
                print(f"  VarianceNeuMiss epoch {epoch}: loss={avg_loss:.6f}")

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
# NeuMiss original (SGD-based, matching paper settings)
# ============================================================================
class NeuMissOriginal(BaseEstimator, RegressorMixin):
    """Original NeuMiss with SGD training matching the paper settings."""

    def __init__(self, depth=3, n_epochs=200, batch_size=10, lr=None,
                 early_stopping=True, residual_connection=False,
                 mlp_depth=0, verbose=False):
        self.depth = depth
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.lr = lr  # will be set as 0.01/d if None
        self.early_stopping = early_stopping
        self.residual_connection = residual_connection
        self.mlp_depth = mlp_depth
        self.verbose = verbose

    def _build_net(self, d):
        """Build the Neumann network module."""
        net = _NeumannModule(d, self.depth, self.residual_connection,
                             self.mlp_depth)
        return net

    def fit(self, X, y, X_val=None, y_val=None):
        M = np.isnan(X).astype(np.float32)
        X_clean = np.nan_to_num(X).astype(np.float32)
        n, d = X.shape

        lr = self.lr if self.lr is not None else 0.01 / d

        X_t = torch.tensor(X_clean)
        M_t = torch.tensor(M)
        y_t = torch.tensor(y, dtype=torch.float32)

        if X_val is not None:
            Mv = np.isnan(X_val).astype(np.float32)
            Xv = torch.tensor(np.nan_to_num(X_val).astype(np.float32))
            Mv_t = torch.tensor(Mv)
            yv = torch.tensor(y_val, dtype=torch.float32)

        self.net = self._build_net(d)
        optimizer = optim.SGD(self.net.parameters(), lr=lr)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.2,
                                       patience=2, threshold=1e-4)
        criterion = nn.MSELoss()

        if self.early_stopping and X_val is not None:
            es = EarlyStopping(patience=15, verbose=self.verbose)

        for epoch in range(self.n_epochs):
            idx = torch.randperm(n)
            X_t, M_t, y_t = X_t[idx], M_t[idx], y_t[idx]

            bx = torch.split(X_t, self.batch_size)
            bm = torch.split(M_t, self.batch_size)
            by = torch.split(y_t, self.batch_size)

            running_loss = 0
            for x_, m_, y_ in zip(bx, bm, by):
                optimizer.zero_grad()
                y_hat = self.net(x_, m_)
                loss = criterion(y_hat, y_)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

            scheduler.step(running_loss / len(bx))

            if optimizer.param_groups[0]['lr'] < 5e-6:
                break

            if X_val is not None and self.early_stopping:
                with torch.no_grad():
                    y_hat_val = self.net(Xv, Mv_t)
                    val_loss = criterion(y_hat_val, yv).item()
                es(val_loss, self.net)
                if es.early_stop:
                    break

        if self.early_stopping and X_val is not None and es.checkpoint:
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


class _NeumannModule(nn.Module):
    """Core Neumann network matching the original implementation."""

    def __init__(self, n_features, depth, residual_connection=False,
                 mlp_depth=0):
        super().__init__()
        self.depth = depth
        self.n_features = n_features
        self.residual_connection = residual_connection
        self.mlp_depth = mlp_depth
        self.relu = nn.ReLU()

        d = n_features
        self.l_W = nn.ParameterList([
            nn.Parameter(torch.empty(d, d)) for _ in range(depth)
        ])
        self.Wc = nn.Parameter(torch.empty(d, d))
        self.beta = nn.Parameter(torch.empty(d))
        self.mu = nn.Parameter(torch.empty(d))
        self.b = nn.Parameter(torch.empty(1))

        self.l_W_mlp = nn.ParameterList([
            nn.Parameter(torch.empty(d, d)) for _ in range(mlp_depth)
        ])
        self.l_b_mlp = nn.ParameterList([
            nn.Parameter(torch.empty(d)) for _ in range(mlp_depth)
        ])

        self._init_weights()

    def _init_weights(self):
        bound = 1 / math.sqrt(self.n_features)
        for W in self.l_W:
            nn.init.kaiming_uniform_(W, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.Wc, a=math.sqrt(5))
        nn.init.uniform_(self.beta, -bound, bound)
        nn.init.uniform_(self.mu, -bound, bound)
        nn.init.normal_(self.b)
        for W in self.l_W_mlp:
            nn.init.kaiming_uniform_(W, a=math.sqrt(5))
        for b in self.l_b_mlp:
            nn.init.uniform_(b, -bound, bound)

    def forward(self, x, m):
        h0 = x + m * self.mu
        h = x - (1 - m) * self.mu
        h_res = x - (1 - m) * self.mu

        if len(self.l_W) > 0:
            h = torch.matmul(h, self.l_W[0]) * (1 - m)

        for W in list(self.l_W)[1:self.depth]:
            h = torch.matmul(h, W) * (1 - m)
            if self.residual_connection:
                h += h_res

        h = torch.matmul(h, self.Wc) * m + h0

        for W, b in zip(self.l_W_mlp, self.l_b_mlp):
            h = torch.matmul(h, W) + b
            h = self.relu(h)

        y = torch.matmul(h, self.beta) + self.b
        return y


# ============================================================================
# Experiment runner
# ============================================================================
def run_experiment():
    print("=" * 80)
    print("MOMENT-AWARE NEUMISS EXPERIMENT")
    print("Theoretically motivated architecture for quadratic/nonlinear responses")
    print("=" * 80)
    print()

    # Settings
    d = 10
    p = 0.5
    n_train = 10000
    n_val = 2500
    n_test = 3000
    seeds = [42, 123, 456]
    n_epochs = 200

    # Scenarios
    scenarios = [
        DataScenario('gaussian', 'linear', 'MCAR', d, p, snr=10.0),
        DataScenario('gaussian', 'quadratic', 'MCAR', d, p, snr=10.0),
        DataScenario('gaussian', 'interaction', 'MCAR', d, p, snr=10.0),
        DataScenario('gaussian', 'cubic', 'MCAR', d, p, snr=10.0),
        DataScenario('mixture_gaussian', 'quadratic', 'MCAR', d, p, snr=10.0,
                     distribution_params={'n_components': 3}),
        DataScenario('student_t', 'quadratic', 'MCAR', d, p, snr=10.0,
                     distribution_params={'df': 5}),
    ]

    # Methods
    def make_methods():
        return {
            'NeuMiss(d=3)': NeuMissOriginal(
                depth=3, n_epochs=n_epochs, batch_size=10, lr=0.01 / d,
                early_stopping=True
            ),
            'NeuMiss(d=5)': NeuMissOriginal(
                depth=5, n_epochs=n_epochs, batch_size=10, lr=0.01 / d,
                early_stopping=True
            ),
            'NeuMiss+C': NeuMissPlus(
                variant='C', depth=3, activation='gelu', expansion_factor=2,
                n_epochs=n_epochs, batch_size=64, lr=0.001, early_stopping=True
            ),
            'PretrainEnc': PretrainEncoder(
                depth=3, mlp_layers=(128,), activation='gelu',
                pretrain_epochs=50, train_epochs=n_epochs,
                batch_size=64, lr=0.001, early_stopping=True
            ),
            'ImputeMLP': ImputeMLP(
                hidden_layers=(128, 64, 32), activation='gelu',
                n_epochs=n_epochs, batch_size=64, lr=0.001, early_stopping=True
            ),
            'MomentNeuMiss': MomentNeuMiss(
                depth=3, activation='gelu', cross_proj_dim=16,
                mlp_layers=(128, 64), dropout=0.1,
                n_epochs=n_epochs, batch_size=64, lr=0.001,
                early_stopping=True
            ),
            'VarianceNeuMiss': VarianceNeuMiss(
                depth=3, activation='gelu',
                mlp_layers=(128, 64), dropout=0.1,
                n_epochs=n_epochs, batch_size=64, lr=0.001,
                early_stopping=True
            ),
        }

    # Results storage
    all_results = {}  # scenario_name -> method_name -> list of R2

    for scenario in scenarios:
        sname = scenario.name
        print(f"\n{'='*70}")
        print(f"SCENARIO: {sname}")
        print(f"  Distribution: {scenario.distribution}, Response: {scenario.response}")
        print(f"  Missing: {scenario.missing_mechanism}, Rate: {scenario.missing_rate}")
        print(f"{'='*70}")

        all_results[sname] = {}

        for seed in seeds:
            print(f"\n  --- Seed {seed} ---")
            data = scenario.generate(n_train, n_val, n_test, random_state=seed)

            methods = make_methods()

            for mname, model in methods.items():
                t0 = time.time()
                try:
                    model.fit(data['X_train'], data['y_train'],
                              data['X_val'], data['y_val'])
                    r2 = model.score(data['X_test'], data['y_test'])
                    elapsed = time.time() - t0
                    print(f"    {mname:20s}: R2 = {r2:+.4f}  ({elapsed:.1f}s)")
                except Exception as e:
                    r2 = float('nan')
                    elapsed = time.time() - t0
                    print(f"    {mname:20s}: FAILED ({e})  ({elapsed:.1f}s)")

                if mname not in all_results[sname]:
                    all_results[sname][mname] = []
                all_results[sname][mname].append(r2)

    # ========================================================================
    # Summary table
    # ========================================================================
    print("\n\n")
    print("=" * 100)
    print("SUMMARY: Mean R2 (std) across seeds")
    print("=" * 100)

    method_names = list(make_methods().keys())

    # Header
    header = f"{'Scenario':<35s}"
    for mname in method_names:
        header += f" {mname:>17s}"
    print(header)
    print("-" * len(header))

    for sname in all_results:
        row = f"{sname:<35s}"
        for mname in method_names:
            scores = all_results[sname].get(mname, [])
            valid = [s for s in scores if not np.isnan(s)]
            if valid:
                mean_r2 = np.mean(valid)
                std_r2 = np.std(valid)
                row += f" {mean_r2:+.4f}({std_r2:.3f})"
            else:
                row += f"          NaN     "
        print(row)

    # ========================================================================
    # Analysis: Where do moment-aware methods help?
    # ========================================================================
    print("\n\n")
    print("=" * 80)
    print("ANALYSIS: Improvement of moment-aware methods over baselines")
    print("=" * 80)

    for sname in all_results:
        print(f"\n{sname}:")
        res = all_results[sname]

        for base in ['NeuMiss(d=3)', 'ImputeMLP', 'PretrainEnc']:
            base_scores = [s for s in res.get(base, []) if not np.isnan(s)]
            if not base_scores:
                continue
            base_mean = np.mean(base_scores)

            for new in ['MomentNeuMiss', 'VarianceNeuMiss']:
                new_scores = [s for s in res.get(new, []) if not np.isnan(s)]
                if not new_scores:
                    continue
                new_mean = np.mean(new_scores)
                diff = new_mean - base_mean
                pct = diff / max(abs(base_mean), 1e-8) * 100
                marker = "+++" if diff > 0.02 else "++" if diff > 0.01 else "+" if diff > 0 else "-" if diff > -0.01 else "--"
                print(f"  {new:>17s} vs {base:<15s}: {diff:+.4f} ({pct:+.1f}%) [{marker}]")

    print("\n\nDone.")


if __name__ == '__main__':
    run_experiment()

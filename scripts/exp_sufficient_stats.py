"""Sufficient Statistics Experiment: Theoretically Optimal Architectures.

THEORETICAL INSIGHT:
For ANY f, the Bayes predictor is E[f(X)|X_obs, M]. If we could perfectly
impute X_mis (sample from P(X_mis|X_obs, M)), then:
    E[f(X)|X_obs] = E_{X_mis|X_obs}[f(X_obs, X_mis)]

For Gaussian X:
    P(X_mis|X_obs) = N(mu_{mis|obs}, Sigma_{mis|obs})

The sufficient statistics are mu_{mis|obs} and Sigma_{mis|obs}.

KEY INSIGHT: Sigma_{mis|obs} = Sigma_{mm} - Sigma_{mo} Sigma_{oo}^{-1} Sigma_{om}
This ONLY depends on M (the mask pattern) and Sigma, NOT on X_obs!
So the conditional variance is a function of M only (given fixed Sigma).

This experiment tests three architectures motivated by this theory:

1. SufficientStatNeuMiss:
   - NeuMiss layers -> mu_{mis|obs} approximation (d features)
   - Learned mask function: W_var @ M + b_var -> d features (approx diag(Sigma_{mis|obs}))
   - Input to MLP: [imputed_X (d), var_proxy (d), X*obs (d), M (d)] = 4d features
   - MLP head -> prediction

2. MultiImputeNeuMiss:
   - Use NeuMiss to get mu_{mis|obs}
   - Learn a variance proxy from M
   - Generate K "pseudo-samples": X_k = X_obs + mu_{mis|obs}*M + eps_k * sqrt(var_proxy) * M
   - Pass each through shared MLP, average predictions
   - Differentiable approximation of multiple imputation

3. OracleTest:
   - Use TRUE conditional mean and variance (computed from known Sigma)
   - Feed [X_obs, true_mu_{mis|obs}, true_diag(Sigma_{mis|obs}), M] -> MLP
   - Gives UPPER BOUND on what is achievable

Baselines: PretrainEncoder, ImputeMLP, original NeuMiss.
"""
import sys
sys.path.insert(0, '/Users/yukang/Desktop/NeuroMiss/src')
sys.path.insert(0, '/Users/yukang/Desktop/NeuroMiss/NeuMiss_original/python')

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from copy import deepcopy
from sklearn.base import BaseEstimator, RegressorMixin

from data_generation import DataScenario
from neumiss_plus import ImputeMLP, EarlyStopping, PretrainEncoder, NeuMissPlus


# ============================================================================
# Helper: standard training loop (shared by multiple estimators)
# ============================================================================
def train_loop(net, X_t, M_t, y_t, Xv_t, Mv_t, yv_t,
               n_epochs=200, batch_size=64, lr=0.001, weight_decay=1e-5,
               patience=25, verbose=False):
    """Standard training loop with early stopping. Returns best state dict."""
    opt = optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)
    sched = ReduceLROnPlateau(opt, factor=0.2, patience=7)
    crit = nn.MSELoss()
    es = EarlyStopping(patience=patience)
    n = X_t.shape[0]

    for epoch in range(n_epochs):
        net.train()
        idx = torch.randperm(n)
        _X, _M, _y = X_t[idx], M_t[idx], y_t[idx]

        bx = torch.split(_X, batch_size)
        bm = torch.split(_M, batch_size)
        by = torch.split(_y, batch_size)

        loss_sum = 0
        for x_, m_, y_ in zip(bx, bm, by):
            opt.zero_grad()
            pred = net(x_, m_)
            loss = crit(pred, y_)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            loss_sum += loss.item()

        sched.step(loss_sum / len(bx))

        if Xv_t is not None:
            net.eval()
            with torch.no_grad():
                vl = crit(net(Xv_t, Mv_t), yv_t).item()
            es(vl, net)
            if es.early_stop:
                if verbose:
                    print(f"  Early stop epoch {epoch}, val_loss={vl:.6f}")
                break

        if opt.param_groups[0]['lr'] < 1e-7:
            break

    if es.checkpoint:
        net.load_state_dict(es.checkpoint)


# ============================================================================
# 1. SufficientStatNeuMiss
# ============================================================================
class SufficientStatNet(nn.Module):
    """NeuMiss imputation + learned variance from mask + MLP.

    Input to MLP: [imputed_X (d), var_proxy (d), X*obs (d), M (d)] = 4d
    The variance proxy is a function of M ONLY (not X_obs),
    which matches the theory that Sigma_{mis|obs} depends only on
    the mask pattern M and the population covariance Sigma.
    """
    def __init__(self, n_features, depth=3, activation='gelu',
                 mlp_layers=(128, 64), dropout=0.1):
        super().__init__()
        d = n_features
        self.depth = depth

        # --- NeuMiss imputation pathway (approximates mu_{mis|obs}) ---
        self.mu = nn.Parameter(torch.empty(d))
        self.l_W = nn.ParameterList([
            nn.Parameter(torch.empty(d, d)) for _ in range(depth)
        ])
        self.l_bias = nn.ParameterList([
            nn.Parameter(torch.zeros(d)) for _ in range(depth)
        ])
        self.Wc = nn.Parameter(torch.empty(d, d))
        self.acts = nn.ModuleList([
            self._act(activation) for _ in range(depth)
        ])

        # --- Variance proxy from mask: linear map M -> var_proxy ---
        # Sigma_{mis|obs} depends only on M (and fixed Sigma), so a
        # learned function of M suffices. We use a small MLP on M.
        self.var_net = nn.Sequential(
            nn.Linear(d, 2 * d),
            self._act(activation),
            nn.Linear(2 * d, d),
            nn.Softplus(),  # variance must be positive
        )

        # --- MLP head: [imputed(d), var_proxy(d), obs_features(d), mask(d)] ---
        mlp_in = 4 * d
        layers = []
        for h in mlp_layers:
            layers.append(nn.Linear(mlp_in, h))
            layers.append(self._act(activation))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            mlp_in = h
        layers.append(nn.Linear(mlp_in, 1))
        self.mlp_head = nn.Sequential(*layers)

        self._init()

    def _act(self, name):
        return {'relu': nn.ReLU(), 'gelu': nn.GELU(), 'tanh': nn.Tanh(),
                'silu': nn.SiLU(), 'elu': nn.ELU()}.get(name, nn.GELU())

    def _init(self):
        for W in list(self.l_W) + [self.Wc]:
            nn.init.xavier_normal_(W)
        nn.init.normal_(self.mu)
        for m in list(self.var_net.modules()) + list(self.mlp_head.modules()):
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x, m):
        obs = 1 - m

        # NeuMiss imputation -> mu_{mis|obs} approximation
        h0 = x + m * self.mu
        h = x - obs * self.mu
        for i in range(self.depth):
            h_in = h
            h = torch.matmul(h, self.l_W[i]) * obs + self.l_bias[i]
            h = self.acts[i](h)
            if i > 0:
                h = h + h_in
        imputed = torch.matmul(h, self.Wc) * m + h0

        # Variance proxy from mask only
        var_proxy = self.var_net(m)

        # Concatenate sufficient statistics
        features = torch.cat([imputed, var_proxy, x * obs, m], dim=1)
        return self.mlp_head(features).squeeze(-1)


class SufficientStatNeuMiss(BaseEstimator, RegressorMixin):
    """sklearn wrapper for SufficientStatNet."""
    def __init__(self, depth=3, mlp_layers=(128, 64), activation='gelu',
                 dropout=0.1, n_epochs=200, batch_size=64, lr=0.001,
                 weight_decay=1e-5, verbose=False):
        self.depth = depth
        self.mlp_layers = mlp_layers
        self.activation = activation
        self.dropout = dropout
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.verbose = verbose

    def fit(self, X, y, X_val=None, y_val=None, **kwargs):
        M = np.isnan(X).astype(np.float32)
        Xc = np.nan_to_num(X).astype(np.float32)
        n, d = X.shape

        X_t = torch.tensor(Xc); M_t = torch.tensor(M)
        y_t = torch.tensor(y, dtype=torch.float32)

        Xv_t, Mv_t, yv_t = None, None, None
        if X_val is not None:
            Mv = np.isnan(X_val).astype(np.float32)
            Xv_t = torch.tensor(np.nan_to_num(X_val).astype(np.float32))
            Mv_t = torch.tensor(Mv)
            yv_t = torch.tensor(y_val, dtype=torch.float32)

        self.net = SufficientStatNet(d, self.depth, self.activation,
                                     self.mlp_layers, self.dropout)
        train_loop(self.net, X_t, M_t, y_t, Xv_t, Mv_t, yv_t,
                   self.n_epochs, self.batch_size, self.lr,
                   self.weight_decay, verbose=self.verbose)
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
# 2. MultiImputeNeuMiss
# ============================================================================
class MultiImputeNet(nn.Module):
    """Differentiable multiple imputation via NeuMiss + learned variance.

    Training: Draw K pseudo-samples from N(mu_{mis|obs}, var_proxy) at
    missing positions, pass each through shared MLP, average predictions.

    Test: Use deterministic imputation (just the mean, since
    E[f(X)] ~ average over samples converges to the same thing with K->inf,
    but we also try K>1 at test time for better approximation).
    """
    def __init__(self, n_features, depth=3, K_samples=5,
                 activation='gelu', mlp_layers=(128, 64), dropout=0.1):
        super().__init__()
        d = n_features
        self.depth = depth
        self.K = K_samples

        # --- NeuMiss imputation pathway ---
        self.mu = nn.Parameter(torch.empty(d))
        self.l_W = nn.ParameterList([
            nn.Parameter(torch.empty(d, d)) for _ in range(depth)
        ])
        self.l_bias = nn.ParameterList([
            nn.Parameter(torch.zeros(d)) for _ in range(depth)
        ])
        self.Wc = nn.Parameter(torch.empty(d, d))
        self.acts = nn.ModuleList([
            self._act(activation) for _ in range(depth)
        ])

        # --- Variance proxy from mask ---
        self.var_net = nn.Sequential(
            nn.Linear(d, 2 * d),
            self._act(activation),
            nn.Linear(2 * d, d),
            nn.Softplus(),
        )

        # --- Shared MLP: takes d-dimensional "completed" X ---
        mlp_in = d
        layers = []
        for h in mlp_layers:
            layers.append(nn.Linear(mlp_in, h))
            layers.append(self._act(activation))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            mlp_in = h
        layers.append(nn.Linear(mlp_in, 1))
        self.mlp_head = nn.Sequential(*layers)

        self._init()

    def _act(self, name):
        return {'relu': nn.ReLU(), 'gelu': nn.GELU(), 'tanh': nn.Tanh(),
                'silu': nn.SiLU(), 'elu': nn.ELU()}.get(name, nn.GELU())

    def _init(self):
        for W in list(self.l_W) + [self.Wc]:
            nn.init.xavier_normal_(W)
        nn.init.normal_(self.mu)
        for mod in list(self.var_net.modules()) + list(self.mlp_head.modules()):
            if isinstance(mod, nn.Linear):
                nn.init.kaiming_normal_(mod.weight)
                nn.init.zeros_(mod.bias)

    def _neumiss_impute(self, x, m):
        """Run NeuMiss layers to get imputed X."""
        obs = 1 - m
        h0 = x + m * self.mu
        h = x - obs * self.mu
        for i in range(self.depth):
            h_in = h
            h = torch.matmul(h, self.l_W[i]) * obs + self.l_bias[i]
            h = self.acts[i](h)
            if i > 0:
                h = h + h_in
        return torch.matmul(h, self.Wc) * m + h0

    def forward(self, x, m):
        obs = 1 - m
        imputed = self._neumiss_impute(x, m)  # (batch, d)
        var_proxy = self.var_net(m)            # (batch, d)
        std_proxy = torch.sqrt(var_proxy + 1e-8)

        if self.training:
            # Draw K pseudo-samples and average predictions
            preds = []
            for _ in range(self.K):
                eps = torch.randn_like(x)
                # X_k = X_obs + mu_{mis|obs} * M + eps * std * M
                # = imputed + eps * std * M  (since imputed already has X_obs at obs positions)
                x_k = imputed + eps * std_proxy * m
                pred_k = self.mlp_head(x_k).squeeze(-1)
                preds.append(pred_k)
            return torch.stack(preds, dim=0).mean(dim=0)
        else:
            # At test time, also use K samples for better E[f] approximation
            preds = []
            K_test = max(self.K, 10)
            for _ in range(K_test):
                eps = torch.randn_like(x)
                x_k = imputed + eps * std_proxy * m
                pred_k = self.mlp_head(x_k).squeeze(-1)
                preds.append(pred_k)
            return torch.stack(preds, dim=0).mean(dim=0)


class MultiImputeNeuMiss(BaseEstimator, RegressorMixin):
    """sklearn wrapper for MultiImputeNet."""
    def __init__(self, depth=3, K_samples=5, mlp_layers=(128, 64),
                 activation='gelu', dropout=0.1, n_epochs=200,
                 batch_size=64, lr=0.001, weight_decay=1e-5, verbose=False):
        self.depth = depth
        self.K_samples = K_samples
        self.mlp_layers = mlp_layers
        self.activation = activation
        self.dropout = dropout
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.verbose = verbose

    def fit(self, X, y, X_val=None, y_val=None, **kwargs):
        M = np.isnan(X).astype(np.float32)
        Xc = np.nan_to_num(X).astype(np.float32)
        n, d = X.shape

        X_t = torch.tensor(Xc); M_t = torch.tensor(M)
        y_t = torch.tensor(y, dtype=torch.float32)

        Xv_t, Mv_t, yv_t = None, None, None
        if X_val is not None:
            Mv = np.isnan(X_val).astype(np.float32)
            Xv_t = torch.tensor(np.nan_to_num(X_val).astype(np.float32))
            Mv_t = torch.tensor(Mv)
            yv_t = torch.tensor(y_val, dtype=torch.float32)

        self.net = MultiImputeNet(d, self.depth, self.K_samples,
                                  self.activation, self.mlp_layers,
                                  self.dropout)
        train_loop(self.net, X_t, M_t, y_t, Xv_t, Mv_t, yv_t,
                   self.n_epochs, self.batch_size, self.lr,
                   self.weight_decay, verbose=self.verbose)
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
# 3. OracleTest: TRUE conditional mean and variance
# ============================================================================
class OracleNet(nn.Module):
    """MLP on top of true sufficient statistics.

    Input: [X_obs (d), true_mu_{mis|obs} (d), true_diag(Sigma_{mis|obs}) (d), M (d)] = 4d
    This provides an UPPER BOUND on what any learned architecture can achieve,
    since it uses the exact Bayes-optimal sufficient statistics.
    """
    def __init__(self, n_features, activation='gelu', mlp_layers=(128, 64),
                 dropout=0.1):
        super().__init__()
        d = n_features
        mlp_in = 4 * d
        layers = []
        for h in mlp_layers:
            layers.append(nn.Linear(mlp_in, h))
            layers.append(self._act(activation))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            mlp_in = h
        layers.append(nn.Linear(mlp_in, 1))
        self.mlp_head = nn.Sequential(*layers)
        self._init()

    def _act(self, name):
        return {'relu': nn.ReLU(), 'gelu': nn.GELU(), 'tanh': nn.Tanh(),
                'silu': nn.SiLU()}.get(name, nn.GELU())

    def _init(self):
        for m in self.mlp_head.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, features, m_unused=None):
        """features is already the concatenated [X_obs, cond_mean, cond_var, M]."""
        return self.mlp_head(features).squeeze(-1)


def compute_oracle_features(X_missing, X_complete, mean, cov, mask_bool):
    """Compute true conditional mean and variance for Gaussian X.

    For each sample i:
        obs_i = indices where mask[i] == 0 (observed)
        mis_i = indices where mask[i] == 1 (missing)
        mu_{mis|obs} = mean_mis + Sigma_{mo} Sigma_{oo}^{-1} (x_obs - mean_obs)
        Sigma_{mis|obs} = Sigma_{mm} - Sigma_{mo} Sigma_{oo}^{-1} Sigma_{om}
    """
    n, d = X_missing.shape
    M = mask_bool.astype(np.float32)  # 1=missing, 0=observed
    X_zero = np.nan_to_num(X_missing).astype(np.float32)

    cond_mean = np.zeros((n, d), dtype=np.float32)
    cond_var = np.zeros((n, d), dtype=np.float32)

    # Cache Sigma inverse by mask pattern for efficiency
    pattern_cache = {}

    for i in range(n):
        obs_idx = np.where(~mask_bool[i])[0]
        mis_idx = np.where(mask_bool[i])[0]

        # Fill observed positions in cond_mean with observed values
        cond_mean[i, obs_idx] = X_complete[i, obs_idx]

        if len(mis_idx) == 0:
            # Nothing missing
            cond_mean[i] = X_complete[i]
            continue
        if len(obs_idx) == 0:
            # Everything missing -> conditional mean is marginal mean
            cond_mean[i, mis_idx] = mean[mis_idx]
            cond_var[i, mis_idx] = np.diag(cov)[mis_idx]
            continue

        # Use cached computation by mask pattern
        pattern_key = tuple(mask_bool[i].astype(int))
        if pattern_key in pattern_cache:
            Sigma_mo_Sigma_oo_inv, Sigma_cond_diag, obs_idx_c, mis_idx_c = pattern_cache[pattern_key]
        else:
            Sigma_oo = cov[np.ix_(obs_idx, obs_idx)]
            Sigma_mo = cov[np.ix_(mis_idx, obs_idx)]
            Sigma_mm = cov[np.ix_(mis_idx, mis_idx)]

            try:
                Sigma_oo_inv = np.linalg.solve(
                    Sigma_oo, np.eye(len(obs_idx)))
            except np.linalg.LinAlgError:
                Sigma_oo_inv = np.linalg.pinv(Sigma_oo)

            Sigma_mo_Sigma_oo_inv = Sigma_mo @ Sigma_oo_inv
            Sigma_cond = Sigma_mm - Sigma_mo_Sigma_oo_inv @ Sigma_mo.T
            Sigma_cond_diag = np.diag(Sigma_cond).clip(min=0)
            pattern_cache[pattern_key] = (
                Sigma_mo_Sigma_oo_inv, Sigma_cond_diag, obs_idx, mis_idx)
            obs_idx_c, mis_idx_c = obs_idx, mis_idx

        # Conditional mean: mean_mis + Sigma_mo Sigma_oo^{-1} (x_obs - mean_obs)
        x_obs = X_complete[i, obs_idx_c]
        cond_mean[i, mis_idx_c] = (
            mean[mis_idx_c] + Sigma_mo_Sigma_oo_inv @ (x_obs - mean[obs_idx_c])
        )
        cond_var[i, mis_idx_c] = Sigma_cond_diag

    return cond_mean, cond_var, M


class OracleTest(BaseEstimator, RegressorMixin):
    """Oracle: MLP on true conditional mean + variance (upper bound)."""
    def __init__(self, mlp_layers=(128, 64), activation='gelu', dropout=0.1,
                 n_epochs=200, batch_size=64, lr=0.001, weight_decay=1e-5,
                 verbose=False):
        self.mlp_layers = mlp_layers
        self.activation = activation
        self.dropout = dropout
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.verbose = verbose

    def fit(self, X, y, X_val=None, y_val=None,
            X_complete_train=None, X_complete_val=None,
            mean=None, cov=None, **kwargs):
        n, d = X.shape
        M_train = np.isnan(X)

        # Compute oracle features
        cond_mean_tr, cond_var_tr, M_float_tr = compute_oracle_features(
            X, X_complete_train, mean, cov, M_train)
        X_obs_tr = np.nan_to_num(X).astype(np.float32)
        feat_tr = np.concatenate([X_obs_tr, cond_mean_tr, cond_var_tr, M_float_tr], axis=1)

        F_t = torch.tensor(feat_tr)
        y_t = torch.tensor(y, dtype=torch.float32)

        Fv_t, yv_t = None, None
        if X_val is not None:
            M_val = np.isnan(X_val)
            cond_mean_v, cond_var_v, M_float_v = compute_oracle_features(
                X_val, X_complete_val, mean, cov, M_val)
            X_obs_v = np.nan_to_num(X_val).astype(np.float32)
            feat_v = np.concatenate([X_obs_v, cond_mean_v, cond_var_v, M_float_v], axis=1)
            Fv_t = torch.tensor(feat_v)
            yv_t = torch.tensor(y_val, dtype=torch.float32)

        self.net = OracleNet(d, self.activation, self.mlp_layers, self.dropout)

        # Store for predict
        self._mean = mean
        self._cov = cov

        # Training loop (oracle net takes features directly, m is unused)
        opt = optim.Adam(self.net.parameters(), lr=self.lr,
                         weight_decay=self.weight_decay)
        sched = ReduceLROnPlateau(opt, factor=0.2, patience=7)
        crit = nn.MSELoss()
        es = EarlyStopping(patience=25)

        # Create dummy mask tensors of correct shape for the loop
        M_dummy = torch.zeros(n, 1)
        Mv_dummy = torch.zeros(X_val.shape[0], 1) if X_val is not None else None

        for epoch in range(self.n_epochs):
            self.net.train()
            idx = torch.randperm(n)
            _F, _y = F_t[idx], y_t[idx]
            bf = torch.split(_F, self.batch_size)
            by = torch.split(_y, self.batch_size)

            loss_sum = 0
            for f_, y_ in zip(bf, by):
                opt.zero_grad()
                loss = crit(self.net(f_), y_)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
                opt.step()
                loss_sum += loss.item()

            sched.step(loss_sum / len(bf))

            if Fv_t is not None:
                self.net.eval()
                with torch.no_grad():
                    vl = crit(self.net(Fv_t), yv_t).item()
                es(vl, self.net)
                if es.early_stop:
                    if self.verbose:
                        print(f"  Oracle early stop epoch {epoch}, val={vl:.6f}")
                    break
            if opt.param_groups[0]['lr'] < 1e-7:
                break

        if es.checkpoint:
            self.net.load_state_dict(es.checkpoint)
        return self

    def predict(self, X, X_complete=None):
        M = np.isnan(X)
        if X_complete is None:
            raise ValueError("OracleTest.predict requires X_complete")
        cond_mean, cond_var, M_float = compute_oracle_features(
            X, X_complete, self._mean, self._cov, M)
        X_obs = np.nan_to_num(X).astype(np.float32)
        feat = np.concatenate([X_obs, cond_mean, cond_var, M_float], axis=1)
        self.net.eval()
        with torch.no_grad():
            return self.net(torch.tensor(feat)).numpy()

    def score(self, X, y, X_complete=None):
        pred = self.predict(X, X_complete)
        return 1 - np.mean((y - pred) ** 2) / np.var(y)


# ============================================================================
# 4. Oracle Multiple Imputation (true parameters + sampling)
# ============================================================================
class OracleMultiImputeNet(nn.Module):
    """Oracle version of MultiImpute: uses TRUE conditional stats for sampling."""
    def __init__(self, n_features, K_samples=20,
                 activation='gelu', mlp_layers=(128, 64), dropout=0.1):
        super().__init__()
        self.K = K_samples
        d = n_features
        mlp_in = d
        layers = []
        for h in mlp_layers:
            layers.append(nn.Linear(mlp_in, h))
            layers.append(self._act(activation))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            mlp_in = h
        layers.append(nn.Linear(mlp_in, 1))
        self.mlp_head = nn.Sequential(*layers)
        self._init()

    def _act(self, name):
        return {'relu': nn.ReLU(), 'gelu': nn.GELU(), 'tanh': nn.Tanh(),
                'silu': nn.SiLU()}.get(name, nn.GELU())

    def _init(self):
        for m in self.mlp_head.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x_obs, cond_mean, cond_std, m):
        """
        x_obs: (batch, d) observed values (0 at missing)
        cond_mean: (batch, d) true conditional mean
        cond_std: (batch, d) sqrt of true conditional variance
        m: (batch, d) mask (1=missing)
        """
        K = self.K if self.training else max(self.K, 20)
        preds = []
        for _ in range(K):
            eps = torch.randn_like(x_obs)
            # Sample: at observed positions use x_obs, at missing use cond_mean + noise
            x_k = x_obs * (1 - m) + (cond_mean + eps * cond_std) * m
            pred_k = self.mlp_head(x_k).squeeze(-1)
            preds.append(pred_k)
        return torch.stack(preds, dim=0).mean(dim=0)


class OracleMultiImpute(BaseEstimator, RegressorMixin):
    """Oracle multiple imputation with true conditional parameters."""
    def __init__(self, K_samples=20, mlp_layers=(128, 64), activation='gelu',
                 dropout=0.1, n_epochs=200, batch_size=64, lr=0.001,
                 weight_decay=1e-5, verbose=False):
        self.K_samples = K_samples
        self.mlp_layers = mlp_layers
        self.activation = activation
        self.dropout = dropout
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.verbose = verbose

    def fit(self, X, y, X_val=None, y_val=None,
            X_complete_train=None, X_complete_val=None,
            mean=None, cov=None, **kwargs):
        n, d = X.shape
        self._mean = mean
        self._cov = cov

        # Compute oracle conditional stats
        M_tr = np.isnan(X)
        cm_tr, cv_tr, Mf_tr = compute_oracle_features(
            X, X_complete_train, mean, cov, M_tr)
        cs_tr = np.sqrt(cv_tr)  # conditional std

        X_obs_tr = np.nan_to_num(X).astype(np.float32)

        Xo_t = torch.tensor(X_obs_tr)
        Cm_t = torch.tensor(cm_tr)
        Cs_t = torch.tensor(cs_tr)
        M_t = torch.tensor(Mf_tr)
        y_t = torch.tensor(y, dtype=torch.float32)

        has_val = X_val is not None
        if has_val:
            M_v = np.isnan(X_val)
            cm_v, cv_v, Mf_v = compute_oracle_features(
                X_val, X_complete_val, mean, cov, M_v)
            cs_v = np.sqrt(cv_v)
            Xo_v = torch.tensor(np.nan_to_num(X_val).astype(np.float32))
            Cm_v = torch.tensor(cm_v)
            Cs_v = torch.tensor(cs_v)
            M_vt = torch.tensor(Mf_v)
            yv_t = torch.tensor(y_val, dtype=torch.float32)

        self.net = OracleMultiImputeNet(d, self.K_samples, self.activation,
                                        self.mlp_layers, self.dropout)

        opt = optim.Adam(self.net.parameters(), lr=self.lr,
                         weight_decay=self.weight_decay)
        sched = ReduceLROnPlateau(opt, factor=0.2, patience=7)
        crit = nn.MSELoss()
        es = EarlyStopping(patience=25)

        for epoch in range(self.n_epochs):
            self.net.train()
            idx = torch.randperm(n)
            _Xo, _Cm, _Cs, _M, _y = (Xo_t[idx], Cm_t[idx], Cs_t[idx],
                                       M_t[idx], y_t[idx])

            bxo = torch.split(_Xo, self.batch_size)
            bcm = torch.split(_Cm, self.batch_size)
            bcs = torch.split(_Cs, self.batch_size)
            bm = torch.split(_M, self.batch_size)
            by = torch.split(_y, self.batch_size)

            loss_sum = 0
            for xo_, cm_, cs_, m_, y_ in zip(bxo, bcm, bcs, bm, by):
                opt.zero_grad()
                pred = self.net(xo_, cm_, cs_, m_)
                loss = crit(pred, y_)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
                opt.step()
                loss_sum += loss.item()

            sched.step(loss_sum / len(bxo))

            if has_val:
                self.net.eval()
                with torch.no_grad():
                    vl = crit(self.net(Xo_v, Cm_v, Cs_v, M_vt), yv_t).item()
                es(vl, self.net)
                if es.early_stop:
                    if self.verbose:
                        print(f"  OracleMI early stop epoch {epoch}, val={vl:.6f}")
                    break
            if opt.param_groups[0]['lr'] < 1e-7:
                break

        if es.checkpoint:
            self.net.load_state_dict(es.checkpoint)
        return self

    def predict(self, X, X_complete=None):
        if X_complete is None:
            raise ValueError("OracleMultiImpute.predict requires X_complete")
        M = np.isnan(X)
        cm, cv, Mf = compute_oracle_features(
            X, X_complete, self._mean, self._cov, M)
        cs = np.sqrt(cv)
        X_obs = np.nan_to_num(X).astype(np.float32)
        self.net.eval()
        with torch.no_grad():
            return self.net(
                torch.tensor(X_obs),
                torch.tensor(cm),
                torch.tensor(cs),
                torch.tensor(Mf)
            ).numpy()

    def score(self, X, y, X_complete=None):
        pred = self.predict(X, X_complete)
        return 1 - np.mean((y - pred) ** 2) / np.var(y)


# ============================================================================
# Experiment runner
# ============================================================================
def run_experiment():
    n_train, n_val, n_test = 10000, 2500, 3000
    seeds = [42, 123, 456]

    scenarios = [
        DataScenario('gaussian', 'linear', 'MCAR', 10, 0.5, 10),
        DataScenario('gaussian', 'quadratic', 'MCAR', 10, 0.5, 10),
        DataScenario('gaussian', 'cubic', 'MCAR', 10, 0.5, 10),
        DataScenario('gaussian', 'interaction', 'MCAR', 10, 0.5, 10),
    ]

    # Model configs: (name, constructor, needs_oracle_data)
    models = [
        # --- Baselines ---
        ("NeuMiss_orig",
         lambda: NeuMissPlus(variant='original', depth=5, n_epochs=200,
                             batch_size=64, lr=0.001, early_stopping=True),
         False),
        ("ImputeMLP",
         lambda: ImputeMLP(hidden_layers=(128, 64, 32), n_epochs=200),
         False),
        ("PretrainEncoder",
         lambda: PretrainEncoder(depth=3, mlp_layers=(128, 64),
                                 pretrain_epochs=50, train_epochs=200),
         False),

        # --- New architectures ---
        ("SuffStat_NeuMiss",
         lambda: SufficientStatNeuMiss(depth=3, mlp_layers=(128, 64),
                                       n_epochs=200),
         False),
        ("MultiImpute_K5",
         lambda: MultiImputeNeuMiss(depth=3, K_samples=5,
                                    mlp_layers=(128, 64), n_epochs=200),
         False),
        ("MultiImpute_K10",
         lambda: MultiImputeNeuMiss(depth=3, K_samples=10,
                                    mlp_layers=(128, 64), n_epochs=200),
         False),

        # --- Oracle upper bounds ---
        ("Oracle_SuffStat",
         lambda: OracleTest(mlp_layers=(128, 64), n_epochs=200),
         True),
        ("Oracle_MultiImp",
         lambda: OracleMultiImpute(K_samples=20, mlp_layers=(128, 64),
                                   n_epochs=200),
         True),
    ]

    total = len(scenarios) * len(models) * len(seeds)
    done = 0
    results = []

    print(f"Running {total} experiments ({len(scenarios)} scenarios x "
          f"{len(models)} models x {len(seeds)} seeds)")
    print("=" * 90)

    for scenario in scenarios:
        print(f"\n>>> Scenario: {scenario.name}")

        for seed in seeds:
            data = scenario.generate(n_train, n_val, n_test, random_state=seed)
            X_complete = data['X_complete']
            Xc_train = X_complete[:n_train]
            Xc_val = X_complete[n_train:n_train + n_val]
            Xc_test = X_complete[n_train + n_val:]
            pop_mean = data['mean']
            pop_cov = data['cov']

            for model_name, model_fn, needs_oracle in models:
                done += 1
                model = model_fn()

                try:
                    if needs_oracle:
                        model.fit(
                            data['X_train'], data['y_train'],
                            data['X_val'], data['y_val'],
                            X_complete_train=Xc_train,
                            X_complete_val=Xc_val,
                            mean=pop_mean, cov=pop_cov,
                        )
                        pred = model.predict(data['X_test'],
                                             X_complete=Xc_test)
                        r2 = 1 - (np.mean((data['y_test'] - pred) ** 2)
                                  / np.var(data['y_test']))
                    else:
                        model.fit(
                            data['X_train'], data['y_train'],
                            data['X_val'], data['y_val'],
                        )
                        pred = model.predict(data['X_test'])
                        r2 = 1 - (np.mean((data['y_test'] - pred) ** 2)
                                  / np.var(data['y_test']))
                    mse = float(np.mean((data['y_test'] - pred) ** 2))

                except Exception as e:
                    r2 = float('nan')
                    mse = float('nan')
                    import traceback
                    print(f"  ERROR {model_name}: {e}")
                    traceback.print_exc()

                results.append({
                    'scenario': scenario.name,
                    'model': model_name,
                    'seed': seed,
                    'r2': r2,
                    'mse': mse,
                })
                print(f"  [{done}/{total}] {model_name:22s} seed={seed} "
                      f"R2={r2:.4f}  MSE={mse:.4f}")

    # ---- Summary ----
    print("\n" + "=" * 90)
    print("SUMMARY: Mean R2 +/- std across seeds")
    print("=" * 90)

    import pandas as pd
    df = pd.DataFrame(results)
    summary = df.groupby(['scenario', 'model']).agg(
        r2_mean=('r2', 'mean'),
        r2_std=('r2', 'std'),
        mse_mean=('mse', 'mean'),
        mse_std=('mse', 'std'),
    ).reset_index()

    for scenario_name in df['scenario'].unique():
        print(f"\n--- {scenario_name} ---")
        sub = summary[summary['scenario'] == scenario_name].sort_values(
            'r2_mean', ascending=False)
        for _, row in sub.iterrows():
            oracle_marker = " ***" if "Oracle" in row['model'] else ""
            print(f"  {row['model']:22s}  R2={row['r2_mean']:.4f}+/-{row['r2_std']:.4f}"
                  f"  MSE={row['mse_mean']:.4f}+/-{row['mse_std']:.4f}{oracle_marker}")

    # ---- Analysis: Gap to oracle ----
    print("\n" + "=" * 90)
    print("ANALYSIS: Gap between learned and oracle architectures")
    print("=" * 90)

    for scenario_name in df['scenario'].unique():
        print(f"\n--- {scenario_name} ---")
        sub = summary[summary['scenario'] == scenario_name]

        def get_r2(name):
            row = sub[sub['model'] == name]
            if len(row) == 0:
                return float('nan'), float('nan')
            return row['r2_mean'].values[0], row['r2_std'].values[0]

        oracle_ss, _ = get_r2('Oracle_SuffStat')
        oracle_mi, _ = get_r2('Oracle_MultiImp')
        oracle_best = max(oracle_ss, oracle_mi)

        ss_r2, ss_std = get_r2('SuffStat_NeuMiss')
        mi5_r2, _ = get_r2('MultiImpute_K5')
        mi10_r2, _ = get_r2('MultiImpute_K10')
        imp_r2, _ = get_r2('ImputeMLP')
        nm_r2, _ = get_r2('NeuMiss_orig')
        pt_r2, _ = get_r2('PretrainEncoder')

        learned_best = max(ss_r2, mi5_r2, mi10_r2)
        learned_best_name = {ss_r2: 'SuffStat_NeuMiss',
                             mi5_r2: 'MultiImpute_K5',
                             mi10_r2: 'MultiImpute_K10'}[learned_best]

        print(f"  Oracle upper bound:    R2={oracle_best:.4f}")
        print(f"    Oracle_SuffStat:     R2={oracle_ss:.4f}")
        print(f"    Oracle_MultiImpute:  R2={oracle_mi:.4f}")
        print(f"  Best learned (ours):   R2={learned_best:.4f} ({learned_best_name})")
        print(f"    Gap to oracle:       {oracle_best - learned_best:.4f}")
        print(f"  Baselines:")
        print(f"    ImputeMLP:           R2={imp_r2:.4f} (gap: {oracle_best - imp_r2:.4f})")
        print(f"    PretrainEncoder:     R2={pt_r2:.4f} (gap: {oracle_best - pt_r2:.4f})")
        print(f"    NeuMiss_orig:        R2={nm_r2:.4f} (gap: {oracle_best - nm_r2:.4f})")

        print(f"\n  Verdict: ", end="")
        if learned_best > imp_r2 + 0.005 and learned_best > pt_r2 + 0.005:
            print(f"Sufficient stats approach BEATS all baselines "
                  f"(+{learned_best - max(imp_r2, pt_r2):.4f} over best baseline)")
        elif learned_best > max(imp_r2, pt_r2):
            print(f"Marginal improvement over baselines "
                  f"(+{learned_best - max(imp_r2, pt_r2):.4f})")
        else:
            print(f"Baselines still competitive. Best baseline: "
                  f"R2={max(imp_r2, pt_r2):.4f}")

        if oracle_best - learned_best > 0.02:
            print(f"  NOTE: Large gap ({oracle_best - learned_best:.4f}) to oracle "
                  f"suggests room for improvement in learning the sufficient stats.")
        elif oracle_best - learned_best < 0.005:
            print(f"  NOTE: Near-oracle performance! The architecture is capturing "
                  f"the conditional distribution well.")

    # ---- Theoretical insight: variance matters more for nonlinear f ----
    print("\n" + "=" * 90)
    print("KEY INSIGHT: Does conditional variance matter?")
    print("(Compare Oracle_SuffStat vs Oracle with mean only)")
    print("=" * 90)

    for scenario_name in df['scenario'].unique():
        sub = summary[summary['scenario'] == scenario_name]
        oracle_ss, _ = get_r2_from(sub, 'Oracle_SuffStat')
        oracle_mi, _ = get_r2_from(sub, 'Oracle_MultiImp')
        nm_r2, _ = get_r2_from(sub, 'NeuMiss_orig')
        imp_r2, _ = get_r2_from(sub, 'ImputeMLP')

        response = scenario_name.split('_')[1]
        print(f"\n  {scenario_name}:")
        print(f"    Oracle w/ variance (SuffStat):   R2={oracle_ss:.4f}")
        print(f"    Oracle w/ sampling (MultiImp):   R2={oracle_mi:.4f}")
        print(f"    NeuMiss (mean only approx):      R2={nm_r2:.4f}")
        print(f"    ImputeMLP (mean impute + MLP):   R2={imp_r2:.4f}")
        gain_ss = oracle_ss - imp_r2
        gain_mi = oracle_mi - imp_r2
        if response in ('quadratic', 'cubic', 'interaction'):
            print(f"    --> For {response} f: variance info gives "
                  f"+{max(gain_ss, gain_mi):.4f} R2 over mean-only imputation")

    # Save
    df.to_csv('/Users/yukang/Desktop/NeuroMiss/results/exp_sufficient_stats.csv',
              index=False)
    print(f"\nResults saved to results/exp_sufficient_stats.csv")

    return df


def get_r2_from(sub, name):
    row = sub[sub['model'] == name]
    if len(row) == 0:
        return float('nan'), float('nan')
    return row['r2_mean'].values[0], row['r2_std'].values[0]


if __name__ == '__main__':
    df = run_experiment()

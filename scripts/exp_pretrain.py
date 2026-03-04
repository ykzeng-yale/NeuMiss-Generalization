"""Pre-training experiment for NeuMiss-Encoder.

Hypothesis: NeuMiss-Encoder underperforms ImputeMLP because jointly learning
imputation and prediction creates conflicting gradients. ImputeMLP uses FIXED
mean imputation (no gradient conflict), so its MLP head focuses entirely on
learning the response function.

Solution: Pre-train the Neumann pathways on an imputation task, then train
(or fine-tune) the MLP head for prediction.

Approaches:
1. PretrainEncoder: Phase 1 (imputation) -> Phase 2 (prediction, encoder frozen or fine-tuned)
2. PretrainEncoderV2: Joint multi-task loss L = L_pred + alpha * L_imputation

Baselines:
- NeuMiss-Encoder (standard, no pre-training)
- ImputeMLP
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
from neumiss_plus import ImputeMLP, EarlyStopping


# ============================================================================
# NeuMiss-Encoder with extractable imputation output
# ============================================================================
class NeuMissEncoderPretrain(nn.Module):
    """NeuMiss-Encoder with separate encode() for pretraining the pathways."""

    def __init__(self, n_features, depth=3, activation='gelu',
                 mlp_layers=(128, 64), dropout=0.1):
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

        # --- MLP head: [imputed, var_repr, obs, mask] = 4d ---
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
        for W in list(self.l_W_mean) + list(self.l_W_var) + [self.Wc_mean, self.Wc_var]:
            nn.init.xavier_normal_(W)
        nn.init.normal_(self.mu)
        for m in self.mlp_head.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def encode(self, x, m):
        """Run encoder pathways only. Returns the imputed representation."""
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

        return imputed, var_repr

    def forward(self, x, m):
        obs = 1 - m
        imputed, var_repr = self.encode(x, m)
        features = torch.cat([imputed, var_repr, x * obs, m], dim=1)
        return self.mlp_head(features).squeeze(-1)

    def encoder_parameters(self):
        """Parameters for the encoder (Neumann pathways) only."""
        params = [self.mu]
        params += list(self.l_W_mean) + list(self.l_bias_mean) + [self.Wc_mean]
        params += list(self.l_W_var) + list(self.l_bias_var) + [self.Wc_var]
        params += list(self.mean_acts.parameters())
        params += list(self.var_acts.parameters())
        return params

    def mlp_parameters(self):
        """Parameters for the MLP head only."""
        return list(self.mlp_head.parameters())


# ============================================================================
# PretrainEncoder: Two-phase training
# ============================================================================
class PretrainEncoder(BaseEstimator, RegressorMixin):
    """
    Phase 1: Train encoder pathways on imputation MSE (reconstruct X_true at missing positions).
    Phase 2: Train full network or freeze encoder and train MLP only.

    Parameters
    ----------
    freeze_encoder : bool
        If True, freeze encoder in Phase 2 (only MLP trains).
    pretrain_epochs : int
        Number of epochs for Phase 1.
    finetune_epochs : int
        Number of epochs for Phase 2.
    """
    def __init__(self, depth=3, mlp_layers=(128, 64), activation='gelu',
                 dropout=0.1, freeze_encoder=False,
                 pretrain_epochs=50, finetune_epochs=150,
                 batch_size=64, lr=0.001, weight_decay=1e-5,
                 verbose=False):
        self.depth = depth
        self.mlp_layers = mlp_layers
        self.activation = activation
        self.dropout = dropout
        self.freeze_encoder = freeze_encoder
        self.pretrain_epochs = pretrain_epochs
        self.finetune_epochs = finetune_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.verbose = verbose

    def fit(self, X, y, X_val=None, y_val=None,
            X_complete_train=None, X_complete_val=None):
        M = np.isnan(X).astype(np.float32)
        X_clean = np.nan_to_num(X).astype(np.float32)
        n, d = X.shape

        X_t = torch.tensor(X_clean)
        M_t = torch.tensor(M)
        y_t = torch.tensor(y, dtype=torch.float32)

        has_val = X_val is not None
        if has_val:
            Mv = np.isnan(X_val).astype(np.float32)
            Xv = torch.tensor(np.nan_to_num(X_val).astype(np.float32))
            Mv_t = torch.tensor(Mv)
            yv = torch.tensor(y_val, dtype=torch.float32)

        # Complete X for imputation loss
        Xc_t = torch.tensor(X_complete_train.astype(np.float32))
        if has_val and X_complete_val is not None:
            Xcv_t = torch.tensor(X_complete_val.astype(np.float32))

        self.net = NeuMissEncoderPretrain(d, self.depth, self.activation,
                                          self.mlp_layers, self.dropout)

        # ---- PHASE 1: Pretrain encoder on imputation ----
        enc_params = self.net.encoder_parameters()
        opt1 = optim.Adam(enc_params, lr=self.lr, weight_decay=self.weight_decay)
        sched1 = ReduceLROnPlateau(opt1, factor=0.3, patience=5)
        es1 = EarlyStopping(patience=15)

        if self.verbose:
            print("=== Phase 1: Encoder pretraining (imputation) ===")

        for epoch in range(self.pretrain_epochs):
            self.net.train()
            idx = torch.randperm(n)
            _X, _M, _Xc = X_t[idx], M_t[idx], Xc_t[idx]

            bx = torch.split(_X, self.batch_size)
            bm = torch.split(_M, self.batch_size)
            bxc = torch.split(_Xc, self.batch_size)

            loss_sum = 0
            for x_, m_, xc_ in zip(bx, bm, bxc):
                opt1.zero_grad()
                imputed, _ = self.net.encode(x_, m_)
                # Imputation loss: MSE on missing positions only
                miss_mask = m_  # 1 where missing
                diff = (imputed - xc_) * miss_mask
                n_miss = miss_mask.sum().clamp(min=1)
                loss = (diff ** 2).sum() / n_miss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(enc_params, 1.0)
                opt1.step()
                loss_sum += loss.item()

            avg = loss_sum / len(bx)
            sched1.step(avg)

            if has_val and X_complete_val is not None:
                self.net.eval()
                with torch.no_grad():
                    imp_v, _ = self.net.encode(Xv, Mv_t)
                    diff_v = (imp_v - Xcv_t) * Mv_t
                    vl = (diff_v ** 2).sum() / Mv_t.sum().clamp(min=1)
                    vl = vl.item()
                es1(vl, self.net)
                if es1.early_stop:
                    if self.verbose:
                        print(f"  Phase 1 early stop at epoch {epoch}, val_impute_mse={vl:.6f}")
                    break
            if self.verbose and epoch % 10 == 0:
                msg = f"  Epoch {epoch}: impute_mse={avg:.6f}"
                if has_val and X_complete_val is not None:
                    msg += f", val_impute_mse={vl:.6f}"
                print(msg)
            if opt1.param_groups[0]['lr'] < 1e-7:
                break

        if es1.checkpoint:
            self.net.load_state_dict(es1.checkpoint)

        # ---- PHASE 2: Train prediction ----
        if self.freeze_encoder:
            # Freeze encoder parameters
            for p in self.net.encoder_parameters():
                p.requires_grad = False
            params2 = self.net.mlp_parameters()
        else:
            # Fine-tune everything but with lower LR for encoder
            params2 = [
                {'params': self.net.mlp_parameters(), 'lr': self.lr},
                {'params': self.net.encoder_parameters(), 'lr': self.lr * 0.1},
            ]

        opt2 = optim.Adam(params2, lr=self.lr, weight_decay=self.weight_decay)
        sched2 = ReduceLROnPlateau(opt2, factor=0.2, patience=7)
        crit = nn.MSELoss()
        es2 = EarlyStopping(patience=25)

        if self.verbose:
            freeze_str = "frozen" if self.freeze_encoder else "fine-tuned"
            print(f"=== Phase 2: Prediction training (encoder {freeze_str}) ===")

        for epoch in range(self.finetune_epochs):
            self.net.train()
            idx = torch.randperm(n)
            _X, _M, _y = X_t[idx], M_t[idx], y_t[idx]

            bx = torch.split(_X, self.batch_size)
            bm = torch.split(_M, self.batch_size)
            by = torch.split(_y, self.batch_size)

            loss_sum = 0
            for x_, m_, y_ in zip(bx, bm, by):
                opt2.zero_grad()
                pred = self.net(x_, m_)
                loss = crit(pred, y_)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
                opt2.step()
                loss_sum += loss.item()

            avg = loss_sum / len(bx)
            sched2.step(avg)

            if has_val:
                self.net.eval()
                with torch.no_grad():
                    vl = crit(self.net(Xv, Mv_t), yv).item()
                es2(vl, self.net)
                if es2.early_stop:
                    if self.verbose:
                        print(f"  Phase 2 early stop at epoch {epoch}, val_loss={vl:.6f}")
                    break
            if self.verbose and epoch % 20 == 0:
                msg = f"  Epoch {epoch}: pred_loss={avg:.6f}"
                if has_val:
                    msg += f", val_pred_loss={vl:.6f}"
                print(msg)
            if opt2.param_groups[0]['lr'] < 1e-7:
                break

        if es2.checkpoint:
            self.net.load_state_dict(es2.checkpoint)

        # Unfreeze for predict
        for p in self.net.parameters():
            p.requires_grad = True

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
# PretrainEncoderV2: Joint multi-task training
# ============================================================================
class PretrainEncoderV2(BaseEstimator, RegressorMixin):
    """
    Joint training with multi-task loss:
        L = L_prediction + alpha * L_imputation

    The imputation auxiliary loss acts as a regularizer that guides the
    encoder pathways to produce good imputations while also learning
    to predict.
    """
    def __init__(self, depth=3, mlp_layers=(128, 64), activation='gelu',
                 dropout=0.1, alpha=0.5,
                 n_epochs=200, batch_size=64, lr=0.001, weight_decay=1e-5,
                 verbose=False):
        self.depth = depth
        self.mlp_layers = mlp_layers
        self.activation = activation
        self.dropout = dropout
        self.alpha = alpha
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.verbose = verbose

    def fit(self, X, y, X_val=None, y_val=None,
            X_complete_train=None, X_complete_val=None):
        M = np.isnan(X).astype(np.float32)
        X_clean = np.nan_to_num(X).astype(np.float32)
        n, d = X.shape

        X_t = torch.tensor(X_clean)
        M_t = torch.tensor(M)
        y_t = torch.tensor(y, dtype=torch.float32)
        Xc_t = torch.tensor(X_complete_train.astype(np.float32))

        has_val = X_val is not None
        if has_val:
            Mv = np.isnan(X_val).astype(np.float32)
            Xv = torch.tensor(np.nan_to_num(X_val).astype(np.float32))
            Mv_t = torch.tensor(Mv)
            yv = torch.tensor(y_val, dtype=torch.float32)
            if X_complete_val is not None:
                Xcv_t = torch.tensor(X_complete_val.astype(np.float32))

        self.net = NeuMissEncoderPretrain(d, self.depth, self.activation,
                                          self.mlp_layers, self.dropout)

        opt = optim.Adam(self.net.parameters(), lr=self.lr,
                         weight_decay=self.weight_decay)
        sched = ReduceLROnPlateau(opt, factor=0.2, patience=7)
        crit = nn.MSELoss()
        es = EarlyStopping(patience=25)

        if self.verbose:
            print(f"=== Joint training with alpha={self.alpha} ===")

        for epoch in range(self.n_epochs):
            self.net.train()
            idx = torch.randperm(n)
            _X, _M, _y, _Xc = X_t[idx], M_t[idx], y_t[idx], Xc_t[idx]

            bx = torch.split(_X, self.batch_size)
            bm = torch.split(_M, self.batch_size)
            by = torch.split(_y, self.batch_size)
            bxc = torch.split(_Xc, self.batch_size)

            loss_sum = 0
            imp_sum = 0
            pred_sum = 0
            for x_, m_, y_, xc_ in zip(bx, bm, by, bxc):
                opt.zero_grad()

                # Prediction loss
                pred = self.net(x_, m_)
                l_pred = crit(pred, y_)

                # Imputation loss (on missing positions)
                imputed, _ = self.net.encode(x_, m_)
                diff = (imputed - xc_) * m_
                n_miss = m_.sum().clamp(min=1)
                l_imp = (diff ** 2).sum() / n_miss

                loss = l_pred + self.alpha * l_imp
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
                opt.step()
                loss_sum += loss.item()
                pred_sum += l_pred.item()
                imp_sum += l_imp.item()

            nb = len(bx)
            sched.step(loss_sum / nb)

            if has_val:
                self.net.eval()
                with torch.no_grad():
                    vl = crit(self.net(Xv, Mv_t), yv).item()
                es(vl, self.net)
                if es.early_stop:
                    if self.verbose:
                        print(f"  Early stop at epoch {epoch}, val_pred={vl:.6f}")
                    break

            if self.verbose and epoch % 20 == 0:
                msg = (f"  Epoch {epoch}: total={loss_sum/nb:.6f}, "
                       f"pred={pred_sum/nb:.6f}, imp={imp_sum/nb:.6f}")
                if has_val:
                    msg += f", val_pred={vl:.6f}"
                print(msg)

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
# Standard NeuMiss-Encoder (no pretraining, for fair comparison)
# ============================================================================
class StandardEncoder(BaseEstimator, RegressorMixin):
    """Standard NeuMiss-Encoder baseline (no pretraining)."""
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
        X_clean = np.nan_to_num(X).astype(np.float32)
        n, d = X.shape

        X_t = torch.tensor(X_clean)
        M_t = torch.tensor(M)
        y_t = torch.tensor(y, dtype=torch.float32)

        has_val = X_val is not None
        if has_val:
            Mv = np.isnan(X_val).astype(np.float32)
            Xv = torch.tensor(np.nan_to_num(X_val).astype(np.float32))
            Mv_t = torch.tensor(Mv)
            yv = torch.tensor(y_val, dtype=torch.float32)

        self.net = NeuMissEncoderPretrain(d, self.depth, self.activation,
                                          self.mlp_layers, self.dropout)
        opt = optim.Adam(self.net.parameters(), lr=self.lr,
                         weight_decay=self.weight_decay)
        sched = ReduceLROnPlateau(opt, factor=0.2, patience=7)
        crit = nn.MSELoss()
        es = EarlyStopping(patience=25)

        for epoch in range(self.n_epochs):
            self.net.train()
            idx = torch.randperm(n)
            _X, _M, _y = X_t[idx], M_t[idx], y_t[idx]
            bx = torch.split(_X, self.batch_size)
            bm = torch.split(_M, self.batch_size)
            by = torch.split(_y, self.batch_size)

            loss_sum = 0
            for x_, m_, y_ in zip(bx, bm, by):
                opt.zero_grad()
                loss = crit(self.net(x_, m_), y_)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
                opt.step()
                loss_sum += loss.item()

            sched.step(loss_sum / len(bx))

            if has_val:
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
# Experiment
# ============================================================================
def run_experiment():
    n_train, n_val, n_test = 10000, 2500, 3000
    seeds = [42, 123, 456]

    scenarios = [
        DataScenario('gaussian', 'quadratic', 'MCAR', 10, 0.5, 10),
        DataScenario('gaussian', 'interaction', 'MCAR', 10, 0.5, 10),
        DataScenario('mixture_gaussian', 'quadratic', 'MCAR', 10, 0.5, 10,
                     distribution_params={'n_components': 3}),
    ]

    # Model configurations: (name, constructor)
    models = [
        ("StandardEncoder", lambda: StandardEncoder(
            depth=3, mlp_layers=(128, 64), n_epochs=200, verbose=False)),
        ("ImputeMLP", lambda: ImputeMLP(
            hidden_layers=(128, 64, 32), n_epochs=200, verbose=False)),
        ("Pretrain_finetune", lambda: PretrainEncoder(
            depth=3, mlp_layers=(128, 64), freeze_encoder=False,
            pretrain_epochs=50, finetune_epochs=150, verbose=False)),
        ("Pretrain_frozen", lambda: PretrainEncoder(
            depth=3, mlp_layers=(128, 64), freeze_encoder=True,
            pretrain_epochs=50, finetune_epochs=150, verbose=False)),
        ("MultiTask_a0.1", lambda: PretrainEncoderV2(
            depth=3, mlp_layers=(128, 64), alpha=0.1, n_epochs=200, verbose=False)),
        ("MultiTask_a0.5", lambda: PretrainEncoderV2(
            depth=3, mlp_layers=(128, 64), alpha=0.5, n_epochs=200, verbose=False)),
        ("MultiTask_a1.0", lambda: PretrainEncoderV2(
            depth=3, mlp_layers=(128, 64), alpha=1.0, n_epochs=200, verbose=False)),
    ]

    total = len(scenarios) * len(models) * len(seeds)
    done = 0
    results = []

    print(f"Running {total} experiments ({len(scenarios)} scenarios x "
          f"{len(models)} models x {len(seeds)} seeds)")
    print("=" * 80)

    for scenario in scenarios:
        print(f"\n>>> Scenario: {scenario.name}")

        for seed in seeds:
            data = scenario.generate(n_train, n_val, n_test, random_state=seed)

            # Extract complete X for pretraining (split same way as missing X)
            X_complete = data['X_complete']
            Xc_train = X_complete[:n_train]
            Xc_val = X_complete[n_train:n_train + n_val]

            for model_name, model_fn in models:
                done += 1
                model = model_fn()

                try:
                    # Models that need X_complete
                    if isinstance(model, (PretrainEncoder, PretrainEncoderV2)):
                        model.fit(
                            data['X_train'], data['y_train'],
                            data['X_val'], data['y_val'],
                            X_complete_train=Xc_train,
                            X_complete_val=Xc_val,
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
                    'scenario': scenario.name,
                    'model': model_name,
                    'seed': seed,
                    'r2': r2,
                    'mse': mse,
                })
                print(f"  [{done}/{total}] {model_name:20s} seed={seed} "
                      f"R2={r2:.4f}  MSE={mse:.4f}")

    # ---- Summary ----
    print("\n" + "=" * 80)
    print("SUMMARY: Mean R2 +/- std across seeds")
    print("=" * 80)

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
        sub = summary[summary['scenario'] == scenario_name].sort_values('r2_mean', ascending=False)
        for _, row in sub.iterrows():
            print(f"  {row['model']:22s}  R2={row['r2_mean']:.4f}+/-{row['r2_std']:.4f}"
                  f"  MSE={row['mse_mean']:.4f}+/-{row['mse_std']:.4f}")

    # ---- Key comparisons ----
    print("\n" + "=" * 80)
    print("KEY COMPARISONS (Does pretraining help?)")
    print("=" * 80)

    for scenario_name in df['scenario'].unique():
        print(f"\n--- {scenario_name} ---")
        sub = summary[summary['scenario'] == scenario_name]

        def get_r2(name):
            row = sub[sub['model'] == name]
            if len(row) == 0:
                return float('nan'), float('nan')
            return row['r2_mean'].values[0], row['r2_std'].values[0]

        std_r2, std_std = get_r2('StandardEncoder')
        imp_r2, imp_std = get_r2('ImputeMLP')
        ft_r2, ft_std = get_r2('Pretrain_finetune')
        fr_r2, fr_std = get_r2('Pretrain_frozen')

        print(f"  StandardEncoder (baseline): R2={std_r2:.4f}+/-{std_std:.4f}")
        print(f"  ImputeMLP (target):         R2={imp_r2:.4f}+/-{imp_std:.4f}")
        print(f"  Pretrain+finetune:          R2={ft_r2:.4f}+/-{ft_std:.4f}  "
              f"(vs std: {'+' if ft_r2 > std_r2 else ''}{ft_r2-std_r2:.4f})")
        print(f"  Pretrain+frozen:            R2={fr_r2:.4f}+/-{fr_std:.4f}  "
              f"(vs std: {'+' if fr_r2 > std_r2 else ''}{fr_r2-std_r2:.4f})")

        best_mt_name = None
        best_mt_r2 = -999
        for alpha in [0.1, 0.5, 1.0]:
            name = f"MultiTask_a{alpha}"
            r2, std = get_r2(name)
            if r2 > best_mt_r2:
                best_mt_r2 = r2
                best_mt_name = name
            print(f"  {name:28s} R2={r2:.4f}+/-{std:.4f}  "
                  f"(vs std: {'+' if r2 > std_r2 else ''}{r2-std_r2:.4f})")

        print(f"\n  Verdict: ", end="")
        best_pretrain = max(ft_r2, fr_r2, best_mt_r2)
        if best_pretrain > std_r2 + 0.005:
            if best_pretrain > imp_r2:
                print("PRETRAINING HELPS and BEATS ImputeMLP!")
            else:
                print(f"Pretraining improves over standard "
                      f"(+{best_pretrain-std_r2:.4f}) but still behind ImputeMLP "
                      f"(gap: {imp_r2-best_pretrain:.4f})")
        else:
            print("Pretraining does NOT meaningfully help. "
                  "Gradient conflict may not be the main issue.")

    return df


if __name__ == '__main__':
    df = run_experiment()

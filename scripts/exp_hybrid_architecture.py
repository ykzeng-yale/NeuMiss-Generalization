"""Focused experiment: Hybrid NeuMiss+ combining best elements.
Try combining Variant C (wider layers) with Variant D (polynomial interactions)."""
import sys; sys.path.insert(0, '/Users/yukang/Desktop/NeuroMiss/src')
import numpy as np, pandas as pd, torch, torch.nn as nn
from data_generation import DataScenario
from neumiss_plus import NeuMissPlus, NeuMissPlusC, NeuMissPlusD, EarlyStopping
from experiment_runner import run_single_experiment
from sklearn.base import BaseEstimator, RegressorMixin
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from copy import deepcopy

# New hybrid architecture: combines wider layers + polynomial interactions
class NeuMissPlusHybrid(nn.Module):
    """Hybrid: Variant C (expand-activate-compress) + Variant D (polynomial interaction)."""
    def __init__(self, n_features, depth, activation='gelu', expansion_factor=3):
        super().__init__()
        self.depth = depth
        self.n_features = n_features
        hidden = n_features * expansion_factor

        act_map = {'relu': nn.ReLU(), 'gelu': nn.GELU(), 'tanh': nn.Tanh(), 'silu': nn.SiLU()}
        self.activation = act_map.get(activation, nn.GELU())

        # Wider layers (from Variant C)
        self.l_W_expand = nn.ParameterList([nn.Parameter(torch.empty(n_features, hidden)) for _ in range(depth)])
        self.l_W_compress = nn.ParameterList([nn.Parameter(torch.empty(hidden, n_features)) for _ in range(depth)])
        self.l_bias1 = nn.ParameterList([nn.Parameter(torch.zeros(hidden)) for _ in range(depth)])
        self.l_bias2 = nn.ParameterList([nn.Parameter(torch.zeros(n_features)) for _ in range(depth)])

        # Polynomial interaction (from Variant D)
        self.W_int1 = nn.Parameter(torch.empty(n_features, n_features))
        self.W_int2 = nn.Parameter(torch.empty(n_features, n_features))
        self.W_combine = nn.Parameter(torch.empty(2 * n_features, n_features))

        self.Wc = nn.Parameter(torch.empty(n_features, n_features))
        self.mu = nn.Parameter(torch.empty(n_features))
        self.beta = nn.Parameter(torch.empty(n_features))
        self.b = nn.Parameter(torch.empty(1))

        self._init()

    def _init(self):
        for p in [self.l_W_expand, self.l_W_compress]:
            for W in p:
                nn.init.xavier_normal_(W)
        nn.init.xavier_normal_(self.W_int1)
        nn.init.xavier_normal_(self.W_int2)
        nn.init.xavier_normal_(self.W_combine)
        nn.init.xavier_normal_(self.Wc)
        nn.init.normal_(self.mu)
        nn.init.normal_(self.beta)
        nn.init.zeros_(self.b)

    def forward(self, x, m):
        obs = 1 - m
        h0 = x + m * self.mu
        h = x - obs * self.mu

        # Wider Neumann-like layers with activations
        for i in range(self.depth):
            h = torch.matmul(h, self.l_W_expand[i]) + self.l_bias1[i]
            h = self.activation(h)
            h = torch.matmul(h, self.l_W_compress[i]) + self.l_bias2[i]
            h = h * obs

        # Polynomial interaction
        h1 = torch.matmul(h * obs, self.W_int1) * obs
        h2 = torch.matmul(h * obs, self.W_int2) * obs
        h_poly = h1 * h2

        h_combined = torch.cat([h * obs, h_poly], dim=1)
        h = torch.matmul(h_combined, self.W_combine) * obs

        h = torch.matmul(h, self.Wc) * m + h0
        return torch.matmul(h, self.beta) + self.b


class NeuMissPlusHybridEstimator(BaseEstimator, RegressorMixin):
    def __init__(self, depth=3, activation='gelu', expansion_factor=3, n_epochs=150, batch_size=64, lr=0.001):
        self.depth = depth
        self.activation = activation
        self.expansion_factor = expansion_factor
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.lr = lr

    def fit(self, X, y, X_val=None, y_val=None):
        M = np.isnan(X).astype(np.float32)
        X_c = np.nan_to_num(X).astype(np.float32)
        n, d = X.shape
        X_t = torch.tensor(X_c); M_t = torch.tensor(M); y_t = torch.tensor(y, dtype=torch.float32)

        if X_val is not None:
            Mv = np.isnan(X_val).astype(np.float32)
            Xv = torch.tensor(np.nan_to_num(X_val).astype(np.float32))
            Mv_t = torch.tensor(Mv); yv = torch.tensor(y_val, dtype=torch.float32)

        self.net = NeuMissPlusHybrid(d, self.depth, self.activation, self.expansion_factor)
        opt = optim.Adam(self.net.parameters(), lr=self.lr)
        sched = ReduceLROnPlateau(opt, factor=0.2, patience=5)
        crit = nn.MSELoss()
        es = EarlyStopping(patience=20)

        for epoch in range(self.n_epochs):
            self.net.train()
            idx = torch.randperm(n); X_t, M_t, y_t = X_t[idx], M_t[idx], y_t[idx]
            bx = torch.split(X_t, self.batch_size); bm = torch.split(M_t, self.batch_size); by = torch.split(y_t, self.batch_size)
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
                if es.early_stop: break
            if opt.param_groups[0]['lr'] < 1e-7: break

        if es.checkpoint: self.net.load_state_dict(es.checkpoint)
        return self

    def predict(self, X):
        M = torch.tensor(np.isnan(X).astype(np.float32))
        X_t = torch.tensor(np.nan_to_num(X).astype(np.float32))
        self.net.eval()
        with torch.no_grad(): return self.net(X_t, M).numpy()

# Run experiments
scenarios = [
    DataScenario('gaussian', 'quadratic', 'MCAR', 10, 0.5, 10),
    DataScenario('gaussian', 'cubic', 'MCAR', 10, 0.5, 10),
    DataScenario('gaussian', 'interaction', 'MCAR', 10, 0.5, 10),
    DataScenario('mixture_gaussian', 'quadratic', 'MCAR', 10, 0.5, 10, distribution_params={'n_components': 3}),
    DataScenario('mixture_gaussian', 'cubic', 'MCAR', 10, 0.5, 10, distribution_params={'n_components': 3}),
    DataScenario('student_t', 'sinusoidal', 'MCAR', 10, 0.5, 10, distribution_params={'df': 5}),
]

results = []
for s in scenarios:
    for seed in range(3):
        data = s.generate(8000, 2000, 3000, random_state=seed)

        # Baseline NeuMiss
        for depth in [3, 5]:
            est = NeuMissPlus(variant='original', depth=depth, n_epochs=100, batch_size=64, lr=0.001, early_stopping=True)
            est.fit(data['X_train'], data['y_train'], data['X_val'], data['y_val'])
            pred = est.predict(data['X_test'])
            r2 = 1 - np.mean((data['y_test'] - pred)**2) / np.var(data['y_test'])
            results.append({'scenario': s.name, 'method': f'NeuMiss_d{depth}', 'r2_test': r2, 'seed': seed})
            print(f"{s.name} | NeuMiss_d{depth} R2={r2:.4f}")

        # NeuMiss+ A
        est_a = NeuMissPlus(variant='A', depth=5, activation='gelu', n_epochs=100, batch_size=64, lr=0.001, early_stopping=True)
        est_a.fit(data['X_train'], data['y_train'], data['X_val'], data['y_val'])
        pred_a = est_a.predict(data['X_test'])
        r2_a = 1 - np.mean((data['y_test'] - pred_a)**2) / np.var(data['y_test'])
        results.append({'scenario': s.name, 'method': 'NM+A_gelu_d5', 'r2_test': r2_a, 'seed': seed})
        print(f"{s.name} | NM+A_gelu_d5 R2={r2_a:.4f}")

        # Hybrid
        for ef in [2, 3, 4]:
            for depth in [3, 5]:
                est_h = NeuMissPlusHybridEstimator(depth=depth, activation='gelu', expansion_factor=ef, n_epochs=100)
                est_h.fit(data['X_train'], data['y_train'], data['X_val'], data['y_val'])
                pred_h = est_h.predict(data['X_test'])
                r2_h = 1 - np.mean((data['y_test'] - pred_h)**2) / np.var(data['y_test'])
                results.append({'scenario': s.name, 'method': f'Hybrid_d{depth}_ef{ef}', 'r2_test': r2_h, 'seed': seed})
                print(f"{s.name} | Hybrid_d{depth}_ef{ef} R2={r2_h:.4f}")

df = pd.DataFrame(results)
df.to_csv('/Users/yukang/Desktop/NeuroMiss/results/exp_hybrid.csv', index=False)
agg = df.groupby(['scenario','method'])['r2_test'].mean().reset_index()
for sc in agg['scenario'].unique():
    print(f"\n{sc}:")
    sub = agg[agg['scenario']==sc].sort_values('r2_test', ascending=False)
    for _,row in sub.iterrows():
        print(f"  {row['method']:30s} R2={row['r2_test']:.4f}")

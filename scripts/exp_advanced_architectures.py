"""Advanced architecture experiments: Systematic ablation of NeuMiss+MLP components.

Tests:
1. Effect of MLP head depth/width
2. Effect of Neumann activation vs no activation in imputation layers
3. Effect of residual connections
4. Effect of layer normalization
5. Effect of dropout
6. Comparison across different response types and distributions
"""
import sys; sys.path.insert(0, '/Users/yukang/Desktop/NeuroMiss/src')
import numpy as np, pandas as pd
from data_generation import DataScenario
from neumiss_plus import NeuMissMLPEstimator

# Key scenarios covering the design space
scenarios = [
    DataScenario('gaussian', 'linear', 'MCAR', 10, 0.5, 10),
    DataScenario('gaussian', 'quadratic', 'MCAR', 10, 0.5, 10),
    DataScenario('gaussian', 'cubic', 'MCAR', 10, 0.5, 10),
    DataScenario('gaussian', 'interaction', 'MCAR', 10, 0.5, 10),
    DataScenario('mixture_gaussian', 'quadratic', 'MCAR', 10, 0.5, 10, distribution_params={'n_components': 3}),
    DataScenario('student_t', 'sinusoidal', 'MCAR', 10, 0.5, 10, distribution_params={'df': 5}),
]

configs = []

# --- Ablation 1: MLP head size ---
for mlp in [(32,), (64,), (128,), (64, 32), (128, 64), (128, 64, 32), (256, 128, 64)]:
    tag = 'x'.join(map(str, mlp))
    configs.append((f'NM+MLP_d3_h{tag}', dict(
        variant='mlp', depth=3, mlp_layers=mlp, activation='gelu',
        neumann_activation=True, residual_connection=True, n_epochs=120)))

# --- Ablation 2: Neumann depth with fixed MLP ---
for d in [1, 2, 3, 5, 7]:
    configs.append((f'NM+MLP_d{d}_h64x32', dict(
        variant='mlp', depth=d, mlp_layers=(64, 32), activation='gelu',
        neumann_activation=True, residual_connection=True, n_epochs=120)))

# --- Ablation 3: With/without Neumann activation ---
configs.append(('NM+MLP_d3_act', dict(
    variant='mlp', depth=3, mlp_layers=(64, 32), activation='gelu',
    neumann_activation=True, residual_connection=True, n_epochs=120)))
configs.append(('NM+MLP_d3_noact', dict(
    variant='mlp', depth=3, mlp_layers=(64, 32), activation='gelu',
    neumann_activation=False, residual_connection=True, n_epochs=120)))

# --- Ablation 4: With/without residual ---
configs.append(('NM+MLP_d5_res', dict(
    variant='mlp', depth=5, mlp_layers=(64, 32), activation='gelu',
    neumann_activation=True, residual_connection=True, n_epochs=120)))
configs.append(('NM+MLP_d5_nores', dict(
    variant='mlp', depth=5, mlp_layers=(64, 32), activation='gelu',
    neumann_activation=True, residual_connection=False, n_epochs=120)))

# --- Ablation 5: Different activations ---
for act in ['relu', 'gelu', 'silu', 'tanh']:
    configs.append((f'NM+MLP_d3_{act}', dict(
        variant='mlp', depth=3, mlp_layers=(64, 32), activation=act,
        neumann_activation=True, residual_connection=True, n_epochs=120)))

# --- Ablation 6: Dropout ---
for dp in [0.0, 0.05, 0.1, 0.2, 0.3]:
    configs.append((f'NM+MLP_d3_dp{dp}', dict(
        variant='mlp', depth=3, mlp_layers=(64, 32), activation='gelu',
        neumann_activation=True, residual_connection=True, dropout=dp, n_epochs=120)))

# --- Ablation 7: Wide NeuMiss + MLP ---
for ef in [2, 3, 4]:
    for d in [3, 5]:
        configs.append((f'NMwide+MLP_d{d}_ef{ef}', dict(
            variant='wide_mlp', depth=d, expansion_factor=ef,
            mlp_layers=(64, 32), activation='gelu', n_epochs=120)))

results = []
total = len(scenarios) * len(configs) * 2
done = 0

for s in scenarios:
    for seed in range(2):
        data = s.generate(8000, 2000, 3000, random_state=seed)
        for name, cfg in configs:
            done += 1
            try:
                est = NeuMissMLPEstimator(**cfg)
                est.fit(data['X_train'], data['y_train'], data['X_val'], data['y_val'])
                pred = est.predict(data['X_test'])
                r2 = 1 - np.mean((data['y_test'] - pred)**2) / np.var(data['y_test'])
                results.append({
                    'scenario': s.name, 'method': name, 'r2_test': r2, 'seed': seed,
                    'response': s.response, 'distribution': s.distribution,
                    'status': 'success'
                })
                print(f"[{done}/{total}] {s.name} | {name:30s} R2={r2:.4f}")
            except Exception as e:
                results.append({'scenario': s.name, 'method': name, 'r2_test': float('nan'),
                               'seed': seed, 'status': 'error', 'error': str(e)})
                print(f"[{done}/{total}] ERROR {s.name} | {name}: {e}")

df = pd.DataFrame(results)
df.to_csv('/Users/yukang/Desktop/NeuroMiss/results/exp_advanced_arch.csv', index=False)

# Analysis
df_ok = df[df['status']=='success']
print("\n" + "="*80)
print("ABLATION RESULTS")
print("="*80)
agg = df_ok.groupby(['scenario','method'])['r2_test'].mean().reset_index()
for sc in sorted(agg['scenario'].unique()):
    print(f"\n{sc}:")
    sub = agg[agg['scenario']==sc].sort_values('r2_test', ascending=False)
    for _,row in sub.head(15).iterrows():
        print(f"  {row['method']:35s} R2={row['r2_test']:.4f}")

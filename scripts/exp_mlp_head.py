"""Critical experiment: NeuMiss+MLP (two-stage) vs single-stage architectures.

Key hypothesis: Adding an MLP prediction head on top of NeuMiss imputation layers
should dramatically improve performance on nonlinear response functions.
"""
import sys; sys.path.insert(0, '/Users/yukang/Desktop/NeuroMiss/src')
import numpy as np, pandas as pd
from data_generation import DataScenario
from neumiss_plus import NeuMissPlus, NeuMissMLPEstimator, ImputeMLP

scenarios = [
    DataScenario('gaussian', 'linear', 'MCAR', 10, 0.5, 10),
    DataScenario('gaussian', 'quadratic', 'MCAR', 10, 0.5, 10),
    DataScenario('gaussian', 'cubic', 'MCAR', 10, 0.5, 10),
    DataScenario('gaussian', 'interaction', 'MCAR', 10, 0.5, 10),
    DataScenario('gaussian', 'sinusoidal', 'MCAR', 10, 0.5, 10),
    DataScenario('mixture_gaussian', 'quadratic', 'MCAR', 10, 0.5, 10, distribution_params={'n_components': 3}),
    DataScenario('mixture_gaussian', 'cubic', 'MCAR', 10, 0.5, 10, distribution_params={'n_components': 3}),
    DataScenario('student_t', 'quadratic', 'MCAR', 10, 0.5, 10, distribution_params={'df': 5}),
    DataScenario('gaussian', 'quadratic', 'MAR', 10, 0.5, 10),
    DataScenario('gaussian', 'quadratic', 'MNAR_censoring', 10, 0.5, 10),
]

methods = []

# --- Baselines ---
methods.append(('NeuMiss_d5', lambda: NeuMissPlus(variant='original', depth=5, n_epochs=100, batch_size=64, lr=0.001, early_stopping=True)))
methods.append(('NM+A_gelu_d5', lambda: NeuMissPlus(variant='A', depth=5, activation='gelu', n_epochs=100, batch_size=64, lr=0.001, early_stopping=True)))
methods.append(('NM+C_gelu_d5_ef3', lambda: NeuMissPlus(variant='C', depth=5, activation='gelu', expansion_factor=3, n_epochs=100, batch_size=64, lr=0.001, early_stopping=True)))

# --- NEW: NeuMiss+MLP variants ---
# Basic NeuMiss+MLP (small head)
methods.append(('NM+MLP_d3_h64', lambda: NeuMissMLPEstimator(
    variant='mlp', depth=3, mlp_layers=(64,), activation='gelu',
    neumann_activation=True, residual_connection=True, n_epochs=150)))
# NeuMiss+MLP (medium head)
methods.append(('NM+MLP_d3_h64x32', lambda: NeuMissMLPEstimator(
    variant='mlp', depth=3, mlp_layers=(64, 32), activation='gelu',
    neumann_activation=True, residual_connection=True, n_epochs=150)))
# NeuMiss+MLP (larger head)
methods.append(('NM+MLP_d5_h128x64', lambda: NeuMissMLPEstimator(
    variant='mlp', depth=5, mlp_layers=(128, 64), activation='gelu',
    neumann_activation=True, residual_connection=True, n_epochs=150)))
# NeuMiss+MLP (deep head)
methods.append(('NM+MLP_d3_h128x64x32', lambda: NeuMissMLPEstimator(
    variant='mlp', depth=3, mlp_layers=(128, 64, 32), activation='gelu',
    neumann_activation=True, residual_connection=True, n_epochs=150)))
# Without neumann activation (pure NeuMiss + MLP)
methods.append(('NM+MLP_d5_noact', lambda: NeuMissMLPEstimator(
    variant='mlp', depth=5, mlp_layers=(128, 64), activation='gelu',
    neumann_activation=False, residual_connection=False, n_epochs=150)))
# Wide NeuMiss+MLP
methods.append(('NMwide+MLP_d3_ef3', lambda: NeuMissMLPEstimator(
    variant='wide_mlp', depth=3, expansion_factor=3,
    mlp_layers=(64, 32), activation='gelu', n_epochs=150)))
methods.append(('NMwide+MLP_d5_ef3', lambda: NeuMissMLPEstimator(
    variant='wide_mlp', depth=5, expansion_factor=3,
    mlp_layers=(128, 64), activation='gelu', n_epochs=150)))

# --- Baseline: MLP-only (no NeuMiss structure) ---
methods.append(('ImputeMLP_128x64x32', lambda: ImputeMLP(
    hidden_layers=(128, 64, 32), activation='gelu', n_epochs=150)))

results = []
total = len(scenarios) * len(methods) * 3
done = 0

for s in scenarios:
    for seed in range(3):
        data = s.generate(8000, 2000, 3000, random_state=seed)
        for name, make_est in methods:
            done += 1
            try:
                est = make_est()
                est.fit(data['X_train'], data['y_train'], data['X_val'], data['y_val'])
                pred = est.predict(data['X_test'])
                r2 = 1 - np.mean((data['y_test'] - pred)**2) / np.var(data['y_test'])
                results.append({
                    'scenario': s.name, 'method': name, 'r2_test': r2, 'seed': seed,
                    'distribution': s.distribution, 'response': s.response,
                    'missing_mechanism': s.missing_mechanism, 'status': 'success'
                })
                print(f"[{done}/{total}] {s.name} | {name:30s} R2={r2:.4f}")
            except Exception as e:
                results.append({
                    'scenario': s.name, 'method': name, 'r2_test': float('nan'),
                    'seed': seed, 'status': 'error', 'error': str(e)
                })
                print(f"[{done}/{total}] {s.name} | {name:30s} ERROR: {e}")

df = pd.DataFrame(results)
df.to_csv('/Users/yukang/Desktop/NeuroMiss/results/exp_mlp_head.csv', index=False)

# Analysis
df_ok = df[df['status']=='success']
print("\n" + "="*80)
print("RESULTS: NeuMiss+MLP vs Single-Stage Architectures")
print("="*80)

agg = df_ok.groupby(['scenario','method'])['r2_test'].agg(['mean','std']).reset_index()
for sc in sorted(agg['scenario'].unique()):
    print(f"\n{'='*60}")
    print(f"  {sc}")
    print(f"{'='*60}")
    sub = agg[agg['scenario']==sc].sort_values('mean', ascending=False)
    for _,row in sub.iterrows():
        print(f"  {row['method']:35s} R2={row['mean']:.4f} +/- {row['std']:.4f}")

# Summary: best method per response type
print("\n\n" + "="*80)
print("SUMMARY: Best method per response type")
print("="*80)
best_by_resp = df_ok.groupby(['response','method'])['r2_test'].mean().reset_index()
for resp in sorted(best_by_resp['response'].unique()):
    print(f"\n{resp}:")
    sub = best_by_resp[best_by_resp['response']==resp].sort_values('r2_test', ascending=False)
    for _, row in sub.head(5).iterrows():
        print(f"  {row['method']:35s} R2={row['r2_test']:.4f}")

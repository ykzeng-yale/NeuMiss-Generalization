"""FINAL DEFINITIVE COMPARISON: All architectures across all scenarios.

Tests the best configurations from all previous experiments:
- Original NeuMiss (baseline)
- NM+C_gelu_d3 with residual (best single-stage)
- NM-Encoder_d3 with large MLP (best two-stage NeuMiss)
- ImputeMLP (non-NeuMiss baseline)

With proper settings: 10K training, 150 epochs, 5 seeds.
"""
import sys; sys.path.insert(0, '/Users/yukang/Desktop/NeuroMiss/src')
import numpy as np, pandas as pd
from data_generation import DataScenario
from neumiss_plus import NeuMissPlus, NeuMissEncoderEstimator, ImputeMLP

# All key scenarios
scenarios = [
    DataScenario('gaussian', 'linear', 'MCAR', 10, 0.5, 10),
    DataScenario('gaussian', 'quadratic', 'MCAR', 10, 0.5, 10),
    DataScenario('gaussian', 'cubic', 'MCAR', 10, 0.5, 10),
    DataScenario('gaussian', 'interaction', 'MCAR', 10, 0.5, 10),
    DataScenario('gaussian', 'sinusoidal', 'MCAR', 10, 0.5, 10),
    DataScenario('mixture_gaussian', 'linear', 'MCAR', 10, 0.5, 10, distribution_params={'n_components': 3}),
    DataScenario('mixture_gaussian', 'quadratic', 'MCAR', 10, 0.5, 10, distribution_params={'n_components': 3}),
    DataScenario('mixture_gaussian', 'cubic', 'MCAR', 10, 0.5, 10, distribution_params={'n_components': 3}),
    DataScenario('student_t', 'quadratic', 'MCAR', 10, 0.5, 10, distribution_params={'df': 5}),
    DataScenario('gaussian', 'quadratic', 'MAR', 10, 0.5, 10),
    DataScenario('gaussian', 'quadratic', 'MNAR_censoring', 10, 0.5, 10),
    DataScenario('gaussian', 'quadratic', 'MNAR_selfmasking', 10, 0.5, 10),
]

methods = [
    ('NeuMiss_d5', lambda: NeuMissPlus(
        variant='original', depth=5, n_epochs=150, batch_size=64, lr=0.001, early_stopping=True)),
    ('NM+C_gelu_d3', lambda: NeuMissPlus(
        variant='C', depth=3, activation='gelu', expansion_factor=3,
        n_epochs=150, batch_size=64, lr=0.001, early_stopping=True,
        residual_connection=True)),
    ('NM-Encoder_d3', lambda: NeuMissEncoderEstimator(
        variant='encoder', depth=3, mlp_layers=(256, 128), activation='gelu',
        dropout=0.1, n_epochs=200, batch_size=64, lr=0.001)),
    ('ImputeMLP', lambda: ImputeMLP(
        hidden_layers=(256, 128, 64), activation='gelu',
        n_epochs=200, batch_size=64, lr=0.001)),
]

results = []
total = len(scenarios) * len(methods) * 5
done = 0

for s in scenarios:
    for seed in range(5):
        data = s.generate(10000, 2500, 3000, random_state=seed)
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
                print(f"[{done}/{total}] {s.name} | {name:20s} R2={r2:.4f}")
            except Exception as e:
                results.append({
                    'scenario': s.name, 'method': name, 'r2_test': float('nan'),
                    'seed': seed, 'status': 'error', 'error': str(e)
                })
                print(f"[{done}/{total}] {s.name} | {name:20s} ERROR: {e}")

df = pd.DataFrame(results)
df.to_csv('/Users/yukang/Desktop/NeuroMiss/results/exp_final_comparison.csv', index=False)

# Comprehensive analysis
df_ok = df[df['status']=='success']
agg = df_ok.groupby(['scenario','method'])['r2_test'].agg(['mean','std']).reset_index()

print("\n" + "="*85)
print("FINAL COMPREHENSIVE COMPARISON")
print("="*85)

for sc in sorted(agg['scenario'].unique()):
    sub = agg[agg['scenario']==sc].sort_values('mean', ascending=False)
    print(f"\n{sc}:")
    for _, row in sub.iterrows():
        stars = " ***" if row['mean'] == sub['mean'].max() else ""
        print(f"  {row['method']:25s} R2={row['mean']:.4f} +/- {row['std']:.4f}{stars}")

# Summary table
print("\n\n" + "="*85)
print("SUMMARY: BEST R2 PER SCENARIO")
print("="*85)
print(f"{'Scenario':<45s} {'NeuMiss':>10s} {'NM+C':>10s} {'Encoder':>10s} {'ImputeMLP':>10s} {'Winner':>15s}")
print("-"*100)
for sc in sorted(agg['scenario'].unique()):
    sub = agg[agg['scenario']==sc]
    vals = {}
    for _, row in sub.iterrows():
        vals[row['method']] = row['mean']
    winner = sub.loc[sub['mean'].idxmax(), 'method']
    nm = vals.get('NeuMiss_d5', float('nan'))
    nc = vals.get('NM+C_gelu_d3', float('nan'))
    enc = vals.get('NM-Encoder_d3', float('nan'))
    imp = vals.get('ImputeMLP', float('nan'))
    print(f"{sc:<45s} {nm:>10.4f} {nc:>10.4f} {enc:>10.4f} {imp:>10.4f} {winner:>15s}")

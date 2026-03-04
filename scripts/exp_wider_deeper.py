"""Focused experiment: Effect of width (expansion factor) and depth on NeuMiss+ Variant C."""
import sys; sys.path.insert(0, '/Users/yukang/Desktop/NeuroMiss/src')
import numpy as np, pandas as pd
from data_generation import DataScenario
from experiment_runner import run_single_experiment

scenarios = [
    DataScenario('gaussian', 'quadratic', 'MCAR', 10, 0.5, 10),
    DataScenario('gaussian', 'cubic', 'MCAR', 10, 0.5, 10),
    DataScenario('mixture_gaussian', 'cubic', 'MCAR', 10, 0.5, 10, distribution_params={'n_components': 3}),
    DataScenario('student_t', 'sinusoidal', 'MCAR', 10, 0.5, 10, distribution_params={'df': 5}),
]

methods = []
# Baseline
methods.append({'name': 'NeuMiss_d5', 'variant': 'original', 'depth': 5, 'activation': 'relu', 'residual_connection': False})
# Variant C with different widths and depths
for ef in [2, 3, 4, 6, 8]:
    for depth in [2, 3, 5, 7]:
        for act in ['gelu', 'relu']:
            methods.append({'name': f'NM+C_{act}_d{depth}_ef{ef}', 'variant': 'C', 'depth': depth,
                           'activation': act, 'expansion_factor': ef, 'residual_connection': False})

results = []
total = len(scenarios) * len(methods) * 2
done = 0
for s in scenarios:
    for m in methods:
        for seed in range(2):
            done += 1
            m_full = {**m, 'degree': 2}
            m_full.setdefault('expansion_factor', 2)
            r = run_single_experiment(s, m_full, n_train=8000, n_val=2000, n_test=3000, n_epochs=100, random_state=seed)
            results.append(r)
            if r['status'] == 'success':
                print(f"[{done}/{total}] {s.name} | {m['name']} R2={r['r2_test']:.4f}")

df = pd.DataFrame(results)
df.to_csv('/Users/yukang/Desktop/NeuroMiss/results/exp_wider_deeper.csv', index=False)

df_ok = df[df['status']=='success']
agg = df_ok.groupby(['scenario','method'])['r2_test'].mean().reset_index()
for sc in agg['scenario'].unique():
    print(f"\n{sc}:")
    sub = agg[agg['scenario']==sc].sort_values('r2_test', ascending=False)
    for _,row in sub.head(10).iterrows():
        print(f"  {row['method']:40s} R2={row['r2_test']:.4f}")

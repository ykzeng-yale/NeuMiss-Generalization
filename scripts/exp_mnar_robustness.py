"""Focused experiment: MNAR robustness - testing the meeting notes claim that MNAR doesn't affect prediction."""
import sys; sys.path.insert(0, '/Users/yukang/Desktop/NeuroMiss/src')
import numpy as np, pandas as pd
from data_generation import DataScenario
from experiment_runner import run_single_experiment

# Same response functions under different missing mechanisms
base_configs = [
    ('gaussian', 'quadratic', {}),
    ('gaussian', 'cubic', {}),
    ('mixture_gaussian', 'linear', {'n_components': 3}),
    ('mixture_gaussian', 'quadratic', {'n_components': 3}),
]

scenarios = []
for dist, resp, dp in base_configs:
    for mech in ['MCAR', 'MAR', 'MNAR_censoring', 'MNAR_selfmasking']:
        scenarios.append(DataScenario(dist, resp, mech, 10, 0.5, 10, distribution_params=dp))

methods = [
    {'name': 'NeuMiss_d3', 'variant': 'original', 'depth': 3, 'activation': 'relu', 'residual_connection': False},
    {'name': 'NeuMiss_d5', 'variant': 'original', 'depth': 5, 'activation': 'relu', 'residual_connection': False},
    {'name': 'NM+A_gelu_d3', 'variant': 'A', 'depth': 3, 'activation': 'gelu', 'residual_connection': False},
    {'name': 'NM+A_gelu_d5', 'variant': 'A', 'depth': 5, 'activation': 'gelu', 'residual_connection': False},
    {'name': 'NM+D_gelu_d3', 'variant': 'D', 'depth': 3, 'activation': 'gelu', 'degree': 2, 'residual_connection': False},
]

results = []
total = len(scenarios) * len(methods) * 2
done = 0
for s in scenarios:
    for m in methods:
        for seed in range(2):
            done += 1
            m_full = {**m}
            m_full.setdefault('expansion_factor', 2)
            m_full.setdefault('degree', 2)
            r = run_single_experiment(s, m_full, n_train=8000, n_val=2000, n_test=3000, n_epochs=100, random_state=seed)
            results.append(r)
            if r['status'] == 'success':
                print(f"[{done}/{total}] {s.name} | {m['name']} R2={r['r2_test']:.4f}")

df = pd.DataFrame(results)
df.to_csv('/Users/yukang/Desktop/NeuroMiss/results/exp_mnar.csv', index=False)

# Compare across missing mechanisms
df_ok = df[df['status']=='success']
print("\n=== MNAR Robustness Analysis ===")
for method in df_ok['method'].unique():
    print(f"\n{method}:")
    sub = df_ok[df_ok['method']==method]
    mech_perf = sub.groupby('missing_mechanism')['r2_test'].mean()
    for mech, r2 in mech_perf.items():
        print(f"  {mech:20s} avg R2={r2:.4f}")

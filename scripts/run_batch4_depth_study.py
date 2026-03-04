"""Batch 4: Depth study - How does depth affect NeuMiss+ performance?
For the best variants, test depth from 1 to 10."""
import sys
sys.path.insert(0, '/Users/yukang/Desktop/NeuroMiss/src')

from data_generation import DataScenario
from experiment_runner import run_experiment_suite, analyze_results

# Test on key scenarios spanning the difficulty spectrum
scenarios = [
    DataScenario('gaussian', 'linear', 'MCAR', n_features=10, missing_rate=0.5, snr=10),
    DataScenario('gaussian', 'quadratic', 'MCAR', n_features=10, missing_rate=0.5, snr=10),
    DataScenario('mixture_gaussian', 'cubic', 'MCAR', n_features=10, missing_rate=0.5, snr=10,
                 distribution_params={'n_components': 3}),
    DataScenario('student_t', 'sinusoidal', 'MCAR', n_features=10, missing_rate=0.5, snr=10,
                 distribution_params={'df': 5}),
]

methods = []
for depth in [1, 2, 3, 4, 5, 7, 10]:
    # Original NeuMiss
    methods.append({
        'name': f'NeuMiss_d{depth}',
        'variant': 'original', 'depth': depth,
        'activation': 'relu', 'residual_connection': False,
    })
    # NeuMiss+ A with GELU
    methods.append({
        'name': f'NeuMiss+A_gelu_d{depth}',
        'variant': 'A', 'depth': depth,
        'activation': 'gelu', 'residual_connection': False,
    })
    # NeuMiss+ B with GELU
    methods.append({
        'name': f'NeuMiss+B_gelu_d{depth}',
        'variant': 'B', 'depth': depth,
        'activation': 'gelu', 'residual_connection': False,
    })

print("=" * 60)
print("BATCH 4: DEPTH STUDY - Performance vs depth across scenarios")
print("=" * 60)

df = run_experiment_suite(
    scenarios, methods,
    n_train=10000, n_val=2000, n_test=5000,
    n_epochs=150, batch_size=64, n_repeats=3,
    output_file='/Users/yukang/Desktop/NeuroMiss/results/batch4_depth.csv')

analyze_results(df)

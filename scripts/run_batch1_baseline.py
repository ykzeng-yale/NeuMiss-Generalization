"""Batch 1: Baseline experiments - Gaussian + Linear with all missing mechanisms.
Tests original NeuMiss vs NeuMiss+ on the original paper's territory."""
import sys
sys.path.insert(0, '/Users/yukang/Desktop/NeuroMiss/src')

from data_generation import DataScenario
from experiment_runner import run_experiment_suite, analyze_results

scenarios = [
    DataScenario('gaussian', 'linear', 'MCAR', n_features=10, missing_rate=0.5, snr=10),
    DataScenario('gaussian', 'linear', 'MAR', n_features=10, missing_rate=0.5, snr=10),
    DataScenario('gaussian', 'linear', 'MNAR_selfmasking', n_features=10, missing_rate=0.5, snr=10),
    DataScenario('gaussian', 'linear', 'MNAR_censoring', n_features=10, missing_rate=0.5, snr=10),
]

methods = [
    {'name': 'NeuMiss_d1', 'variant': 'original', 'depth': 1, 'activation': 'relu', 'residual_connection': False},
    {'name': 'NeuMiss_d3', 'variant': 'original', 'depth': 3, 'activation': 'relu', 'residual_connection': False},
    {'name': 'NeuMiss_d5', 'variant': 'original', 'depth': 5, 'activation': 'relu', 'residual_connection': False},
    {'name': 'NeuMiss+A_gelu_d3', 'variant': 'A', 'depth': 3, 'activation': 'gelu', 'residual_connection': False},
    {'name': 'NeuMiss+A_relu_d3', 'variant': 'A', 'depth': 3, 'activation': 'relu', 'residual_connection': False},
    {'name': 'NeuMiss+B_gelu_d3', 'variant': 'B', 'depth': 3, 'activation': 'gelu', 'residual_connection': False},
    {'name': 'NeuMiss+B_relu_d3', 'variant': 'B', 'depth': 3, 'activation': 'relu', 'residual_connection': False},
    {'name': 'NeuMiss+C_gelu_d3', 'variant': 'C', 'depth': 3, 'activation': 'gelu', 'residual_connection': False, 'expansion_factor': 2},
    {'name': 'NeuMiss+D_gelu_d3', 'variant': 'D', 'depth': 3, 'activation': 'gelu', 'residual_connection': False, 'degree': 2},
]

print("=" * 60)
print("BATCH 1: BASELINE - Gaussian + Linear (original NeuMiss territory)")
print("=" * 60)

df = run_experiment_suite(
    scenarios, methods,
    n_train=10000, n_val=2000, n_test=5000,
    n_epochs=150, batch_size=64, n_repeats=3,
    output_file='/Users/yukang/Desktop/NeuroMiss/results/batch1_baseline.csv')

analyze_results(df)

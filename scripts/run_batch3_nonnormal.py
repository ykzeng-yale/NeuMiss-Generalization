"""Batch 3: Non-normal distribution experiments.
Tests mixture of Gaussians, Student-t, skewed distributions.
This is the harder extension from the meeting notes."""
import sys
sys.path.insert(0, '/Users/yukang/Desktop/NeuroMiss/src')

from data_generation import DataScenario
from experiment_runner import run_experiment_suite, analyze_results

scenarios = [
    # Mixture of Gaussians
    DataScenario('mixture_gaussian', 'linear', 'MCAR', n_features=10, missing_rate=0.5, snr=10,
                 distribution_params={'n_components': 3}),
    DataScenario('mixture_gaussian', 'quadratic', 'MCAR', n_features=10, missing_rate=0.5, snr=10,
                 distribution_params={'n_components': 3}),
    DataScenario('mixture_gaussian', 'cubic', 'MCAR', n_features=10, missing_rate=0.5, snr=10,
                 distribution_params={'n_components': 3}),
    DataScenario('mixture_gaussian', 'linear', 'MNAR_censoring', n_features=10, missing_rate=0.5, snr=10,
                 distribution_params={'n_components': 3}),

    # Student-t (heavy tails)
    DataScenario('student_t', 'linear', 'MCAR', n_features=10, missing_rate=0.5, snr=10,
                 distribution_params={'df': 5}),
    DataScenario('student_t', 'quadratic', 'MCAR', n_features=10, missing_rate=0.5, snr=10,
                 distribution_params={'df': 5}),
    DataScenario('student_t', 'sinusoidal', 'MCAR', n_features=10, missing_rate=0.5, snr=10,
                 distribution_params={'df': 5}),

    # Skewed
    DataScenario('skewed', 'linear', 'MCAR', n_features=10, missing_rate=0.5, snr=10),
    DataScenario('skewed', 'quadratic', 'MCAR', n_features=10, missing_rate=0.5, snr=10),
]

methods = [
    {'name': 'NeuMiss_d1', 'variant': 'original', 'depth': 1, 'activation': 'relu', 'residual_connection': False},
    {'name': 'NeuMiss_d3', 'variant': 'original', 'depth': 3, 'activation': 'relu', 'residual_connection': False},
    {'name': 'NeuMiss_d5', 'variant': 'original', 'depth': 5, 'activation': 'relu', 'residual_connection': False},
    {'name': 'NeuMiss+A_gelu_d3', 'variant': 'A', 'depth': 3, 'activation': 'gelu', 'residual_connection': False},
    {'name': 'NeuMiss+A_relu_d3', 'variant': 'A', 'depth': 3, 'activation': 'relu', 'residual_connection': False},
    {'name': 'NeuMiss+B_gelu_d3', 'variant': 'B', 'depth': 3, 'activation': 'gelu', 'residual_connection': False},
    {'name': 'NeuMiss+C_gelu_d3', 'variant': 'C', 'depth': 3, 'activation': 'gelu', 'residual_connection': False, 'expansion_factor': 2},
    {'name': 'NeuMiss+C_gelu_d3_ef4', 'variant': 'C', 'depth': 3, 'activation': 'gelu', 'residual_connection': False, 'expansion_factor': 4},
    {'name': 'NeuMiss+D_gelu_d3', 'variant': 'D', 'depth': 3, 'activation': 'gelu', 'residual_connection': False, 'degree': 2},
    {'name': 'NeuMiss+A_gelu_d5_res', 'variant': 'A', 'depth': 5, 'activation': 'gelu', 'residual_connection': True},
]

print("=" * 60)
print("BATCH 3: NON-NORMAL - Mixture/Student-t/Skewed distributions")
print("=" * 60)

df = run_experiment_suite(
    scenarios, methods,
    n_train=10000, n_val=2000, n_test=5000,
    n_epochs=150, batch_size=64, n_repeats=3,
    output_file='/Users/yukang/Desktop/NeuroMiss/results/batch3_nonnormal.csv')

analyze_results(df)

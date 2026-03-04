#!/usr/bin/env python3
"""Quick head-to-head comparison of existing architectures on 4 key scenarios.

Establishes current performance baselines before introducing new architectures.
"""

import os
import sys
import time
import warnings
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from data_generation import DataScenario
from neumiss_plus import NeuMissPlus, ImputeMLP, PretrainEncoder

warnings.filterwarnings('ignore')

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
N_TRAIN = 2000
N_VAL = 500
N_TEST = 1000
SEEDS = [0, 1, 2]

SCENARIOS = [
    {
        'name': 'gaussian + quadratic + MCAR',
        'desc': 'Needs variance correction',
        'kwargs': dict(distribution='gaussian', response='quadratic',
                       missing_mechanism='MCAR', n_features=10,
                       missing_rate=0.5, snr=10.0),
    },
    {
        'name': 'mixture_gaussian + cubic + MCAR',
        'desc': 'Hardest scenario',
        'kwargs': dict(distribution='mixture_gaussian', response='cubic',
                       missing_mechanism='MCAR', n_features=10,
                       missing_rate=0.5, snr=10.0,
                       distribution_params={'n_components': 3}),
    },
    {
        'name': 'student_t + quadratic + MCAR',
        'desc': 'Heavy-tailed distribution',
        'kwargs': dict(distribution='student_t', response='quadratic',
                       missing_mechanism='MCAR', n_features=10,
                       missing_rate=0.5, snr=10.0,
                       distribution_params={'df': 5}),
    },
    {
        'name': 'gaussian + interaction + MCAR',
        'desc': 'Pairwise interactions',
        'kwargs': dict(distribution='gaussian', response='interaction',
                       missing_mechanism='MCAR', n_features=10,
                       missing_rate=0.5, snr=10.0),
    },
]

METHODS = {
    'NeuMiss (original)': lambda: NeuMissPlus(
        variant='original', depth=3, n_epochs=200, batch_size=64, lr=0.001),
    'NeuMiss+ (C, gelu)': lambda: NeuMissPlus(
        variant='C', depth=3, activation='gelu', n_epochs=200,
        batch_size=64, lr=0.001),
    'ImputeMLP': lambda: ImputeMLP(
        hidden_layers=(256, 128), n_epochs=200, batch_size=64, lr=0.001),
    'PretrainEncoder': lambda: PretrainEncoder(
        depth=3, mlp_layers=(128,), pretrain_epochs=50, train_epochs=200,
        batch_size=64, lr=0.001),
}


def r2_score(y_true, y_pred):
    ss_res = np.mean((y_true - y_pred) ** 2)
    ss_tot = np.var(y_true)
    return 1.0 - ss_res / ss_tot


def run_single(method_factory, data):
    model = method_factory()
    model.fit(data['X_train'], data['y_train'],
              X_val=data['X_val'], y_val=data['y_val'])
    y_pred = model.predict(data['X_test'])
    return r2_score(data['y_test'], y_pred)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    print("=" * 80)
    print("HEAD-TO-HEAD BASELINE COMPARISON")
    print("=" * 80)
    print(f"n_train={N_TRAIN}  n_val={N_VAL}  n_test={N_TEST}  seeds={SEEDS}")
    print(f"Methods: {list(METHODS.keys())}")
    print()

    # results[scenario_name][method_name] = list of R2 across seeds
    results = {}

    for sc_cfg in SCENARIOS:
        sc_name = sc_cfg['name']
        sc_desc = sc_cfg['desc']
        print("-" * 80)
        print(f"Scenario: {sc_name}  ({sc_desc})")
        print("-" * 80)
        results[sc_name] = {}

        for method_name, method_factory in METHODS.items():
            r2_list = []
            for seed in SEEDS:
                sc = DataScenario(**sc_cfg['kwargs'])
                data = sc.generate(n_train=N_TRAIN, n_val=N_VAL,
                                   n_test=N_TEST, random_state=seed)
                t0 = time.time()
                r2 = run_single(method_factory, data)
                elapsed = time.time() - t0
                r2_list.append(r2)
                print(f"  {method_name:25s}  seed={seed}  R2={r2:.4f}  ({elapsed:.1f}s)")

            results[sc_name][method_name] = r2_list
            mean_r2 = np.mean(r2_list)
            std_r2 = np.std(r2_list)
            print(f"  {method_name:25s}  => mean R2 = {mean_r2:.4f} +/- {std_r2:.4f}")
            print()

    # ---------------------------------------------------------------------------
    # Summary table
    # ---------------------------------------------------------------------------
    print()
    print("=" * 80)
    print("SUMMARY TABLE:  mean R2 +/- std  (3 seeds)")
    print("=" * 80)

    method_names = list(METHODS.keys())
    header = f"{'Scenario':42s}" + "".join(f"{m:>22s}" for m in method_names)
    print(header)
    print("-" * len(header))

    for sc_cfg in SCENARIOS:
        sc_name = sc_cfg['name']
        row = f"{sc_name:42s}"
        best_mean = -np.inf
        for m in method_names:
            mean_r2 = np.mean(results[sc_name][m])
            if mean_r2 > best_mean:
                best_mean = mean_r2
        for m in method_names:
            vals = results[sc_name][m]
            mean_r2 = np.mean(vals)
            std_r2 = np.std(vals)
            marker = " *" if abs(mean_r2 - best_mean) < 1e-6 else "  "
            row += f"  {mean_r2:+.4f}+/-{std_r2:.4f}{marker}"
        print(row)

    print()
    print("* = best method for that scenario")
    print()

    # ---------------------------------------------------------------------------
    # Per-scenario winner
    # ---------------------------------------------------------------------------
    print("WINNERS:")
    for sc_cfg in SCENARIOS:
        sc_name = sc_cfg['name']
        best_m = max(method_names,
                     key=lambda m: np.mean(results[sc_name][m]))
        best_r2 = np.mean(results[sc_name][best_m])
        print(f"  {sc_name:42s} -> {best_m} (R2={best_r2:.4f})")

    print()
    print("Done.")

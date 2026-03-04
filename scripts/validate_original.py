"""
Validate the original NeuMiss code by reproducing baseline results.

Tests MCAR, MAR, and selfmasking settings with d=10, n=10000.
Computes Bayes rate and reports R2 relative to Bayes rate for NeuMiss
at depth 1, 3, 5 and an MLP baseline.
"""

import sys
import time
import warnings
import numpy as np

# Insert original NeuMiss code path
sys.path.insert(0, '/Users/yukang/Desktop/NeuroMiss/NeuMiss_original/python')

from ground_truth import (
    gen_params, gen_data,
    gen_params_selfmasking, gen_data_selfmasking,
    BayesPredictor_MCAR_MAR,
    BayesPredictor_gaussian_selfmasking,
)
from neumannS0_mlp import Neumann_mlp
from mlp import MLP_reg

warnings.filterwarnings('ignore')
np.random.seed(42)


# ------------------------------------------------------------------ #
#  Helpers
# ------------------------------------------------------------------ #
def r2_score(y_true, y_pred):
    ss_res = np.mean((y_true - y_pred) ** 2)
    ss_tot = np.mean((y_true - np.mean(y_true)) ** 2)
    return 1.0 - ss_res / ss_tot


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test).flatten()
    return r2_score(y_test, y_pred)


# ------------------------------------------------------------------ #
#  Settings
# ------------------------------------------------------------------ #
N_FEATURES = 10
N_TRAIN = 10000
N_VAL = 2000
N_TEST = 5000
N_TOT = N_TRAIN + N_VAL + N_TEST
MISSING_RATE = 0.5
PROP_LATENT = 0.5
SNR = 10
RANDOM_STATE = 0

NEUMANN_DEPTHS = [1, 3, 5]
N_EPOCHS_NEUMANN = 100
BATCH_SIZE_NEUMANN = 10
LR = 0.01 / N_FEATURES  # 0.001

# MLP settings (from their launch_simu_perf.py)
MLP_WIDTH_MULT = 5  # hidden = 5 * 2*n_features = 100
MLP_N_EPOCHS = 200
MLP_BATCH_SIZE = 200


scenarios = {
    'MCAR': {
        'data_type': 'MCAR',
        'gen_params_kwargs': {
            'n_features': N_FEATURES,
            'missing_rate': MISSING_RATE,
            'prop_latent': PROP_LATENT,
            'snr': SNR,
            'masking': 'MCAR',
            'random_state': RANDOM_STATE,
        },
        'bayes_class': BayesPredictor_MCAR_MAR,
    },
    'MAR_logistic': {
        'data_type': 'MAR_logistic',
        'gen_params_kwargs': {
            'n_features': N_FEATURES,
            'missing_rate': MISSING_RATE,
            'prop_latent': PROP_LATENT,
            'snr': SNR,
            'masking': 'MAR_logistic',
            'prop_for_masking': 0.1,
            'random_state': RANDOM_STATE,
        },
        'bayes_class': BayesPredictor_MCAR_MAR,
    },
    'Gaussian_selfmasking': {
        'data_type': 'selfmasking',
        'gen_params_kwargs': {
            'n_features': N_FEATURES,
            'missing_rate': MISSING_RATE,
            'prop_latent': PROP_LATENT,
            'sm_type': 'gaussian',
            'sm_param': 2,
            'snr': SNR,
            'perm': False,
            'random_state': RANDOM_STATE,
        },
        'bayes_class': BayesPredictor_gaussian_selfmasking,
    },
}


# ------------------------------------------------------------------ #
#  Run experiments
# ------------------------------------------------------------------ #
print("=" * 78)
print("NeuMiss Original Code Validation")
print("=" * 78)
print(f"n_features={N_FEATURES}, n_train={N_TRAIN}, n_val={N_VAL}, "
      f"n_test={N_TEST}, missing_rate={MISSING_RATE}, snr={SNR}")
print(f"Neumann: SGD, lr={LR}, batch_size={BATCH_SIZE_NEUMANN}, "
      f"n_epochs={N_EPOCHS_NEUMANN}")
print(f"MLP baseline: Adam, lr={LR}, batch_size={MLP_BATCH_SIZE}, "
      f"n_epochs={MLP_N_EPOCHS}, hidden=[{MLP_WIDTH_MULT}*2d={MLP_WIDTH_MULT*2*N_FEATURES}]")
print("=" * 78)

all_results = {}

for scenario_name, scenario in scenarios.items():
    print(f"\n{'─' * 78}")
    print(f"  Scenario: {scenario_name}")
    print(f"{'─' * 78}")

    # Generate data
    t0 = time.time()
    if scenario['data_type'] == 'selfmasking':
        data_params = gen_params_selfmasking(**scenario['gen_params_kwargs'])
        gen = gen_data_selfmasking([N_TOT], data_params, random_state=RANDOM_STATE)
    else:
        data_params = gen_params(**scenario['gen_params_kwargs'])
        gen = gen_data([N_TOT], data_params, random_state=RANDOM_STATE)

    # Consume the generator (one yield for one n_size)
    for X_full, y_full in gen:
        pass

    # Split: test first, then val, then train (matching their convention)
    X_test = X_full[:N_TEST]
    y_test = y_full[:N_TEST]
    X_val = X_full[N_TEST:N_TEST + N_VAL]
    y_val = y_full[N_TEST:N_TEST + N_VAL]
    X_train = X_full[N_TEST + N_VAL:]
    y_train = y_full[N_TEST + N_VAL:]

    miss_rate_actual = np.isnan(X_full).mean()
    print(f"  Data generated in {time.time() - t0:.1f}s")
    print(f"  Actual missing rate: {miss_rate_actual:.3f}")
    print(f"  y_train var: {np.var(y_train):.2f}, y_test var: {np.var(y_test):.2f}")

    results = {}

    # ---- Bayes rate ---- #
    print("\n  Computing Bayes rate...")
    t0 = time.time()
    bayes = scenario['bayes_class'](data_params)
    bayes.fit(X_train, y_train)
    r2_bayes = evaluate_model(bayes, X_test, y_test)
    results['Bayes'] = r2_bayes
    print(f"    Bayes R2 (test): {r2_bayes:.4f}  ({time.time() - t0:.1f}s)")

    # ---- NeuMiss at various depths ---- #
    for depth in NEUMANN_DEPTHS:
        print(f"\n  Training NeuMiss depth={depth}...")
        t0 = time.time()
        model = Neumann_mlp(
            depth=depth,
            n_epochs=N_EPOCHS_NEUMANN,
            batch_size=BATCH_SIZE_NEUMANN,
            lr=LR,
            early_stopping=True,
            residual_connection=False,
            mlp_depth=0,
            init_type='normal',
            verbose=False,
        )
        model.fit(X_train.copy(), y_train.copy(),
                  X_val=X_val.copy(), y_val=y_val.copy())
        r2_nm = evaluate_model(model, X_test, y_test)
        results[f'NeuMiss_d{depth}'] = r2_nm
        pct = (r2_nm / r2_bayes * 100) if r2_bayes > 0 else float('nan')
        print(f"    NeuMiss(d={depth}) R2 (test): {r2_nm:.4f}  "
              f"({pct:.1f}% of Bayes)  ({time.time() - t0:.1f}s)")

    # ---- MLP baseline ---- #
    print(f"\n  Training MLP baseline...")
    t0 = time.time()
    hidden_size = MLP_WIDTH_MULT * 2 * N_FEATURES  # MLP sees [X, mask], so 2d input
    mlp = MLP_reg(
        hidden_layer_sizes=[hidden_size],
        lr=LR,
        batch_size=MLP_BATCH_SIZE,
        n_epochs=MLP_N_EPOCHS,
        early_stopping=True,
        verbose=False,
    )
    mlp.fit(X_train.copy(), y_train.copy(),
            X_val=X_val.copy(), y_val=y_val.copy())
    r2_mlp = evaluate_model(mlp, X_test, y_test)
    results['MLP'] = r2_mlp
    pct = (r2_mlp / r2_bayes * 100) if r2_bayes > 0 else float('nan')
    print(f"    MLP R2 (test): {r2_mlp:.4f}  "
          f"({pct:.1f}% of Bayes)  ({time.time() - t0:.1f}s)")

    all_results[scenario_name] = results


# ------------------------------------------------------------------ #
#  Summary table
# ------------------------------------------------------------------ #
print("\n\n")
print("=" * 78)
print("  SUMMARY TABLE")
print("=" * 78)

header = f"{'Scenario':<22} {'Bayes':>8} {'NM_d1':>8} {'NM_d3':>8} {'NM_d5':>8} {'MLP':>8}"
print(header)
print("-" * len(header))

for scenario_name, results in all_results.items():
    row = f"{scenario_name:<22}"
    row += f" {results.get('Bayes', float('nan')):>8.4f}"
    for depth in NEUMANN_DEPTHS:
        key = f'NeuMiss_d{depth}'
        row += f" {results.get(key, float('nan')):>8.4f}"
    row += f" {results.get('MLP', float('nan')):>8.4f}"
    print(row)

print()
print("R2 as % of Bayes rate:")
print("-" * len(header))
for scenario_name, results in all_results.items():
    r2b = results.get('Bayes', 0)
    row = f"{scenario_name:<22}"
    row += f" {'100.0%':>8}"
    for depth in NEUMANN_DEPTHS:
        key = f'NeuMiss_d{depth}'
        val = results.get(key, 0)
        pct = val / r2b * 100 if r2b > 0 else float('nan')
        row += f" {pct:>7.1f}%"
    val = results.get('MLP', 0)
    pct = val / r2b * 100 if r2b > 0 else float('nan')
    row += f" {pct:>7.1f}%"
    print(row)

print("\n" + "=" * 78)
print("Validation complete.")
print("=" * 78)

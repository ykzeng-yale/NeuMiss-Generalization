#!/usr/bin/env python3
"""Ralph Loop V2: Theory-Driven Iterative Improvement.

Based on the theoretical extension showing that for nonlinear f:
- Mean pathway (NeuMiss): approximates E[X_mis|X_obs]
- Variance pathway (NEW): captures tr(A_mm Sigma_mis|obs) correction
- For non-Gaussian X: mixture-of-experts structure needed

This Ralph loop:
Round 0: Comprehensive baseline across ALL scenarios and methods
Round 1: Focus on where theory-motivated architectures (NeuMiss-NL, SuffStat)
         improve over baselines
Round 2: Hyperparameter sweep on promising combinations
Round 3: Ablation study and final validation

Available architectures (actual class names in neumiss_plus.py):
- NeuMissPlus(variant='original'): Original NeuMiss (linear baseline)
- NeuMissPlus(variant='C', activation='gelu'): NeuMiss+ Variant C
- NeuMissMLPEstimator: NeuMiss + MLP head for nonlinear response
- ImputeMLP: Mean impute + MLP baseline
- NeuMissEncoderEstimator: Dual pathway encoder + MLP
- PretrainEncoder: Two-phase pretrained encoder (previous best)
- NeuMissNLEstimator: NEW two-pathway mean+variance architecture (if available)
- SuffStatNeuMissEstimator: NEW sufficient statistics approach (if available)
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))

import numpy as np
import pandas as pd
import time
import traceback
from itertools import product
from data_generation import DataScenario
from neumiss_plus import (
    NeuMissPlus,
    NeuMissMLPEstimator,
    ImputeMLP,
    NeuMissEncoderEstimator,
    PretrainEncoder,
)

# Attempt to import NEW theory-motivated architectures (may not exist yet)
_HAS_NEUMISS_NL = False
_HAS_SUFFSTAT = False

try:
    from neumiss_plus import NeuMissNLEstimator
    _HAS_NEUMISS_NL = True
except ImportError:
    print("[WARN] NeuMissNLEstimator not found in neumiss_plus.py -- will skip.")
    NeuMissNLEstimator = None

try:
    from neumiss_plus import SuffStatNeuMissEstimator
    _HAS_SUFFSTAT = True
except ImportError:
    print("[WARN] SuffStatNeuMissEstimator not found in neumiss_plus.py -- will skip.")
    SuffStatNeuMissEstimator = None

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)


# ============================================================================
# Scenario definitions
# ============================================================================
def get_scenarios():
    """Comprehensive scenario set organized by theoretical difficulty.

    Groups:
    1. Linear f, Gaussian X (NeuMiss baseline territory)
    2. Polynomial f, Gaussian X (theory: need variance correction)
    3. Non-Gaussian X, Linear f (theory: need nonlinear conditional expectation)
    4. Non-Gaussian X + Nonlinear f (hardest, "best case" for new methods)
    5. MNAR + Nonlinear (theory: MNAR doesn't hurt prediction)
    6. Varying dimensions (scaling test)
    7. Varying missing rates
    """
    scenarios = {}

    # GROUP 1: Linear f (NeuMiss baseline territory)
    scenarios['gauss_linear_MCAR'] = DataScenario(
        'gaussian', 'linear', 'MCAR', 10, 0.5, 10.0)
    scenarios['gauss_linear_MAR'] = DataScenario(
        'gaussian', 'linear', 'MAR', 10, 0.5, 10.0)
    scenarios['gauss_linear_MNAR'] = DataScenario(
        'gaussian', 'linear', 'MNAR_selfmasking', 10, 0.5, 10.0)

    # GROUP 2: Polynomial f, Gaussian X (theory says need variance correction)
    scenarios['gauss_quadratic_MCAR'] = DataScenario(
        'gaussian', 'quadratic', 'MCAR', 10, 0.5, 10.0)
    scenarios['gauss_cubic_MCAR'] = DataScenario(
        'gaussian', 'cubic', 'MCAR', 10, 0.5, 10.0)
    scenarios['gauss_interaction_MCAR'] = DataScenario(
        'gaussian', 'interaction', 'MCAR', 10, 0.5, 10.0)

    # GROUP 3: Non-Gaussian X, Linear f (theory says need nonlinear computation)
    scenarios['mixture_linear_MCAR'] = DataScenario(
        'mixture_gaussian', 'linear', 'MCAR', 10, 0.5, 10.0,
        distribution_params={'n_components': 3})
    scenarios['student_linear_MCAR'] = DataScenario(
        'student_t', 'linear', 'MCAR', 10, 0.5, 10.0,
        distribution_params={'df': 5})
    scenarios['skewed_linear_MCAR'] = DataScenario(
        'skewed', 'linear', 'MCAR', 10, 0.5, 10.0)

    # GROUP 4: Non-Gaussian + Nonlinear (hardest, "best case scenario" for new methods)
    scenarios['mixture_quadratic_MCAR'] = DataScenario(
        'mixture_gaussian', 'quadratic', 'MCAR', 10, 0.5, 10.0,
        distribution_params={'n_components': 3})
    scenarios['mixture_cubic_MCAR'] = DataScenario(
        'mixture_gaussian', 'cubic', 'MCAR', 10, 0.5, 10.0,
        distribution_params={'n_components': 3})
    scenarios['student_quadratic_MCAR'] = DataScenario(
        'student_t', 'quadratic', 'MCAR', 10, 0.5, 10.0,
        distribution_params={'df': 5})
    scenarios['student_cubic_MCAR'] = DataScenario(
        'student_t', 'cubic', 'MCAR', 10, 0.5, 10.0,
        distribution_params={'df': 5})

    # GROUP 5: MNAR + Nonlinear (theory says MNAR doesn't hurt prediction)
    scenarios['gauss_quadratic_MNAR'] = DataScenario(
        'gaussian', 'quadratic', 'MNAR_censoring', 10, 0.5, 10.0)
    scenarios['mixture_cubic_MNAR'] = DataScenario(
        'mixture_gaussian', 'cubic', 'MNAR_censoring', 10, 0.5, 10.0,
        distribution_params={'n_components': 3})

    # GROUP 6: Varying dimensions (scaling test)
    scenarios['gauss_cubic_d20'] = DataScenario(
        'gaussian', 'cubic', 'MCAR', 20, 0.5, 10.0)
    scenarios['gauss_cubic_d50'] = DataScenario(
        'gaussian', 'cubic', 'MCAR', 50, 0.3, 10.0)

    # GROUP 7: Varying missing rates
    scenarios['gauss_quadratic_mr30'] = DataScenario(
        'gaussian', 'quadratic', 'MCAR', 10, 0.3, 10.0)
    scenarios['gauss_quadratic_mr70'] = DataScenario(
        'gaussian', 'quadratic', 'MCAR', 10, 0.7, 10.0)

    return scenarios


# ============================================================================
# Method definitions (factory functions returning fresh estimator instances)
# ============================================================================
def get_methods_round0():
    """All methods for initial comprehensive baseline.

    Returns dict of name -> callable that produces a fresh estimator.
    Skips methods whose classes are not available.
    """
    methods = {
        # Original NeuMiss (linear baseline)
        'NeuMiss_d3': lambda: NeuMissPlus(
            variant='original', depth=3,
            n_epochs=200, batch_size=64, lr=0.001),

        # NeuMiss+ Variant C (best variant from prior experiments)
        'NM+C_gelu_d3': lambda: NeuMissPlus(
            variant='C', depth=3, activation='gelu',
            n_epochs=200, batch_size=64, lr=0.001),
        'NM+C_gelu_d5': lambda: NeuMissPlus(
            variant='C', depth=5, activation='gelu',
            n_epochs=200, batch_size=64, lr=0.001),

        # NeuMiss + MLP (two-stage: Neumann imputation -> MLP head)
        'NeuMissMLP_d3': lambda: NeuMissMLPEstimator(
            variant='mlp', depth=3, mlp_layers=(128, 64),
            activation='gelu', n_epochs=200, batch_size=64, lr=0.001),

        # Impute-then-MLP baseline
        'ImputeMLP': lambda: ImputeMLP(
            hidden_layers=(256, 128), n_epochs=200,
            batch_size=64, lr=0.001),

        # Dual pathway encoder + MLP
        'Encoder_d3': lambda: NeuMissEncoderEstimator(
            variant='encoder', depth=3, mlp_layers=(128, 64),
            activation='gelu', n_epochs=200, batch_size=64, lr=0.001),

        # PretrainEncoder (previous best)
        'PretrainEnc': lambda: PretrainEncoder(
            depth=3, mlp_layers=(128,), activation='gelu',
            pretrain_epochs=50, train_epochs=200,
            batch_size=64, lr=0.001),
    }

    # Theory-motivated NEW architectures (added only if available)
    if _HAS_NEUMISS_NL:
        methods['NeuMiss_NL'] = lambda: NeuMissNLEstimator(
            depth=3, activation='gelu', mlp_layers=(128,),
            n_epochs=300, batch_size=64, lr=0.001)

    if _HAS_SUFFSTAT:
        methods['SuffStat'] = lambda: SuffStatNeuMissEstimator(
            depth=3, mlp_layers=(256, 128),
            n_epochs=300, batch_size=64, lr=0.001)

    return methods


# ============================================================================
# Experiment runner
# ============================================================================
def run_experiment(scenario_name, scenario, method_name, method_fn, seed):
    """Run a single experiment and return results dict.

    Parameters
    ----------
    scenario_name : str
    scenario : DataScenario
    method_name : str
    method_fn : callable returning a fresh estimator
    seed : int

    Returns
    -------
    dict with keys: scenario, method, seed, r2_test, r2_val, r2_train,
                    train_time, distribution, response, missing_mech,
                    n_features, missing_rate, error
    """
    np.random.seed(seed)

    base_result = {
        'scenario': scenario_name,
        'method': method_name,
        'seed': seed,
        'distribution': scenario.distribution,
        'response': scenario.response,
        'missing_mech': scenario.missing_mechanism,
        'n_features': scenario.n_features,
        'missing_rate': scenario.missing_rate,
    }

    try:
        data = scenario.generate(
            n_train=2000, n_val=500, n_test=1000, random_state=seed)

        model = method_fn()
        t0 = time.time()
        model.fit(data['X_train'], data['y_train'],
                  data['X_val'], data['y_val'])
        train_time = time.time() - t0

        r2_test = model.score(data['X_test'], data['y_test'])
        r2_val = model.score(data['X_val'], data['y_val'])
        r2_train = model.score(data['X_train'], data['y_train'])

        base_result.update({
            'r2_test': r2_test,
            'r2_val': r2_val,
            'r2_train': r2_train,
            'train_time': train_time,
            'error': '',
        })
        return base_result

    except Exception as e:
        tb = traceback.format_exc()
        print(f"  ERROR: {scenario_name}/{method_name}/seed={seed}: {e}")
        print(f"  {tb[-200:]}")
        base_result.update({
            'r2_test': float('nan'),
            'r2_val': float('nan'),
            'r2_train': float('nan'),
            'train_time': 0,
            'error': str(e),
        })
        return base_result


# ============================================================================
# Analysis
# ============================================================================
def analyze_round(df, round_num):
    """Analyze results from a round and determine focus for next round.

    Prints summary tables, winner analysis, and theory-validation checks.
    Returns (pivot_wide, winners) for downstream consumption.
    """
    print(f"\n{'='*80}")
    print(f"  RALPH LOOP V2 - ROUND {round_num} ANALYSIS")
    print(f"{'='*80}")

    # Drop rows with NaN R2 for analysis
    df_valid = df.dropna(subset=['r2_test'])
    if len(df_valid) == 0:
        print("  No valid results to analyze.")
        return pd.DataFrame(), pd.DataFrame()

    # Overall ranking: mean R2 by (scenario, method)
    pivot = df_valid.groupby(['scenario', 'method'])['r2_test'].mean().reset_index()
    pivot_wide = pivot.pivot(index='scenario', columns='method', values='r2_test')

    print("\n--- Mean R2 (test) by Scenario x Method ---")
    print(pivot_wide.round(4).to_string())

    # Winner per scenario
    winners = pivot.loc[pivot.groupby('scenario')['r2_test'].idxmax()]
    print("\n--- Winner per Scenario ---")
    for _, row in winners.iterrows():
        print(f"  {row['scenario']:40s} -> {row['method']:20s} (R2={row['r2_test']:.4f})")

    # Win count
    win_count = winners['method'].value_counts()
    print(f"\n--- Win Count ---")
    for method, count in win_count.items():
        print(f"  {method:20s}: {count} wins")

    # Mean R2 by method (overall)
    method_mean = df_valid.groupby('method')['r2_test'].mean().sort_values(ascending=False)
    print(f"\n--- Overall Mean R2 ---")
    for method, r2 in method_mean.items():
        print(f"  {method:20s}: {r2:.4f}")

    # Theory validation: nonlinear scenarios
    nonlinear_mask = df_valid['response'].isin(['quadratic', 'cubic', 'interaction'])
    nonlinear_df = df_valid[nonlinear_mask]
    if len(nonlinear_df) > 0:
        nl_means = nonlinear_df.groupby('method')['r2_test'].mean().sort_values(ascending=False)
        print(f"\n--- Nonlinear Scenarios Only (quadratic/cubic/interaction) ---")
        for method, r2 in nl_means.items():
            print(f"  {method:20s}: {r2:.4f}")

    # Non-Gaussian scenarios
    nongauss_mask = df_valid['distribution'].isin(['mixture_gaussian', 'student_t', 'skewed'])
    nongauss_df = df_valid[nongauss_mask]
    if len(nongauss_df) > 0:
        ng_means = nongauss_df.groupby('method')['r2_test'].mean().sort_values(ascending=False)
        print(f"\n--- Non-Gaussian Scenarios Only ---")
        for method, r2 in ng_means.items():
            print(f"  {method:20s}: {r2:.4f}")

    # Non-Gaussian AND Nonlinear (the "best case" for theory-motivated methods)
    hard_mask = nonlinear_mask & nongauss_mask
    hard_df = df_valid[hard_mask]
    if len(hard_df) > 0:
        hard_means = hard_df.groupby('method')['r2_test'].mean().sort_values(ascending=False)
        print(f"\n--- Non-Gaussian + Nonlinear (hardest) ---")
        for method, r2 in hard_means.items():
            print(f"  {method:20s}: {r2:.4f}")

    # Identify theory-motivated method wins
    theory_method_names = []
    if _HAS_NEUMISS_NL:
        theory_method_names.append('NeuMiss_NL')
    if _HAS_SUFFSTAT:
        theory_method_names.append('SuffStat')

    if theory_method_names:
        theory_wins = winners[winners['method'].isin(theory_method_names)]
        print(f"\n--- Theory-Motivated Method Wins ---")
        if len(theory_wins) > 0:
            for _, row in theory_wins.iterrows():
                print(f"  {row['scenario']} -> {row['method']} (R2={row['r2_test']:.4f})")
        else:
            print("  None yet -- need to investigate and improve")
    else:
        print("\n--- Theory-Motivated Methods ---")
        print("  NeuMissNLEstimator / SuffStatNeuMissEstimator not available.")
        print("  Comparing existing architectures on theory-relevant scenarios instead.")

    # Overfitting diagnostic: train vs test gap
    gap = df_valid.groupby('method').apply(
        lambda g: (g['r2_train'] - g['r2_test']).mean()
    ).sort_values(ascending=False)
    print(f"\n--- Overfitting Gap (mean train R2 - test R2) ---")
    for method, g in gap.items():
        print(f"  {method:20s}: {g:.4f}")

    # Training time
    time_means = df_valid.groupby('method')['train_time'].mean().sort_values()
    print(f"\n--- Mean Training Time (seconds) ---")
    for method, t in time_means.items():
        print(f"  {method:20s}: {t:.1f}s")

    return pivot_wide, winners


# ============================================================================
# Round 0: Comprehensive baseline
# ============================================================================
def round0_comprehensive(scenarios, seeds=None):
    """Round 0: Comprehensive baseline across all scenarios and methods.

    Runs every method on every scenario with 3 seeds for initial coverage.
    """
    if seeds is None:
        seeds = [0, 1, 2]

    print("\n" + "=" * 80)
    print("  ROUND 0: Comprehensive Baseline")
    print("=" * 80)

    methods = get_methods_round0()
    results = []
    total = len(scenarios) * len(methods) * len(seeds)
    i = 0

    for sc_name, sc in scenarios.items():
        for m_name, m_fn in methods.items():
            for seed in seeds:
                i += 1
                print(f"[{i}/{total}] {sc_name} / {m_name} / seed={seed}")
                result = run_experiment(sc_name, sc, m_name, m_fn, seed)
                results.append(result)
                # Save incrementally so partial results survive crashes
                pd.DataFrame(results).to_csv(
                    os.path.join(RESULTS_DIR, 'ralph_v2_round0.csv'), index=False)

    df = pd.DataFrame(results)
    return df


# ============================================================================
# Round 1: Focused evaluation with more seeds on promising combos
# ============================================================================
def round1_focused(df_r0, scenarios, seeds=None):
    """Round 1: Focus on promising combinations with more seeds.

    Strategy:
    - Drop trivial (linear Gaussian) scenarios where NeuMiss already dominates
    - Keep top-3 methods per scenario from Round 0
    - Always include theory-motivated methods and PretrainEncoder
    - Run 5 seeds for tighter confidence intervals
    """
    if seeds is None:
        seeds = [0, 1, 2, 3, 4]

    print("\n" + "=" * 80)
    print("  ROUND 1: Focused Evaluation (5 seeds)")
    print("=" * 80)

    # Determine top methods per scenario from Round 0
    r0_valid = df_r0.dropna(subset=['r2_test'])
    r0_means = r0_valid.groupby(['scenario', 'method'])['r2_test'].mean()

    focus_methods = set()
    for sc in r0_valid['scenario'].unique():
        if sc in r0_means.index.get_level_values(0):
            sc_means = r0_means[sc].sort_values(ascending=False)
            focus_methods.update(sc_means.head(3).index.tolist())

    # Always include key methods
    always_include = ['PretrainEnc', 'NeuMissMLP_d3', 'Encoder_d3']
    if _HAS_NEUMISS_NL:
        always_include.append('NeuMiss_NL')
    if _HAS_SUFFSTAT:
        always_include.append('SuffStat')
    focus_methods.update(always_include)

    # Focus on non-trivial scenarios: drop pure linear+Gaussian
    focus_scenarios = {
        k: v for k, v in scenarios.items()
        if not (v.response == 'linear' and v.distribution == 'gaussian')
    }

    methods_all = get_methods_round0()
    methods_to_run = {k: v for k, v in methods_all.items() if k in focus_methods}

    results = []
    total = len(focus_scenarios) * len(methods_to_run) * len(seeds)
    i = 0

    for sc_name, sc in focus_scenarios.items():
        for m_name, m_fn in methods_to_run.items():
            for seed in seeds:
                i += 1
                print(f"[{i}/{total}] {sc_name} / {m_name} / seed={seed}")
                result = run_experiment(sc_name, sc, m_name, m_fn, seed)
                results.append(result)
                pd.DataFrame(results).to_csv(
                    os.path.join(RESULTS_DIR, 'ralph_v2_round1.csv'), index=False)

    return pd.DataFrame(results)


# ============================================================================
# Round 2: Hyperparameter sweep on key scenarios
# ============================================================================
def round2_hyperparam(df_r1, scenarios, seeds=None):
    """Round 2: Hyperparameter tuning for top methods on key scenarios.

    Sweeps depth, MLP width, learning rate, and activation for:
    - NeuMissMLPEstimator (most flexible NeuMiss-based architecture)
    - NeuMissEncoderEstimator (dual pathway)
    - PretrainEncoder (previous best)
    - NeuMissNLEstimator / SuffStatNeuMissEstimator (if available)
    """
    if seeds is None:
        seeds = [0, 1, 2]

    print("\n" + "=" * 80)
    print("  ROUND 2: Hyperparameter Sweep")
    print("=" * 80)

    # Key scenarios where nonlinear/non-Gaussian effects matter most
    key_scenario_names = [
        'gauss_quadratic_MCAR', 'gauss_cubic_MCAR',
        'mixture_quadratic_MCAR', 'student_quadratic_MCAR',
        'gauss_interaction_MCAR', 'mixture_cubic_MCAR',
    ]

    # ---- NeuMissMLP configs ----
    mlp_configs = {
        'MLP_d3_128_64': lambda: NeuMissMLPEstimator(
            depth=3, mlp_layers=(128, 64), activation='gelu',
            n_epochs=250, batch_size=64, lr=0.001),
        'MLP_d3_256_128': lambda: NeuMissMLPEstimator(
            depth=3, mlp_layers=(256, 128), activation='gelu',
            n_epochs=250, batch_size=64, lr=0.001),
        'MLP_d5_128_64': lambda: NeuMissMLPEstimator(
            depth=5, mlp_layers=(128, 64), activation='gelu',
            n_epochs=250, batch_size=64, lr=0.001),
        'MLP_d3_256_128_silu': lambda: NeuMissMLPEstimator(
            depth=3, mlp_layers=(256, 128), activation='silu',
            n_epochs=250, batch_size=64, lr=0.001),
        'MLP_d3_128_64_lr5e4': lambda: NeuMissMLPEstimator(
            depth=3, mlp_layers=(128, 64), activation='gelu',
            n_epochs=250, batch_size=64, lr=5e-4),
        'MLP_wide_d3': lambda: NeuMissMLPEstimator(
            variant='wide_mlp', depth=3, expansion_factor=3,
            mlp_layers=(128, 64), activation='gelu',
            n_epochs=250, batch_size=64, lr=0.001),
    }

    # ---- Encoder configs ----
    enc_configs = {
        'Enc_d3_128_64': lambda: NeuMissEncoderEstimator(
            depth=3, mlp_layers=(128, 64), activation='gelu',
            n_epochs=250, batch_size=64, lr=0.001),
        'Enc_d3_256_128': lambda: NeuMissEncoderEstimator(
            depth=3, mlp_layers=(256, 128), activation='gelu',
            n_epochs=250, batch_size=64, lr=0.001),
        'Enc_d5_128_64': lambda: NeuMissEncoderEstimator(
            depth=5, mlp_layers=(128, 64), activation='gelu',
            n_epochs=250, batch_size=64, lr=0.001),
    }

    # ---- PretrainEncoder configs ----
    pt_configs = {
        'PT_d3_128_pt50': lambda: PretrainEncoder(
            depth=3, mlp_layers=(128,), pretrain_epochs=50,
            train_epochs=250, batch_size=64, lr=0.001),
        'PT_d3_256_pt100': lambda: PretrainEncoder(
            depth=3, mlp_layers=(256,), pretrain_epochs=100,
            train_epochs=250, batch_size=64, lr=0.001),
        'PT_d5_128_pt50': lambda: PretrainEncoder(
            depth=5, mlp_layers=(128,), pretrain_epochs=50,
            train_epochs=250, batch_size=64, lr=0.001),
        'PT_d3_128_pt50_lr5e4': lambda: PretrainEncoder(
            depth=3, mlp_layers=(128,), pretrain_epochs=50,
            train_epochs=250, batch_size=64, lr=5e-4),
    }

    all_methods = {}
    all_methods.update(mlp_configs)
    all_methods.update(enc_configs)
    all_methods.update(pt_configs)

    # ---- NeuMissNL configs (if available) ----
    if _HAS_NEUMISS_NL:
        nl_configs = {
            'NL_d3_mlp128': lambda: NeuMissNLEstimator(
                depth=3, mlp_layers=(128,), lr=0.001, n_epochs=300),
            'NL_d3_mlp256': lambda: NeuMissNLEstimator(
                depth=3, mlp_layers=(256,), lr=0.001, n_epochs=300),
            'NL_d3_mlp256_128': lambda: NeuMissNLEstimator(
                depth=3, mlp_layers=(256, 128), lr=0.001, n_epochs=300),
            'NL_d5_mlp128': lambda: NeuMissNLEstimator(
                depth=5, mlp_layers=(128,), lr=0.001, n_epochs=300),
            'NL_d3_mlp128_lr5e4': lambda: NeuMissNLEstimator(
                depth=3, mlp_layers=(128,), lr=5e-4, n_epochs=300),
            'NL_d3_mlp128_silu': lambda: NeuMissNLEstimator(
                depth=3, mlp_layers=(128,), activation='silu', n_epochs=300),
        }
        all_methods.update(nl_configs)

    # ---- SuffStat configs (if available) ----
    if _HAS_SUFFSTAT:
        ss_configs = {
            'SS_d3_256_128': lambda: SuffStatNeuMissEstimator(
                depth=3, mlp_layers=(256, 128), n_epochs=300),
            'SS_d3_512_256': lambda: SuffStatNeuMissEstimator(
                depth=3, mlp_layers=(512, 256), n_epochs=300),
            'SS_d5_256_128': lambda: SuffStatNeuMissEstimator(
                depth=5, mlp_layers=(256, 128), n_epochs=300),
            'SS_d3_256_128_lr5e4': lambda: SuffStatNeuMissEstimator(
                depth=3, mlp_layers=(256, 128), lr=5e-4, n_epochs=300),
        }
        all_methods.update(ss_configs)

    # Also include baselines for comparison
    all_methods['ImputeMLP'] = lambda: ImputeMLP(
        hidden_layers=(256, 128), n_epochs=200, batch_size=64, lr=0.001)
    all_methods['PretrainEnc_base'] = lambda: PretrainEncoder(
        depth=3, mlp_layers=(128,), pretrain_epochs=50, train_epochs=200,
        batch_size=64, lr=0.001)

    results = []
    key_scenarios = {k: scenarios[k] for k in key_scenario_names if k in scenarios}
    total = len(key_scenarios) * len(all_methods) * len(seeds)
    i = 0

    for sc_name, sc in key_scenarios.items():
        for m_name, m_fn in all_methods.items():
            for seed in seeds:
                i += 1
                print(f"[{i}/{total}] {sc_name} / {m_name} / seed={seed}")
                result = run_experiment(sc_name, sc, m_name, m_fn, seed)
                results.append(result)
                pd.DataFrame(results).to_csv(
                    os.path.join(RESULTS_DIR, 'ralph_v2_round2.csv'), index=False)

    return pd.DataFrame(results)


def select_best_config(df_r2, prefix, default_name, default_fn):
    """Select the best hyperparameter config from Round 2 results.

    Parameters
    ----------
    df_r2 : DataFrame from Round 2
    prefix : str, e.g. 'NL_' or 'SS_' or 'MLP_' or 'PT_'
    default_name : str, fallback config name
    default_fn : callable, fallback factory function

    Returns
    -------
    (best_name, best_fn) tuple
    """
    subset = df_r2[df_r2['method'].str.startswith(prefix)]
    subset = subset.dropna(subset=['r2_test'])
    if len(subset) == 0:
        return default_name, default_fn
    best_name = subset.groupby('method')['r2_test'].mean().idxmax()
    print(f"  Best {prefix}* config: {best_name}")
    return best_name, None  # fn will be looked up from round2 methods


# ============================================================================
# Round 3: Ablation and final validation
# ============================================================================
def round3_ablation(scenarios, best_methods, seeds=None):
    """Round 3: Final validation of best configs on ALL scenarios, 5 seeds.

    Parameters
    ----------
    scenarios : dict of DataScenario
    best_methods : dict of method_name -> factory_fn
    seeds : list of int
    """
    if seeds is None:
        seeds = [0, 1, 2, 3, 4]

    print("\n" + "=" * 80)
    print("  ROUND 3: Ablation & Final Validation (5 seeds)")
    print("=" * 80)

    results = []
    total = len(scenarios) * len(best_methods) * len(seeds)
    i = 0

    for sc_name, sc in scenarios.items():
        for m_name, m_fn in best_methods.items():
            for seed in seeds:
                i += 1
                print(f"[{i}/{total}] {sc_name} / {m_name} / seed={seed}")
                result = run_experiment(sc_name, sc, m_name, m_fn, seed)
                results.append(result)
                pd.DataFrame(results).to_csv(
                    os.path.join(RESULTS_DIR, 'ralph_v2_round3.csv'), index=False)

    return pd.DataFrame(results)


# ============================================================================
# Main entry point
# ============================================================================
if __name__ == '__main__':
    print("=" * 80)
    print("  RALPH LOOP V2: Theory-Driven Iterative Improvement")
    print("  Based on: NeuMiss extension for nonlinear f and non-Gaussian X")
    print("=" * 80)

    print(f"\n  Available theory-motivated methods:")
    print(f"    NeuMissNLEstimator:      {'YES' if _HAS_NEUMISS_NL else 'NO (not implemented yet)'}")
    print(f"    SuffStatNeuMissEstimator: {'YES' if _HAS_SUFFSTAT else 'NO (not implemented yet)'}")

    scenarios = get_scenarios()
    print(f"\n  Total scenarios: {len(scenarios)}")
    for name, sc in scenarios.items():
        print(f"    {name}: {sc.distribution}/{sc.response}/{sc.missing_mechanism} "
              f"(d={sc.n_features}, mr={sc.missing_rate})")

    # ==== Round 0: Comprehensive baseline ====
    df_r0 = round0_comprehensive(scenarios, seeds=[0, 1, 2])
    pivot_r0, winners_r0 = analyze_round(df_r0, 0)

    # ==== Round 1: Focused evaluation with more seeds ====
    df_r1 = round1_focused(df_r0, scenarios, seeds=[0, 1, 2, 3, 4])
    pivot_r1, winners_r1 = analyze_round(df_r1, 1)

    # ==== Round 2: Hyperparameter sweep ====
    df_r2 = round2_hyperparam(df_r1, scenarios, seeds=[0, 1, 2])
    pivot_r2, winners_r2 = analyze_round(df_r2, 2)

    # ==== Determine best configs from Round 2 ====
    print("\n" + "-" * 60)
    print("  Selecting best configs from Round 2...")
    print("-" * 60)

    # Best NeuMissMLP config
    best_mlp_name, _ = select_best_config(
        df_r2, 'MLP_', 'MLP_d3_128_64',
        lambda: NeuMissMLPEstimator(depth=3, mlp_layers=(128, 64), n_epochs=250))

    # Best PretrainEncoder config
    best_pt_name, _ = select_best_config(
        df_r2, 'PT_', 'PT_d3_128_pt50',
        lambda: PretrainEncoder(depth=3, mlp_layers=(128,), pretrain_epochs=50, train_epochs=250))

    # Best Encoder config
    best_enc_name, _ = select_best_config(
        df_r2, 'Enc_', 'Enc_d3_128_64',
        lambda: NeuMissEncoderEstimator(depth=3, mlp_layers=(128, 64), n_epochs=250))

    # Build the Round 2 method lookup table for reconstructing best configs
    r2_method_lookup = {}

    # MLP configs
    r2_method_lookup['MLP_d3_128_64'] = lambda: NeuMissMLPEstimator(
        depth=3, mlp_layers=(128, 64), activation='gelu', n_epochs=250, batch_size=64, lr=0.001)
    r2_method_lookup['MLP_d3_256_128'] = lambda: NeuMissMLPEstimator(
        depth=3, mlp_layers=(256, 128), activation='gelu', n_epochs=250, batch_size=64, lr=0.001)
    r2_method_lookup['MLP_d5_128_64'] = lambda: NeuMissMLPEstimator(
        depth=5, mlp_layers=(128, 64), activation='gelu', n_epochs=250, batch_size=64, lr=0.001)
    r2_method_lookup['MLP_d3_256_128_silu'] = lambda: NeuMissMLPEstimator(
        depth=3, mlp_layers=(256, 128), activation='silu', n_epochs=250, batch_size=64, lr=0.001)
    r2_method_lookup['MLP_d3_128_64_lr5e4'] = lambda: NeuMissMLPEstimator(
        depth=3, mlp_layers=(128, 64), activation='gelu', n_epochs=250, batch_size=64, lr=5e-4)
    r2_method_lookup['MLP_wide_d3'] = lambda: NeuMissMLPEstimator(
        variant='wide_mlp', depth=3, expansion_factor=3, mlp_layers=(128, 64),
        activation='gelu', n_epochs=250, batch_size=64, lr=0.001)

    # Encoder configs
    r2_method_lookup['Enc_d3_128_64'] = lambda: NeuMissEncoderEstimator(
        depth=3, mlp_layers=(128, 64), activation='gelu', n_epochs=250, batch_size=64, lr=0.001)
    r2_method_lookup['Enc_d3_256_128'] = lambda: NeuMissEncoderEstimator(
        depth=3, mlp_layers=(256, 128), activation='gelu', n_epochs=250, batch_size=64, lr=0.001)
    r2_method_lookup['Enc_d5_128_64'] = lambda: NeuMissEncoderEstimator(
        depth=5, mlp_layers=(128, 64), activation='gelu', n_epochs=250, batch_size=64, lr=0.001)

    # PretrainEncoder configs
    r2_method_lookup['PT_d3_128_pt50'] = lambda: PretrainEncoder(
        depth=3, mlp_layers=(128,), pretrain_epochs=50, train_epochs=250, batch_size=64, lr=0.001)
    r2_method_lookup['PT_d3_256_pt100'] = lambda: PretrainEncoder(
        depth=3, mlp_layers=(256,), pretrain_epochs=100, train_epochs=250, batch_size=64, lr=0.001)
    r2_method_lookup['PT_d5_128_pt50'] = lambda: PretrainEncoder(
        depth=5, mlp_layers=(128,), pretrain_epochs=50, train_epochs=250, batch_size=64, lr=0.001)
    r2_method_lookup['PT_d3_128_pt50_lr5e4'] = lambda: PretrainEncoder(
        depth=3, mlp_layers=(128,), pretrain_epochs=50, train_epochs=250, batch_size=64, lr=5e-4)

    # NeuMissNL configs
    if _HAS_NEUMISS_NL:
        best_nl_name, _ = select_best_config(
            df_r2, 'NL_', 'NL_d3_mlp128',
            lambda: NeuMissNLEstimator(depth=3, mlp_layers=(128,), n_epochs=300, lr=0.001))
        r2_method_lookup['NL_d3_mlp128'] = lambda: NeuMissNLEstimator(
            depth=3, mlp_layers=(128,), lr=0.001, n_epochs=300)
        r2_method_lookup['NL_d3_mlp256'] = lambda: NeuMissNLEstimator(
            depth=3, mlp_layers=(256,), lr=0.001, n_epochs=300)
        r2_method_lookup['NL_d3_mlp256_128'] = lambda: NeuMissNLEstimator(
            depth=3, mlp_layers=(256, 128), lr=0.001, n_epochs=300)
        r2_method_lookup['NL_d5_mlp128'] = lambda: NeuMissNLEstimator(
            depth=5, mlp_layers=(128,), lr=0.001, n_epochs=300)
        r2_method_lookup['NL_d3_mlp128_lr5e4'] = lambda: NeuMissNLEstimator(
            depth=3, mlp_layers=(128,), lr=5e-4, n_epochs=300)
        r2_method_lookup['NL_d3_mlp128_silu'] = lambda: NeuMissNLEstimator(
            depth=3, mlp_layers=(128,), activation='silu', n_epochs=300)

    # SuffStat configs
    if _HAS_SUFFSTAT:
        best_ss_name, _ = select_best_config(
            df_r2, 'SS_', 'SS_d3_256_128',
            lambda: SuffStatNeuMissEstimator(depth=3, mlp_layers=(256, 128), n_epochs=300, lr=0.001))
        r2_method_lookup['SS_d3_256_128'] = lambda: SuffStatNeuMissEstimator(
            depth=3, mlp_layers=(256, 128), n_epochs=300)
        r2_method_lookup['SS_d3_512_256'] = lambda: SuffStatNeuMissEstimator(
            depth=3, mlp_layers=(512, 256), n_epochs=300)
        r2_method_lookup['SS_d5_256_128'] = lambda: SuffStatNeuMissEstimator(
            depth=5, mlp_layers=(256, 128), n_epochs=300)
        r2_method_lookup['SS_d3_256_128_lr5e4'] = lambda: SuffStatNeuMissEstimator(
            depth=3, mlp_layers=(256, 128), lr=5e-4, n_epochs=300)

    # ==== Round 3: Final validation ====
    # Assemble final method set: baselines + best config from each architecture family
    final_methods = {
        # Baselines
        'NeuMiss_d3': lambda: NeuMissPlus(
            variant='original', depth=3, n_epochs=200, batch_size=64, lr=0.001),
        'NM+C_gelu_d3': lambda: NeuMissPlus(
            variant='C', depth=3, activation='gelu',
            n_epochs=200, batch_size=64, lr=0.001),
        'ImputeMLP': lambda: ImputeMLP(
            hidden_layers=(256, 128), n_epochs=200, batch_size=64, lr=0.001),
    }

    # Add best from each architecture family
    if best_mlp_name in r2_method_lookup:
        final_methods[f'Best_MLP ({best_mlp_name})'] = r2_method_lookup[best_mlp_name]
    else:
        final_methods['Best_MLP'] = lambda: NeuMissMLPEstimator(
            depth=3, mlp_layers=(128, 64), n_epochs=250, batch_size=64, lr=0.001)

    if best_pt_name in r2_method_lookup:
        final_methods[f'Best_PT ({best_pt_name})'] = r2_method_lookup[best_pt_name]
    else:
        final_methods['Best_PT'] = lambda: PretrainEncoder(
            depth=3, mlp_layers=(128,), pretrain_epochs=50, train_epochs=250)

    if best_enc_name in r2_method_lookup:
        final_methods[f'Best_Enc ({best_enc_name})'] = r2_method_lookup[best_enc_name]
    else:
        final_methods['Best_Enc'] = lambda: NeuMissEncoderEstimator(
            depth=3, mlp_layers=(128, 64), n_epochs=250, batch_size=64, lr=0.001)

    if _HAS_NEUMISS_NL and best_nl_name in r2_method_lookup:
        final_methods[f'Best_NL ({best_nl_name})'] = r2_method_lookup[best_nl_name]

    if _HAS_SUFFSTAT and best_ss_name in r2_method_lookup:
        final_methods[f'Best_SS ({best_ss_name})'] = r2_method_lookup[best_ss_name]

    df_r3 = round3_ablation(scenarios, final_methods, seeds=[0, 1, 2, 3, 4])
    pivot_r3, winners_r3 = analyze_round(df_r3, 3)

    # ==== Final Summary ====
    print("\n" + "=" * 80)
    print("  FINAL SUMMARY")
    print("=" * 80)

    # Combine all results
    all_results = pd.concat([df_r0, df_r1, df_r2, df_r3], ignore_index=True)
    all_results.to_csv(os.path.join(RESULTS_DIR, 'ralph_v2_all.csv'), index=False)

    # Final detailed results table (Round 3 only)
    df_r3_valid = df_r3.dropna(subset=['r2_test'])
    if len(df_r3_valid) > 0:
        final_stats = df_r3_valid.groupby(['scenario', 'method'])['r2_test'].agg(
            ['mean', 'std', 'count']
        )
        print("\nFinal Results (Round 3, 5 seeds):")
        print(final_stats.round(4).to_string())

        # Winner per scenario
        final_winners = df_r3_valid.groupby('scenario').apply(
            lambda g: g.groupby('method')['r2_test'].mean().idxmax()
        )
        print("\nFinal Winners:")
        for sc, winner in final_winners.items():
            r2 = df_r3_valid[
                (df_r3_valid['scenario'] == sc) & (df_r3_valid['method'] == winner)
            ]['r2_test'].mean()
            print(f"  {sc:40s} -> {winner} (R2={r2:.4f})")

        # Summary by scenario group
        print("\n--- Summary by Scenario Group ---")

        group_labels = {
            'Linear+Gaussian': lambda r: r['response'] == 'linear' and r['distribution'] == 'gaussian',
            'Nonlinear+Gaussian': lambda r: r['response'] != 'linear' and r['distribution'] == 'gaussian',
            'Linear+NonGaussian': lambda r: r['response'] == 'linear' and r['distribution'] != 'gaussian',
            'Nonlinear+NonGaussian': lambda r: r['response'] != 'linear' and r['distribution'] != 'gaussian',
        }
        for group_name, group_filter in group_labels.items():
            mask = df_r3_valid.apply(group_filter, axis=1)
            group_df = df_r3_valid[mask]
            if len(group_df) > 0:
                group_means = group_df.groupby('method')['r2_test'].mean().sort_values(ascending=False)
                print(f"\n  {group_name}:")
                for method, r2 in group_means.items():
                    print(f"    {method:40s}: {r2:.4f}")

    print(f"\nTotal experiments run: {len(all_results)}")
    print(f"Total valid experiments: {len(all_results.dropna(subset=['r2_test']))}")
    print(f"Results saved to: {RESULTS_DIR}/ralph_v2_*.csv")
    print("\nDone.")

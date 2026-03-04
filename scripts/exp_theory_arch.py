"""
Comprehensive Theory-Driven Architecture Comparison.

Tests ALL theory-driven architectures across 10 carefully chosen scenarios
that probe different aspects of the missing-data problem:
  - Distribution type: gaussian, mixture_gaussian, student_t
  - Response type: linear, quadratic, cubic, interaction
  - Missing mechanism: MCAR, MAR, MNAR_censoring

Architectures tested:
  1. NeuMiss (original)           -- linear Neumann baseline
  2. NM+C (gelu, depth=3)        -- nonlinear NeuMiss variant
  3. ImputeMLP                    -- impute-then-predict baseline
  4. PretrainEncoder              -- two-phase denoising + prediction
  5. NeuMiss-NL (depth=3, 128)   -- two-pathway architecture
  6. NeuMiss-NL (depth=5, 128)   -- deeper two-pathway
  7. NeuMiss-NL (depth=3, 256/128) -- wider two-pathway
  8. SuffStatNeuMiss              -- sufficient statistics architecture

Settings: n_train=2000, n_val=500, n_test=1000, n_features=10, 3 seeds.
"""

import sys
import os
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import pandas as pd
from data_generation import DataScenario
from neumiss_plus import (
    NeuMissPlus,
    ImputeMLP,
    PretrainEncoder,
    NeuMissNLEstimator,
    SuffStatNeuMissEstimator,
)

# ============================================================================
# Configuration
# ============================================================================
N_TRAIN = 2000
N_VAL = 500
N_TEST = 1000
N_FEATURES = 10
SEEDS = [0, 1, 2]
RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
RESULTS_PATH = os.path.join(RESULTS_DIR, 'exp_theory_arch.csv')

# ============================================================================
# 10 Scenarios
# ============================================================================
scenarios = [
    # 1. Baseline: Gaussian + linear + MCAR
    DataScenario('gaussian', 'linear', 'MCAR', N_FEATURES, 0.5, 10.0),
    # 2. Variance correction test: Gaussian + quadratic + MCAR
    DataScenario('gaussian', 'quadratic', 'MCAR', N_FEATURES, 0.5, 10.0),
    # 3. Higher-order test: Gaussian + cubic + MCAR
    DataScenario('gaussian', 'cubic', 'MCAR', N_FEATURES, 0.5, 10.0),
    # 4. Interaction test: Gaussian + interaction + MCAR
    DataScenario('gaussian', 'interaction', 'MCAR', N_FEATURES, 0.5, 10.0),
    # 5. Non-Gaussian test: mixture_gaussian + linear + MCAR
    DataScenario('mixture_gaussian', 'linear', 'MCAR', N_FEATURES, 0.5, 10.0,
                 distribution_params={'n_components': 3}),
    # 6. Non-Gaussian + nonlinear: mixture_gaussian + quadratic + MCAR
    DataScenario('mixture_gaussian', 'quadratic', 'MCAR', N_FEATURES, 0.5, 10.0,
                 distribution_params={'n_components': 3}),
    # 7. Hardest: mixture_gaussian + cubic + MCAR
    DataScenario('mixture_gaussian', 'cubic', 'MCAR', N_FEATURES, 0.5, 10.0,
                 distribution_params={'n_components': 3}),
    # 8. Heavy tails: student_t + quadratic + MCAR
    DataScenario('student_t', 'quadratic', 'MCAR', N_FEATURES, 0.5, 10.0,
                 distribution_params={'df': 5}),
    # 9. MAR mechanism: Gaussian + quadratic + MAR
    DataScenario('gaussian', 'quadratic', 'MAR', N_FEATURES, 0.5, 10.0),
    # 10. MNAR mechanism: Gaussian + cubic + MNAR_censoring
    DataScenario('gaussian', 'cubic', 'MNAR_censoring', N_FEATURES, 0.5, 10.0),
]

# ============================================================================
# Methods (8 architectures including NeuMiss-NL depth/width variants)
# ============================================================================
methods = [
    # 1. Original NeuMiss (linear Neumann, depth 3)
    ('NeuMiss_orig_d3', lambda: NeuMissPlus(
        variant='original', depth=3,
        n_epochs=200, batch_size=64, lr=0.001, early_stopping=True)),

    # 2. NM+C with GELU (nonlinear NeuMiss variant)
    ('NM+C_gelu_d3', lambda: NeuMissPlus(
        variant='C', activation='gelu', depth=3, expansion_factor=2,
        residual_connection=True,
        n_epochs=200, batch_size=64, lr=0.001, early_stopping=True)),

    # 3. Impute+MLP baseline (256, 128)
    ('ImputeMLP', lambda: ImputeMLP(
        hidden_layers=(256, 128), activation='gelu',
        n_epochs=150, batch_size=64, lr=0.001, dropout=0.1,
        early_stopping=True)),

    # 4. PretrainEncoder (denoising pretrain + fine-tune)
    ('PretrainEnc_d3', lambda: PretrainEncoder(
        depth=3, mlp_layers=(128,), activation='gelu', dropout=0.1,
        pretrain_epochs=50, train_epochs=200,
        batch_size=64, lr=0.001, early_stopping=True)),

    # 5. NeuMiss-NL depth=3, mlp_layers=(128,)
    ('NM-NL_d3_128', lambda: NeuMissNLEstimator(
        depth=3, mlp_layers=(128,), activation='gelu',
        residual_connection=True, dropout=0.1,
        n_epochs=300, batch_size=64, lr=0.001, early_stopping=True)),

    # 6. NeuMiss-NL depth=5, mlp_layers=(128,) -- deeper
    ('NM-NL_d5_128', lambda: NeuMissNLEstimator(
        depth=5, mlp_layers=(128,), activation='gelu',
        residual_connection=True, dropout=0.1,
        n_epochs=300, batch_size=64, lr=0.001, early_stopping=True)),

    # 7. NeuMiss-NL depth=3, mlp_layers=(256, 128) -- wider
    ('NM-NL_d3_256_128', lambda: NeuMissNLEstimator(
        depth=3, mlp_layers=(256, 128), activation='gelu',
        residual_connection=True, dropout=0.1,
        n_epochs=300, batch_size=64, lr=0.001, early_stopping=True)),

    # 8. SuffStatNeuMiss (sufficient statistics)
    ('SuffStat_d3', lambda: SuffStatNeuMissEstimator(
        depth=3, mlp_layers=(256, 128), activation='gelu', dropout=0.1,
        n_epochs=300, batch_size=64, lr=0.001, early_stopping=True)),
]

# ============================================================================
# Run experiment
# ============================================================================
def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    total = len(scenarios) * len(methods) * len(SEEDS)
    print(f"Theory-Driven Architecture Experiment")
    print(f"  {len(scenarios)} scenarios x {len(methods)} methods x {len(SEEDS)} seeds = {total} runs")
    print("=" * 90)

    results = []
    done = 0
    t0_global = time.time()

    for sc in scenarios:
        for seed in SEEDS:
            # Generate data once per (scenario, seed) pair
            data = sc.generate(N_TRAIN, N_VAL, N_TEST, random_state=seed)

            for method_name, make_est in methods:
                done += 1
                t0 = time.time()
                try:
                    est = make_est()
                    est.fit(data['X_train'], data['y_train'],
                            data['X_val'], data['y_val'])
                    pred = est.predict(data['X_test'])
                    mse = float(np.mean((data['y_test'] - pred) ** 2))
                    var_y = float(np.var(data['y_test']))
                    r2 = 1.0 - mse / var_y if var_y > 0 else float('nan')
                    elapsed = time.time() - t0

                    results.append({
                        'scenario': sc.name,
                        'distribution': sc.distribution,
                        'response': sc.response,
                        'missing_mechanism': sc.missing_mechanism,
                        'method': method_name,
                        'seed': seed,
                        'r2_test': r2,
                        'mse_test': mse,
                        'time_s': elapsed,
                        'status': 'success',
                    })
                    print(f"[{done:3d}/{total}] {sc.name:40s} | "
                          f"{method_name:20s} | R2={r2:+.4f} | {elapsed:.1f}s")

                except Exception as e:
                    elapsed = time.time() - t0
                    results.append({
                        'scenario': sc.name,
                        'distribution': sc.distribution,
                        'response': sc.response,
                        'missing_mechanism': sc.missing_mechanism,
                        'method': method_name,
                        'seed': seed,
                        'r2_test': float('nan'),
                        'mse_test': float('nan'),
                        'time_s': elapsed,
                        'status': 'error',
                        'error': str(e),
                    })
                    print(f"[{done:3d}/{total}] {sc.name:40s} | "
                          f"{method_name:20s} | ERROR: {e}")

        # Save incrementally after each scenario
        pd.DataFrame(results).to_csv(RESULTS_PATH, index=False)

    total_time = time.time() - t0_global
    print(f"\nTotal time: {total_time:.0f}s ({total_time/60:.1f} min)")

    # ========================================================================
    # Summary table
    # ========================================================================
    df = pd.DataFrame(results)
    df.to_csv(RESULTS_PATH, index=False)
    print(f"\nResults saved to: {RESULTS_PATH}")

    df_ok = df[df['status'] == 'success'].copy()
    if df_ok.empty:
        print("No successful runs.")
        return

    # Mean R2 per (scenario, method)
    agg = (df_ok.groupby(['scenario', 'method'])['r2_test']
           .agg(['mean', 'std'])
           .reset_index())
    agg.columns = ['scenario', 'method', 'r2_mean', 'r2_std']

    # Pivot table: scenarios as rows, methods as columns
    pivot = agg.pivot(index='scenario', columns='method', values='r2_mean')
    pivot_std = agg.pivot(index='scenario', columns='method', values='r2_std')

    # Order scenarios as defined
    scenario_order = [sc.name for sc in scenarios]
    pivot = pivot.reindex(scenario_order)
    pivot_std = pivot_std.reindex(scenario_order)

    # Order methods as defined
    method_order = [m[0] for m in methods]
    pivot = pivot[[c for c in method_order if c in pivot.columns]]
    pivot_std = pivot_std[[c for c in method_order if c in pivot_std.columns]]

    print("\n" + "=" * 120)
    print("SUMMARY: Mean R2 (test) across 3 seeds")
    print("=" * 120)

    # Print header
    header = f"{'Scenario':<45s}"
    for m in pivot.columns:
        header += f" {m:>14s}"
    header += f" {'Best':>16s}"
    print(header)
    print("-" * 120)

    for sc_name in pivot.index:
        row = f"{sc_name:<45s}"
        vals = {}
        for m in pivot.columns:
            v = pivot.loc[sc_name, m]
            s = pivot_std.loc[sc_name, m] if sc_name in pivot_std.index else 0
            if pd.notna(v):
                row += f" {v:>+7.4f}({s:.3f})"
                vals[m] = v
            else:
                row += f" {'N/A':>14s}"
        # Mark best method
        if vals:
            best_m = max(vals, key=vals.get)
            row += f"  {best_m}"
        print(row)

    print("-" * 120)

    # Mean rank across scenarios
    print("\nMean rank across scenarios (lower is better):")
    ranks = pivot.rank(axis=1, ascending=False, method='min')
    mean_ranks = ranks.mean(axis=0).sort_values()
    for m, r in mean_ranks.items():
        print(f"  {m:20s}: {r:.2f}")

    # NeuMiss-NL depth comparison
    print("\n--- NeuMiss-NL Depth Comparison (d3 vs d5, width=128) ---")
    for sc_name in pivot.index:
        d3 = pivot.loc[sc_name, 'NM-NL_d3_128'] if 'NM-NL_d3_128' in pivot.columns else float('nan')
        d5 = pivot.loc[sc_name, 'NM-NL_d5_128'] if 'NM-NL_d5_128' in pivot.columns else float('nan')
        winner = "d5" if (pd.notna(d5) and (pd.isna(d3) or d5 > d3)) else "d3"
        print(f"  {sc_name:<45s}  d3={d3:+.4f}  d5={d5:+.4f}  => {winner}")

    # NeuMiss-NL width comparison
    print("\n--- NeuMiss-NL Width Comparison (128 vs 256,128 at depth=3) ---")
    for sc_name in pivot.index:
        w1 = pivot.loc[sc_name, 'NM-NL_d3_128'] if 'NM-NL_d3_128' in pivot.columns else float('nan')
        w2 = pivot.loc[sc_name, 'NM-NL_d3_256_128'] if 'NM-NL_d3_256_128' in pivot.columns else float('nan')
        winner = "256,128" if (pd.notna(w2) and (pd.isna(w1) or w2 > w1)) else "128"
        print(f"  {sc_name:<45s}  128={w1:+.4f}  256,128={w2:+.4f}  => {winner}")


if __name__ == '__main__':
    main()

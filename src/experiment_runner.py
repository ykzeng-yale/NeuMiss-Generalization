"""
Experiment runner framework for NeuMiss+ research.

Runs all method variants against all data scenarios, collects metrics,
and produces comparison analysis.
"""

import numpy as np
import pandas as pd
import json
import time
import traceback
from itertools import product

from data_generation import DataScenario, get_key_scenarios, get_all_scenarios
from neumiss_plus import NeuMissPlus


def get_method_configs():
    """Define all method configurations to test."""
    configs = []

    # Original NeuMiss (baseline) at various depths
    for depth in [1, 3, 5]:
        configs.append({
            'name': f'NeuMiss_d{depth}',
            'variant': 'original',
            'depth': depth,
            'activation': 'relu',  # not used for original
            'residual_connection': False,
        })

    # NeuMiss+ Variant A (activation after mask) - various activations
    for act in ['relu', 'gelu', 'tanh', 'silu']:
        for depth in [3, 5]:
            configs.append({
                'name': f'NeuMiss+A_{act}_d{depth}',
                'variant': 'A',
                'depth': depth,
                'activation': act,
                'residual_connection': False,
            })

    # NeuMiss+ Variant B (activation before mask)
    for act in ['relu', 'gelu', 'tanh']:
        for depth in [3, 5]:
            configs.append({
                'name': f'NeuMiss+B_{act}_d{depth}',
                'variant': 'B',
                'depth': depth,
                'activation': act,
                'residual_connection': False,
            })

    # NeuMiss+ Variant C (wider hidden layers)
    for act in ['relu', 'gelu']:
        for depth in [3, 5]:
            for ef in [2, 4]:
                configs.append({
                    'name': f'NeuMiss+C_{act}_d{depth}_ef{ef}',
                    'variant': 'C',
                    'depth': depth,
                    'activation': act,
                    'expansion_factor': ef,
                    'residual_connection': False,
                })

    # NeuMiss+ Variant D (polynomial interaction)
    for act in ['relu', 'gelu']:
        for depth in [3, 5]:
            configs.append({
                'name': f'NeuMiss+D_{act}_d{depth}',
                'variant': 'D',
                'depth': depth,
                'activation': act,
                'degree': 2,
                'residual_connection': False,
            })

    # With residual connections (select variants)
    for variant in ['A', 'B']:
        configs.append({
            'name': f'NeuMiss+{variant}_gelu_d5_res',
            'variant': variant,
            'depth': 5,
            'activation': 'gelu',
            'residual_connection': True,
        })

    return configs


def get_compact_method_configs():
    """Smaller set of methods for faster iteration."""
    configs = []

    # Baseline
    for depth in [1, 3, 5]:
        configs.append({
            'name': f'NeuMiss_d{depth}',
            'variant': 'original',
            'depth': depth,
            'activation': 'relu',
            'residual_connection': False,
        })

    # Best candidates from each variant
    for variant, act in [('A', 'gelu'), ('A', 'relu'),
                          ('B', 'gelu'), ('B', 'relu'),
                          ('C', 'gelu'), ('D', 'gelu')]:
        depth = 3
        cfg = {
            'name': f'NeuMiss+{variant}_{act}_d{depth}',
            'variant': variant,
            'depth': depth,
            'activation': act,
            'residual_connection': False,
        }
        if variant == 'C':
            cfg['expansion_factor'] = 2
        if variant == 'D':
            cfg['degree'] = 2
        configs.append(cfg)

    return configs


def run_single_experiment(scenario, method_config, n_train=10000,
                          n_val=2000, n_test=5000, n_epochs=200,
                          batch_size=64, random_state=0):
    """Run a single experiment: one scenario + one method."""
    result = {
        'scenario': scenario.name,
        'distribution': scenario.distribution,
        'response': scenario.response,
        'missing_mechanism': scenario.missing_mechanism,
        'method': method_config['name'],
        'variant': method_config['variant'],
        'random_state': random_state,
    }

    try:
        # Generate data
        data = scenario.generate(n_train, n_val, n_test,
                                 random_state=random_state)

        # Build estimator
        est_params = {
            'variant': method_config['variant'],
            'depth': method_config['depth'],
            'activation': method_config.get('activation', 'relu'),
            'residual_connection': method_config.get('residual_connection', False),
            'expansion_factor': method_config.get('expansion_factor', 2),
            'degree': method_config.get('degree', 2),
            'n_epochs': n_epochs,
            'batch_size': batch_size,
            'lr': 0.001,
            'early_stopping': True,
            'verbose': False,
        }
        est = NeuMissPlus(**est_params)

        # Fit
        t0 = time.time()
        est.fit(data['X_train'], data['y_train'],
                X_val=data['X_val'], y_val=data['y_val'])
        fit_time = time.time() - t0

        # Evaluate
        pred_test = est.predict(data['X_test'])
        pred_train = est.predict(data['X_train'])

        mse_test = np.mean((data['y_test'] - pred_test) ** 2)
        mse_train = np.mean((data['y_train'] - pred_train) ** 2)

        var_test = np.var(data['y_test'])
        var_train = np.var(data['y_train'])

        r2_test = 1 - mse_test / var_test if var_test > 0 else float('nan')
        r2_train = 1 - mse_train / var_train if var_train > 0 else float('nan')

        result.update({
            'mse_test': mse_test,
            'mse_train': mse_train,
            'r2_test': r2_test,
            'r2_train': r2_train,
            'fit_time': fit_time,
            'n_epochs_actual': len(est.train_losses),
            'status': 'success',
        })

    except Exception as e:
        result.update({
            'mse_test': float('nan'),
            'mse_train': float('nan'),
            'r2_test': float('nan'),
            'r2_train': float('nan'),
            'fit_time': float('nan'),
            'n_epochs_actual': 0,
            'status': f'error: {str(e)}',
        })
        traceback.print_exc()

    return result


def run_experiment_suite(scenarios, method_configs, n_train=10000,
                         n_val=2000, n_test=5000, n_epochs=200,
                         batch_size=64, n_repeats=3, output_file=None):
    """Run all scenarios x methods x repeats."""
    results = []
    total = len(scenarios) * len(method_configs) * n_repeats
    done = 0

    for scenario in scenarios:
        for method_config in method_configs:
            for seed in range(n_repeats):
                done += 1
                print(f"\n[{done}/{total}] {scenario.name} | "
                      f"{method_config['name']} | seed={seed}")

                result = run_single_experiment(
                    scenario, method_config,
                    n_train=n_train, n_val=n_val, n_test=n_test,
                    n_epochs=n_epochs, batch_size=batch_size,
                    random_state=seed)
                results.append(result)

                if result['status'] == 'success':
                    print(f"  R2_test={result['r2_test']:.4f}, "
                          f"MSE_test={result['mse_test']:.4f}, "
                          f"time={result['fit_time']:.1f}s")
                else:
                    print(f"  FAILED: {result['status']}")

    df = pd.DataFrame(results)

    if output_file:
        df.to_csv(output_file, index=False)
        print(f"\nResults saved to {output_file}")

    return df


def analyze_results(df):
    """Analyze experiment results and produce summary."""
    print("\n" + "=" * 80)
    print("EXPERIMENT RESULTS ANALYSIS")
    print("=" * 80)

    # Filter successful runs
    df_ok = df[df['status'] == 'success'].copy()
    if len(df_ok) == 0:
        print("No successful experiments!")
        return

    # Aggregate over seeds
    agg = df_ok.groupby(['scenario', 'method']).agg({
        'r2_test': ['mean', 'std'],
        'mse_test': ['mean', 'std'],
        'fit_time': 'mean',
    }).round(4)
    agg.columns = ['r2_mean', 'r2_std', 'mse_mean', 'mse_std', 'time_mean']
    agg = agg.reset_index()

    # Best method per scenario
    print("\n--- Best Method per Scenario (by R2 test) ---")
    for scenario in agg['scenario'].unique():
        sub = agg[agg['scenario'] == scenario].sort_values('r2_mean',
                                                            ascending=False)
        best = sub.iloc[0]
        print(f"\n{scenario}:")
        print(f"  Best: {best['method']} | R2={best['r2_mean']:.4f} "
              f"(+/-{best['r2_std']:.4f})")

        # Also show baseline
        baseline = sub[sub['method'].str.startswith('NeuMiss_')]
        if len(baseline) > 0:
            best_base = baseline.iloc[0]
            print(f"  Baseline: {best_base['method']} | "
                  f"R2={best_base['r2_mean']:.4f}")
            improvement = best['r2_mean'] - best_base['r2_mean']
            print(f"  Improvement: {improvement:+.4f}")

    # Overall comparison: NeuMiss vs NeuMiss+
    print("\n--- NeuMiss vs NeuMiss+ Overall ---")
    neumiss_mask = agg['method'].str.startswith('NeuMiss_')
    plus_mask = agg['method'].str.contains(r'NeuMiss\+')

    if neumiss_mask.any() and plus_mask.any():
        avg_baseline = agg[neumiss_mask].groupby('scenario')['r2_mean'].max()
        avg_plus = agg[plus_mask].groupby('scenario')['r2_mean'].max()
        comparison = pd.DataFrame({
            'baseline_best': avg_baseline,
            'plus_best': avg_plus,
        })
        comparison['improvement'] = comparison['plus_best'] - comparison['baseline_best']
        print(comparison.to_string())
        print(f"\nAvg improvement: {comparison['improvement'].mean():.4f}")

    # By scenario type
    print("\n--- Performance by Distribution Type ---")
    dist_perf = df_ok.groupby(['distribution', 'method'])['r2_test'].mean()
    for dist in df_ok['distribution'].unique():
        print(f"\n{dist}:")
        sub = dist_perf[dist].sort_values(ascending=False)
        for method, r2 in sub.head(5).items():
            print(f"  {method}: R2={r2:.4f}")

    print("\n--- Performance by Response Type ---")
    resp_perf = df_ok.groupby(['response', 'method'])['r2_test'].mean()
    for resp in df_ok['response'].unique():
        print(f"\n{resp}:")
        sub = resp_perf[resp].sort_values(ascending=False)
        for method, r2 in sub.head(5).items():
            print(f"  {method}: R2={r2:.4f}")

    return agg


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', default='compact',
                        choices=['compact', 'full', 'key'])
    parser.add_argument('--n_train', type=int, default=10000)
    parser.add_argument('--n_epochs', type=int, default=200)
    parser.add_argument('--n_repeats', type=int, default=3)
    parser.add_argument('--output', default='../results/experiment_results.csv')
    args = parser.parse_args()

    if args.mode == 'compact':
        scenarios = get_key_scenarios()[:5]  # Start with fewer scenarios
        methods = get_compact_method_configs()
    elif args.mode == 'key':
        scenarios = get_key_scenarios()
        methods = get_compact_method_configs()
    else:
        scenarios = get_all_scenarios()
        methods = get_method_configs()

    df = run_experiment_suite(
        scenarios, methods,
        n_train=args.n_train, n_epochs=args.n_epochs,
        n_repeats=args.n_repeats, output_file=args.output)

    analyze_results(df)

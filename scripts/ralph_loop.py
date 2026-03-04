"""
Ralph Loop: Iterative Research-Algorithm-Learn-Prove-Hypothesis loop.

This implements the iterative cycle:
1. Hypothesize: Based on current results, form a hypothesis about what works
2. Modify: Implement algorithmic changes based on hypothesis
3. Experiment: Run targeted experiments
4. Analyze: Compare results, identify improvements
5. Iterate: Update hypothesis and repeat

The loop tracks all iterations and builds a knowledge base of what works.
"""

import sys
sys.path.insert(0, '/Users/yukang/Desktop/NeuroMiss/src')

import numpy as np
import pandas as pd
import json
from datetime import datetime

from data_generation import DataScenario
from neumiss_plus import NeuMissPlus
from experiment_runner import run_single_experiment


class RalphLoop:
    """Research-Algorithm-Learn-Prove-Hypothesis iterative improvement loop."""

    def __init__(self, log_dir='/Users/yukang/Desktop/NeuroMiss/results'):
        self.log_dir = log_dir
        self.iterations = []
        self.best_configs = {}  # scenario -> best method config
        self.knowledge_base = []

    def run_iteration(self, iteration_name, hypothesis, scenarios, methods,
                      n_train=10000, n_val=2000, n_test=5000,
                      n_epochs=150, n_repeats=3):
        """Run one iteration of the Ralph loop."""
        print(f"\n{'='*70}")
        print(f"RALPH LOOP ITERATION: {iteration_name}")
        print(f"Hypothesis: {hypothesis}")
        print(f"{'='*70}")

        results = []
        total = len(scenarios) * len(methods) * n_repeats

        for i, scenario in enumerate(scenarios):
            for j, method in enumerate(methods):
                for seed in range(n_repeats):
                    idx = i * len(methods) * n_repeats + j * n_repeats + seed + 1
                    print(f"  [{idx}/{total}] {scenario.name} | {method['name']} | seed={seed}")

                    result = run_single_experiment(
                        scenario, method,
                        n_train=n_train, n_val=n_val, n_test=n_test,
                        n_epochs=n_epochs, random_state=seed)
                    results.append(result)

                    if result['status'] == 'success':
                        print(f"    R2={result['r2_test']:.4f}")

        df = pd.DataFrame(results)

        # Analyze this iteration
        analysis = self._analyze_iteration(df)

        # Record iteration
        iteration_record = {
            'name': iteration_name,
            'hypothesis': hypothesis,
            'timestamp': datetime.now().isoformat(),
            'n_experiments': len(results),
            'analysis': analysis,
        }
        self.iterations.append(iteration_record)

        # Update best configs
        self._update_best_configs(df)

        # Save results
        df.to_csv(f"{self.log_dir}/ralph_{iteration_name}.csv", index=False)

        return df, analysis

    def _analyze_iteration(self, df):
        """Analyze results from one iteration."""
        df_ok = df[df['status'] == 'success']
        if len(df_ok) == 0:
            return {'status': 'all_failed'}

        agg = df_ok.groupby(['scenario', 'method']).agg({
            'r2_test': ['mean', 'std'],
            'mse_test': 'mean',
        }).reset_index()
        agg.columns = ['scenario', 'method', 'r2_mean', 'r2_std', 'mse_mean']

        # Find best method per scenario
        best_per_scenario = {}
        improvements = {}
        for scenario in agg['scenario'].unique():
            sub = agg[agg['scenario'] == scenario].sort_values('r2_mean', ascending=False)
            best = sub.iloc[0]
            best_per_scenario[scenario] = {
                'method': best['method'],
                'r2': float(best['r2_mean']),
                'r2_std': float(best['r2_std']),
            }

            # Compare NeuMiss+ vs baseline
            baseline = sub[sub['method'].str.startswith('NeuMiss_')]
            plus = sub[sub['method'].str.contains(r'\+')]
            if len(baseline) > 0 and len(plus) > 0:
                improvements[scenario] = float(plus.iloc[0]['r2_mean'] - baseline.iloc[0]['r2_mean'])

        analysis = {
            'best_per_scenario': best_per_scenario,
            'improvements': improvements,
            'avg_improvement': np.mean(list(improvements.values())) if improvements else 0,
            'overall_best_methods': agg.groupby('method')['r2_mean'].mean().sort_values(ascending=False).head(5).to_dict(),
        }

        print(f"\n--- Iteration Analysis ---")
        print(f"Best per scenario:")
        for s, info in best_per_scenario.items():
            print(f"  {s}: {info['method']} (R2={info['r2']:.4f})")
        print(f"Average improvement over baseline: {analysis['avg_improvement']:.4f}")
        print(f"Overall best methods:")
        for m, r2 in analysis['overall_best_methods'].items():
            print(f"  {m}: avg R2={r2:.4f}")

        return analysis

    def _update_best_configs(self, df):
        """Track best configurations across iterations."""
        df_ok = df[df['status'] == 'success']
        for scenario in df_ok['scenario'].unique():
            sub = df_ok[df_ok['scenario'] == scenario]
            best_idx = sub.groupby('method')['r2_test'].mean().idxmax()
            best_r2 = sub[sub['method'] == best_idx]['r2_test'].mean()

            if scenario not in self.best_configs or best_r2 > self.best_configs[scenario]['r2']:
                self.best_configs[scenario] = {
                    'method': best_idx,
                    'r2': float(best_r2),
                }

    def get_insights(self):
        """Get accumulated insights from all iterations."""
        print("\n" + "=" * 70)
        print("RALPH LOOP: ACCUMULATED INSIGHTS")
        print("=" * 70)

        print(f"\nTotal iterations: {len(self.iterations)}")

        print("\nBest known configs per scenario:")
        for scenario, info in sorted(self.best_configs.items()):
            print(f"  {scenario}: {info['method']} (R2={info['r2']:.4f})")

        # Track improvement over iterations
        if len(self.iterations) > 1:
            print("\nImprovement trajectory:")
            for it in self.iterations:
                print(f"  {it['name']}: avg_improvement={it['analysis'].get('avg_improvement', 'N/A')}")


def run_ralph_iteration_1():
    """First Ralph iteration: Test basic hypothesis that activations help."""
    loop = RalphLoop()

    scenarios = [
        DataScenario('gaussian', 'linear', 'MCAR', n_features=10, missing_rate=0.5, snr=10),
        DataScenario('gaussian', 'quadratic', 'MCAR', n_features=10, missing_rate=0.5, snr=10),
        DataScenario('mixture_gaussian', 'linear', 'MCAR', n_features=10, missing_rate=0.5, snr=10,
                     distribution_params={'n_components': 3}),
        DataScenario('mixture_gaussian', 'cubic', 'MCAR', n_features=10, missing_rate=0.5, snr=10,
                     distribution_params={'n_components': 3}),
    ]

    methods = [
        {'name': 'NeuMiss_d3', 'variant': 'original', 'depth': 3, 'activation': 'relu', 'residual_connection': False},
        {'name': 'NeuMiss+A_gelu_d3', 'variant': 'A', 'depth': 3, 'activation': 'gelu', 'residual_connection': False},
        {'name': 'NeuMiss+B_gelu_d3', 'variant': 'B', 'depth': 3, 'activation': 'gelu', 'residual_connection': False},
        {'name': 'NeuMiss+C_gelu_d3', 'variant': 'C', 'depth': 3, 'activation': 'gelu', 'residual_connection': False, 'expansion_factor': 2},
        {'name': 'NeuMiss+D_gelu_d3', 'variant': 'D', 'depth': 3, 'activation': 'gelu', 'residual_connection': False, 'degree': 2},
    ]

    df, analysis = loop.run_iteration(
        'iter1_basic_hypothesis',
        'Adding activations between mask multiplications should improve performance on nonlinear/non-normal scenarios',
        scenarios, methods,
        n_train=10000, n_val=2000, n_test=5000, n_epochs=150, n_repeats=3)

    loop.get_insights()
    return loop, df, analysis


if __name__ == '__main__':
    loop, df, analysis = run_ralph_iteration_1()

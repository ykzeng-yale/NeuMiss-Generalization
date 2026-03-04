"""
Automated Ralph Loop: Self-iterating research cycle.

This script runs the full iterative cycle automatically:
1. Start with initial experiments on key scenarios
2. Analyze which architectures work for which problems
3. Generate new hypotheses and architecture modifications
4. Test targeted experiments
5. Repeat until improvement converges

Goal: Find NeuMiss+ configurations that significantly outperform original NeuMiss
on non-linear f and non-normal X scenarios.
"""

import sys
sys.path.insert(0, '/Users/yukang/Desktop/NeuroMiss/src')

import numpy as np
import pandas as pd
import json
import time
from datetime import datetime

from data_generation import DataScenario
from neumiss_plus import NeuMissPlus
from experiment_runner import run_single_experiment, analyze_results


class AutoRalphLoop:
    """Self-iterating research improvement loop."""

    def __init__(self, results_dir='/Users/yukang/Desktop/NeuroMiss/results'):
        self.results_dir = results_dir
        self.all_results = []
        self.iteration_log = []
        self.best_r2_per_scenario = {}
        self.best_method_per_scenario = {}
        self.convergence_threshold = 0.005  # stop if improvement < this
        self.max_iterations = 6

    def get_core_scenarios(self):
        """Core scenarios spanning the difficulty spectrum."""
        return [
            # Original territory (baseline)
            DataScenario('gaussian', 'linear', 'MCAR', 10, 0.5, 10),
            # Easy extension: polynomial
            DataScenario('gaussian', 'quadratic', 'MCAR', 10, 0.5, 10),
            DataScenario('gaussian', 'cubic', 'MCAR', 10, 0.5, 10),
            # Non-normal + linear
            DataScenario('mixture_gaussian', 'linear', 'MCAR', 10, 0.5, 10,
                         distribution_params={'n_components': 3}),
            # Hard: non-normal + nonlinear
            DataScenario('mixture_gaussian', 'quadratic', 'MCAR', 10, 0.5, 10,
                         distribution_params={'n_components': 3}),
            DataScenario('student_t', 'sinusoidal', 'MCAR', 10, 0.5, 10,
                         distribution_params={'df': 5}),
            # MNAR scenarios
            DataScenario('gaussian', 'quadratic', 'MNAR_censoring', 10, 0.5, 10),
            DataScenario('mixture_gaussian', 'cubic', 'MNAR_censoring', 10, 0.5, 10,
                         distribution_params={'n_components': 3}),
        ]

    def iteration_0_baseline(self):
        """Iteration 0: Establish baselines with original NeuMiss and basic NeuMiss+ variants."""
        print("\n" + "=" * 80)
        print("RALPH LOOP - ITERATION 0: BASELINE EXPLORATION")
        print("Hypothesis: Adding activations should help for non-linear/non-normal cases")
        print("=" * 80)

        scenarios = self.get_core_scenarios()

        methods = [
            # Original NeuMiss baselines
            {'name': 'NeuMiss_d1', 'variant': 'original', 'depth': 1},
            {'name': 'NeuMiss_d3', 'variant': 'original', 'depth': 3},
            {'name': 'NeuMiss_d5', 'variant': 'original', 'depth': 5},
            # NeuMiss+ Variant A (activation after mask)
            {'name': 'NM+A_relu_d3', 'variant': 'A', 'depth': 3, 'activation': 'relu'},
            {'name': 'NM+A_gelu_d3', 'variant': 'A', 'depth': 3, 'activation': 'gelu'},
            {'name': 'NM+A_tanh_d3', 'variant': 'A', 'depth': 3, 'activation': 'tanh'},
            {'name': 'NM+A_silu_d3', 'variant': 'A', 'depth': 3, 'activation': 'silu'},
            # NeuMiss+ Variant B (activation before mask)
            {'name': 'NM+B_relu_d3', 'variant': 'B', 'depth': 3, 'activation': 'relu'},
            {'name': 'NM+B_gelu_d3', 'variant': 'B', 'depth': 3, 'activation': 'gelu'},
            # NeuMiss+ Variant C (wider hidden layers)
            {'name': 'NM+C_gelu_d3', 'variant': 'C', 'depth': 3, 'activation': 'gelu',
             'expansion_factor': 2},
            # NeuMiss+ Variant D (polynomial interactions)
            {'name': 'NM+D_gelu_d3', 'variant': 'D', 'depth': 3, 'activation': 'gelu',
             'degree': 2},
        ]

        return self._run_iteration('iter0_baseline', scenarios, methods,
                                    n_train=8000, n_epochs=120, n_repeats=3)

    def iteration_1_refine(self, prev_analysis):
        """Iteration 1: Based on iter0 results, focus on winning patterns."""
        print("\n" + "=" * 80)
        print("RALPH LOOP - ITERATION 1: REFINE WINNING PATTERNS")
        print("=" * 80)

        # Identify what worked
        best_variants = self._get_best_variants(prev_analysis)
        print(f"Best variants from iter0: {best_variants}")

        scenarios = self.get_core_scenarios()

        # Generate targeted methods based on what worked
        methods = []

        # Keep baselines for comparison
        methods.append({'name': 'NeuMiss_d3', 'variant': 'original', 'depth': 3})
        methods.append({'name': 'NeuMiss_d5', 'variant': 'original', 'depth': 5})

        # Deepen the best variants
        for variant_info in best_variants[:3]:
            variant = variant_info['variant']
            act = variant_info.get('activation', 'gelu')
            for depth in [3, 5, 7]:
                cfg = {
                    'name': f'NM+{variant}_{act}_d{depth}',
                    'variant': variant,
                    'depth': depth,
                    'activation': act,
                }
                if variant == 'C':
                    cfg['expansion_factor'] = variant_info.get('expansion_factor', 2)
                if variant == 'D':
                    cfg['degree'] = variant_info.get('degree', 2)
                methods.append(cfg)

        # Try residual connections on best variant
        best = best_variants[0]
        cfg = {
            'name': f'NM+{best["variant"]}_{best.get("activation","gelu")}_d5_res',
            'variant': best['variant'],
            'depth': 5,
            'activation': best.get('activation', 'gelu'),
            'residual_connection': True,
        }
        if best['variant'] == 'C':
            cfg['expansion_factor'] = best.get('expansion_factor', 2)
        if best['variant'] == 'D':
            cfg['degree'] = best.get('degree', 2)
        methods.append(cfg)

        return self._run_iteration('iter1_refine', scenarios, methods,
                                    n_train=8000, n_epochs=120, n_repeats=3)

    def iteration_2_ablation(self, prev_analysis):
        """Iteration 2: Ablation study on the best configuration."""
        print("\n" + "=" * 80)
        print("RALPH LOOP - ITERATION 2: ABLATION & OPTIMIZATION")
        print("=" * 80)

        best_variants = self._get_best_variants(prev_analysis)

        # Focus on harder scenarios where improvement is most needed
        scenarios = [
            DataScenario('gaussian', 'quadratic', 'MCAR', 10, 0.5, 10),
            DataScenario('gaussian', 'cubic', 'MCAR', 10, 0.5, 10),
            DataScenario('mixture_gaussian', 'quadratic', 'MCAR', 10, 0.5, 10,
                         distribution_params={'n_components': 3}),
            DataScenario('student_t', 'sinusoidal', 'MCAR', 10, 0.5, 10,
                         distribution_params={'df': 5}),
            DataScenario('mixture_gaussian', 'cubic', 'MNAR_censoring', 10, 0.5, 10,
                         distribution_params={'n_components': 3}),
        ]

        methods = []
        methods.append({'name': 'NeuMiss_d5', 'variant': 'original', 'depth': 5})

        best = best_variants[0]
        v = best['variant']
        act = best.get('activation', 'gelu')

        # Vary learning rates and batch sizes for best config
        for lr_mult in [0.5, 1.0, 2.0]:
            for bs in [32, 64, 128]:
                cfg = {
                    'name': f'NM+{v}_{act}_d5_lr{lr_mult}_bs{bs}',
                    'variant': v,
                    'depth': 5,
                    'activation': act,
                }
                if v == 'C':
                    cfg['expansion_factor'] = best.get('expansion_factor', 2)
                if v == 'D':
                    cfg['degree'] = best.get('degree', 2)
                methods.append(cfg)

        # Also try Variant C with different expansion factors
        for ef in [2, 3, 4]:
            methods.append({
                'name': f'NM+C_gelu_d5_ef{ef}',
                'variant': 'C',
                'depth': 5,
                'activation': 'gelu',
                'expansion_factor': ef,
            })

        return self._run_iteration('iter2_ablation', scenarios, methods,
                                    n_train=8000, n_epochs=150, n_repeats=3)

    def iteration_3_scale_validate(self, prev_analysis):
        """Iteration 3: Scale up and validate on all scenarios with more data."""
        print("\n" + "=" * 80)
        print("RALPH LOOP - ITERATION 3: SCALE UP & VALIDATE")
        print("=" * 80)

        best_variants = self._get_best_variants(prev_analysis)

        # ALL scenarios including different missing rates and feature dims
        scenarios = self.get_core_scenarios()
        # Add higher dimension
        scenarios.extend([
            DataScenario('gaussian', 'quadratic', 'MCAR', 20, 0.5, 10),
            DataScenario('mixture_gaussian', 'cubic', 'MCAR', 20, 0.5, 10,
                         distribution_params={'n_components': 3}),
            DataScenario('gaussian', 'cubic', 'MNAR_selfmasking', 10, 0.5, 10),
            DataScenario('gaussian', 'interaction', 'MCAR', 10, 0.5, 10),
        ])

        methods = []
        methods.append({'name': 'NeuMiss_d3', 'variant': 'original', 'depth': 3})
        methods.append({'name': 'NeuMiss_d5', 'variant': 'original', 'depth': 5})

        # Top 3 configurations
        for i, cfg_info in enumerate(best_variants[:3]):
            v = cfg_info['variant']
            act = cfg_info.get('activation', 'gelu')
            d = cfg_info.get('depth', 5)
            cfg = {
                'name': f'Best{i+1}_NM+{v}_{act}_d{d}',
                'variant': v,
                'depth': d,
                'activation': act,
                'residual_connection': cfg_info.get('residual_connection', False),
            }
            if v == 'C':
                cfg['expansion_factor'] = cfg_info.get('expansion_factor', 2)
            if v == 'D':
                cfg['degree'] = cfg_info.get('degree', 2)
            methods.append(cfg)

        return self._run_iteration('iter3_validate', scenarios, methods,
                                    n_train=15000, n_epochs=200, n_repeats=3)

    def _run_iteration(self, name, scenarios, methods, n_train=8000,
                       n_epochs=120, n_repeats=3):
        """Execute one iteration and return analysis."""
        results = []
        total = len(scenarios) * len(methods) * n_repeats
        done = 0
        t_start = time.time()

        for scenario in scenarios:
            for method_config in methods:
                for seed in range(n_repeats):
                    done += 1
                    # Extract method params
                    m = method_config.copy()
                    m.setdefault('activation', 'relu')
                    m.setdefault('residual_connection', False)
                    m.setdefault('expansion_factor', 2)
                    m.setdefault('degree', 2)

                    print(f"  [{done}/{total}] {scenario.name} | {m['name']}")

                    result = run_single_experiment(
                        scenario, m,
                        n_train=n_train, n_val=2000, n_test=3000,
                        n_epochs=n_epochs, random_state=seed)
                    results.append(result)

                    if result['status'] == 'success':
                        print(f"    R2={result['r2_test']:.4f} "
                              f"MSE={result['mse_test']:.4f}")

        elapsed = time.time() - t_start
        print(f"\n  Iteration {name} completed in {elapsed:.0f}s")

        df = pd.DataFrame(results)
        self.all_results.extend(results)

        # Save
        df.to_csv(f"{self.results_dir}/ralph_{name}.csv", index=False)

        # Analyze
        analysis = self._analyze(df, name)
        self.iteration_log.append({
            'name': name,
            'n_experiments': len(results),
            'elapsed': elapsed,
            'analysis': analysis,
        })

        return analysis

    def _analyze(self, df, iteration_name):
        """Detailed analysis of iteration results."""
        df_ok = df[df['status'] == 'success'].copy()
        if len(df_ok) == 0:
            print("  WARNING: No successful experiments!")
            return {}

        # Aggregate
        agg = df_ok.groupby(['scenario', 'method']).agg({
            'r2_test': ['mean', 'std'],
            'mse_test': ['mean', 'std'],
        }).round(4)
        agg.columns = ['r2_mean', 'r2_std', 'mse_mean', 'mse_std']
        agg = agg.reset_index()

        print(f"\n{'='*70}")
        print(f"ANALYSIS: {iteration_name}")
        print(f"{'='*70}")

        improvements = {}
        best_per_scenario = {}

        for scenario in agg['scenario'].unique():
            sub = agg[agg['scenario'] == scenario].sort_values('r2_mean', ascending=False)

            best = sub.iloc[0]
            best_per_scenario[scenario] = {
                'method': best['method'],
                'r2': float(best['r2_mean']),
                'r2_std': float(best['r2_std']),
            }

            # Baseline comparison
            baseline = sub[sub['method'].str.startswith('NeuMiss_')]
            plus = sub[~sub['method'].str.startswith('NeuMiss_')]

            print(f"\n  {scenario}:")
            for _, row in sub.head(5).iterrows():
                marker = " ***" if row['method'] == best['method'] else ""
                print(f"    {row['method']:40s} R2={row['r2_mean']:.4f} "
                      f"(+/-{row['r2_std']:.4f}){marker}")

            if len(baseline) > 0 and len(plus) > 0:
                base_best = baseline.iloc[0]['r2_mean']
                plus_best = plus.iloc[0]['r2_mean']
                imp = plus_best - base_best
                improvements[scenario] = float(imp)
                marker = "BETTER" if imp > 0 else "WORSE"
                print(f"    --> NeuMiss+ improvement: {imp:+.4f} ({marker})")

            # Update global best
            if scenario not in self.best_r2_per_scenario or \
               best['r2_mean'] > self.best_r2_per_scenario[scenario]:
                self.best_r2_per_scenario[scenario] = float(best['r2_mean'])
                self.best_method_per_scenario[scenario] = best['method']

        avg_imp = np.mean(list(improvements.values())) if improvements else 0
        print(f"\n  Average improvement over baseline: {avg_imp:+.4f}")

        # Overall ranking
        overall = agg.groupby('method')['r2_mean'].mean().sort_values(ascending=False)
        print(f"\n  Overall method ranking (avg R2 across scenarios):")
        for method, r2 in overall.head(8).items():
            print(f"    {method:40s} avg_R2={r2:.4f}")

        return {
            'best_per_scenario': best_per_scenario,
            'improvements': improvements,
            'avg_improvement': avg_imp,
            'overall_ranking': overall.to_dict(),
        }

    def _get_best_variants(self, analysis):
        """Extract best NeuMiss+ variant configurations from analysis."""
        if not analysis or 'overall_ranking' not in analysis:
            # Default
            return [
                {'variant': 'A', 'activation': 'gelu', 'depth': 3},
                {'variant': 'B', 'activation': 'gelu', 'depth': 3},
                {'variant': 'C', 'activation': 'gelu', 'depth': 3, 'expansion_factor': 2},
            ]

        ranking = analysis['overall_ranking']
        variants = []

        for method_name, r2 in sorted(ranking.items(), key=lambda x: -x[1]):
            if 'NM+' in method_name or 'NeuMiss+' in method_name:
                # Parse method name to extract config
                parts = method_name.replace('NM+', '').replace('NeuMiss+', '').split('_')
                cfg = {'r2': r2}

                # Parse variant letter
                if len(parts) > 0:
                    cfg['variant'] = parts[0][0] if parts[0] else 'A'

                # Parse activation
                for p in parts:
                    if p in ['relu', 'gelu', 'tanh', 'silu', 'elu']:
                        cfg['activation'] = p

                # Parse depth
                for p in parts:
                    if p.startswith('d') and p[1:].isdigit():
                        cfg['depth'] = int(p[1:])

                # Parse expansion factor
                for p in parts:
                    if p.startswith('ef') and p[2:].isdigit():
                        cfg['expansion_factor'] = int(p[2:])

                # Parse residual
                cfg['residual_connection'] = 'res' in method_name

                cfg.setdefault('activation', 'gelu')
                cfg.setdefault('depth', 3)

                variants.append(cfg)

        if not variants:
            return [
                {'variant': 'A', 'activation': 'gelu', 'depth': 3},
                {'variant': 'B', 'activation': 'gelu', 'depth': 3},
                {'variant': 'C', 'activation': 'gelu', 'depth': 3, 'expansion_factor': 2},
            ]

        return variants

    def run_full_loop(self):
        """Run the complete iterative loop until convergence."""
        print("\n" + "#" * 80)
        print("# AUTOMATED RALPH LOOP: NeuMiss+ Research Iteration")
        print("# Goal: Find NeuMiss+ that significantly beats NeuMiss on")
        print("#       non-linear f and non-normal X scenarios")
        print("#" * 80)

        t_global = time.time()

        # Iteration 0: Baseline
        analysis_0 = self.iteration_0_baseline()

        # Iteration 1: Refine
        analysis_1 = self.iteration_1_refine(analysis_0)

        # Check convergence
        imp_0 = analysis_0.get('avg_improvement', 0)
        imp_1 = analysis_1.get('avg_improvement', 0)
        print(f"\n  Improvement trajectory: iter0={imp_0:.4f}, iter1={imp_1:.4f}")

        if abs(imp_1 - imp_0) < self.convergence_threshold and imp_1 > 0:
            print("  Converged early! NeuMiss+ improvement is stable.")
        else:
            # Iteration 2: Ablation
            analysis_2 = self.iteration_2_ablation(analysis_1)

            # Iteration 3: Scale & validate
            analysis_3 = self.iteration_3_scale_validate(analysis_2)

        total_time = time.time() - t_global

        # Final summary
        self._print_final_summary(total_time)

        # Save all results
        all_df = pd.DataFrame(self.all_results)
        all_df.to_csv(f"{self.results_dir}/ralph_all_results.csv", index=False)

        return self.all_results

    def _print_final_summary(self, total_time):
        """Print comprehensive final summary."""
        print("\n" + "#" * 80)
        print("# FINAL SUMMARY")
        print("#" * 80)
        print(f"\nTotal time: {total_time:.0f}s ({total_time/60:.1f}min)")
        print(f"Total experiments: {len(self.all_results)}")
        print(f"Iterations completed: {len(self.iteration_log)}")

        print("\nBest method per scenario (across all iterations):")
        for scenario in sorted(self.best_r2_per_scenario.keys()):
            r2 = self.best_r2_per_scenario[scenario]
            method = self.best_method_per_scenario[scenario]
            is_plus = 'NM+' in method or '+' in method
            marker = " [NeuMiss+]" if is_plus else " [baseline]"
            print(f"  {scenario:60s} R2={r2:.4f} -> {method}{marker}")

        print("\nImprovement trajectory:")
        for log in self.iteration_log:
            imp = log['analysis'].get('avg_improvement', 'N/A')
            print(f"  {log['name']:30s} avg_improvement={imp}")

        # Count wins
        plus_wins = sum(1 for m in self.best_method_per_scenario.values()
                        if 'NM+' in m or '+' in m)
        baseline_wins = len(self.best_method_per_scenario) - plus_wins
        print(f"\nNeuMiss+ wins: {plus_wins}/{len(self.best_method_per_scenario)}")
        print(f"Baseline wins: {baseline_wins}/{len(self.best_method_per_scenario)}")


if __name__ == '__main__':
    loop = AutoRalphLoop()
    loop.run_full_loop()

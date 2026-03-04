"""Real-world dataset benchmark for NeuMiss+ architectures.

Validates NeuMiss+ beyond synthetic data using real regression datasets with
artificially introduced missing values (MCAR and MAR mechanisms) at various
rates. Compares all major NeuMiss+ variants against traditional baselines
using 5-fold cross-validation with statistical significance testing.

Datasets:
  1. California Housing (sklearn) - 20640 samples, 8 features
  2. Diabetes (sklearn) - 442 samples, 10 features (small sample regime)
  3. Ames Housing (OpenML) - ~1460 samples, numeric features
  4. Wine Quality (OpenML) - ~6497 samples, 11 features

Methods:
  - NeuMiss Original (depth 3, 5)
  - NeuMiss+C (best single-stage variant)
  - PretrainEncoder (best two-stage architecture)
  - ImputeMLP (baseline)
  - IterativeImputer + Ridge (traditional ML baseline)
  - IterativeImputer + RandomForest (traditional ML baseline)
"""
import sys
sys.path.insert(0, '/Users/yukang/Desktop/NeuroMiss/src')

import os
import warnings
import numpy as np
import pandas as pd
from itertools import product
from collections import defaultdict

import torch
from sklearn.datasets import (
    fetch_california_housing, load_diabetes, load_wine
)
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score
from scipy import stats

from neumiss_plus import NeuMissPlus, PretrainEncoder, ImputeMLP

warnings.filterwarnings('ignore')

# Reproducibility
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

# ============================================================================
# Output directory
# ============================================================================
RESULTS_DIR = '/Users/yukang/Desktop/NeuroMiss/results'
os.makedirs(RESULTS_DIR, exist_ok=True)

# ============================================================================
# Dataset loading and preparation
# ============================================================================

def load_datasets():
    """Load and standardize all real-world datasets."""
    datasets = {}

    # 1. California Housing
    print("Loading California Housing...")
    cal = fetch_california_housing()
    datasets['CaliforniaHousing'] = {
        'X': cal.data, 'y': cal.target,
        'n_samples': cal.data.shape[0], 'n_features': cal.data.shape[1],
        'description': 'Regression, 20640 samples, 8 features'
    }

    # 2. Diabetes
    print("Loading Diabetes...")
    diab = load_diabetes()
    datasets['Diabetes'] = {
        'X': diab.data, 'y': diab.target,
        'n_samples': diab.data.shape[0], 'n_features': diab.data.shape[1],
        'description': 'Regression, 442 samples, 10 features (small-sample)'
    }

    # 3. Ames Housing (OpenML, numeric features only)
    print("Loading Ames Housing from OpenML...")
    try:
        ames = fetch_openml(name='house_prices', version=1, as_frame=True,
                            parser='auto')
        df = ames.data
        y_ames = ames.target.values.astype(float)
        # Keep only numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        X_ames = df[numeric_cols].values.astype(float)
        # Drop rows with any NaN in original data (we introduce our own)
        valid_mask = ~np.isnan(X_ames).any(axis=1) & ~np.isnan(y_ames)
        X_ames = X_ames[valid_mask]
        y_ames = y_ames[valid_mask]
        datasets['AmesHousing'] = {
            'X': X_ames, 'y': y_ames,
            'n_samples': X_ames.shape[0], 'n_features': X_ames.shape[1],
            'description': f'Regression, {X_ames.shape[0]} samples, {X_ames.shape[1]} numeric features'
        }
    except Exception as e:
        print(f"  Ames Housing failed ({e}), trying alternative...")
        try:
            houses = fetch_openml(data_id=537, as_frame=True, parser='auto')
            df = houses.data
            y_h = houses.target.values.astype(float)
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            X_h = df[numeric_cols].values.astype(float)
            valid_mask = ~np.isnan(X_h).any(axis=1) & ~np.isnan(y_h)
            X_h = X_h[valid_mask]
            y_h = y_h[valid_mask]
            datasets['Houses'] = {
                'X': X_h, 'y': y_h,
                'n_samples': X_h.shape[0], 'n_features': X_h.shape[1],
                'description': f'Regression, {X_h.shape[0]} samples, {X_h.shape[1]} features'
            }
        except Exception as e2:
            print(f"  Alternative also failed ({e2}), skipping.")

    # 4. Wine Quality (classification-turned-regression)
    print("Loading Wine Quality...")
    try:
        wine_data = fetch_openml(name='wine-quality-red', version=1,
                                 as_frame=True, parser='auto')
        X_wine = wine_data.data.values.astype(float)
        y_wine = wine_data.target.values.astype(float)
        valid_mask = ~np.isnan(X_wine).any(axis=1) & ~np.isnan(y_wine)
        X_wine = X_wine[valid_mask]
        y_wine = y_wine[valid_mask]
        datasets['WineQuality'] = {
            'X': X_wine, 'y': y_wine,
            'n_samples': X_wine.shape[0], 'n_features': X_wine.shape[1],
            'description': f'Regression, {X_wine.shape[0]} samples, {X_wine.shape[1]} features'
        }
    except Exception as e:
        print(f"  Wine Quality failed ({e}), using sklearn Wine as fallback...")
        wine = load_wine()
        # Use first target class as a pseudo-regression target via class probabilities
        # Better: use original features, predict alcohol content (feature 0)
        X_wine = wine.data[:, 1:]  # all features except alcohol
        y_wine = wine.data[:, 0]   # predict alcohol
        datasets['WineAlcohol'] = {
            'X': X_wine, 'y': y_wine,
            'n_samples': X_wine.shape[0], 'n_features': X_wine.shape[1],
            'description': f'Regression (predict alcohol), {X_wine.shape[0]} samples, {X_wine.shape[1]} features'
        }

    # Standardize all datasets
    for name, d in datasets.items():
        scaler_X = StandardScaler()
        scaler_y = StandardScaler()
        d['X'] = scaler_X.fit_transform(d['X'])
        d['y'] = scaler_y.fit_transform(d['y'].reshape(-1, 1)).ravel()
        print(f"  {name}: {d['description']}")

    return datasets


# ============================================================================
# Missing value introduction
# ============================================================================

def introduce_mcar(X, p, rng):
    """Introduce Missing Completely At Random (MCAR).
    Each entry is independently masked with probability p."""
    X_miss = X.copy()
    mask = rng.random(X.shape) < p
    X_miss[mask] = np.nan
    # Ensure no row is fully missing (would cause issues)
    fully_missing = np.isnan(X_miss).all(axis=1)
    if fully_missing.any():
        for i in np.where(fully_missing)[0]:
            j = rng.integers(X.shape[1])
            X_miss[i, j] = X[i, j]
    return X_miss


def introduce_mar(X, p, rng):
    """Introduce Missing At Random (MAR).
    Missingness of feature j depends on feature (j-1) % d via a logistic model.
    The overall missing rate is calibrated to approximately p."""
    X_miss = X.copy()
    n, d = X.shape
    for j in range(d):
        # Feature that determines missingness
        predictor_idx = (j - 1) % d
        predictor = X[:, predictor_idx]
        # Logistic probability: higher values of predictor -> more missing
        # Calibrate intercept so that mean probability ~ p
        # logit(p) = intercept + slope * predictor
        slope = 2.0  # moderate dependence
        intercept = np.log(p / (1 - p))  # base rate
        logits = intercept + slope * predictor
        probs = 1.0 / (1.0 + np.exp(-logits))
        mask = rng.random(n) < probs
        X_miss[mask, j] = np.nan
    # Ensure no row is fully missing
    fully_missing = np.isnan(X_miss).all(axis=1)
    if fully_missing.any():
        for i in np.where(fully_missing)[0]:
            j = rng.integers(d)
            X_miss[i, j] = X[i, j]
    return X_miss


# ============================================================================
# Method definitions
# ============================================================================

def get_methods():
    """Return dict of method_name -> constructor lambda."""
    methods = {}

    # 1. NeuMiss Original (depth 3)
    methods['NeuMiss_d3'] = lambda: NeuMissPlus(
        variant='original', depth=3, n_epochs=200, batch_size=64,
        lr=0.001, early_stopping=True, verbose=False
    )

    # 2. NeuMiss Original (depth 5)
    methods['NeuMiss_d5'] = lambda: NeuMissPlus(
        variant='original', depth=5, n_epochs=200, batch_size=64,
        lr=0.001, early_stopping=True, verbose=False
    )

    # 3. NeuMiss+C (best single-stage, with GELU + residual)
    methods['NM+C_gelu'] = lambda: NeuMissPlus(
        variant='C', depth=3, activation='gelu', expansion_factor=3,
        residual_connection=True, n_epochs=200, batch_size=64,
        lr=0.001, early_stopping=True, verbose=False
    )

    # 4. PretrainEncoder (best two-stage architecture)
    methods['PretrainEnc'] = lambda: PretrainEncoder(
        depth=3, mlp_layers=(128,), activation='gelu', dropout=0.1,
        pretrain_epochs=50, train_epochs=200, batch_size=64,
        lr=0.001, weight_decay=1e-5, early_stopping=True, verbose=False
    )

    # 5. ImputeMLP baseline
    methods['ImputeMLP'] = lambda: ImputeMLP(
        hidden_layers=(128, 64, 32), activation='gelu',
        n_epochs=200, batch_size=64, lr=0.001, dropout=0.1,
        early_stopping=True, verbose=False
    )

    # 6. IterativeImputer + Ridge
    methods['IterImp+Ridge'] = lambda: Pipeline([
        ('imputer', IterativeImputer(max_iter=10, random_state=SEED,
                                     sample_posterior=False)),
        ('model', Ridge(alpha=1.0))
    ])

    # 7. IterativeImputer + RandomForest
    methods['IterImp+RF'] = lambda: Pipeline([
        ('imputer', IterativeImputer(max_iter=10, random_state=SEED,
                                     sample_posterior=False)),
        ('model', RandomForestRegressor(n_estimators=100, max_depth=10,
                                        random_state=SEED, n_jobs=-1))
    ])

    return methods


# ============================================================================
# Custom cross-validation for NeuMiss estimators
# ============================================================================

def evaluate_method(method_name, method_constructor, X_miss, y, train_idx,
                    test_idx):
    """Fit and evaluate a method on one fold.

    For NeuMiss-family estimators, we split training into train/val for
    early stopping. For sklearn pipelines, we just fit directly.
    """
    X_train_full, y_train_full = X_miss[train_idx], y[train_idx]
    X_test, y_test = X_miss[test_idx], y[test_idx]

    est = method_constructor()

    if isinstance(est, Pipeline):
        # sklearn pipeline: fit directly
        est.fit(X_train_full, y_train_full)
        pred = est.predict(X_test)
    else:
        # NeuMiss-family: split train into train/val for early stopping
        n_train = len(train_idx)
        n_val = max(int(0.15 * n_train), 50)
        perm = np.random.permutation(n_train)
        val_sub = perm[:n_val]
        train_sub = perm[n_val:]

        X_tr = X_train_full[train_sub]
        y_tr = y_train_full[train_sub]
        X_vl = X_train_full[val_sub]
        y_vl = y_train_full[val_sub]

        est.fit(X_tr, y_tr, X_val=X_vl, y_val=y_vl)
        pred = est.predict(X_test)

    r2 = r2_score(y_test, pred)
    return r2


# ============================================================================
# Main benchmark
# ============================================================================

def run_benchmark():
    print("=" * 80)
    print("REAL-WORLD DATASET BENCHMARK FOR NEUMISS+")
    print("=" * 80)
    print()

    # Load datasets
    datasets = load_datasets()
    print()

    # Settings
    missing_rates = [0.1, 0.3, 0.5]
    missing_mechanisms = ['MCAR', 'MAR']
    n_folds = 5
    methods = get_methods()

    all_results = []

    # Count total experiments
    total_exps = (len(datasets) * len(missing_rates) * len(missing_mechanisms)
                  * len(methods) * n_folds)
    print(f"Total experiments: {total_exps}")
    print(f"Datasets: {list(datasets.keys())}")
    print(f"Missing rates: {missing_rates}")
    print(f"Mechanisms: {missing_mechanisms}")
    print(f"Methods: {list(methods.keys())}")
    print(f"Folds: {n_folds}")
    print("=" * 80)
    print()

    exp_count = 0

    for ds_name, ds in datasets.items():
        X_full, y = ds['X'], ds['y']
        n_samples = ds['n_samples']

        # For large datasets, subsample to keep runtime manageable
        max_samples = 5000
        if n_samples > max_samples:
            rng_sub = np.random.default_rng(SEED)
            idx = rng_sub.choice(n_samples, max_samples, replace=False)
            X_full = X_full[idx]
            y = y[idx]
            n_samples = max_samples

        print(f"\n{'='*80}")
        print(f"DATASET: {ds_name} ({n_samples} samples, {ds['n_features']} features)")
        print(f"{'='*80}")

        for p_miss in missing_rates:
            for mechanism in missing_mechanisms:
                print(f"\n--- {ds_name} | {mechanism} p={p_miss} ---")

                # Introduce missing values (same missing pattern for all methods
                # within each fold, but re-generated per fold for proper CV)
                kf = KFold(n_splits=n_folds, shuffle=True, random_state=SEED)

                for fold_idx, (train_idx, test_idx) in enumerate(kf.split(X_full)):
                    rng = np.random.default_rng(SEED + fold_idx * 1000)

                    # Introduce missing values to the FULL data, then split
                    if mechanism == 'MCAR':
                        X_miss = introduce_mcar(X_full, p_miss, rng)
                    else:
                        X_miss = introduce_mar(X_full, p_miss, rng)

                    # Verify actual missing rate
                    actual_rate = np.isnan(X_miss).mean()

                    for method_name, method_ctor in methods.items():
                        exp_count += 1
                        try:
                            np.random.seed(SEED + fold_idx)
                            torch.manual_seed(SEED + fold_idx)

                            r2 = evaluate_method(
                                method_name, method_ctor,
                                X_miss, y, train_idx, test_idx
                            )

                            all_results.append({
                                'dataset': ds_name,
                                'missing_rate': p_miss,
                                'mechanism': mechanism,
                                'method': method_name,
                                'fold': fold_idx,
                                'r2': r2,
                                'actual_miss_rate': actual_rate,
                                'status': 'success'
                            })

                            print(f"  [{exp_count}/{total_exps}] Fold {fold_idx} | "
                                  f"{method_name:16s} | R2={r2:+.4f}")

                        except Exception as e:
                            all_results.append({
                                'dataset': ds_name,
                                'missing_rate': p_miss,
                                'mechanism': mechanism,
                                'method': method_name,
                                'fold': fold_idx,
                                'r2': float('nan'),
                                'actual_miss_rate': actual_rate,
                                'status': f'error: {str(e)[:80]}'
                            })
                            print(f"  [{exp_count}/{total_exps}] Fold {fold_idx} | "
                                  f"{method_name:16s} | ERROR: {str(e)[:60]}")

    # ========================================================================
    # Save raw results
    # ========================================================================
    df = pd.DataFrame(all_results)
    csv_path = os.path.join(RESULTS_DIR, 'exp_real_data.csv')
    df.to_csv(csv_path, index=False)
    print(f"\nRaw results saved to {csv_path}")

    # ========================================================================
    # Analysis and reporting
    # ========================================================================
    print("\n" + "=" * 100)
    print("RESULTS SUMMARY")
    print("=" * 100)

    df_ok = df[df['status'] == 'success'].copy()

    if df_ok.empty:
        print("No successful results to report!")
        return df

    # Aggregate: mean +/- std R2 per (dataset, mechanism, missing_rate, method)
    agg = (df_ok.groupby(['dataset', 'mechanism', 'missing_rate', 'method'])['r2']
           .agg(['mean', 'std', 'count'])
           .reset_index())
    agg.columns = ['dataset', 'mechanism', 'missing_rate', 'method',
                    'mean_r2', 'std_r2', 'n_folds']

    # --- Table 1: Results by dataset and missing rate (MCAR) ---
    print("\n" + "=" * 100)
    print("TABLE 1: R2 scores under MCAR missingness (mean +/- std over 5 folds)")
    print("=" * 100)

    method_names = list(methods.keys())
    for ds_name in datasets.keys():
        print(f"\n{'─'*90}")
        print(f"  {ds_name}")
        print(f"{'─'*90}")
        header = f"  {'p_miss':>6s}"
        for mn in method_names:
            header += f" | {mn:>16s}"
        print(header)
        print(f"  {'─'*6}" + "─┼─".join(["─" * 16] * len(method_names)) + "─")

        for p in missing_rates:
            row = f"  {p:6.1f}"
            for mn in method_names:
                sub = agg[(agg['dataset'] == ds_name) &
                          (agg['mechanism'] == 'MCAR') &
                          (agg['missing_rate'] == p) &
                          (agg['method'] == mn)]
                if len(sub) > 0 and not np.isnan(sub.iloc[0]['mean_r2']):
                    row += f" | {sub.iloc[0]['mean_r2']:+.3f}+/-{sub.iloc[0]['std_r2']:.3f}"
                else:
                    row += f" | {'N/A':>16s}"
            print(row)

    # --- Table 2: Results by dataset and missing rate (MAR) ---
    print("\n" + "=" * 100)
    print("TABLE 2: R2 scores under MAR missingness (mean +/- std over 5 folds)")
    print("=" * 100)

    for ds_name in datasets.keys():
        print(f"\n{'─'*90}")
        print(f"  {ds_name}")
        print(f"{'─'*90}")
        header = f"  {'p_miss':>6s}"
        for mn in method_names:
            header += f" | {mn:>16s}"
        print(header)
        print(f"  {'─'*6}" + "─┼─".join(["─" * 16] * len(method_names)) + "─")

        for p in missing_rates:
            row = f"  {p:6.1f}"
            for mn in method_names:
                sub = agg[(agg['dataset'] == ds_name) &
                          (agg['mechanism'] == 'MAR') &
                          (agg['missing_rate'] == p) &
                          (agg['method'] == mn)]
                if len(sub) > 0 and not np.isnan(sub.iloc[0]['mean_r2']):
                    row += f" | {sub.iloc[0]['mean_r2']:+.3f}+/-{sub.iloc[0]['std_r2']:.3f}"
                else:
                    row += f" | {'N/A':>16s}"
            print(row)

    # --- Table 3: Average R2 across all settings per method ---
    print("\n" + "=" * 100)
    print("TABLE 3: Overall average R2 per method (across all datasets/rates/mechanisms)")
    print("=" * 100)

    overall = (df_ok.groupby('method')['r2']
               .agg(['mean', 'std', 'count'])
               .sort_values('mean', ascending=False))
    print(f"\n  {'Method':>16s} | {'Mean R2':>10s} | {'Std R2':>10s} | {'N':>5s}")
    print(f"  {'─'*16}─┼─{'─'*10}─┼─{'─'*10}─┼─{'─'*5}")
    for method, row in overall.iterrows():
        print(f"  {method:>16s} | {row['mean']:+10.4f} | {row['std']:10.4f} | {int(row['count']):5d}")

    # --- Table 4: Average rank per method ---
    print("\n" + "=" * 100)
    print("TABLE 4: Average rank per method (lower is better)")
    print("=" * 100)

    # Compute rank within each (dataset, mechanism, missing_rate, fold)
    df_ok['rank'] = (df_ok.groupby(['dataset', 'mechanism', 'missing_rate', 'fold'])['r2']
                     .rank(ascending=False, method='min'))
    avg_rank = (df_ok.groupby('method')['rank']
                .mean()
                .sort_values())
    print(f"\n  {'Method':>16s} | {'Avg Rank':>10s}")
    print(f"  {'─'*16}─┼─{'─'*10}")
    for method, rank_val in avg_rank.items():
        print(f"  {method:>16s} | {rank_val:10.2f}")

    # --- Statistical significance tests ---
    print("\n" + "=" * 100)
    print("TABLE 5: Paired t-test p-values (PretrainEnc vs each method)")
    print("         Tests whether PretrainEnc is significantly different.")
    print("=" * 100)

    ref_method = 'PretrainEnc'
    if ref_method in df_ok['method'].values:
        print(f"\n  {'Method':>16s} | {'Mean diff':>10s} | {'p-value':>10s} | {'Significant':>12s}")
        print(f"  {'─'*16}─┼─{'─'*10}─┼─{'─'*10}─┼─{'─'*12}")

        for mn in method_names:
            if mn == ref_method:
                continue
            # Pair up by (dataset, mechanism, missing_rate, fold)
            ref_scores = df_ok[df_ok['method'] == ref_method].sort_values(
                ['dataset', 'mechanism', 'missing_rate', 'fold'])['r2'].values
            comp_scores = df_ok[df_ok['method'] == mn].sort_values(
                ['dataset', 'mechanism', 'missing_rate', 'fold'])['r2'].values

            min_len = min(len(ref_scores), len(comp_scores))
            if min_len > 1:
                ref_scores = ref_scores[:min_len]
                comp_scores = comp_scores[:min_len]
                diff = ref_scores - comp_scores
                t_stat, p_val = stats.ttest_rel(ref_scores, comp_scores)
                sig = "YES (p<0.05)" if p_val < 0.05 else "no"
                print(f"  {mn:>16s} | {diff.mean():+10.4f} | {p_val:10.4f} | {sig:>12s}")
            else:
                print(f"  {mn:>16s} | {'N/A':>10s} | {'N/A':>10s} | {'N/A':>12s}")

    # --- Wins per method ---
    print("\n" + "=" * 100)
    print("TABLE 6: Number of wins per method (highest R2 in each setting)")
    print("=" * 100)

    wins = defaultdict(int)
    total_settings = 0
    for (ds, mech, p, fold), group in df_ok.groupby(
            ['dataset', 'mechanism', 'missing_rate', 'fold']):
        if len(group) == 0:
            continue
        total_settings += 1
        best_idx = group['r2'].idxmax()
        winner = group.loc[best_idx, 'method']
        wins[winner] += 1

    print(f"\n  {'Method':>16s} | {'Wins':>6s} | {'Win %':>6s}")
    print(f"  {'─'*16}─┼─{'─'*6}─┼─{'─'*6}")
    for mn in sorted(wins, key=wins.get, reverse=True):
        pct = 100.0 * wins[mn] / total_settings if total_settings > 0 else 0
        print(f"  {mn:>16s} | {wins[mn]:6d} | {pct:5.1f}%")
    print(f"\n  Total settings evaluated: {total_settings}")

    # --- Performance degradation with missing rate ---
    print("\n" + "=" * 100)
    print("TABLE 7: R2 degradation from p=0.1 to p=0.5 (MCAR, per method)")
    print("=" * 100)
    print(f"\n  {'Dataset':>20s} | {'Method':>16s} | {'R2@p=0.1':>9s} | {'R2@p=0.5':>9s} | {'Drop':>8s}")
    print(f"  {'─'*20}─┼─{'─'*16}─┼─{'─'*9}─┼─{'─'*9}─┼─{'─'*8}")

    for ds_name in datasets.keys():
        for mn in method_names:
            r2_low = agg[(agg['dataset'] == ds_name) &
                         (agg['mechanism'] == 'MCAR') &
                         (agg['missing_rate'] == 0.1) &
                         (agg['method'] == mn)]
            r2_high = agg[(agg['dataset'] == ds_name) &
                          (agg['mechanism'] == 'MCAR') &
                          (agg['missing_rate'] == 0.5) &
                          (agg['method'] == mn)]
            if len(r2_low) > 0 and len(r2_high) > 0:
                v_low = r2_low.iloc[0]['mean_r2']
                v_high = r2_high.iloc[0]['mean_r2']
                drop = v_low - v_high
                print(f"  {ds_name:>20s} | {mn:>16s} | {v_low:+9.4f} | {v_high:+9.4f} | {drop:+8.4f}")

    print("\n" + "=" * 100)
    print("BENCHMARK COMPLETE")
    print("=" * 100)

    return df


if __name__ == '__main__':
    results = run_benchmark()

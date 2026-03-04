"""
Bayes Oracle Predictors for Missing Data.

Computes the exact Bayes-optimal predictor for Gaussian X with MCAR missing data
under linear and quadratic response functions. These establish the theoretical
UPPER BOUND on achievable R^2 for each scenario.

Theory:
  For Gaussian X and MCAR missingness, the Bayes predictor conditions on the
  observed pattern (x_obs, m). The conditional distribution X_mis | X_obs is
  Gaussian with known mean and covariance, allowing closed-form computation.

References:
  - Le Morvan et al. (2020): "NeuMiss networks" (linear Bayes predictor)
  - This script: extension to quadratic response functions
"""

import os
import sys
import warnings
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.metrics import r2_score

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from data_generation import DataScenario

warnings.filterwarnings('ignore')


# ============================================================================
# Helper: get conditional Gaussian parameters for a given missing pattern
# ============================================================================
def _get_conditional_params(mean, cov, obs_idx, mis_idx):
    """Compute conditional mean and covariance for X_mis | X_obs under Gaussian.

    Returns:
        mu_obs, mu_mis, regression_matrix, Sigma_cond
    where:
        E[X_mis | X_obs = x_obs] = mu_mis + regression_matrix @ (x_obs - mu_obs)
        Cov[X_mis | X_obs] = Sigma_cond
    """
    mu_obs = mean[obs_idx]
    mu_mis = mean[mis_idx]
    Sigma_oo = cov[np.ix_(obs_idx, obs_idx)]
    Sigma_mo = cov[np.ix_(mis_idx, obs_idx)]
    Sigma_mm = cov[np.ix_(mis_idx, mis_idx)]

    try:
        Sigma_oo_inv = np.linalg.solve(
            Sigma_oo + 1e-8 * np.eye(len(obs_idx)),
            np.eye(len(obs_idx))
        )
    except np.linalg.LinAlgError:
        Sigma_oo_inv = np.linalg.pinv(Sigma_oo)

    regression_matrix = Sigma_mo @ Sigma_oo_inv
    Sigma_cond = Sigma_mm - regression_matrix @ Sigma_mo.T

    return mu_obs, mu_mis, regression_matrix, Sigma_cond


# ============================================================================
# Bayes Linear Predictor
# ============================================================================
class BayesLinearPredictor(BaseEstimator, RegressorMixin):
    """
    Exact Bayes predictor for linear response Y = beta0 + X @ beta + noise,
    with Gaussian X and MCAR missing data.

    For each sample with observed pattern m (mask), the prediction is:
        f*(x_obs, m) = beta0 + beta_obs^T x_obs
                       + beta_mis^T [mu_mis + Sigma_mo Sigma_oo^{-1} (x_obs - mu_obs)]

    This is the conditional expectation E[Y | X_obs, M=m] and is Bayes-optimal
    under squared loss for this setting.
    """

    def __init__(self, mean=None, cov=None, beta=None, beta0=1.0):
        self.mean = mean
        self.cov = cov
        self.beta = beta
        self.beta0 = beta0

    def fit(self, X=None, y=None):
        """No fitting needed -- uses true parameters. Included for sklearn API."""
        return self

    def predict(self, X):
        n, d = X.shape
        predictions = np.zeros(n)
        mask = np.isnan(X)
        pattern_cache = {}

        for i in range(n):
            m = mask[i]
            obs_idx = np.where(~m)[0]
            mis_idx = np.where(m)[0]

            if len(obs_idx) == 0:
                predictions[i] = self.beta0 + self.beta @ self.mean
                continue
            if len(mis_idx) == 0:
                predictions[i] = self.beta0 + self.beta @ X[i]
                continue

            pattern_key = tuple(m)
            if pattern_key not in pattern_cache:
                mu_obs, mu_mis, reg_mat, _ = _get_conditional_params(
                    self.mean, self.cov, obs_idx, mis_idx)
                pattern_cache[pattern_key] = (
                    obs_idx, mis_idx, mu_obs, mu_mis, reg_mat)

            obs_idx, mis_idx, mu_obs, mu_mis, reg_mat = pattern_cache[pattern_key]
            x_obs = X[i, obs_idx]
            mu_hat = mu_mis + reg_mat @ (x_obs - mu_obs)

            predictions[i] = (self.beta0
                              + self.beta[obs_idx] @ x_obs
                              + self.beta[mis_idx] @ mu_hat)

        return predictions

    def score(self, X, y):
        return r2_score(y, self.predict(X))


# ============================================================================
# Bayes Quadratic Predictor
# ============================================================================
class BayesQuadraticPredictor(BaseEstimator, RegressorMixin):
    """
    Exact Bayes predictor for quadratic response
        Y = (beta0 + X@beta) + (beta0 + X@beta)^2 + noise
    with Gaussian X and MCAR missing data.

    We expand the noiseless response as Y = c + b^T X + X^T A X, where:
        c = beta0 + beta0^2
        b = (1 + 2*beta0) * beta
        A = beta @ beta^T

    Then the Bayes prediction E[Y | X_obs, M=m] is:
        f*(x_obs, m) = c + b_obs^T x_obs + b_mis^T mu_hat
                       + x_obs^T A_oo x_obs + 2 x_obs^T A_om mu_hat
                       + mu_hat^T A_mm mu_hat + tr(A_mm Sigma_mis|obs)

    where mu_hat = E[X_mis | X_obs], Sigma_mis|obs = Cov[X_mis | X_obs].
    """

    def __init__(self, mean=None, cov=None, beta=None, beta0=1.0):
        self.mean = mean
        self.cov = cov
        self.beta = beta
        self.beta0 = beta0

    def fit(self, X=None, y=None):
        """Precompute the quadratic form decomposition."""
        self.c_ = self.beta0 + self.beta0 ** 2
        self.b_ = (1.0 + 2.0 * self.beta0) * self.beta
        self.A_ = np.outer(self.beta, self.beta)
        return self

    def predict(self, X):
        n, d = X.shape
        predictions = np.zeros(n)
        mask = np.isnan(X)
        pattern_cache = {}

        for i in range(n):
            m = mask[i]
            obs_idx = np.where(~m)[0]
            mis_idx = np.where(m)[0]

            if len(obs_idx) == 0:
                mu = self.mean
                predictions[i] = (self.c_ + self.b_ @ mu
                                  + mu @ self.A_ @ mu
                                  + np.trace(self.A_ @ self.cov))
                continue
            if len(mis_idx) == 0:
                x = X[i]
                predictions[i] = self.c_ + self.b_ @ x + x @ self.A_ @ x
                continue

            pattern_key = tuple(m)
            if pattern_key not in pattern_cache:
                mu_obs, mu_mis, reg_mat, Sigma_cond = _get_conditional_params(
                    self.mean, self.cov, obs_idx, mis_idx)
                A_oo = self.A_[np.ix_(obs_idx, obs_idx)]
                A_om = self.A_[np.ix_(obs_idx, mis_idx)]
                A_mm = self.A_[np.ix_(mis_idx, mis_idx)]
                b_obs = self.b_[obs_idx]
                b_mis = self.b_[mis_idx]
                trace_term = np.trace(A_mm @ Sigma_cond)
                pattern_cache[pattern_key] = (
                    obs_idx, mis_idx, mu_obs, mu_mis, reg_mat,
                    A_oo, A_om, A_mm, b_obs, b_mis, trace_term)

            (obs_idx, mis_idx, mu_obs, mu_mis, reg_mat,
             A_oo, A_om, A_mm, b_obs, b_mis, trace_term) = \
                pattern_cache[pattern_key]

            x_obs = X[i, obs_idx]
            mu_hat = mu_mis + reg_mat @ (x_obs - mu_obs)

            predictions[i] = (
                self.c_
                + b_obs @ x_obs + b_mis @ mu_hat
                + x_obs @ A_oo @ x_obs
                + 2.0 * x_obs @ A_om @ mu_hat
                + mu_hat @ A_mm @ mu_hat
                + trace_term
            )

        return predictions

    def score(self, X, y):
        return r2_score(y, self.predict(X))


# ============================================================================
# Monte Carlo Bayes predictor (works for any response, but uses sampling)
# ============================================================================
class BayesMCPredictor(BaseEstimator, RegressorMixin):
    """
    Monte Carlo approximation of the Bayes predictor E[f(X) | X_obs, M=m].

    For each sample, we draw K samples from X_mis | X_obs (using Gaussian
    conditional), fill in the missing values, compute f(X_complete) for each,
    and average. This works for ANY response function, not just linear/quadratic.

    This is slower but provides a universal Bayes oracle for any f.
    """

    def __init__(self, mean=None, cov=None, beta=None, beta0=1.0,
                 response_fn=None, response_kwargs=None, n_mc=500):
        self.mean = mean
        self.cov = cov
        self.beta = beta
        self.beta0 = beta0
        self.response_fn = response_fn
        self.response_kwargs = response_kwargs or {}
        self.n_mc = n_mc

    def fit(self, X=None, y=None):
        return self

    def predict(self, X):
        n, d = X.shape
        predictions = np.zeros(n)
        mask = np.isnan(X)
        pattern_cache = {}
        rng = np.random.RandomState(0)

        for i in range(n):
            m = mask[i]
            obs_idx = np.where(~m)[0]
            mis_idx = np.where(m)[0]

            if len(mis_idx) == 0:
                # No missing: exact
                x_full = X[i:i+1].copy()
                predictions[i] = self.response_fn(
                    x_full, self.beta, **self.response_kwargs)[0]
                continue

            if len(obs_idx) == 0:
                # All missing: sample from marginal
                X_samples = rng.multivariate_normal(
                    self.mean, self.cov, size=self.n_mc)
                y_samples = self.response_fn(
                    X_samples, self.beta, **self.response_kwargs)
                predictions[i] = np.mean(y_samples)
                continue

            pattern_key = tuple(m)
            if pattern_key not in pattern_cache:
                mu_obs, mu_mis, reg_mat, Sigma_cond = _get_conditional_params(
                    self.mean, self.cov, obs_idx, mis_idx)
                # Ensure Sigma_cond is PSD
                Sigma_cond = (Sigma_cond + Sigma_cond.T) / 2
                eigvals = np.linalg.eigvalsh(Sigma_cond)
                if eigvals.min() < 0:
                    Sigma_cond += (-eigvals.min() + 1e-8) * np.eye(len(mis_idx))
                pattern_cache[pattern_key] = (
                    obs_idx, mis_idx, mu_obs, mu_mis, reg_mat, Sigma_cond)

            obs_idx, mis_idx, mu_obs, mu_mis, reg_mat, Sigma_cond = \
                pattern_cache[pattern_key]

            x_obs = X[i, obs_idx]
            mu_hat = mu_mis + reg_mat @ (x_obs - mu_obs)

            # Sample X_mis from N(mu_hat, Sigma_cond)
            mis_samples = rng.multivariate_normal(mu_hat, Sigma_cond,
                                                   size=self.n_mc)

            # Build full X samples
            X_samples = np.tile(x_obs, (self.n_mc, 1))
            X_full = np.zeros((self.n_mc, d))
            X_full[:, obs_idx] = X_samples
            X_full[:, mis_idx] = mis_samples

            y_samples = self.response_fn(
                X_full, self.beta, **self.response_kwargs)
            predictions[i] = np.mean(y_samples)

        return predictions

    def score(self, X, y):
        return r2_score(y, self.predict(X))


# ============================================================================
# Response functions (imported for MC predictor)
# ============================================================================
from data_generation import (linear_response, quadratic_response,
                              cubic_response)


def _make_cubic_response_fixed_std(std_value):
    """Create a cubic response function with a FIXED normalization constant.

    The original cubic_response normalizes by the batch std, which varies.
    During data generation, the std is computed on the full n_total sample.
    For the Bayes MC predictor, we need a fixed std so that predictions are
    consistent regardless of batch size.
    """
    def cubic_fixed(X, beta, beta0=1.0):
        linear_part = beta0 + X.dot(beta)
        normalized = linear_part / std_value
        return normalized + normalized ** 3
    return cubic_fixed


# ============================================================================
# Experiment runner
# ============================================================================
def run_single_experiment(scenario_config, seed, use_neural=True):
    """Run a single experiment with one seed.

    Returns dict of method_name -> test R^2.
    """
    scenario = DataScenario(**scenario_config)
    data = scenario.generate(n_train=2000, n_val=500, n_test=1000,
                             random_state=seed)

    X_train = data['X_train']
    y_train = data['y_train']
    X_val = data['X_val']
    y_val = data['y_val']
    X_test = data['X_test']
    y_test = data['y_test']

    mean = data['mean']
    cov = data['cov']
    beta = data['beta']

    response_type = scenario_config['response']

    results = {}

    # ---- Oracle: R^2 on complete data (no missing) ----
    X_complete = data['X_complete']
    y_clean = data['y_clean']
    n_train = 2000
    n_val = 500
    y_test_clean = y_clean[n_train + n_val:]
    ss_noise = np.mean((y_test.astype(np.float64) - y_test_clean) ** 2)
    ss_tot = np.var(y_test)
    results['Oracle (no missing)'] = 1.0 - ss_noise / ss_tot if ss_tot > 0 else 0.0

    # ---- Bayes Linear Predictor ----
    bayes_lin = BayesLinearPredictor(mean=mean, cov=cov, beta=beta, beta0=1.0)
    bayes_lin.fit()
    results['Bayes Linear'] = bayes_lin.score(X_test, y_test)

    # ---- Bayes Quadratic Predictor (only meaningful for quadratic response) ----
    if response_type == 'quadratic':
        bayes_quad = BayesQuadraticPredictor(
            mean=mean, cov=cov, beta=beta, beta0=1.0)
        bayes_quad.fit()
        results['Bayes Quadratic'] = bayes_quad.score(X_test, y_test)

    # ---- Monte Carlo Bayes Predictor (works for any response) ----
    if response_type == 'linear':
        resp_fn = linear_response
        resp_kwargs = {}
    elif response_type == 'quadratic':
        resp_fn = quadratic_response
        resp_kwargs = {}
    elif response_type == 'cubic':
        # Compute the normalization std from the FULL complete data
        # (matching what cubic_response computed during data generation)
        linear_part_full = 1.0 + X_complete.dot(beta)
        std_full = np.std(linear_part_full)
        resp_fn = _make_cubic_response_fixed_std(std_full)
        resp_kwargs = {}
    else:
        resp_fn = None

    if resp_fn is not None:
        bayes_mc = BayesMCPredictor(
            mean=mean, cov=cov, beta=beta, beta0=1.0,
            response_fn=resp_fn, response_kwargs=resp_kwargs,
            n_mc=500)
        bayes_mc.fit()
        results['Bayes MC'] = bayes_mc.score(X_test, y_test)

    # ---- Neural baselines ----
    if use_neural:
        try:
            from neumiss_plus import NeuMissPlus, NeuMissMLPEstimator, ImputeMLP

            # NeuMiss (original, linear architecture)
            neumiss = NeuMissPlus(variant='original', depth=5, n_epochs=200,
                                  batch_size=64, lr=0.001, verbose=False)
            neumiss.fit(X_train, y_train, X_val=X_val, y_val=y_val)
            results['NeuMiss (original)'] = neumiss.score(X_test, y_test)

            # NeuMiss+ (variant A with activations)
            neumiss_plus = NeuMissPlus(variant='A', depth=5, activation='relu',
                                       n_epochs=200, batch_size=64, lr=0.001,
                                       verbose=False)
            neumiss_plus.fit(X_train, y_train, X_val=X_val, y_val=y_val)
            results['NeuMiss+ (A)'] = neumiss_plus.score(X_test, y_test)

            # NeuMiss+MLP (two-stage)
            neumiss_mlp = NeuMissMLPEstimator(
                variant='mlp', depth=3, mlp_layers=(64, 32),
                activation='gelu', n_epochs=200, batch_size=64,
                lr=0.001, verbose=False)
            neumiss_mlp.fit(X_train, y_train, X_val=X_val, y_val=y_val)
            results['NeuMiss+MLP'] = neumiss_mlp.score(X_test, y_test)

            # ImputeMLP baseline
            impute_mlp = ImputeMLP(hidden_layers=(128, 64, 32), n_epochs=200,
                                   batch_size=64, lr=0.001, verbose=False)
            impute_mlp.fit(X_train, y_train, X_val=X_val, y_val=y_val)
            results['ImputeMLP'] = impute_mlp.score(X_test, y_test)

        except Exception as e:
            print(f"  [WARN] Neural methods failed: {e}")

    return results


def main():
    print("=" * 80)
    print("BAYES ORACLE EXPERIMENT")
    print("Theoretical upper bounds on R^2 for missing data prediction")
    print("=" * 80)
    print()

    seeds = [42, 123, 7]

    # Define scenarios
    scenarios = {
        'gaussian + linear + MCAR': dict(
            distribution='gaussian', response='linear',
            missing_mechanism='MCAR', n_features=10, missing_rate=0.5, snr=10.0,
        ),
        'gaussian + quadratic + MCAR': dict(
            distribution='gaussian', response='quadratic',
            missing_mechanism='MCAR', n_features=10, missing_rate=0.5, snr=10.0,
        ),
        'gaussian + cubic + MCAR': dict(
            distribution='gaussian', response='cubic',
            missing_mechanism='MCAR', n_features=10, missing_rate=0.5, snr=10.0,
        ),
        'mixture_gaussian + linear + MCAR': dict(
            distribution='mixture_gaussian', response='linear',
            missing_mechanism='MCAR', n_features=10, missing_rate=0.5, snr=10.0,
            distribution_params={'n_components': 3},
        ),
        'student_t + quadratic + MCAR': dict(
            distribution='student_t', response='quadratic',
            missing_mechanism='MCAR', n_features=10, missing_rate=0.5, snr=10.0,
            distribution_params={'df': 5},
        ),
    }

    # Collect results across seeds
    all_results = {name: {} for name in scenarios}

    for scenario_name, config in scenarios.items():
        print(f"\n{'=' * 70}")
        print(f"Scenario: {scenario_name}")
        print(f"{'=' * 70}")

        seed_results = []
        for seed in seeds:
            print(f"  Seed {seed}...", end=" ", flush=True)
            res = run_single_experiment(config, seed, use_neural=True)
            seed_results.append(res)
            print("done")

        # Aggregate across seeds
        all_methods = set()
        for sr in seed_results:
            all_methods.update(sr.keys())
        all_methods = sorted(all_methods)

        print(f"\n  {'Method':<25s}  {'R^2 (mean +/- std)':>22s}")
        print(f"  {'=' * 25}  {'=' * 22}")

        for method in all_methods:
            vals = [sr[method] for sr in seed_results if method in sr]
            if len(vals) > 0:
                mean_r2 = np.mean(vals)
                std_r2 = np.std(vals)
                all_results[scenario_name][method] = (mean_r2, std_r2)
                marker = ""
                if "Bayes" in method or "Oracle" in method:
                    marker = " <-- BOUND"
                print(f"  {method:<25s}  {mean_r2:>8.4f} +/- {std_r2:.4f}{marker}")

    # ========================================================================
    # Summary table
    # ========================================================================
    print("\n\n")
    print("=" * 90)
    print("SUMMARY TABLE: Mean R^2 (std) across 3 seeds")
    print("=" * 90)

    # Determine which methods appear
    all_method_names = set()
    for scenario_name in scenarios:
        all_method_names.update(all_results[scenario_name].keys())
    method_order = [
        'Oracle (no missing)',
        'Bayes Linear',
        'Bayes Quadratic',
        'Bayes MC',
        'NeuMiss (original)',
        'NeuMiss+ (A)',
        'NeuMiss+MLP',
        'ImputeMLP',
    ]
    method_order = [m for m in method_order if m in all_method_names]

    # Print as a clean table
    col_width = 16
    header = f"{'Scenario':<40s}"
    for m in method_order:
        short = m[:15]
        header += f" {short:>{col_width}s}"
    print(header)
    print("-" * len(header))

    for scenario_name in scenarios:
        row = f"{scenario_name:<40s}"
        for m in method_order:
            if m in all_results[scenario_name]:
                mean_r2, std_r2 = all_results[scenario_name][m]
                cell = f"{mean_r2:.4f}({std_r2:.3f})"
                row += f" {cell:>{col_width}s}"
            else:
                row += f" {'--':>{col_width}s}"
        print(row)

    # ========================================================================
    # Analysis
    # ========================================================================
    print("\n\n")
    print("=" * 80)
    print("ANALYSIS")
    print("=" * 80)

    for scenario_name in scenarios:
        res = all_results[scenario_name]
        print(f"\n--- {scenario_name} ---")

        oracle = res.get('Oracle (no missing)', (0, 0))[0]
        bayes_lin = res.get('Bayes Linear', (0, 0))[0]
        bayes_quad = res.get('Bayes Quadratic', (None, 0))[0]
        bayes_mc = res.get('Bayes MC', (None, 0))[0]

        best_bayes = bayes_lin
        best_bayes_name = "Bayes Linear"
        if bayes_quad is not None and bayes_quad > best_bayes:
            best_bayes = bayes_quad
            best_bayes_name = "Bayes Quadratic"
        if bayes_mc is not None and bayes_mc > best_bayes:
            best_bayes = bayes_mc
            best_bayes_name = "Bayes MC"

        print(f"  Oracle R^2 (no missing):     {oracle:.4f}")
        print(f"  Best Bayes R^2:              {best_bayes:.4f}  ({best_bayes_name})")
        print(f"  Gap (Oracle - Best Bayes):   {oracle - best_bayes:.4f}")
        print(f"  This gap is the irreducible cost of missing data.")

        # Neural methods
        for mname in ['NeuMiss (original)', 'NeuMiss+ (A)', 'NeuMiss+MLP',
                       'ImputeMLP']:
            if mname in res:
                mr2 = res[mname][0]
                gap = best_bayes - mr2
                pct = (mr2 / best_bayes * 100) if best_bayes > 0 else float('nan')
                print(f"  {mname:<25s} R^2={mr2:.4f}  "
                      f"({pct:.1f}% of Bayes bound, gap={gap:+.4f})")

    print("\n")
    print("=" * 80)
    print("KEY TAKEAWAYS")
    print("=" * 80)
    print("""
  1. gaussian + linear + MCAR:
     Bayes Linear is the EXACT optimal predictor. Any neural method that
     approaches this R^2 has learned the optimal missing-data strategy.

  2. gaussian + quadratic + MCAR:
     Bayes Quadratic is the EXACT optimal predictor. The gap between
     Bayes Linear and Bayes Quadratic shows the value of modeling
     nonlinearity in the response. Neural methods should aim for the
     Bayes Quadratic bound.

  3. gaussian + cubic + MCAR:
     Neither Bayes Linear nor Quadratic is exact. The Bayes MC predictor
     approximates the true optimum. The gap between Bayes Quadratic and
     Bayes MC shows how much the cubic term matters beyond quadratic.

  4. mixture_gaussian + linear + MCAR:
     Bayes Linear (which assumes Gaussian X) is NOT the true optimal
     predictor because X is mixture Gaussian. The Bayes MC predictor
     (which also assumes Gaussian conditionals) is similarly misspecified.
     Neural methods can potentially EXCEED the Bayes Linear bound here.

  5. student_t + quadratic + MCAR:
     The Gaussian-based Bayes predictors are approximate since X is
     Student-t. The conditional X_mis|X_obs is NOT Gaussian, so
     neural methods could potentially do better.
""")


if __name__ == '__main__':
    main()

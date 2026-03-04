"""
Comprehensive data generation for NeuMiss+ experiments.

Scenarios from the meeting notes:
1. Gaussian X + Linear f (original NeuMiss setting) - baseline
2. Gaussian X + Polynomial f (low-degree: quadratic, cubic)
3. Gaussian X + General nonlinear f (y + y^3, sin, etc.)
4. Non-normal X + Linear f (mixture of Gaussians, Student-t)
5. Non-normal X + Nonlinear f (hardest case, "best case scenario" from notes)

Missing mechanisms:
- MCAR: Missing Completely At Random
- MAR: Missing At Random (logistic model)
- MNAR_censoring: values above/below threshold more likely missing
- MNAR_selfmasking: probability of missing depends on value itself
"""

import numpy as np
from sklearn.utils import check_random_state
from scipy.optimize import fsolve


def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))


# ============================================================================
# X distributions
# ============================================================================
def generate_gaussian_X(n_samples, n_features, prop_latent=0.5,
                        random_state=None):
    """Standard Gaussian X (original NeuMiss setting)."""
    rng = check_random_state(random_state)
    B = rng.randn(n_features, max(1, int(prop_latent * n_features)))
    cov = B.dot(B.T) + np.diag(
        rng.uniform(low=0.01, high=0.1, size=n_features))
    mean = rng.randn(n_features)
    X = rng.multivariate_normal(mean=mean, cov=cov, size=n_samples)
    return X, mean, cov


def generate_mixture_gaussian_X(n_samples, n_features, n_components=3,
                                prop_latent=0.5, random_state=None):
    """Mixture of Gaussians X (non-normal)."""
    rng = check_random_state(random_state)
    X = np.zeros((n_samples, n_features))

    # Random mixing weights
    weights = rng.dirichlet(np.ones(n_components))

    means = []
    covs = []
    for k in range(n_components):
        B = rng.randn(n_features, max(1, int(prop_latent * n_features)))
        cov_k = B.dot(B.T) + np.diag(
            rng.uniform(low=0.01, high=0.1, size=n_features))
        mean_k = rng.randn(n_features) * 2  # spread out means
        means.append(mean_k)
        covs.append(cov_k)

    # Sample component assignments
    components = rng.choice(n_components, size=n_samples, p=weights)
    for k in range(n_components):
        mask_k = components == k
        n_k = mask_k.sum()
        if n_k > 0:
            X[mask_k] = rng.multivariate_normal(
                mean=means[k], cov=covs[k], size=n_k)

    # Compute overall mean and covariance for reference
    mean = np.sum([w * m for w, m in zip(weights, means)], axis=0)
    cov = np.sum([w * (c + np.outer(m, m)) for w, m, c in
                  zip(weights, means, covs)], axis=0) - np.outer(mean, mean)

    return X, mean, cov


def generate_student_t_X(n_samples, n_features, df=5, prop_latent=0.5,
                         random_state=None):
    """Student-t distributed X (heavy tails, non-normal)."""
    rng = check_random_state(random_state)
    B = rng.randn(n_features, max(1, int(prop_latent * n_features)))
    cov = B.dot(B.T) + np.diag(
        rng.uniform(low=0.01, high=0.1, size=n_features))
    mean = rng.randn(n_features)

    # Generate multivariate t via Gaussian / sqrt(chi2/df)
    Z = rng.multivariate_normal(np.zeros(n_features), cov, size=n_samples)
    chi2 = rng.chisquare(df, size=(n_samples, 1))
    X = mean + Z / np.sqrt(chi2 / df)

    return X, mean, cov


def generate_skewed_X(n_samples, n_features, skewness=2.0,
                      random_state=None):
    """Skewed distribution X using log-normal transformation."""
    rng = check_random_state(random_state)
    Z = rng.randn(n_samples, n_features)
    # Apply exponential to some features to create skewness
    n_skewed = n_features // 2
    X = Z.copy()
    X[:, :n_skewed] = np.exp(Z[:, :n_skewed] * 0.5) - 1

    mean = X.mean(axis=0)
    cov = np.cov(X.T)
    return X, mean, cov


# ============================================================================
# Response functions f(X)
# ============================================================================
def linear_response(X, beta, beta0=1.0):
    """Linear: y = beta0 + X @ beta."""
    return beta0 + X.dot(beta)


def quadratic_response(X, beta, beta0=1.0):
    """Quadratic: y = beta0 + X@beta + (X@beta)^2."""
    linear_part = beta0 + X.dot(beta)
    return linear_part + linear_part ** 2


def cubic_response(X, beta, beta0=1.0):
    """Cubic: y = beta0 + X@beta + (X@beta)^3 (from meeting notes: y + y^3)."""
    linear_part = beta0 + X.dot(beta)
    # Normalize to avoid explosion
    std = np.std(linear_part)
    if std > 0:
        normalized = linear_part / std
        return normalized + normalized ** 3
    return linear_part


def polynomial_response(X, beta, degree=3, beta0=1.0):
    """General polynomial: sum of terms up to given degree."""
    linear_part = X.dot(beta)
    std = np.std(linear_part)
    if std > 0:
        linear_part = linear_part / std

    y = beta0
    for d in range(1, degree + 1):
        y = y + linear_part ** d / np.math.factorial(d)
    return y


def sinusoidal_response(X, beta, beta0=1.0):
    """Sinusoidal nonlinearity: y = sin(X@beta)."""
    return beta0 + np.sin(X.dot(beta))


def piecewise_linear_response(X, beta, beta0=1.0):
    """Piecewise linear (absolute value): y = |X@beta|."""
    return beta0 + np.abs(X.dot(beta))


def interaction_response(X, beta, beta0=1.0):
    """Includes pairwise interactions: y = X@beta + sum(x_i * x_j)."""
    n, d = X.shape
    linear_part = beta0 + X.dot(beta)
    # Add pairwise interactions (top few)
    interaction = np.zeros(n)
    for i in range(min(d, 5)):
        for j in range(i + 1, min(d, 5)):
            interaction += X[:, i] * X[:, j] * beta[i] * beta[j]
    return linear_part + interaction


# ============================================================================
# Missing mechanisms
# ============================================================================
def apply_MCAR(X, missing_rate=0.5, random_state=None):
    """Missing Completely At Random."""
    rng = check_random_state(random_state)
    mask = rng.rand(*X.shape) < missing_rate
    X_missing = X.copy()
    X_missing[mask] = np.nan
    return X_missing, mask


def apply_MAR(X, missing_rate=0.5, p_obs=0.1, random_state=None):
    """Missing At Random (logistic model based on observed variables)."""
    rng = check_random_state(random_state)
    n, d = X.shape
    mask = np.zeros((n, d), dtype=bool)

    d_obs = max(int(p_obs * d), 1)
    idxs_obs = rng.choice(d, d_obs, replace=False)
    idxs_nas = np.array([i for i in range(d) if i not in idxs_obs])
    d_na = len(idxs_nas)

    if d_na == 0:
        X_missing = X.copy()
        return X_missing, mask

    mu = X.mean(0)
    cov = (X - mu).T.dot(X - mu) / n
    cov_obs = cov[np.ix_(idxs_obs, idxs_obs)]
    coeffs = rng.randn(d_obs, d_na)
    v = np.array([coeffs[:, j].dot(cov_obs).dot(coeffs[:, j])
                  for j in range(d_na)])
    steepness = rng.uniform(low=0.1, high=0.5, size=d_na)
    coeffs /= (steepness * np.sqrt(np.maximum(v, 1e-10)))

    intercepts = np.zeros(d_na)
    for j in range(d_na):
        w = coeffs[:, j]

        def f(b):
            s = sigmoid(X[:, idxs_obs].dot(w) + b) - missing_rate
            return s.mean()

        try:
            res = fsolve(f, x0=0)
            intercepts[j] = res[0]
        except Exception:
            intercepts[j] = 0

    ps = sigmoid(X[:, idxs_obs].dot(coeffs) + intercepts)
    ber = rng.rand(n, d_na)
    mask[:, idxs_nas] = ber < ps

    X_missing = X.copy()
    X_missing[mask] = np.nan
    return X_missing, mask


def apply_MNAR_censoring(X, missing_rate=0.5, direction='high',
                         random_state=None):
    """MNAR: high/low values more likely to be missing (censoring).

    This is the scenario from the meeting notes:
    "values above threshold -> missing probability increases"
    """
    rng = check_random_state(random_state)
    n, d = X.shape
    mask = np.zeros((n, d), dtype=bool)

    for j in range(d):
        x_j = X[:, j]
        if direction == 'high':
            # Higher values more likely to be missing
            quantile = np.quantile(x_j, 1 - missing_rate * 1.5)
            prob = sigmoid(3 * (x_j - quantile) / np.std(x_j))
        elif direction == 'low':
            # Lower values more likely to be missing
            quantile = np.quantile(x_j, missing_rate * 1.5)
            prob = sigmoid(3 * (quantile - x_j) / np.std(x_j))
        else:
            # Both extremes
            center = np.mean(x_j)
            prob = sigmoid(2 * (np.abs(x_j - center) / np.std(x_j) - 1))

        # Adjust to achieve target missing rate
        scale = missing_rate / np.mean(prob)
        prob = np.clip(prob * scale, 0, 0.95)
        mask[:, j] = rng.rand(n) < prob

    X_missing = X.copy()
    X_missing[mask] = np.nan
    return X_missing, mask


def apply_MNAR_selfmasking(X, missing_rate=0.5, random_state=None):
    """MNAR self-masking: P(M_j=1|X_j) depends on X_j via Gaussian kernel."""
    rng = check_random_state(random_state)
    n, d = X.shape
    mask = np.zeros((n, d), dtype=bool)

    for j in range(d):
        x_j = X[:, j]
        mu_j = np.mean(x_j)
        std_j = np.std(x_j)
        # Gaussian self-masking: missing probability is Gaussian centered
        # at mean + k*std
        k = 2.0
        mu_tilde = mu_j + k * std_j
        sigma_tilde_sq = std_j ** 2 * 2.0

        prob = np.exp(-0.5 * (x_j - mu_tilde) ** 2 / sigma_tilde_sq)
        # Scale to achieve target rate
        scale = missing_rate / np.mean(np.clip(prob, 1e-10, 1.0))
        prob = np.clip(prob * scale, 0, 0.95)
        mask[:, j] = rng.rand(n) < prob

    X_missing = X.copy()
    X_missing[mask] = np.nan
    return X_missing, mask


# ============================================================================
# Complete data scenario generator
# ============================================================================
class DataScenario:
    """Generates complete datasets for a given scenario configuration."""

    DISTRIBUTIONS = {
        'gaussian': generate_gaussian_X,
        'mixture_gaussian': generate_mixture_gaussian_X,
        'student_t': generate_student_t_X,
        'skewed': generate_skewed_X,
    }

    RESPONSES = {
        'linear': linear_response,
        'quadratic': quadratic_response,
        'cubic': cubic_response,
        'polynomial': polynomial_response,
        'sinusoidal': sinusoidal_response,
        'piecewise_linear': piecewise_linear_response,
        'interaction': interaction_response,
    }

    MISSING_MECHS = {
        'MCAR': apply_MCAR,
        'MAR': apply_MAR,
        'MNAR_censoring': apply_MNAR_censoring,
        'MNAR_selfmasking': apply_MNAR_selfmasking,
    }

    def __init__(self, distribution='gaussian', response='linear',
                 missing_mechanism='MCAR', n_features=10,
                 missing_rate=0.5, snr=10.0,
                 distribution_params=None, response_params=None,
                 missing_params=None):
        self.distribution = distribution
        self.response = response
        self.missing_mechanism = missing_mechanism
        self.n_features = n_features
        self.missing_rate = missing_rate
        self.snr = snr
        self.distribution_params = distribution_params or {}
        self.response_params = response_params or {}
        self.missing_params = missing_params or {}

    def generate(self, n_train, n_val, n_test, random_state=None):
        """Generate train/val/test data for this scenario."""
        rng = check_random_state(random_state)
        n_total = n_train + n_val + n_test

        # 1. Generate X
        dist_fn = self.DISTRIBUTIONS[self.distribution]
        X, mean, cov = dist_fn(
            n_total, self.n_features, random_state=rng,
            **self.distribution_params)

        # 2. Generate beta
        beta = np.ones(self.n_features)

        # 3. Generate Y = f(X) + noise
        response_fn = self.RESPONSES[self.response]
        y_clean = response_fn(X, beta, **self.response_params)

        # Add noise to achieve desired SNR
        var_y = np.var(y_clean)
        noise_var = var_y / self.snr
        noise = rng.normal(0, np.sqrt(max(noise_var, 1e-10)), size=n_total)
        y = y_clean + noise

        # 4. Apply missing mechanism
        missing_fn = self.MISSING_MECHS[self.missing_mechanism]
        X_missing, mask = missing_fn(
            X, missing_rate=self.missing_rate, random_state=rng,
            **self.missing_params)

        # 5. Split
        data = {
            'X_train': X_missing[:n_train],
            'y_train': y[:n_train].astype(np.float32),
            'X_val': X_missing[n_train:n_train + n_val],
            'y_val': y[n_train:n_train + n_val].astype(np.float32),
            'X_test': X_missing[n_train + n_val:],
            'y_test': y[n_train + n_val:].astype(np.float32),
            'X_complete': X,  # for reference
            'y_clean': y_clean,
            'mean': mean,
            'cov': cov,
            'beta': beta,
            'mask': mask,
        }
        return data

    @property
    def name(self):
        return f"{self.distribution}_{self.response}_{self.missing_mechanism}"


def get_all_scenarios(n_features=10, missing_rate=0.5, snr=10.0):
    """Generate all data scenario configurations for comprehensive testing."""
    scenarios = []

    distributions = ['gaussian', 'mixture_gaussian', 'student_t', 'skewed']
    responses = ['linear', 'quadratic', 'cubic', 'sinusoidal', 'interaction']
    missing_mechs = ['MCAR', 'MAR', 'MNAR_censoring', 'MNAR_selfmasking']

    for dist in distributions:
        for resp in responses:
            for mech in missing_mechs:
                dist_params = {}
                if dist == 'mixture_gaussian':
                    dist_params = {'n_components': 3}
                elif dist == 'student_t':
                    dist_params = {'df': 5}

                scenarios.append(DataScenario(
                    distribution=dist,
                    response=resp,
                    missing_mechanism=mech,
                    n_features=n_features,
                    missing_rate=missing_rate,
                    snr=snr,
                    distribution_params=dist_params,
                ))

    return scenarios


def get_key_scenarios(n_features=10, missing_rate=0.5, snr=10.0):
    """Get the most important scenarios for initial testing.

    Based on meeting notes priority:
    1. Gaussian + Linear (baseline)
    2. Gaussian + Polynomial (easy extension)
    3. Non-normal + Nonlinear (target: "best case scenario")
    4. Various MNAR (MNAR doesn't affect prediction per discussion)
    """
    scenarios = [
        # Baseline: original NeuMiss territory
        DataScenario('gaussian', 'linear', 'MCAR', n_features, missing_rate, snr),
        DataScenario('gaussian', 'linear', 'MAR', n_features, missing_rate, snr),
        DataScenario('gaussian', 'linear', 'MNAR_selfmasking', n_features, missing_rate, snr),

        # Easy extension: Gaussian X + polynomial f
        DataScenario('gaussian', 'quadratic', 'MCAR', n_features, missing_rate, snr),
        DataScenario('gaussian', 'cubic', 'MCAR', n_features, missing_rate, snr),

        # Non-normal X + linear f
        DataScenario('mixture_gaussian', 'linear', 'MCAR', n_features, missing_rate, snr,
                     distribution_params={'n_components': 3}),
        DataScenario('student_t', 'linear', 'MCAR', n_features, missing_rate, snr,
                     distribution_params={'df': 5}),

        # Hard: Non-normal X + nonlinear f ("best case scenario")
        DataScenario('mixture_gaussian', 'quadratic', 'MCAR', n_features, missing_rate, snr,
                     distribution_params={'n_components': 3}),
        DataScenario('mixture_gaussian', 'cubic', 'MCAR', n_features, missing_rate, snr,
                     distribution_params={'n_components': 3}),
        DataScenario('student_t', 'sinusoidal', 'MCAR', n_features, missing_rate, snr,
                     distribution_params={'df': 5}),

        # MNAR with nonlinear (from meeting notes: MNAR doesn't affect prediction)
        DataScenario('gaussian', 'quadratic', 'MNAR_censoring', n_features, missing_rate, snr),
        DataScenario('mixture_gaussian', 'cubic', 'MNAR_censoring', n_features, missing_rate, snr,
                     distribution_params={'n_components': 3}),

        # Interaction terms
        DataScenario('gaussian', 'interaction', 'MCAR', n_features, missing_rate, snr),
    ]
    return scenarios

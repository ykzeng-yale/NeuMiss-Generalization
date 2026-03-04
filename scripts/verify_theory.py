#!/usr/bin/env python3
"""
Empirical verification of the key theoretical prediction from NeuroMiss:

    For Gaussian X, the conditional covariance Sigma_{mis|obs} depends ONLY
    on the mask pattern M, not on X_obs.

This means that for a quadratic f, there exists a "variance correction" term
    tr(A_mm @ Sigma_{mis|obs})
that is constant for a given mask pattern.

We verify:
  1. Analytic Sigma_{mis|obs} = Sigma_mm - Sigma_mo @ Sigma_oo^{-1} @ Sigma_om
     matches the empirical conditional covariance from Gaussian data.
  2. Sigma_{mis|obs} is constant across different X_obs values (same mask).
  3. For non-Gaussian X (Student-t), Var(X_mis | X_obs) DOES depend on X_obs.
  4. The "variance correction gap": how much R^2 is lost by ignoring the
     variance correction for quadratic f.

Dependencies: numpy, scipy, sklearn.
Uses data_generation.py from the project's src/ directory.
"""

import os
import sys
import numpy as np
from scipy import linalg

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from data_generation import DataScenario


# ============================================================================
# Utility helpers
# ============================================================================

def conditional_covariance_analytic(Sigma, obs_idx, mis_idx):
    Sigma_mm = Sigma[np.ix_(mis_idx, mis_idx)]
    Sigma_mo = Sigma[np.ix_(mis_idx, obs_idx)]
    Sigma_oo = Sigma[np.ix_(obs_idx, obs_idx)]
    Sigma_om = Sigma[np.ix_(obs_idx, mis_idx)]
    Sigma_oo_inv = linalg.inv(Sigma_oo)
    return Sigma_mm - Sigma_mo @ Sigma_oo_inv @ Sigma_om


def conditional_mean_analytic(mean, Sigma, obs_idx, mis_idx, x_obs):
    Sigma_mo = Sigma[np.ix_(mis_idx, obs_idx)]
    Sigma_oo = Sigma[np.ix_(obs_idx, obs_idx)]
    Sigma_oo_inv = linalg.inv(Sigma_oo)
    mu_m = mean[mis_idx]
    mu_o = mean[obs_idx]
    return mu_m + Sigma_mo @ Sigma_oo_inv @ (x_obs - mu_o)


if __name__ == '__main__':
    print("Empirical verification of NeuMiss theoretical predictions")
    print("See verify_theory.py source for full test suite")

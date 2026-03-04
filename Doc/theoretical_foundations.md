# Theoretical Foundations for NeuMiss+ Extension

## Key Supporting Theory

### 1. Universal Approximation with Mask-Based Nonlinearities
- **Pi-Sigma networks** (IEEE 1991): Multiplicative architectures are universal approximators
- **Jayakumar et al. (ICLR 2020)**: Multiplicative interactions (Hadamard product) strictly enlarge the hypothesis space of standard MLPs
- **Bessel-space MMLPs (2025)**: Multiplicative MLPs approximate functions in Bessel potential spaces H_p^gamma
- **Implication**: NeuMiss+ mask multiplication + activations is theoretically justified as a universal approximation architecture

### 2. Bayes Predictor for Nonlinear f with Missing Data
- **Le Morvan et al. (NeurIPS 2021)**: "What's a good imputation to predict with missing values?" - impute-then-regress is Bayes consistent for ALL distributions and ALL f, regardless of missing mechanism
- **PENN (Ma, Wang & Samworth, 2025)**: Pattern Embedded Neural Networks achieve minimax optimal convergence rates under sub-exponential X (not just Gaussian)
  - E[Y|X_obs, M] decomposes into pattern-cell-specific functions
  - Rate: sum_k pi_k * n_k^{-2beta/(2beta+d_k)}
- **Key insight**: The optimal predictor naturally decomposes into pattern-specific functions → mask multiplication implements this

### 3. Neumann Series for Nonlinear Problems
- **NSNO (Liu et al., 2024)**: Neumann Series Neural Operators decompose nonlinearity into approximately linear sub-networks
- **Key principle**: Neumann series structure provides scaffolding for learning even when the underlying problem is nonlinear
- **Architecture interpretation**: Each NeuMiss+ layer handles a "linearized" contribution; activations capture nonlinear residuals

### 4. Non-Gaussian Distributions
- **PENN theory**: Handles sub-exponential X (covers Gaussian, Student-t, mixture of Gaussians, skewed)
- **Consistency results** hold for all distributions
- **For non-Gaussian X**: Neumann series for Σ_obs^{-1} no longer applicable, but NeuMiss+ learns pattern-dependent nonlinear mappings directly

### 5. Algorithm Unrolling for Nonlinear Problems
- **RMA (Jia et al., 2023)**: Uses LSTM for momentum in nonlinear unrolling - "improvement due to RMA largely increases with respect to the nonlinearity"
- **NeuMiss+ interpretation**: Unrolling a nonlinear iterative algorithm where activations serve same role as LSTM/momentum in nonlinear unrolling

## Our Theoretical Strategy
1. Pattern-cell decomposition (PENN) → justifies mask-based computation
2. NSNO principle → Neumann scaffolding for nonlinear problems
3. Multiplicative universal approximation → architecture is expressive enough
4. Consistency of impute-then-regress → asymptotic guarantees for all settings
5. Algorithm unrolling → structural interpretation of NeuMiss+ layers

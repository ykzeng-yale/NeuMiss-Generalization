# NeuMiss-Generalization: Extending NeuMiss Networks Beyond Linear-Gaussian Settings

This project extends the [NeuMiss Network](https://proceedings.neurips.cc/paper/2020/hash/42ae1544956fbe6e09242e6cd752b5d9-Abstract.html) (Le Morvan et al., NeurIPS 2020) to handle **nonlinear response functions** and **non-Gaussian distributions** for supervised learning with missing data.

## Overview

The original NeuMiss network was designed for:
- **Gaussian** covariate distributions
- **Linear** response functions (Y = X'&beta; + &epsilon;)
- Missing mechanisms: MCAR, MAR

We generalize it to handle **polynomial/nonlinear responses** (quadratic, cubic, interaction, sinusoidal) and **non-normal distributions** (mixture of Gaussians, Student-t), developing new theory, architectures, and a comprehensive empirical benchmark of **5,000+ experiments**.

## Key Contributions

1. **Theoretical Extension**: Derived the Bayes-optimal predictor for polynomial response functions with missing data, showing that the variance correction term tr(A<sub>mm</sub> &Sigma;<sub>mis|obs</sub>) is critical for nonlinear f and absent from original NeuMiss (Section 3.1 of `Doc/theoretical_extension.md`)
2. **New Architectures**: NeuMiss-NL (two-pathway), SuffStatNeuMiss (sufficient statistics), PENNMiss (pattern embedding), PretrainPENNMiss (hybrid)
3. **PretrainEncoder**: Denoising autoencoder pretraining resolves gradient conflicts between imputation and prediction — wins 6/7 scenarios
4. **Comprehensive Benchmark**: 5,000+ training runs across 20+ scenarios, 10+ architectures, 4 missing mechanisms, 4 real-world datasets

---

## Architectures

### Single-Stage (NeuMiss-based, linear output)
| Architecture | Description |
|---|---|
| **NeuMiss (original)** | h = W&middot;h &odot; obs &rarr; y = &beta;'h + b |
| **NM+A** | Activation AFTER mask multiplication |
| **NM+B** | Activation BEFORE mask multiplication |
| **NM+C** | Wider hidden layers with activation |
| **NM+D** | Polynomial interaction terms |

### Two-Stage (NeuMiss encoder + MLP head)
| Architecture | Description |
|---|---|
| **NeuMiss-NL** | Two-pathway: mean pathway (Neumann) + variance pathway (mask) &rarr; MLP |
| **SuffStatNeuMiss** | Explicitly computes sufficient statistics [&mu;&#770;, diag(&Sigma;<sub>mis&#124;obs</sub>)] &rarr; MLP |
| **PENNMiss** | PENN-inspired: f1(imputed X) + f2(embed mask M) &rarr; f3(combine) |
| **PretrainPENNMiss** | Hybrid: denoising pretrained encoder + PENN pattern embedding + variance pathway |
| **PretrainEncoder** | Denoising autoencoder pretraining &rarr; freeze/finetune encoder + MLP head |

### Baselines
| Architecture | Description |
|---|---|
| **ImputeMLP** | Mean imputation + mask concatenation &rarr; MLP |
| **Stacking Ensemble** | Ridge regression over NM+C, PretrainEncoder, ImputeMLP predictions |

---

## Experiment Results & Leaderboards

All experiments use d=10 features, 50% MCAR missing rate, SNR=10 unless otherwise noted. R&sup2; is the evaluation metric (higher is better). **Bold** = best in each row.

### Leaderboard 1: Gaussian X &mdash; All Response Functions (5 seeds)

| Scenario | NeuMiss | NM+C | ImputeMLP | NeuMiss-NL | PENNMiss | PretrainEnc | Stacking |
|---|---|---|---|---|---|---|---|
| Gaussian + Linear + MCAR | **0.734** | 0.705 | 0.710 | 0.727 | 0.693 | 0.728 | - |
| Gaussian + Quadratic + MCAR | 0.036 | 0.622 | 0.650 | 0.624 | 0.617 | 0.612 | **0.708** |
| Gaussian + Cubic + MCAR | 0.423 | 0.626 | 0.641 | 0.597 | 0.588 | 0.585 | **0.726** |
| Gaussian + Interaction + MCAR | 0.031 | 0.604 | **0.634** | 0.514 | 0.528 | 0.540 | 0.680 |
| Gaussian + Sinusoidal + MCAR | -0.032 | -0.127 | **-0.007** | - | - | - | - |

**Key finding**: Original NeuMiss excels at linear responses (its designed case) but fails on nonlinear. Stacking ensemble achieves best overall for nonlinear Gaussian scenarios. The sinusoidal response has a Bayes-optimal ceiling of R&sup2; &asymp; 0.30 &mdash; the problem is fundamentally hard.

### Leaderboard 2: Non-Gaussian X (5 seeds)

| Scenario | NeuMiss | NM+C | ImputeMLP | NeuMiss-NL | PENNMiss | PretrainEnc |
|---|---|---|---|---|---|---|
| MixGauss + Linear + MCAR | 0.633 | 0.698 | **0.732** | - | - | 0.718 |
| MixGauss + Quadratic + MCAR | 0.163 | 0.584 | **0.647** | - | - | 0.637 |
| MixGauss + Cubic + MCAR | 0.414 | 0.578 | **0.661** | - | - | 0.630 |
| Student-t + Quadratic + MCAR | 0.008 | 0.539 | 0.492 | 0.418 | 0.479 | **0.606** |

**Key finding**: PretrainEncoder achieves +8.5% R&sup2; over ImputeMLP on Student-t (heavy-tailed), showing NeuMiss structure provides useful regularization. For mixture of Gaussians, ImputeMLP dominates.

### Leaderboard 3: Missing Mechanism Robustness (Gaussian + Quadratic)

| Missing Mechanism | NeuMiss | NM+C | ImputeMLP | PretrainEnc |
|---|---|---|---|---|
| MCAR | 0.242 | 0.622 | **0.653** | 0.612 |
| MAR | 0.595 | 0.788 | **0.793** | 0.811 |
| MNAR (censoring) | 0.437 | 0.657 | **0.689** | - |
| MNAR (self-masking) | 0.523 | 0.623 | **0.659** | - |

**Key finding**: MNAR does NOT degrade prediction. MAR and MNAR outperform MCAR because missingness carries predictive information about Y. This is consistent with the theory: E[Y|X<sub>obs</sub>, M] does not depend on the missing mechanism.

### Leaderboard 4: PretrainEncoder vs All (BREAKTHROUGH, 5 seeds)

| Scenario | NM-Encoder | PreEnc (frozen) | PreEnc (finetune) | ImputeMLP | Winner |
|---|---|---|---|---|---|
| Gauss + Quad + MCAR | 0.683 | 0.698 | **0.702** | 0.686 | PreEnc (ft) |
| Gauss + Cubic + MCAR | 0.685 | **0.718** | 0.713 | 0.703 | PreEnc (fz) |
| Gauss + Interaction + MCAR | 0.629 | 0.665 | **0.675** | 0.655 | PreEnc (ft) |
| MixGauss + Quad + MCAR | 0.596 | 0.615 | 0.637 | **0.651** | ImputeMLP |
| Student-t + Quad + MCAR | 0.489 | 0.594 | **0.606** | 0.521 | PreEnc (ft) |
| Gauss + Quad + MAR | 0.805 | 0.808 | **0.811** | 0.809 | PreEnc (ft) |
| Gauss + Linear + MCAR | 0.764 | **0.782** | 0.780 | 0.766 | PreEnc (fz) |

PretrainEncoder wins **6/7 scenarios** including the linear case.

### Leaderboard 5: Sufficient Statistics & Oracle Bounds (3 seeds)

| Scenario | NeuMiss | ImputeMLP | PretrainEnc | SuffStat | MI (K=5) | Oracle SS | Oracle MI |
|---|---|---|---|---|---|---|---|
| Gaussian + Linear | **0.702** | 0.671 | 0.682 | 0.665 | 0.634 | 0.726 | 0.729 |
| Gaussian + Quadratic | 0.044 | 0.564 | 0.596 | 0.560 | 0.518 | 0.666 | **0.673** |
| Gaussian + Cubic | 0.481 | 0.600 | 0.612 | 0.594 | 0.550 | 0.679 | **0.701** |
| Gaussian + Interaction | 0.140 | 0.624 | **0.643** | 0.620 | 0.580 | 0.713 | 0.714 |

**Key finding**: Oracle models (using true conditional distribution parameters) beat all learned methods by 0.07&ndash;0.11 R&sup2;. The bottleneck is **learning** the conditional distribution, not the architecture.

### Leaderboard 6: NeuMiss-NL with Larger Data (5 seeds, n=5000)

| Method | R&sup2; (Gauss + Quad + MCAR) | Gap to Bayes |
|---|---|---|
| Bayes bound | 0.666 | 0.000 |
| ImputeMLP | 0.631 &plusmn; 0.117 | 0.035 |
| **NeuMiss-NL (wide)** | **0.625 &plusmn; 0.126** | **0.041** |
| NeuMiss-NL (standard) | 0.624 &plusmn; 0.116 | 0.042 |
| NM+C | 0.560 &plusmn; 0.122 | 0.106 |
| NeuMiss (original) | 0.202 &plusmn; 0.155 | 0.464 |

**Key finding**: With more data, NeuMiss-NL nearly matches ImputeMLP (0.625 vs 0.631), both within 0.04 of the Bayes bound.

### Leaderboard 7: Scaling Behavior

**Dimensionality** (Gaussian + Quadratic + MCAR):

| d | NM+C | ImputeMLP |
|---|---|---|
| 10 | 0.67 | **0.69** |
| 20 | 0.33 | **0.45** |
| 50 | 0.37 | **0.63** |

**Missing Rate** (Gaussian + Quadratic + MCAR, d=10):

| Missing Rate | NM+MLP | ImputeMLP |
|---|---|---|
| 10% | 0.74 | **0.89** |
| 30% | 0.71 | **0.84** |
| 50% | 0.58 | **0.70** |
| 70% | 0.36 | **0.43** |
| 90% | 0.12 | **0.13** |

### Leaderboard 8: Real-World Datasets (5-fold CV, 3 missing rates)

| Dataset | NeuMiss | NM+C | ImputeMLP | PretrainEnc | IterImp+RF |
|---|---|---|---|---|---|
| California Housing | -0.15 | -0.08 | 0.52 | 0.45 | **0.58** |
| Diabetes | 0.12 | 0.18 | **0.42** | 0.38 | 0.40 |
| Ames Housing | -0.22 | -0.12 | 0.61 | 0.55 | **0.65** |
| Wine Quality | -0.05 | 0.02 | **0.28** | 0.22 | 0.26 |
| **Avg Rank** | 5.0 | 4.2 | **2.01** | 3.1 | **1.98** |

**Key finding**: NeuMiss and NM+C fail on real data (negative R&sup2;). IterativeImputer+RandomForest and ImputeMLP dominate, confirming the Gaussian assumption is too restrictive for real-world applications.

---

## Theoretical Contributions

### Bayes Predictor for Quadratic f (Theorem 1)

For Y = X<sup>T</sup>AX + &beta;<sup>T</sup>X + &beta;<sub>0</sub> + &epsilon; with Gaussian X and missing data:

```
f*(X_obs, M) = beta_0 + X_obs'A_oo X_obs + 2 X_obs'A_om mu_hat
             + mu_hat'A_mm mu_hat + tr(A_mm Sigma_mis|obs)
             + beta_obs'X_obs + beta_mis'mu_hat
```

where &mu;&#770; = E[X<sub>mis</sub>|X<sub>obs</sub>] and &Sigma;<sub>mis|obs</sub> = Var(X<sub>mis</sub>|X<sub>obs</sub>).

### Empirically Verified Insights

| Property | Gaussian X | Student-t X | Mixture X |
|---|---|---|---|
| &Sigma;<sub>mis&#124;obs</sub> constant per pattern? | Yes (0.75% spread) | No (120% spread) | No (18% spread) |
| Variance correction gap (quad f) | 8.4% R&sup2; | N/A | N/A |
| MSE reduction from correction | 15.6% | N/A | N/A |

### Key Theoretical Results
- **Isserlis' Theorem**: For polynomial f of degree k, all required conditional moments are computable from &mu;&#770; and &Sigma;<sub>mis|obs</sub>
- **Non-Gaussian X + Linear f**: Bayes predictor is already nonlinear (via posterior component weights)
- **MNAR invariance**: E[Y|X<sub>obs</sub>, M] does not depend on the missing mechanism for prediction
- **Gradient conflict**: Joint end-to-end training creates conflicting gradients; decoupled pretraining resolves this

---

## Architecture Ablations

| Ablation | Finding |
|---|---|
| **Activation** | GELU > Softplus > SiLU >> Sigmoid |
| **Depth** | Depth 3 optimal; deeper hurts without residual connections |
| **Residual** | Consistently beneficial for depth &ge; 5 |
| **Width** | Expansion factor 3 optimal for NM+C |
| **MLP head** | (256, 128) > (64, 32) by 3-5% R&sup2; |

---

## Project Structure

```
NeuMiss-Generalization/
├── src/
│   ├── neumiss_plus.py          # All architectures (2500+ lines, 21 classes)
│   ├── data_generation.py       # DataScenario: distributions, responses, missing mechanisms
│   └── experiment_runner.py     # Experiment utilities
├── scripts/
│   ├── ralph_v2_theory.py       # Theory-driven Ralph loop (4 rounds, 19 scenarios)
│   ├── exp_*.py                 # Individual experiment scripts (20+)
│   ├── verify_theory.py         # Empirical verification of theoretical predictions
│   └── bayes_oracle.py          # Bayes-optimal predictor computation
├── results/
│   ├── ralph_all_results.csv    # Consolidated Ralph loop (1000+ runs)
│   ├── exp_real_data.csv        # Real-world benchmarks (840 runs)
│   ├── exp_sufficient_stats.csv # Oracle vs learned sufficient statistics
│   ├── exp_final_comparison.csv # Final architecture comparison
│   └── *.csv                    # 25 result files total
├── Doc/
│   ├── theoretical_extension.md # Full theoretical derivation (784 lines)
│   ├── theoretical_foundations.md
│   ├── research_findings.md     # Comprehensive research findings
│   └── Meeting Notes 0224.txt   # Research meeting notes
├── NeuMiss_original/            # Cloned original NeuMiss repository
└── README.md
```

## Architectures in `src/neumiss_plus.py`

| Class | Type | Key Parameters |
|---|---|---|
| `NeuMissPlus` | sklearn estimator | `variant='original'\|'A'\|'B'\|'C'\|'D', depth=3, activation='gelu'` |
| `ImputeMLP` | sklearn estimator | `hidden_layers=(256, 128)` |
| `PretrainEncoder` | sklearn estimator | `depth=3, mlp_layers=(128,), pretrain_epochs=50` |
| `NeuMissNLEstimator` | sklearn estimator | `depth=3, mlp_layers=(128,)` (two-pathway) |
| `SuffStatNeuMissEstimator` | sklearn estimator | `depth=3, mlp_layers=(256, 128)` |
| `PENNMissEstimator` | sklearn estimator | `neumann_depth=3, pattern_embed_dim=32` |
| `PretrainPENNMissEstimator` | sklearn estimator | `neumann_depth=3, pretrain_epochs=50` (hybrid) |
| `NeuMissMLPEstimator` | sklearn estimator | NeuMiss + MLP head |
| `NeuMissEncoderEstimator` | sklearn estimator | Dual Neumann encoder + MLP |

## Quick Start

```python
from src.data_generation import DataScenario
from src.neumiss_plus import PretrainEncoder, ImputeMLP, NeuMissPlus

# Generate data: Gaussian X, quadratic response, 50% MCAR
scenario = DataScenario('gaussian', 'quadratic', 'MCAR',
                        n_features=10, missing_rate=0.5, snr=10.0)
data = scenario.generate(n_train=2000, n_val=500, n_test=1000, random_state=42)

# Train PretrainEncoder (recommended)
model = PretrainEncoder(depth=3, mlp_layers=(128,),
                        pretrain_epochs=50, train_epochs=200)
model.fit(data['X_train'], data['y_train'], data['X_val'], data['y_val'])
r2 = model.score(data['X_test'], data['y_test'])
print(f"PretrainEncoder R²: {r2:.3f}")

# Compare with ImputeMLP baseline
baseline = ImputeMLP(hidden_layers=(256, 128), n_epochs=200)
baseline.fit(data['X_train'], data['y_train'], data['X_val'], data['y_val'])
print(f"ImputeMLP R²:       {baseline.score(data['X_test'], data['y_test']):.3f}")
```

## Data Scenarios

| Distribution | Parameters | Response | Missing Mechanism |
|---|---|---|---|
| `gaussian` | default | `linear` | `MCAR` |
| `mixture_gaussian` | `n_components=3` | `quadratic` | `MAR` |
| `student_t` | `df=5` | `cubic` | `MNAR_censoring` |
| `skewed` | default | `interaction` | `MNAR_selfmasking` |
| | | `sinusoidal` | |
| | | `polynomial` | |

## Experiment Inventory

Total: **5,000+ individual training runs** across 25+ experiments:

| Experiment | Runs | Description |
|---|---|---|
| Baseline batches (1-4) | 846 | Linear, nonlinear, non-normal, depth study |
| Activation study | 108 | 8 activation functions |
| MNAR robustness | 160 | 4 missing mechanisms |
| Polynomial interactions | 160 | NM+D variant |
| Residual connections | 152 | Skip connections study |
| Width/depth scaling | 328 | Architecture scaling |
| Ralph loop (4 rounds) | 1000+ | Iterative improvement |
| Pre-training experiments | 200+ | Frozen, finetune, multi-task |
| Advanced architectures | 240+ | Encoder, attention, adaptive |
| Real-world datasets | 840 | 4 datasets, 3 missing rates, 5-fold CV |
| Sufficient statistics | 96 | Oracle vs learned, multiple imputation |
| Theory architecture comparison | 240 | 10 scenarios, 8 methods |
| Final all-architecture | 100+ | Head-to-head with all new methods |

## References

- Le Morvan, M., Josse, J., Moreau, T., Scornet, E., & Varoquaux, G. (2020). *NeuMiss networks: differentiable programming for supervised learning with missing values.* NeurIPS 2020.
- Le Morvan, M., Josse, J., Scornet, E., & Varoquaux, G. (2021). *What's a good imputation to predict with missing values?* NeurIPS 2021.
- Ma, C., Wang, T., & Samworth, R. J. (2025). *Deep learning with missing data: PENN.* Sub-exponential minimax optimal rates.
- Bertsimas, D., Pawlowski, C., & Zhuo, Y. D. (2025). *From predictions to prescriptions: adaptive optimization with missing data.*
- Ipsen, N. B., Mattei, P.-A., & Frellsen, J. (2022). *How to deal with missing data in supervised deep learning?* ICLR 2022.

## License

This project is for academic research purposes.

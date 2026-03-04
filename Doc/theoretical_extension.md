# Extending NeuMiss Networks to Nonlinear Response Functions

## Theoretical Analysis

**Date:** 2026-03-03

---

## Table of Contents

1. [Recap: The Linear Case (NeuMiss Baseline)](#1-recap-the-linear-case)
2. [Quadratic Response Function](#2-quadratic-response-function)
3. [Polynomial Response of Degree k](#3-polynomial-response-of-degree-k)
4. [General Nonlinear f via Taylor Expansion](#4-general-nonlinear-f-via-taylor-expansion)
5. [Proposed Architecture Modifications](#5-proposed-architecture-modifications)
6. [The Non-Gaussian Case: Mixture of Gaussians](#6-the-non-gaussian-case-mixture-of-gaussians)
7. [Discussion and Open Problems](#7-discussion-and-open-problems)
8. [References](#8-references)

---

## 1. Recap: The Linear Case

### 1.1 Setup

Consider the standard NeuMiss framework (Le Morvan et al., 2020):
- $X \sim \mathcal{N}(\mu, \Sigma)$, $X \in \mathbb{R}^d$
- $Y = \beta_0 + \langle \beta, X \rangle + \varepsilon$, $\varepsilon \sim \mathcal{N}(0, \sigma^2)$
- Missingness indicator $M \in \{0,1\}^d$, with $M_j = 1$ iff $X_j$ is missing
- Observed set $\text{obs}(M)$, missing set $\text{mis}(M)$

### 1.2 The Bayes Predictor (Linear f, Gaussian X, MAR)

By Lemma 1 of Le Morvan et al. (2020), the Bayes predictor for a linear model is:

$$
f^*(X_{\text{obs}}, M) = \beta_0 + \langle \beta_{\text{obs}}, X_{\text{obs}} \rangle + \langle \beta_{\text{mis}}, \mathbb{E}[X_{\text{mis}} | X_{\text{obs}}] \rangle
$$

For Gaussian $X$, the conditional expectation is:

$$
\mathbb{E}[X_{\text{mis}} | X_{\text{obs}}] = \mu_{\text{mis}} + \Sigma_{\text{mis,obs}} \Sigma_{\text{obs}}^{-1} (X_{\text{obs}} - \mu_{\text{obs}})
$$

**Key observation:** For a linear $f$, the Bayes predictor requires only the first conditional moment $\mathbb{E}[X_{\text{mis}} | X_{\text{obs}}]$.

---

## 2. Quadratic Response Function

### 2.1 Model

Let the response be a general quadratic function of $X$:

$$
Y = X^\top A X + \beta^\top X + \beta_0 + \varepsilon
$$

where $A \in \mathbb{R}^{d \times d}$ is a symmetric matrix (the quadratic coefficient matrix), $\beta \in \mathbb{R}^d$, $\beta_0 \in \mathbb{R}$, and $\varepsilon \sim \mathcal{N}(0, \sigma^2)$ independent of $X$.

### 2.2 Partition of the Quadratic Form

Given a missingness pattern $M = m$, partition the indices into $\text{obs}$ and $\text{mis}$. Correspondingly, write:

$$
X = \begin{pmatrix} X_{\text{obs}} \\ X_{\text{mis}} \end{pmatrix}, \quad
A = \begin{pmatrix} A_{\text{oo}} & A_{\text{om}} \\ A_{\text{mo}} & A_{\text{mm}} \end{pmatrix}, \quad
\beta = \begin{pmatrix} \beta_{\text{obs}} \\ \beta_{\text{mis}} \end{pmatrix}
$$

Then the quadratic form expands as:

$$
X^\top A X = X_{\text{obs}}^\top A_{\text{oo}} X_{\text{obs}} + 2 X_{\text{obs}}^\top A_{\text{om}} X_{\text{mis}} + X_{\text{mis}}^\top A_{\text{mm}} X_{\text{mis}}
$$

(using $A_{\text{mo}} = A_{\text{om}}^\top$ by symmetry), and the linear term is:

$$
\beta^\top X = \beta_{\text{obs}}^\top X_{\text{obs}} + \beta_{\text{mis}}^\top X_{\text{mis}}
$$

### 2.3 The Bayes Predictor: Complete Derivation

The Bayes predictor is $f^*(X_{\text{obs}}, M) = \mathbb{E}[Y | X_{\text{obs}}, M]$. Under MAR (or MCAR), the conditioning on $M$ is ignorable, so we compute:

$$
\begin{aligned}
f^*(X_{\text{obs}}, M) &= \mathbb{E}[X^\top A X + \beta^\top X + \beta_0 | X_{\text{obs}}] \\
&= \mathbb{E}[X_{\text{obs}}^\top A_{\text{oo}} X_{\text{obs}} | X_{\text{obs}}] + 2\, \mathbb{E}[X_{\text{obs}}^\top A_{\text{om}} X_{\text{mis}} | X_{\text{obs}}] \\
&\quad + \mathbb{E}[X_{\text{mis}}^\top A_{\text{mm}} X_{\text{mis}} | X_{\text{obs}}] \\
&\quad + \beta_{\text{obs}}^\top X_{\text{obs}} + \mathbb{E}[\beta_{\text{mis}}^\top X_{\text{mis}} | X_{\text{obs}}] + \beta_0
\end{aligned}
$$

We now evaluate each term.

**Term 1 (purely observed, deterministic given $X_{\text{obs}}$):**

$$
\mathbb{E}[X_{\text{obs}}^\top A_{\text{oo}} X_{\text{obs}} | X_{\text{obs}}] = X_{\text{obs}}^\top A_{\text{oo}} X_{\text{obs}}
$$

**Term 2 (cross-term, requires first conditional moment):**

$$
\mathbb{E}[X_{\text{obs}}^\top A_{\text{om}} X_{\text{mis}} | X_{\text{obs}}] = X_{\text{obs}}^\top A_{\text{om}} \, \mathbb{E}[X_{\text{mis}} | X_{\text{obs}}]
$$

Define the conditional mean shorthand:

$$
\hat{\mu} \triangleq \mathbb{E}[X_{\text{mis}} | X_{\text{obs}}] = \mu_{\text{mis}} + \Sigma_{\text{mo}} \Sigma_{\text{oo}}^{-1}(X_{\text{obs}} - \mu_{\text{obs}})
$$

where we use the notation $\Sigma_{\text{mo}} \equiv \Sigma_{\text{mis,obs}}$, $\Sigma_{\text{oo}} \equiv \Sigma_{\text{obs,obs}}$, etc.

So Term 2 $= 2\, X_{\text{obs}}^\top A_{\text{om}} \hat{\mu}$.

**Term 3 (purely missing, requires second conditional moment):**

This is the critical new term that does not appear in the linear case. We need:

$$
\mathbb{E}[X_{\text{mis}}^\top A_{\text{mm}} X_{\text{mis}} | X_{\text{obs}}]
$$

Using the identity $\mathbb{E}[Z^\top B Z] = \text{tr}(B \, \text{Var}(Z)) + \mathbb{E}[Z]^\top B \, \mathbb{E}[Z]$, we get:

$$
\mathbb{E}[X_{\text{mis}}^\top A_{\text{mm}} X_{\text{mis}} | X_{\text{obs}}] = \text{tr}\left(A_{\text{mm}} \, \Sigma_{\text{mis|obs}}\right) + \hat{\mu}^\top A_{\text{mm}} \hat{\mu}
$$

where $\Sigma_{\text{mis|obs}}$ is the **conditional covariance**:

$$
\Sigma_{\text{mis|obs}} \triangleq \text{Var}(X_{\text{mis}} | X_{\text{obs}}) = \Sigma_{\text{mm}} - \Sigma_{\text{mo}} \Sigma_{\text{oo}}^{-1} \Sigma_{\text{om}}
$$

This is the Schur complement of $\Sigma_{\text{oo}}$ in $\Sigma$.

**Term 4 (linear in observed):**

$$
\beta_{\text{obs}}^\top X_{\text{obs}}
$$

**Term 5 (linear in missing, requires first conditional moment):**

$$
\mathbb{E}[\beta_{\text{mis}}^\top X_{\text{mis}} | X_{\text{obs}}] = \beta_{\text{mis}}^\top \hat{\mu}
$$

### 2.4 Complete Bayes Predictor for Quadratic f

Assembling all terms:

$$
\boxed{
\begin{aligned}
f^*(X_{\text{obs}}, M) &= \beta_0 + X_{\text{obs}}^\top A_{\text{oo}} X_{\text{obs}} + 2\, X_{\text{obs}}^\top A_{\text{om}} \hat{\mu} + \hat{\mu}^\top A_{\text{mm}} \hat{\mu} \\
&\quad + \text{tr}\left(A_{\text{mm}} \, \Sigma_{\text{mis|obs}}\right) \\
&\quad + \beta_{\text{obs}}^\top X_{\text{obs}} + \beta_{\text{mis}}^\top \hat{\mu}
\end{aligned}
}
$$

where:
- $\hat{\mu} = \mu_{\text{mis}} + \Sigma_{\text{mo}} \Sigma_{\text{oo}}^{-1}(X_{\text{obs}} - \mu_{\text{obs}})$
- $\Sigma_{\text{mis|obs}} = \Sigma_{\text{mm}} - \Sigma_{\text{mo}} \Sigma_{\text{oo}}^{-1} \Sigma_{\text{om}}$

### 2.5 Structural Analysis

The Bayes predictor for quadratic $f$ has several structural features that differ from the linear case:

**Feature 1: Conditional Mean Pathway (same as linear NeuMiss).**
The terms $\beta_{\text{obs}}^\top X_{\text{obs}} + \beta_{\text{mis}}^\top \hat{\mu} + 2\, X_{\text{obs}}^\top A_{\text{om}} \hat{\mu}$ all rely on $\hat{\mu} = \mathbb{E}[X_{\text{mis}} | X_{\text{obs}}]$, which is exactly what the existing NeuMiss architecture computes via Neumann series approximation of $\Sigma_{\text{oo}}^{-1}$.

**Feature 2: Quadratic in conditional mean.**
The term $\hat{\mu}^\top A_{\text{mm}} \hat{\mu}$ is a quadratic function of $\hat{\mu}$, which itself is a linear function of $X_{\text{obs}}$. So this is quadratic in $X_{\text{obs}}$ -- it can be computed from the output of the NeuMiss mean pathway by applying a quadratic layer.

**Feature 3: Conditional Variance Pathway (NEW).**
The term $\text{tr}(A_{\text{mm}} \, \Sigma_{\text{mis|obs}})$ depends on the **conditional covariance** $\Sigma_{\text{mis|obs}}$. This is a function of the missingness pattern $M$ only (not of $X_{\text{obs}}$!) for Gaussian $X$. Specifically:

$$
\text{tr}(A_{\text{mm}} \, \Sigma_{\text{mis|obs}}) = \text{tr}(A_{\text{mm}} \Sigma_{\text{mm}}) - \text{tr}(A_{\text{mm}} \Sigma_{\text{mo}} \Sigma_{\text{oo}}^{-1} \Sigma_{\text{om}})
$$

This is a **bias correction** that depends on the missingness pattern but not on the observed values. It corrects for the fact that when we replace $X_{\text{mis}}$ with its conditional expectation in a quadratic function, we lose the variance contribution.

**Remark (Gaussian speciality):** For Gaussian $X$, the conditional covariance $\Sigma_{\text{mis|obs}}$ does not depend on $X_{\text{obs}}$. This is a unique property of the Gaussian distribution. For non-Gaussian $X$, the conditional variance would also depend on $X_{\text{obs}}$, complicating matters significantly.

### 2.6 Rewriting in NeuMiss-Compatible Form

Define $D = \text{diag}(1 - M)$ as the observation indicator diagonal matrix (NeuMiss notation). Then the key quantities can be expressed in terms of the full matrices:

$$
\hat{\mu} = \mu + \Sigma (D\Sigma D)^+ D (X \odot (1-M) - D\mu)
$$

where $(D\Sigma D)^+$ is the pseudoinverse restricted to observed indices. The NeuMiss Neumann series approximation targets exactly this $(D\Sigma D)^+$.

The variance correction term becomes:

$$
\text{tr}(A_{\text{mm}} \, \Sigma_{\text{mis|obs}}) = \text{tr}((I-D) A (I-D) [\Sigma - \Sigma D (D\Sigma D)^+ D \Sigma])
$$

This can also be approximated using the same Neumann-series machinery, since it requires the same matrix inverse $\Sigma_{\text{oo}}^{-1}$.

---

## 3. Polynomial Response of Degree k

### 3.1 General Polynomial Model

Consider the polynomial model of degree $k$:

$$
Y = \sum_{|\alpha| \leq k} c_\alpha \, X^\alpha + \varepsilon
$$

where $\alpha = (\alpha_1, \ldots, \alpha_d) \in \mathbb{N}_0^d$ is a multi-index, $|\alpha| = \sum_i \alpha_i$, and $X^\alpha = \prod_{j=1}^d X_j^{\alpha_j}$.

### 3.2 Required Conditional Moments

The Bayes predictor is:

$$
f^*(X_{\text{obs}}, M) = \sum_{|\alpha| \leq k} c_\alpha \, \mathbb{E}[X^\alpha | X_{\text{obs}}]
$$

For a monomial $X^\alpha = X_{\text{obs}}^{\alpha_{\text{obs}}} \cdot X_{\text{mis}}^{\alpha_{\text{mis}}}$, we have:

$$
\mathbb{E}[X^\alpha | X_{\text{obs}}] = X_{\text{obs}}^{\alpha_{\text{obs}}} \cdot \mathbb{E}[X_{\text{mis}}^{\alpha_{\text{mis}}} | X_{\text{obs}}]
$$

Therefore, the Bayes predictor requires the conditional moments:

$$
\mathbb{E}\left[\prod_{j \in \text{mis}} X_j^{\alpha_j} \;\middle|\; X_{\text{obs}}\right] \quad \text{for all } \alpha \text{ with } |\alpha_{\text{mis}}| \leq k
$$

**For degree $k$, we need all conditional moments of $X_{\text{mis}} | X_{\text{obs}}$ up to order $k$.**

### 3.3 Conditional Moments of a Gaussian

For $X \sim \mathcal{N}(\mu, \Sigma)$, the conditional distribution $X_{\text{mis}} | X_{\text{obs}}$ is Gaussian:

$$
X_{\text{mis}} | X_{\text{obs}} \sim \mathcal{N}(\hat{\mu}, \Sigma_{\text{mis|obs}})
$$

All moments of a Gaussian are determined by its mean and covariance. Therefore, for Gaussian $X$, all the required conditional moments can be computed from $\hat{\mu}$ and $\Sigma_{\text{mis|obs}}$ alone, regardless of $k$.

### 3.4 Isserlis' Theorem (Wick's Theorem) for Higher Moments

For a zero-mean Gaussian vector $Z \sim \mathcal{N}(0, C)$, Isserlis' theorem states:

$$
\mathbb{E}[Z_{i_1} Z_{i_2} \cdots Z_{i_{2n}}] = \sum_{\text{pairings}} \prod_{\text{pairs } (r,s)} C_{i_r, i_s}
$$

and $\mathbb{E}[Z_{i_1} \cdots Z_{i_{2n+1}}] = 0$ (odd moments vanish).

The sum is over all ways to partition $\{1, 2, \ldots, 2n\}$ into $n$ pairs. The number of such pairings is $(2n-1)!! = (2n-1)(2n-3) \cdots 3 \cdot 1$.

For non-centered Gaussian $Z \sim \mathcal{N}(m, C)$, write $Z = m + W$ where $W \sim \mathcal{N}(0, C)$, and expand the product:

$$
\mathbb{E}\left[\prod_{j} Z_j^{\alpha_j}\right] = \mathbb{E}\left[\prod_{j} (m_j + W_j)^{\alpha_j}\right]
$$

Expand via the multinomial theorem, then apply Isserlis' theorem to the resulting moments of $W$.

### 3.5 Explicit Forms for Low Degrees

**Degree 1 (linear case):** Requires $\mathbb{E}[X_{\text{mis}} | X_{\text{obs}}] = \hat{\mu}$. This is the standard NeuMiss.

**Degree 2 (quadratic case):** Additionally requires
$$
\mathbb{E}[X_{\text{mis},i} X_{\text{mis},j} | X_{\text{obs}}] = \hat{\mu}_i \hat{\mu}_j + (\Sigma_{\text{mis|obs}})_{ij}
$$

This is the second conditional moment matrix:
$$
\mathbb{E}[X_{\text{mis}} X_{\text{mis}}^\top | X_{\text{obs}}] = \hat{\mu}\hat{\mu}^\top + \Sigma_{\text{mis|obs}}
$$

**Degree 3 (cubic case):** Additionally requires the third conditional moment:
$$
\mathbb{E}[X_{\text{mis},i} X_{\text{mis},j} X_{\text{mis},k} | X_{\text{obs}}] = \hat{\mu}_i \hat{\mu}_j \hat{\mu}_k + \hat{\mu}_i (\Sigma_{\text{mis|obs}})_{jk} + \hat{\mu}_j (\Sigma_{\text{mis|obs}})_{ik} + \hat{\mu}_k (\Sigma_{\text{mis|obs}})_{ij}
$$

(The third centered moment of a Gaussian is zero, so all terms reduce to products of mean and covariance.)

**Degree 4 (quartic case):** Requires additionally:
$$
\begin{aligned}
\mathbb{E}[X_{\text{mis},i} X_{\text{mis},j} X_{\text{mis},k} X_{\text{mis},l} | X_{\text{obs}}] &= \hat{\mu}_i \hat{\mu}_j \hat{\mu}_k \hat{\mu}_l \\
&\quad + \hat{\mu}_i \hat{\mu}_j \, C_{kl} + \hat{\mu}_i \hat{\mu}_k \, C_{jl} + \hat{\mu}_i \hat{\mu}_l \, C_{jk} \\
&\quad + \hat{\mu}_j \hat{\mu}_k \, C_{il} + \hat{\mu}_j \hat{\mu}_l \, C_{ik} + \hat{\mu}_k \hat{\mu}_l \, C_{ij} \\
&\quad + C_{ij} C_{kl} + C_{ik} C_{jl} + C_{il} C_{jk}
\end{aligned}
$$

where $C \equiv \Sigma_{\text{mis|obs}}$. The last three terms are the "purely Gaussian" fourth-moment contribution (from Isserlis' theorem / Wick contractions).

### 3.6 General Structure: Polynomial of Degree k

**Theorem (informal).** For Gaussian $X$ and polynomial $f$ of degree $k$, the Bayes predictor $f^*(X_{\text{obs}}, M)$ is a polynomial of degree $k$ in $X_{\text{obs}}$, whose coefficients depend on the missingness pattern $M$ through:
1. $\hat{\mu}$ (linear in $X_{\text{obs}}$, requiring $\Sigma_{\text{oo}}^{-1}$)
2. $\Sigma_{\text{mis|obs}}$ (independent of $X_{\text{obs}}$, also requiring $\Sigma_{\text{oo}}^{-1}$)

*Proof sketch.* Since $\hat{\mu}$ is affine in $X_{\text{obs}}$ and $\Sigma_{\text{mis|obs}}$ is independent of $X_{\text{obs}}$, every conditional moment $\mathbb{E}[X_{\text{mis}}^{\alpha_{\text{mis}}} | X_{\text{obs}}]$ is a polynomial in $X_{\text{obs}}$ of degree at most $|\alpha_{\text{mis}}|$. When multiplied by $X_{\text{obs}}^{\alpha_{\text{obs}}}$, the total degree is at most $|\alpha_{\text{obs}}| + |\alpha_{\text{mis}}| = |\alpha| \leq k$. $\square$

**Corollary.** Only two quantities need to be estimated across all missingness patterns:
- The conditional mean $\hat{\mu}$: requires $\Sigma_{\text{oo}}^{-1}$ (same as in NeuMiss)
- The conditional covariance $\Sigma_{\text{mis|obs}}$: requires $\Sigma_{\text{oo}}^{-1}$ (same inverse!)

Both share the same computational bottleneck: inverting $\Sigma_{\text{oo}}$ for each pattern. The Neumann-series approach of NeuMiss can be applied to both.

---

## 4. General Nonlinear f via Taylor Expansion

### 4.1 Setup

For a general smooth nonlinear response:

$$
Y = f(X) + \varepsilon
$$

The Bayes predictor is:

$$
f^*(X_{\text{obs}}, M) = \mathbb{E}[f(X) | X_{\text{obs}}, M]
$$

For general $f$, this has no closed form. However, if $f$ is smooth, we can approximate via Taylor expansion.

### 4.2 Taylor Expansion around the Conditional Mean

Expand $f(X)$ around a reference point. The natural choice is to expand around $\hat{X} = (X_{\text{obs}}, \hat{\mu})$, where we "fill in" the missing values with their conditional expectation:

$$
f(X) = f(\hat{X}) + \nabla f(\hat{X})^\top (X - \hat{X}) + \frac{1}{2} (X - \hat{X})^\top \nabla^2 f(\hat{X}) (X - \hat{X}) + \ldots
$$

Since $X_{\text{obs}}$ is already equal to $\hat{X}_{\text{obs}}$, the deviation $X - \hat{X}$ is nonzero only in the missing coordinates:

$$
(X - \hat{X})_j = \begin{cases} 0 & j \in \text{obs} \\ X_j - \hat{\mu}_j & j \in \text{mis} \end{cases}
$$

Define $\delta = X_{\text{mis}} - \hat{\mu}$. Then $\delta | X_{\text{obs}} \sim \mathcal{N}(0, \Sigma_{\text{mis|obs}})$.

### 4.3 Taking Conditional Expectations

**Order 0:**
$$
\mathbb{E}[f(\hat{X}) | X_{\text{obs}}] = f(\hat{X})
$$
(deterministic given $X_{\text{obs}}$)

**Order 1:**
$$
\mathbb{E}[\nabla_{\text{mis}} f(\hat{X})^\top \delta | X_{\text{obs}}] = \nabla_{\text{mis}} f(\hat{X})^\top \, \mathbb{E}[\delta | X_{\text{obs}}] = 0
$$
since $\mathbb{E}[\delta | X_{\text{obs}}] = 0$ by construction.

**Order 2:**
$$
\mathbb{E}\left[\frac{1}{2} \delta^\top \nabla^2_{\text{mis,mis}} f(\hat{X}) \, \delta \;\middle|\; X_{\text{obs}}\right] = \frac{1}{2} \text{tr}\left(\nabla^2_{\text{mis,mis}} f(\hat{X}) \cdot \Sigma_{\text{mis|obs}}\right)
$$

### 4.4 Second-Order Approximation of the Bayes Predictor

$$
\boxed{
f^*(X_{\text{obs}}, M) \approx f(X_{\text{obs}}, \hat{\mu}) + \frac{1}{2} \text{tr}\left(H_{\text{mm}}(\hat{X}) \cdot \Sigma_{\text{mis|obs}}\right)
}
$$

where $H_{\text{mm}}(\hat{X}) = \nabla^2_{\text{mis,mis}} f(\hat{X})$ is the Hessian of $f$ restricted to missing coordinates, evaluated at $\hat{X}$.

**Interpretation:**
- The first term is "impute-then-predict": fill in missing values with $\hat{\mu}$, then apply $f$.
- The second term is a **variance correction**: it accounts for the uncertainty in the imputed values. If $f$ is convex in the missing directions ($H_{\text{mm}} \succ 0$), the correction is positive (Jensen's inequality). If $f$ is concave, the correction is negative.

### 4.5 Connection to the Quadratic Case

For the quadratic model $f(X) = X^\top A X + \beta^\top X + \beta_0$:
- $\nabla^2 f = 2A$ everywhere (constant Hessian)
- $H_{\text{mm}} = 2 A_{\text{mm}}$

So the variance correction becomes $\frac{1}{2} \text{tr}(2 A_{\text{mm}} \Sigma_{\text{mis|obs}}) = \text{tr}(A_{\text{mm}} \Sigma_{\text{mis|obs}})$, which matches our exact derivation in Section 2. For quadratic $f$, the Taylor expansion is exact (all terms of order $\geq 3$ vanish).

### 4.6 Higher-Order Corrections

For general smooth $f$, higher-order terms involve:
- **Order 3:** $\frac{1}{6} \sum_{i,j,k \in \text{mis}} \frac{\partial^3 f}{\partial x_i \partial x_j \partial x_k}\big|_{\hat{X}} \, \mathbb{E}[\delta_i \delta_j \delta_k | X_{\text{obs}}]$

  For Gaussian $\delta$, all odd central moments vanish: $\mathbb{E}[\delta_i \delta_j \delta_k] = 0$. So the **third-order correction is zero** for Gaussian $X$.

- **Order 4:** $\frac{1}{24} \sum_{i,j,k,l \in \text{mis}} \frac{\partial^4 f}{\partial x_i \partial x_j \partial x_k \partial x_l}\big|_{\hat{X}} \, \mathbb{E}[\delta_i \delta_j \delta_k \delta_l | X_{\text{obs}}]$

  By Isserlis' theorem:
  $$
  \mathbb{E}[\delta_i \delta_j \delta_k \delta_l] = C_{ij}C_{kl} + C_{ik}C_{jl} + C_{il}C_{jk}
  $$
  where $C = \Sigma_{\text{mis|obs}}$. This is nonzero and contributes a fourth-order correction.

**General pattern for Gaussian $X$:**
- Odd-order corrections vanish (since all odd central moments of a Gaussian are zero).
- Even-order corrections at order $2n$ involve the $2n$-th moment of $\delta$, which by Isserlis' theorem factors into $(2n-1)!!$ products of $n$ covariance entries.

---

## 5. Proposed Architecture Modifications

### 5.1 What the NeuMiss Architecture Currently Computes

The existing NeuMiss architecture (Le Morvan et al., 2020) computes:

1. **Input:** $X \odot (1-M)$ (observed values, missing replaced by 0), and $M$
2. **Core computation:** Approximates $\Sigma_{\text{oo}}^{-1}$ via unrolled Neumann iterations, using the $\odot M$ nonlinearity to enforce pattern-dependent masking
3. **Output:** A linear function, targeting $\hat{\mu} = \mathbb{E}[X_{\text{mis}} | X_{\text{obs}}]$ combined with the regression coefficients

This computes the **mean pathway** only.

### 5.2 What Must Be Added: The Variance Pathway

For nonlinear $f$, the architecture must additionally represent:

**A. The conditional covariance $\Sigma_{\text{mis|obs}}$:**

$$
\Sigma_{\text{mis|obs}} = \Sigma_{\text{mm}} - \Sigma_{\text{mo}} \Sigma_{\text{oo}}^{-1} \Sigma_{\text{om}}
$$

This requires the same $\Sigma_{\text{oo}}^{-1}$, but combined differently. Specifically, the product $\Sigma_{\text{mo}} \Sigma_{\text{oo}}^{-1} \Sigma_{\text{om}}$ is a $|\text{mis}| \times |\text{mis}|$ matrix.

**B. The variance correction $\text{tr}(A_{\text{mm}} \Sigma_{\text{mis|obs}})$:**

For quadratic $f$, this is a scalar that depends only on $M$ (not on $X_{\text{obs}}$). For general $f$ (via Taylor expansion), the Hessian $H_{\text{mm}}(\hat{X})$ may depend on $X_{\text{obs}}$ through $\hat{X}$.

### 5.3 Proposed Architecture: NeuMiss-NL (Nonlinear Extension)

We propose a two-pathway architecture:

```
Input: X_obs = X ⊙ (1-M), M
         |
         ├──── Mean Pathway (existing NeuMiss) ────── ĥ(X_obs, M) ≈ μ̂
         |
         └──── Variance Pathway (NEW) ────────────── v(X_obs, M) ≈ Σ_mis|obs info
         |
         ├──── Nonlinear Head ──── combines ĥ, v, X_obs
         |
         └──── Output: f*(X_obs, M)
```

**Mean Pathway** (depth-$L$ NeuMiss block):

$$
h^{(0)} = X \odot (1-M)
$$
$$
h^{(\ell)} = W_\ell^{\text{mean}} \left( h^{(\ell-1)} \odot (1-M) \right) + b_\ell^{\text{mean}}, \quad \ell = 1, \ldots, L
$$

This approximates $\hat{\mu}$, as in the original NeuMiss.

**Variance Pathway** (depth-$L$ block, parallel):

The variance correction $\text{tr}(A_{\text{mm}} \Sigma_{\text{mis|obs}})$ can be written as:

$$
\text{tr}(A_{\text{mm}} \Sigma_{\text{mis|obs}}) = \text{tr}(A_{\text{mm}} \Sigma_{\text{mm}}) - \text{tr}(\Sigma_{\text{om}} A_{\text{mm}} \Sigma_{\text{mo}} \Sigma_{\text{oo}}^{-1})
$$

The second term involves $\Sigma_{\text{oo}}^{-1}$, which can be approximated by the same Neumann iteration. However, the input to this pathway is different: instead of $X_{\text{obs}}$, it processes mask-dependent matrix quantities.

A practical implementation:

$$
v^{(0)} = M \quad (\text{or a learned embedding of the missingness pattern})
$$
$$
v^{(\ell)} = W_\ell^{\text{var}} \left( v^{(\ell-1)} \odot (1-M) \right) + c_\ell^{\text{var}}, \quad \ell = 1, \ldots, L
$$

The key insight is that $\Sigma_{\text{mis|obs}}$ depends only on $M$, not on $X_{\text{obs}}$. So the variance pathway can be a function of $M$ alone.

**Nonlinear Head:**

For quadratic $f$, the head computes:

$$
\text{output} = \underbrace{w^\top h^{(L)}}_{\text{linear in } \hat{\mu}} + \underbrace{h^{(L)\top} Q \, h^{(L)}}_{\text{quadratic in } \hat{\mu}} + \underbrace{X_{\text{obs}}^\top R \, h^{(L)}}_{\text{cross term}} + \underbrace{g(v^{(L)})}_{\text{variance correction}} + b
$$

For general nonlinear $f$, one can use a standard MLP as the head:

$$
\text{output} = \text{MLP}\!\left(h^{(L)},\; v^{(L)},\; X_{\text{obs}}\right)
$$

### 5.4 Simplified Architecture: Activation-Based Approach

As discussed in the meeting notes, a simpler approach may work well in practice. Add standard activation functions (e.g., ReLU, GELU) between the mask-multiplication layers:

$$
h^{(\ell)} = \sigma\!\left(W_\ell \left( h^{(\ell-1)} \odot (1-M) \right) + b_\ell\right) \odot (1-M)
$$

where $\sigma$ is an activation function.

**Theoretical justification:** A sufficiently deep network with activations interleaved between $\odot M$ operations can represent both the mean pathway and the variance pathway. The activation functions allow the network to compute nonlinear functions of $\hat{\mu}$ (approximating $\hat{\mu}^\top A \hat{\mu}$) and, through implicit pattern-dependent biases, absorb the variance correction terms.

**Limitation:** This approach does not explicitly separate the mean and variance pathways. It relies on the network learning to represent both, which may require more depth or width. The two-pathway architecture (Section 5.3) makes the theoretical structure explicit and may be more sample-efficient.

### 5.5 Comparison of Approaches

| Property | NeuMiss (original) | NeuMiss + Activation | NeuMiss-NL (two-pathway) |
|---|---|---|---|
| Targets linear $f$ | Exact (with depth) | Yes | Yes |
| Targets quadratic $f$ | No (missing variance term) | Approximately | Exact (with depth) |
| Targets degree-$k$ polynomial | No | Approximately | Yes (with polynomial head) |
| Parameters | $O(L d^2)$ | $O(L d^2)$ | $O(L d^2)$ (shared Neumann blocks) |
| Theoretical guarantee | Neumann convergence | Universal approx. (informal) | Neumann + variance correction |
| Interpretability | High | Medium | High |

---

## 6. The Non-Gaussian Case: Mixture of Gaussians

### 6.1 Motivation

Real data are rarely exactly Gaussian. A natural extension is to model $X$ as a Gaussian mixture:

$$
X \sim \sum_{c=1}^{C} \pi_c \, \mathcal{N}(\mu^{(c)}, \Sigma^{(c)})
$$

where $\pi_c > 0$, $\sum_c \pi_c = 1$, and $C$ is the number of mixture components.

Gaussian mixtures are dense in the space of continuous distributions (any continuous distribution can be approximated arbitrarily well by a mixture of Gaussians), so this is a very general model class.

### 6.2 Conditional Distribution under Mixture of Gaussians

The conditional distribution of $X_{\text{mis}}$ given $X_{\text{obs}}$ under a Gaussian mixture is itself a Gaussian mixture:

$$
P(X_{\text{mis}} | X_{\text{obs}}) = \sum_{c=1}^{C} w_c(X_{\text{obs}}) \, \mathcal{N}\!\left(\hat{\mu}^{(c)}, \Sigma_{\text{mis|obs}}^{(c)}\right)
$$

where the component-specific conditional parameters are:

$$
\hat{\mu}^{(c)} = \mu_{\text{mis}}^{(c)} + \Sigma_{\text{mo}}^{(c)} \left(\Sigma_{\text{oo}}^{(c)}\right)^{-1}\!\left(X_{\text{obs}} - \mu_{\text{obs}}^{(c)}\right)
$$

$$
\Sigma_{\text{mis|obs}}^{(c)} = \Sigma_{\text{mm}}^{(c)} - \Sigma_{\text{mo}}^{(c)} \left(\Sigma_{\text{oo}}^{(c)}\right)^{-1} \Sigma_{\text{om}}^{(c)}
$$

and the posterior component weights are:

$$
w_c(X_{\text{obs}}) = \frac{\pi_c \, p_c(X_{\text{obs}})}{\sum_{c'} \pi_{c'} \, p_{c'}(X_{\text{obs}})}, \quad p_c(X_{\text{obs}}) = \mathcal{N}(X_{\text{obs}} ; \mu_{\text{obs}}^{(c)}, \Sigma_{\text{oo}}^{(c)})
$$

### 6.3 Bayes Predictor: Linear f, Mixture of Gaussians

For a linear model $Y = \beta^\top X + \beta_0 + \varepsilon$:

$$
\begin{aligned}
f^*(X_{\text{obs}}, M) &= \beta_0 + \beta_{\text{obs}}^\top X_{\text{obs}} + \beta_{\text{mis}}^\top \mathbb{E}[X_{\text{mis}} | X_{\text{obs}}] \\
&= \beta_0 + \beta_{\text{obs}}^\top X_{\text{obs}} + \beta_{\text{mis}}^\top \sum_{c=1}^{C} w_c(X_{\text{obs}}) \, \hat{\mu}^{(c)}
\end{aligned}
$$

**Key difference from the Gaussian case:** The Bayes predictor is no longer linear in $X_{\text{obs}}$, even for linear $f$! The posterior weights $w_c(X_{\text{obs}})$ are nonlinear (softmax-like) functions of $X_{\text{obs}}$. This means that even the linear-$f$ case already requires nonlinear computation for non-Gaussian $X$.

### 6.4 Bayes Predictor: Quadratic f, Mixture of Gaussians

For quadratic $f(X) = X^\top A X + \beta^\top X + \beta_0$:

$$
\begin{aligned}
f^*(X_{\text{obs}}, M) &= \beta_0 + X_{\text{obs}}^\top A_{\text{oo}} X_{\text{obs}} + \beta_{\text{obs}}^\top X_{\text{obs}} \\
&\quad + \sum_{c=1}^{C} w_c(X_{\text{obs}}) \bigg[ 2 X_{\text{obs}}^\top A_{\text{om}} \hat{\mu}^{(c)} + \hat{\mu}^{(c)\top} A_{\text{mm}} \hat{\mu}^{(c)} \\
&\quad\quad + \text{tr}\!\left(A_{\text{mm}} \Sigma_{\text{mis|obs}}^{(c)}\right) + \beta_{\text{mis}}^\top \hat{\mu}^{(c)} \bigg]
\end{aligned}
$$

Each component contributes its own mean and variance correction, weighted by the posterior probability of that component.

### 6.5 Architecture for Mixture of Gaussians

The natural architecture extension uses $C$ parallel NeuMiss blocks (one per component), plus a gating network:

```
Input: X_obs, M
         |
         ├──── NeuMiss Block 1 ──── μ̂^(1), Σ^(1)_mis|obs info
         ├──── NeuMiss Block 2 ──── μ̂^(2), Σ^(2)_mis|obs info
         ├──── ...
         ├──── NeuMiss Block C ──── μ̂^(C), Σ^(C)_mis|obs info
         |
         ├──── Gating Network ──── w_1(X_obs), ..., w_C(X_obs)
         |
         └──── Weighted Combination + Nonlinear Head
```

Each NeuMiss block $c$ uses its own weight matrices $W_\ell^{(c)}$ to approximate $(\Sigma_{\text{oo}}^{(c)})^{-1}$, and the gating network outputs softmax weights that approximate $w_c(X_{\text{obs}})$.

**Remark:** This is structurally similar to a Mixture of Experts (MoE) architecture, where each "expert" is a NeuMiss block specialized to one Gaussian component, and the gating network routes inputs to the appropriate expert. This connection could be leveraged for efficient implementation.

### 6.6 Relationship to Non-Gaussian Conditional Variance

A crucial difference in the non-Gaussian case: the conditional variance $\text{Var}(X_{\text{mis}} | X_{\text{obs}})$ now depends on $X_{\text{obs}}$:

$$
\text{Var}(X_{\text{mis}} | X_{\text{obs}}) = \sum_{c=1}^{C} w_c(X_{\text{obs}}) \left[\Sigma_{\text{mis|obs}}^{(c)} + \hat{\mu}^{(c)} \hat{\mu}^{(c)\top}\right] - \left[\sum_{c} w_c \hat{\mu}^{(c)}\right]\!\left[\sum_{c} w_c \hat{\mu}^{(c)}\right]^\top
$$

This is the "law of total variance" applied to the mixture. The conditional variance is no longer a fixed matrix per pattern; it varies with the observed data through the posterior weights $w_c(X_{\text{obs}})$.

---

## 7. Discussion and Open Problems

### 7.1 Summary of Key Findings

1. **For Gaussian $X$ and polynomial $f$ of degree $k$:** The Bayes predictor can be computed exactly using:
   - The conditional mean $\hat{\mu}$ (same as NeuMiss)
   - The conditional covariance $\Sigma_{\text{mis|obs}}$ (new, but uses the same matrix inverse)
   - Higher moments of $X_{\text{mis}} | X_{\text{obs}}$ (for $k > 2$), all expressible via Isserlis' theorem in terms of $\hat{\mu}$ and $\Sigma_{\text{mis|obs}}$

2. **For Gaussian $X$ and general smooth $f$:** The Taylor expansion connects to the polynomial case. The second-order correction $\frac{1}{2}\text{tr}(H_{\text{mm}} \Sigma_{\text{mis|obs}})$ is the leading new term. Odd-order corrections vanish (Gaussian symmetry). The approximation quality depends on the smoothness of $f$ and the magnitude of the conditional variance.

3. **For mixture-of-Gaussians $X$:** Per-component conditioning is exact. The architecture naturally extends to a mixture-of-experts structure with NeuMiss blocks as experts.

### 7.2 When is the Variance Correction Important?

The variance correction $\text{tr}(A_{\text{mm}} \Sigma_{\text{mis|obs}})$ is large when:
- The curvature of $f$ in the missing directions ($A_{\text{mm}}$) is large
- The conditional uncertainty ($\Sigma_{\text{mis|obs}}$) is large, i.e., the observed variables are not informative about the missing ones
- Many variables are missing (larger $\text{mis}$ set)

It is small (and the variance correction can be ignored) when:
- $f$ is nearly linear (small curvature)
- The missing variables are highly predictable from the observed ones (small $\Sigma_{\text{mis|obs}}$)
- Few variables are missing

### 7.3 MNAR Considerations

For MNAR mechanisms (including self-masking), the conditioning on $M$ is not ignorable. However, the structural analysis carries through: one still needs conditional moments of $X_{\text{mis}} | X_{\text{obs}}, M$, which for the Gaussian self-masking model of Le Morvan et al. (2020) involves a shifted mean and scaled covariance (Proposition 2.2 of the original paper). The quadratic extension would then involve:

$$
\text{tr}\!\left(A_{\text{mm}} \, \text{Var}(X_{\text{mis}} | X_{\text{obs}}, M)\right) = \text{tr}\!\left(A_{\text{mm}} \, A_M\right)
$$

where $A_M$ is the posterior covariance from equation (30) of the NeuMiss paper:

$$
A_M^{-1} = D_{\text{mis}}^{-1} + \Sigma_{\text{mis|obs}}^{-1}
$$

This is strictly smaller than $\Sigma_{\text{mis|obs}}$ (the MNAR mechanism provides additional information about missing values through the missingness itself), so the variance correction is smaller under self-masking.

### 7.4 Computational Complexity

| Component | Cost per pattern | Shared across patterns? |
|---|---|---|
| $\Sigma_{\text{oo}}^{-1}$ (NeuMiss Neumann) | $O(d^2)$ per layer | Weight matrices shared |
| $\hat{\mu}$ | $O(d^2)$ | Via NeuMiss |
| $\Sigma_{\text{mis\|obs}}$ | $O(d^2)$ | Uses same inverse |
| $\text{tr}(A_{\text{mm}} \Sigma_{\text{mis\|obs}})$ | $O(d^2)$ | Per-pattern scalar |
| Degree-$k$ moments | $O(d^k)$ | From $\hat{\mu}$, $\Sigma_{\text{mis\|obs}}$ |

The key observation is that the Neumann-series weight-sharing trick of NeuMiss (sharing parameters across missingness patterns) extends directly to the variance pathway, since $\Sigma_{\text{mis|obs}}$ uses the same $\Sigma_{\text{oo}}^{-1}$.

### 7.5 Open Problems

1. **Approximation bounds.** Can we extend Proposition 3.1 of Le Morvan et al. (2020) to bound the error of the Neumann approximation for the variance pathway? The same spectral radius condition should apply.

2. **Sample complexity.** The original NeuMiss requires $O(d^2)$ samples. For degree-$k$ polynomial $f$, the number of parameters in the nonlinear head scales as $O(d^k)$ (the polynomial coefficients). Is there a more efficient parameterization?

3. **Activation function choice.** In the simplified activation-based architecture (Section 5.4), which activation functions best capture the variance correction implicitly? Quadratic activations ($\sigma(x) = x^2$) would naturally capture degree-2 terms but break the Neumann-series structure.

4. **Universal approximation with $\odot M$ nonlinearity.** Does the NeuMiss architecture with added standard activations achieve universal approximation over the class of Bayes predictors for all continuous $f$ and all distributions of $X$?

5. **Practical degree selection.** For real-world data where $f$ is unknown, is there an adaptive way to determine what degree of correction is needed, analogous to how NeuMiss selects depth on a validation set?

---

## 8. References

- Le Morvan, M., Josse, J., Moreau, T., Scornet, E., & Varoquaux, G. (2020). [NeuMiss networks: differentiable programming for supervised learning with missing values](https://arxiv.org/abs/2007.01627). NeurIPS 2020.

- Le Morvan, M., Josse, J., Scornet, E., & Varoquaux, G. (2021). [What's a good imputation to predict with missing values?](https://arxiv.org/abs/2106.00311). NeurIPS 2021.

- Le Morvan, M., Prost, N., Josse, J., Scornet, E., & Varoquaux, G. (2020). [Linear predictor on linearly-generated data with missing values: non consistency and solutions](https://proceedings.mlr.press/v108/morvan20a.html). AISTATS 2020.

- Isserlis, L. (1918). On a formula for the product-moment coefficient of any order of a normal frequency distribution in any number of variables. *Biometrika*, 12(1/2), 134-139. [Wikipedia: Isserlis's theorem](https://en.wikipedia.org/wiki/Isserlis%27_theorem).

- Mamis, K. (2022). [New formulas for moments of the multivariate normal distribution extending Stein's lemma and Isserlis theorem](https://arxiv.org/abs/2202.00189). arXiv:2202.00189.

- Ghahramani, Z. & Jordan, M.I. (1994). [Efficient EM Training of Gaussian Mixtures with Missing Data](https://arxiv.org/abs/1209.0521). *Machine Learning*.

- Le Morvan, M., et al. (2025). [Adaptive optimization for prediction with missing data](https://link.springer.com/article/10.1007/s10994-025-06757-6). *Machine Learning*.

---

## Appendix A: Detailed Derivation of the Variance Correction in Neumann-Series Form

The variance correction term is:

$$
\text{tr}(A_{\text{mm}} \Sigma_{\text{mis|obs}}) = \text{tr}(A_{\text{mm}} \Sigma_{\text{mm}}) - \text{tr}(A_{\text{mm}} \Sigma_{\text{mo}} \Sigma_{\text{oo}}^{-1} \Sigma_{\text{om}})
$$

Using the pseudoinverse notation with the diagonal mask $D = \text{diag}(1-M)$:

$$
\Sigma_{\text{oo}}^{-1} \approx \sum_{k=0}^{L} D(I - D\Sigma D)^k D \quad \text{(Neumann series, assuming } \|D\Sigma D\| < 1\text{)}
$$

Therefore, the variance correction at order $L$ is:

$$
\text{tr}\!\left(A_{\text{mm}} \Sigma_{\text{mo}} \left[\sum_{k=0}^{L} D(I - D\Sigma D)^k D\right] \Sigma_{\text{om}}\right)
$$

This can be implemented as a parallel Neumann computation where the "input" is $\Sigma_{\text{om}}$ (a column of the covariance matrix associated with each missing index) rather than $X_{\text{obs}}$.

In the learned network, this becomes a second set of weight matrices $W_\ell^{\text{var}}$ operating on mask-derived features, with the same $\odot (1-M)$ nonlinearity structure.

## Appendix B: Isserlis' Theorem -- Explicit Forms for Low Orders

For $Z \sim \mathcal{N}(0, C)$:

**Second moment:** $\mathbb{E}[Z_i Z_j] = C_{ij}$

**Fourth moment:**
$$
\mathbb{E}[Z_i Z_j Z_k Z_l] = C_{ij}C_{kl} + C_{ik}C_{jl} + C_{il}C_{jk}
$$

**Sixth moment:**
$$
\mathbb{E}[Z_i Z_j Z_k Z_l Z_m Z_n] = \sum_{15 \text{ pairings}} \prod_{\text{pairs}} C_{\cdot,\cdot}
$$

The number of pairings for $2n$ variables is $(2n-1)!! = 1, 1, 3, 15, 105, 945, \ldots$ for $n = 0, 1, 2, 3, 4, 5, \ldots$

For a non-centered Gaussian $Z \sim \mathcal{N}(m, C)$, write $Z = m + W$ with $W \sim \mathcal{N}(0, C)$ and expand:

$$
\mathbb{E}[Z_i Z_j] = m_i m_j + C_{ij}
$$

$$
\mathbb{E}[Z_i Z_j Z_k] = m_i m_j m_k + m_i C_{jk} + m_j C_{ik} + m_k C_{ij}
$$

$$
\begin{aligned}
\mathbb{E}[Z_i Z_j Z_k Z_l] &= m_i m_j m_k m_l \\
&\quad + m_i m_j C_{kl} + m_i m_k C_{jl} + m_i m_l C_{jk} + m_j m_k C_{il} + m_j m_l C_{ik} + m_k m_l C_{ij} \\
&\quad + C_{ij}C_{kl} + C_{ik}C_{jl} + C_{il}C_{jk}
\end{aligned}
$$

These are the building blocks for the conditional moment computations needed in the polynomial Bayes predictor.

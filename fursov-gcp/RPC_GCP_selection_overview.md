# Computing RPCs with Robust GCP Selection — Overview

**Paper:** Fursov & Kotov (2018), *"Computing RPC using robust selection of GCPs"*. This is the condensed companion to [`RPC_GCP_selection_comprehensive.md`](RPC_GCP_selection_comprehensive.md), which carries every derivation step, the explicit matrix structures, and all 16 figures. This version keeps the full argument, the key equations, and the headline evidence.

**The argument in one paragraph:** satellite georeferencing uses a rational model (RFM) whose coefficients (RPCs) can be fitted from Ground Control Points by least squares. The accuracy of that fit is governed by the information matrix $A = M^\top M$ — and $A$'s conditioning is set by *where the GCPs are*, not how many or how precise. Badly distributed GCPs make $A$ nearly singular, and measurement noise gets amplified by exactly $\lambda_{\min}^{-1/2}(A)$. So: score candidate GCP subsets by a conditioning criterion and select the best subset — using, if you want it cheap, the eigensolver-free index $\Phi(A) = (\mathrm{tr}A)^2 / \|A\|_F^2$.

Everything below is verified on a synthetic scene with a known true sensor (a pinhole "satellite" over 20×20 km of terrain, 30 GCPs, 0.5 px measurement noise, a 1681-point noiseless check grid).

---

## 1. The setup: GCPs tie the image to the map

A **GCP** is a point known in *both* coordinate systems: surveyed ground coordinates $(E, N, h)$ and identified pixel location $(x, y)$. The georeferencing function is fitted from these pairs — and GCPs are expensive, which makes "which K points should I use?" a real question.

![gcp geometry](figs/01_gcp_geometry.png)

## 2. The model: rational because cameras divide

A pinhole camera is a rotation+translation (affine in ground coordinates) followed by division by depth:

$$x = f\,\frac{X_c}{Z_c} = \frac{f\,\mathbf{r}_1^\top(\mathbf{g}-\mathbf{s})}{\mathbf{r}_3^\top(\mathbf{g}-\mathbf{s})} \qquad\text{— a ratio of degree-1 polynomials: exactly the first-order RFM, eq. (1):}\qquad Y = \frac{a_0 + a_1L + a_2P + a_3H}{1 + b_1L + b_2P + b_3H}$$

The first-order RFM *is* a pinhole camera, not an approximation of one. Counting monomials ($\binom{d+3}{3}$ per polynomial, minus one for the $b_0{=}1$ gauge) gives the minimum GCP counts: **7 / 19 / 39** for orders 1/2/3. Verified: the rational model with 7 parameters fits our camera to **6.5×10⁻¹¹ px**; a quadratic *polynomial* with 10 parameters leaves 0.25 px — form beats parameter count, because the division is the physics.

![why rational](figs/02_why_rational.png)

## 3. Normalization, linearization, least squares

**Normalize** every coordinate to $[-1,1]$ (the OFFSET/SCALE fields in every real RPC file): measured, this alone takes $k(A)$ from $1.3\times10^{18}$ to $292$ — removing the *artificial* (units-driven) ill-conditioning so that what remains is genuinely geometric.

**Linearize** eq. (1) by fixing the gauge ($b_0 = 1$; coefficients are only defined up to common scale) and cross-multiplying:

$$Y_i = a_0 + a_1L_i + a_2P_i + a_3H_i - b_1L_iY_i - b_2P_iY_i - b_3H_iY_i$$

— linear in the 7 unknowns $\mathbf{J}$. Stacking N GCPs gives $\mathbf{Y} = M\mathbf{J} + \boldsymbol{\xi}$ with rows $[1, L, P, H, -YL, -YP, -YH]$, one such system per image axis (each with its own conditioning). Minimizing $\|\mathbf{Y} - M\mathbf{J}\|^2$ yields the normal equations:

$$\boxed{\hat{\mathbf{J}} = (M^\top M)^{-1}M^\top\mathbf{Y}}, \qquad A := M^\top M \text{ (the information matrix — for Gaussian noise, } \sigma^{-2}A \text{ is the Fisher information)}$$

Healthy baseline (12 well-spread GCPs): 0.5 px noise in → **0.65 px RMSE** out. A well-conditioned system passes noise through.

## 4. The failure: distribution moves the error 300×

Same 12-point budget, same noise, three layouts — the paper's Table 1, reproduced:

![distributions](figs/06_distributions.png)

| distribution | $k(A_x)$ | $k(A_y)$ | $\Phi$ (worse) | RMSE total (px) |
|---|---|---|---|---|
| evenly | 292 | 463 | 2.69 | **0.69** |
| diagonally | 24,096 | 28,216 | 2.03 | **12.14** |
| vertically | 13,432,708 | 54,185 | 1.68 | **213.85** |

Same ordering and per-axis asymmetry as the paper (their vertical case: RMSE_X ≫ RMSE_Y — a north–south line of points leaves the East coordinate nearly constant, collapsing the *x*-system's columns into near-collinearity while the *y*-system survives).

## 5. Why: the spectrum of A is the whole story

Eigendecompose $A = V\Lambda V^\top$; then $A^{-1} = V\Lambda^{-1}V^\top$ — **the inverse has reciprocal eigenvalues $1/\lambda_i$ on the same directions.** Whatever direction the GCPs informed least ($\lambda_{\min}$), the inversion amplifies most ($1/\lambda_{\min}$); the eigenvector $v_{\min}$ *names* the blind coefficient combination. The noise-error relationship, derived from $\Delta\hat{\mathbf{J}} = A^{-1}M^\top\boldsymbol{\xi}$ and norm inequalities:

$$\text{eq. (9): } \|\Delta\hat{\mathbf{J}}\| \le \lambda_{\min}^{-1/2}\|\boldsymbol{\xi}\| \qquad \text{eq. (10): } \|\Delta\hat{\mathbf{J}}\| \le \lambda_{\min}^{-1}\|M^\top\boldsymbol{\xi}\| \qquad \text{eq. (13): } \delta_J \le k(A)\,\delta_b$$

with full covariance $\mathrm{Cov}(\hat{\mathbf{J}}) = \sigma^2 A^{-1}$ — an uncertainty ellipsoid with semi-axes $\sigma/\sqrt{\lambda_i}$. All verified: 900 Monte-Carlo trials sit under every bound, the coefficient clouds match the predicted ellipses (spans differing ~100× between even and vertical layouts under *identical* noise), and sweeping σ confirms the structure **error = (spectral multiplier) × (input noise)** with the multiplier matching $\lambda_{\min}^{-1/2}$:

![bounds](figs/08_bounds.png)

**Intuition (the paper's message in one image):** the GCPs are pillars under a foundation. Spread them and the foundation is rigid — noise is absorbed. Collapse them onto a line and the structure balances on a tightrope: tiny noise, multiplied by $\lambda_{\min}^{-1}$, tips the whole model over in exactly the direction nobody measured.

## 6. The cheap surrogate: Φ(A), no eigensolver required

$$\Phi(A) = \frac{(\mathrm{tr}A)^2}{\|A\|_F^2} = \frac{(\sum_i\lambda_i)^2}{\sum_i\lambda_i^2} \in [1, m]$$

computable from matrix entries in two passes, yet a pure function of the spectrum — it is the **participation ratio**: the *effective number of eigen-directions the GCPs informed* (Φ = m iff perfectly conditioned; Φ → 1 as one direction dominates). Our configurations: even ≈ 2.8 informed directions, vertical ≈ 1.7.

The paper's bound eq. (19) turns out to be **Samuelson's inequality** applied to the eigenvalues (mean $\mu = \mathrm{tr}A/m$, std $s = \mu\sqrt{m/\Phi - 1}$, and every value of a set exceeds $\mu - s\sqrt{m-1}$):

$$\lambda_{\min}(A) \ge \frac{\mathrm{tr}A}{m}\Big[1 - \sqrt{(m/\Phi-1)(m-1)}\Big], \qquad \text{positive iff } \Phi > m-1 \text{ (= eq. 18)}$$

Honesty check: real configurations live at Φ ∈ [1.5, 3], far below the guarantee window — yet across 3000 random subsets Φ *ranks* configurations nearly identically to the condition number (Spearman ρ ≈ 0.9). The guarantee dies; the ranking — all that selection needs — survives.

![phi surrogate](figs/09_phi_surrogate.png)

## 7. The algorithm: select the subset that conditions A best

Criteria (eq. 20–22): $Q_1 = \lambda_{\min}$, $Q_2 = 1/k(A)$, $Q_3 = \Phi - m + 1$; score each subset by its worse axis; maximize over K-subsets (eq. 23). Full search is $\binom{30}{12} \approx 8.6\times10^7$ — feasible only for small N (the paper says so), so we use **greedy backward elimination** ($(N{-}K)\cdot N$ evaluations; measured at **100% of the exhaustive optimum** on a pool where both run). This is classical **E-optimal experiment design** — the paper reinvents it plus a cheap surrogate.

![selection](figs/10_selection.png)

**Results:** against 3000 random 12-subsets (median 0.91 px), the selected subsets score **Q₁: 0.63 px (beats 84.6%), Q₂: 0.59 px (88.6%), Q₃: 0.48 px (96.7%)** — the cheapest criterion winning — while §4's pathological layouts sit at 12 and 214 px. Framed as budget: **a well-chosen 12 GCPs matches a random 23**.

![budget sweep](figs/16_budget_sweep.png)

## 8. Takeaways

1. **Geometry is a first-order error source** — same points, different layout, 300× the error. Conditioning metrics predict it *before fitting*.
2. **The amplification law is exact and simple:** error ≈ $\lambda_{\min}^{-1/2}(A)$ × noise. Selection by conditioning is selection by accuracy — decisive against pathological layouts, a mild tune-up among already-good ones (insurance, not a fine-tuner).
3. **Φ is the practical tool:** eigensolver-free, numerically robust, and empirically as good a ranking as $k(A)$. Watch the *height* dimension — spatially perfect GCPs at uniform elevation still starve the H coefficients, and Φ sees it where spatial-coverage heuristics don't.
4. **Pipeline placement (your auto-GCP work):** SIFT proposes correspondences → RANSAC removes outliers → **this selects which inliers fit the RPCs** (greedy $\max Q_j$ directly on the inlier mask). In the common refinement regime (1–5 GCPs correcting vendor RPCs) the logic bites hardest: one badly placed point is 20% of your information matrix.

---

## Files

- Full derivations, matrix structures, and figures 12–16: [`RPC_GCP_selection_comprehensive.md`](RPC_GCP_selection_comprehensive.md)
- Code: `rpc_scratch.py`, `rpc_walkthrough.py`, `rpc_walkthrough_extended.py`; figures in `figs/`
- [`RPC_GCP_selection_pipeline_implementation.md`](RPC_GCP_selection_pipeline_implementation.md) — same paper on DROID + Sentinel-2 (`figs/pipeline/`, `pipeline_walkthrough.py`).

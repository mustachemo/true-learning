# Computing RPCs with Robust GCP Selection — From First Principles

**Paper:** Fursov & Kotov (2018), *"Computing RPC using robust selection of GCPs"*, J. Phys.: Conf. Ser. 1096.

This doc rebuilds the paper end-to-end, with two upgrades: every equation the paper *states* is **derived** here (including eq. 19, which the paper gives without proof — it turns out to be Samuelson's inequality applied to an eigenvalue spectrum), and every claim is **verified numerically** on a synthetic scene with a known true sensor, so ground truth exists for everything.

The paper's argument, compressed:

```
1. Satellite georeferencing uses a RATIONAL model (RFM) with coefficients (RPCs)    §2
2. Without the vendor's sensor model, you fit RPCs to Ground Control Points (GCPs) §1, §4-5
3. Fitting is least squares  ->  accuracy is governed by A = MᵀM                    §5
4. BADLY DISTRIBUTED GCPs make A ill-conditioned -> huge errors from tiny noise     §6-8
5. So: SELECT the GCP subset that best conditions A                                 §10
6. Cheap trick: rank subsets by Φ(A) = (trA)²/‖A‖_F² instead of eigenvalues        §9
```

Everything is implemented in `rpc_scratch.py`; `rpc_walkthrough.py` generates all figures. Headline verifications: the paper's Table 1 reproduces (same 12-point budget, distribution alone moves RMSE **300×**: 0.69 → 12.1 → 213.9 px), all three error bounds (eq. 9/10/13) hold in 900 Monte-Carlo trials, and criterion-selected GCP subsets beat **85–97%** of random subsets.

---

## 1. What a GCP is (starting from zero)

A satellite image is a grid of pixels. A map is a grid of ground coordinates. **Georeferencing** is the function tying them together: given a ground point — here local East/North/height $(E, N, h)$, standing in for the paper's geodetic $(\varphi, \lambda, h)$ — where does it land in the image $(x, y)$?

A **Ground Control Point (GCP)** is one point where you know *both sides* of that function: its ground coordinates (surveyed with GPS, or taken from an existing reference map) *and* its pixel location (a road intersection, a building corner — something you can click on in the image). GCPs are the Rosetta-stone points from which the whole mapping is inferred. They are expensive: each one historically meant a survey crew or careful manual measurement — which is exactly why *"which K points do I use?"* (this paper) and *"can I find them automatically?"* (your thesis) are real questions.

Our synthetic stand-in for the paper's satellite scene — a 20×20 km terrain, a pinhole "satellite" at 450 km altitude viewing 42° off-nadir, and 30 scattered GCPs (mirroring the paper's Figure 1):

![gcp geometry](figs/01_gcp_geometry.png)

The right panel is genuinely rendered *through* the true camera (500k hillshaded ground points projected and binned) — the same 30 physical points appear in both coordinate systems. Everything downstream is about recovering the left↔right mapping from those 30 pairs alone.

---

## 2. The model: why RATIONAL functions — the derivation the paper skips

### The paper's eq. (1)

$$Y = \frac{\mathbf{a}^\top\mathbf{u}}{\mathbf{b}^\top\mathbf{u}} = \frac{a_0 + a_1 L + a_2 P + a_3 H}{b_0 + b_1 L + b_2 P + b_3 H}$$

Why a *ratio* of polynomials and not just a polynomial? Because **cameras divide**. Derive it: a pinhole camera first expresses the world point in camera axes — a rotation plus translation, so each camera coordinate is an *affine* function of the ground coordinates:

$$X_c = \mathbf{r}_1^\top \mathbf{g} + t_1, \quad Y_c = \mathbf{r}_2^\top \mathbf{g} + t_2, \quad Z_c = \mathbf{r}_3^\top \mathbf{g} + t_3, \qquad \mathbf{g} = (E, N, h)$$

then projects by dividing by depth:

$$x = f\,\frac{X_c}{Z_c} = \frac{f(\mathbf{r}_1^\top \mathbf{g} + t_1)}{\mathbf{r}_3^\top \mathbf{g} + t_3}$$

— **a ratio of two degree-1 polynomials in $(E,N,h)$. That is *exactly* the paper's eq. (1).** The first-order RFM isn't an approximation of a pinhole camera; it *is* one (it's the 3D→1D projective map — the same family as the homography from the RANSAC doc, one dimension up). The perspective division is the irreducible nonlinearity, and putting it in the *denominator of the model* rather than asking polynomial terms to Taylor-expand it is the entire design insight of the RFM.

Real satellites aren't pinholes (pushbroom scanning, orbital motion during acquisition, atmospheric refraction, lens distortion), which is why production RPCs use **third-order** polynomials. The coefficient counts the paper quotes follow from counting monomials: the number of monomials of degree ≤ d in 3 variables is $\binom{d+3}{3}$, so

| order d | monomials $\binom{d+3}{3}$ | numerator + denominator − 1 | min. GCPs (per axis) |
|---|---|---|---|
| 1 | 4 | 4 + 3 = **7** | 7 |
| 2 | 10 | 10 + 9 = **19** | 19 |
| 3 | 20 | 20 + 19 = **39** | 39 |

(the −1 is the $b_0 = 1$ gauge fixing, derived in §4) — exactly the paper's 7 / 19 / 39.

### The verification

Fit three models to the same noiseless GCPs and test on a transect (including 25% extrapolation beyond the GCP hull):

![why rational](figs/02_why_rational.png)

Measured: affine polynomial (4 params) leaves **37.8 px** of structured perspective error; a *quadratic* polynomial with **10 params** — more than the rational's 7 — still leaves 0.25 px inside the area and ~1 px extrapolating; the rational model with 7 params sits at **6.5×10⁻¹¹ px everywhere** (machine precision), because for this camera it's not fitting the projection, it *is* the projection. The lesson in one line: **it's the functional form (the division), not the parameter count.**

---

## 3. Normalization: the same disease as the DLT, the same cure

The paper says coordinates are "normalised" and moves on. Here's why it's load-bearing. Real RPC metadata ships each coordinate with an OFFSET and SCALE mapping it to $[-1, 1]$:

$$L = \frac{E - E_{\text{off}}}{E_{\text{scale}}}, \quad P = \frac{N - N_{\text{off}}}{N_{\text{scale}}}, \quad H = \frac{h - h_{\text{off}}}{h_{\text{scale}}}, \quad \text{(same for image } X, Y\text{)}$$

Without it, the design matrix (§4) mixes columns of 1's, coordinates in the tens of thousands (meters, pixels), and *products* of the two in the hundreds of millions. Measured on the same 12 GCPs:

![normalization](figs/03_normalization.png)

$$k(A)_{\text{raw}} = 1.3\times 10^{18} \qquad\longrightarrow\qquad k(A)_{\text{normalized}} = 292$$

**Fifteen orders of magnitude** from a per-coordinate affine rescale. This is Hartley normalization from the RANSAC doc wearing a photogrammetry hat — and it's why every RPC file you'll ever open has those OFF/SCALE fields. Crucially for this paper: normalization removes the *artificial* ill-conditioning from units, so that any conditioning that *remains* is genuinely about the GCP geometry — which is the thing §6 studies.

---

## 4. Linearization: eq. (1) → eq. (3), every step

Eq. (1) is nonlinear in the coefficients (they appear in a quotient). Three moves make it linear:

**Move 1 — gauge fixing, $b_0 = 1$.** Multiply numerator and denominator of eq. (1) by any constant $c \neq 0$: the *function* is unchanged, so $(\mathbf{a}, \mathbf{b})$ are only identifiable up to scale — 8 parameters, 7 degrees of freedom. Fix the gauge by $b_0 = 1$ (safe because after normalization the denominator ≈ 1 near the scene center; a projective map that sent the denominator to 0 inside the scene would be degenerate anyway).

**Move 2 — cross-multiply.** From $Y = \mathbf{a}^\top\mathbf{u} / \mathbf{b}^\top\mathbf{u}$:

$$Y\,(1 + b_1 L + b_2 P + b_3 H) = a_0 + a_1 L + a_2 P + a_3 H$$

**Move 3 — move knowns left, unknowns right** (paper's eq. 3):

$$Y_i = a_0 + a_1 L_i + a_2 P_i + a_3 H_i \;-\; b_1 L_i Y_i - b_2 P_i Y_i - b_3 H_i Y_i$$

Linear in $\mathbf{J} = [a_0, a_1, a_2, a_3, b_1, b_2, b_3]^\top$: one equation per GCP, seven unknowns, hence ≥ 7 GCPs. Stacking $N$ GCPs gives the paper's eq. (4), $\mathbf{Y} = M\mathbf{J} + \boldsymbol{\xi}$, with rows

$$M_i = \big[\,1,\; L_i,\; P_i,\; H_i,\; -Y_i L_i,\; -Y_i P_i,\; -Y_i H_i\,\big]$$

Two things worth seeing rather than imagining — the actual $M$ (12×7) and its Gramian $A = M^\top M$ (7×7):

![linear system](figs/04_linear_system.png)

**A subtlety the paper's $\boldsymbol{\xi}$ hides:** the measured $Y_i$ appears on *both* sides — in the observation vector *and* inside three columns of $M$. Measurement noise therefore perturbs the design matrix too (an errors-in-variables problem; cross-multiplying also reweights each equation by the denominator). At the 0.5 px noise level this is a second-order effect and the linear theory of §7 describes reality well (we verify this) — but it's why heavy-duty RPC solvers iterate or use total least squares. Also note there are **two independent systems** — one for $Y$, one for $X$ — with *different* design matrices (each contains its own image coordinate), hence different conditioning per axis. That asymmetry becomes visible in §6.

---

## 5. Least squares, eq. (5) — derived, then sanity-checked

Minimize $\|\mathbf{Y} - M\mathbf{J}\|_2^2$. Expand and set the gradient to zero:

$$\nabla_\mathbf{J}\,(\mathbf{Y} - M\mathbf{J})^\top(\mathbf{Y} - M\mathbf{J}) = -2M^\top\mathbf{Y} + 2M^\top M\,\mathbf{J} = 0 \;\;\Longrightarrow\;\; \boxed{\hat{\mathbf{J}} = (M^\top M)^{-1} M^\top \mathbf{Y}}$$

the **normal equations** $A\hat{\mathbf{J}} = \mathbf{b}$ with $A = M^\top M$, $\mathbf{b} = M^\top\mathbf{Y}$ — the paper's eq. (5)/(6). ($A$ is called the *information matrix* because, for Gaussian noise, $\sigma^{-2}A$ is exactly the Fisher information about $\mathbf{J}$.) A numerical footnote the paper doesn't make: *solving* via $A$ squares the rounding sensitivity ($k(A) = k(M)^2$), so our code solves with QR (`lstsq`) — but the *statistical* analysis in terms of $A$'s spectrum, which is the paper's whole subject, is solver-independent: it describes how **data noise**, not roundoff, propagates.

A healthy baseline — 12 well-spread GCPs, σ = 0.5 px of measurement noise, evaluated on 1681 noiseless check points via the paper's eq. (24)–(26) RMSE:

![fit residuals](figs/05_fit_residuals.png)

RMSE ≈ 0.65 px from 0.5 px input noise: a well-conditioned system roughly *passes noise through* without amplifying it. Hold that thought.

---

## 6. The failure the paper studies: Table 1, reproduced

Same 12-point budget, same noise, three spatial distributions (the paper's evenly / diagonally / vertically):

![distributions](figs/06_distributions.png)

| distribution | $k(A_x)$ | $k(A_y)$ | $\Phi_x$ | $\Phi_y$ | RMSE_X | RMSE_Y | total (px) |
|---|---|---|---|---|---|---|---|
| evenly | 292 | 463 | 2.79 | 2.69 | 0.46 | 0.44 | **0.69** |
| diagonally | 24,096 | 28,216 | 2.03 | 2.05 | 8.37 | 6.68 | **12.14** |
| vertically | 13,432,708 | 54,185 | 1.68 | 1.79 | 213.70 | 9.23 | **213.85** |

Compare the paper's Table 1 (k: 180 → 2,733 → 6,869; Φ: 2.92 → 2.44 → 1.68; RMSE: 2.2 → 3.4 → 8.2): same ordering, same direction of every quantity — even the paper's most curious detail reproduces, the **per-axis asymmetry**: their vertical case had RMSE_X = 8.2 ≫ RMSE_Y = 0.81, and ours has 213.7 ≫ 9.2. The mechanism is now visible in our per-axis condition numbers: a north–south line of GCPs leaves $L$ (East) nearly constant, so in the *x*-system the columns $[\mathbf{1}]$ and $[L]$ (and their $-XL$ partners) become nearly parallel — $A_x$ approaches singularity ($k = 1.3\times10^7$) while $A_y$ merely degrades. Geometrically: points on a line simply don't *ask* the model any questions about the East direction, so the East-related coefficients are answered by noise.

Note also the error *fields* (bottom row): the damage is worst far from the GCP band — an ill-determined coefficient is a lever, and distance from the data is the lever arm.

The number to internalize: **the same 12 points, arranged differently, move the error by 300×.** Everything after this section is about predicting and preventing that.

---

## 7. Why: perturbation theory — deriving eq. (9)–(14)

Setup for all derivations: the system with true (noiseless) data is solved exactly by $\mathbf{J}$; the measured $\mathbf{Y}$ carries noise $\boldsymbol{\xi}$. Since $\hat{\mathbf{J}}$ is linear in $\mathbf{Y}$:

$$\Delta\hat{\mathbf{J}} = \hat{\mathbf{J}} - \mathbf{J} = (M^\top M)^{-1}M^\top \boldsymbol{\xi} = A^{-1}\boldsymbol{\zeta}, \qquad \boldsymbol{\zeta} := M^\top\boldsymbol{\xi}$$

**Eq. (10)** is immediate: for symmetric positive-definite $A$, $\|A^{-1}\|_2 = 1/\lambda_{\min}(A)$, so

$$\|\Delta\hat{\mathbf{J}}\| \le \|A^{-1}\|\,\|\boldsymbol{\zeta}\| = \lambda_{\min}^{-1}(A)\,\|\boldsymbol{\zeta}\| \qquad \blacksquare$$

**Eq. (9)** needs one SVD step. Write $M = U\Sigma V^\top$. Then $\Delta\hat{\mathbf{J}} = M^+\boldsymbol{\xi}$ where $M^+ = V\Sigma^{-1}U^\top$ is the pseudoinverse, and $\|M^+\|_2 = 1/\sigma_{\min}(M)$. Since the eigenvalues of $A = M^\top M$ are the squared singular values of $M$, $\sigma_{\min}(M) = \lambda_{\min}^{1/2}(A)$:

$$\|\Delta\hat{\mathbf{J}}\| \le \lambda_{\min}^{-1/2}(A)\,\|\boldsymbol{\xi}\| \qquad \blacksquare$$

(So eq. (9) and (10) are the *same* bound measured before vs. after multiplying the noise by $M^\top$ — eq. (9) is the operative one: image noise gets amplified by exactly $\lambda_{\min}^{-1/2}$.)

**Eq. (13)** (relative error ≤ condition number × relative data error): from the normal equations, $\|\mathbf{b}\| = \|A\hat{\mathbf{J}}\| \le \lambda_{\max}\|\hat{\mathbf{J}}\|$, i.e. $1/\|\hat{\mathbf{J}}\| \le \lambda_{\max}/\|\mathbf{b}\|$. Combine with eq. (10):

$$\delta_J = \frac{\|\Delta\hat{\mathbf{J}}\|}{\|\hat{\mathbf{J}}\|} \;\le\; \frac{\|\boldsymbol{\zeta}\|}{\lambda_{\min}}\cdot\frac{\lambda_{\max}}{\|\mathbf{b}\|} \;=\; k(A)\,\delta_b, \qquad \delta_b = \frac{\|\boldsymbol{\zeta}\|}{\|\mathbf{b}\|} \qquad \blacksquare$$

**One more the paper implies but doesn't write** — the full error *distribution*, not just its worst case. For $\boldsymbol{\xi} \sim \mathcal{N}(0, \sigma^2 I)$:

$$\mathrm{Cov}(\hat{\mathbf{J}}) = A^{-1}M^\top(\sigma^2 I)M A^{-1} = \sigma^2 A^{-1}$$

The coefficient uncertainty is an **ellipsoid whose semi-axes are $\sigma/\sqrt{\lambda_i}$ along $A$'s eigenvectors**. Ill-conditioned = one tiny eigenvalue = one enormous ellipsoid axis = a *direction in coefficient space the GCPs never measured*. Verified by Monte Carlo — 500 noisy refits per configuration, against the predicted ellipse:

![perturbation](figs/07_perturbation.png)

Left: the spectra. The vertical configuration's $\lambda_{\min}$ collapses by ~5 orders of magnitude — the "cigar." Right: the $(a_1, b_1)$ coefficient clouds match the $\sigma^2 A^{-1}$ ellipses, and the axis spans differ by ~100× between even and vertical. **The noise is identical in both panels; only the GCP geometry differs.**

And the three bounds, verified across 900 trials (300 per configuration, spanning 5 orders of magnitude of conditioning):

![bounds](figs/08_bounds.png)

Every point sits below the $y = x$ line: the bounds hold, and — the useful part — actual errors *track* the bounds across configurations, which is what licenses using $\lambda_{\min}$ and $k(A)$ as selection criteria in §10. (Worth knowing: eq. (13)'s bound is visibly loose — worst-case over all noise directions — while eq. (9) is nearly tight. Bounds being sortable matters more here than bounds being tight.)

---

## 8. Interlude: what these quantities *mean*

Collecting the cast before the surrogate:

- $\lambda_{\min}(A)$ — how weakly the *least-examined direction* of coefficient space is pinned down by the GCPs. The amplification factor for noise is $\lambda_{\min}^{-1/2}$ (eq. 9).
- $k(A) = \lambda_{\max}/\lambda_{\min}$ — anisotropy of the information: how much better the best-known direction is known than the worst. Scale-free (unit changes cancel), which is why it's the standard diagnostic.
- Both need an eigendecomposition — $O(m^3)$ per evaluation, and numerically delicate exactly when $A$ is nearly singular, i.e. exactly when you care. The paper's response: a spectral quantity you can compute **without touching the spectrum**.

---

## 9. The cheap surrogate Φ — with the derivation the paper omits

### Eq. (15), reread

$$\Phi(A) = \frac{\left(\sum_i a_{ii}\right)^2}{\sum_{i,j} a_{ij}^2} = \frac{(\mathrm{tr}\,A)^2}{\|A\|_F^2}$$

For symmetric $A$, $\mathrm{tr}\,A = \sum_i \lambda_i$ (paper's eq. 16) and $\|A\|_F^2 = \mathrm{tr}\,A^2 = \sum_i \lambda_i^2$ (eq. 17 — since $A^2$'s eigenvalues are $\lambda_i^2$). So:

$$\Phi(A) = \frac{\big(\sum \lambda_i\big)^2}{\sum \lambda_i^2}$$

**Φ is a pure function of the spectrum, computed from matrix entries alone** — two passes over a 7×7 matrix, no eigensolver, numerically bulletproof.

**Its range, derived.** Cauchy–Schwarz on the vectors $(\lambda_1, \dots, \lambda_m)$ and $(1, \dots, 1)$: $(\sum\lambda_i)^2 \le m\sum\lambda_i^2$, with equality iff all $\lambda_i$ equal. And for $\lambda_i \ge 0$, $(\sum\lambda_i)^2 \ge \sum\lambda_i^2$. Hence

$$1 \le \Phi(A) \le m, \qquad \Phi = m \iff \text{all eigenvalues equal (perfect conditioning)}$$

**The interpretation that makes it click:** if $p$ eigenvalues equal some $\lambda$ and the rest are 0, then $\Phi = (p\lambda)^2/(p\lambda^2) = p$. Φ is the **effective number of significant eigenvalues** — physicists call this exact expression the *participation ratio*. So Φ answers: *how many independent directions of the 7-dimensional coefficient space did your GCPs actually inform?* Look back at the §6 table with this lens: even ≈ 2.8 directions, diagonal ≈ 2.0, vertical ≈ 1.7. (None near 7 — even good GCP clouds inform the constant term far more strongly than the height terms. Conditioning is graded, not binary.)

### Eq. (19), derived — it's Samuelson's inequality

The paper asserts: if $m - 1 < \Phi \le m$ (eq. 18), then

$$\lambda_{\min}(A) \ge \frac{\mathrm{tr}A}{m}\left[1 - \sqrt{(m/\Phi - 1)(m-1)}\right] \qquad \text{(eq. 19)}$$

Derivation in three steps.

**Step 1 — spectrum mean and variance from tr and Φ.** Let $\mu = \frac{1}{m}\sum\lambda_i = \frac{\mathrm{tr}A}{m}$ and $s^2 = \frac{1}{m}\sum\lambda_i^2 - \mu^2$. Using $\sum\lambda_i^2 = (\mathrm{tr}A)^2/\Phi$:

$$s^2 = \frac{(\mathrm{tr}A)^2}{m\,\Phi} - \frac{(\mathrm{tr}A)^2}{m^2} = \mu^2\left(\frac{m}{\Phi} - 1\right) \;\;\Longrightarrow\;\; s = \mu\sqrt{m/\Phi - 1}$$

**Step 2 — Samuelson's inequality** (1968): *any* real numbers $x_1,\dots,x_m$ with mean $\mu$ and population std $s$ satisfy $x_i \ge \mu - s\sqrt{m-1}$ for every $i$. Proof: let $x_{\min}$ be the smallest; the other $m-1$ values have mean $\mu' = \frac{m\mu - x_{\min}}{m-1}$, so $\mu' - \mu = \frac{\mu - x_{\min}}{m-1}$. The total variance is at least the contribution of $x_{\min}$'s deviation plus the (minimal) spread of the others sitting at their own mean:

$$m s^2 = \sum(x_i - \mu)^2 \ge (x_{\min}-\mu)^2 + (m-1)(\mu'-\mu)^2 = (x_{\min}-\mu)^2\left(1 + \tfrac{1}{m-1}\right) = (x_{\min}-\mu)^2\tfrac{m}{m-1}$$

hence $|x_{\min} - \mu| \le s\sqrt{m-1}$. $\;\blacksquare$

**Step 3 — combine.** Apply Samuelson to the eigenvalues:

$$\lambda_{\min} \ge \mu - s\sqrt{m-1} = \frac{\mathrm{tr}A}{m}\left[1 - \sqrt{(m/\Phi-1)(m-1)}\right]$$

— **exactly eq. (19).** And the bracket is positive iff $(m/\Phi - 1)(m-1) < 1 \iff \Phi > \frac{m(m-1)}{m} \cdot \frac{m-1}{m-1}\dots$ — solving directly: $m/\Phi < 1 + \frac{1}{m-1} = \frac{m}{m-1} \iff \Phi > m - 1$ — **exactly the window of eq. (18).** The paper's two unexplained conditions fall out of one inequality.

### The honesty check the paper only gestures at

Eq. (18) demands $\Phi > 6$ (for $m = 7$) before the bound says anything — but *every* configuration we've seen, including the paper's own Table 1, lives at $\Phi \in [1.5, 3]$. Is Φ useless there? Measure it — 3000 random 12-GCP subsets:

![phi surrogate](figs/09_phi_surrogate.png)

Left: Φ vs $1/k(A)$ across all 3000 subsets — Spearman ρ ≈ 0.9: **Φ ranks configurations almost identically to the condition number, at a fraction of the cost and with zero eigensolver fragility, far outside its provable window.** Right: the Samuelson bound curve with the eq. (18) window shaded, and the real subsets clustered far to its left. This is the precise content of the paper's remark that Φ "defines the degree of condition over a wider range of values" than eq. (18) suggests: the *guarantee* dies below $\Phi = m-1$, but the *ranking* survives — and selection (§10) only ever needed the ranking.

---

## 10. The selection algorithm: eq. (20)–(23)

The three criteria, normalized to comparable ranges:

$$Q_1 = \lambda_{\min}(A) \qquad Q_2 = k^{-1}(A) = \frac{\lambda_{\min}}{\lambda_{\max}} \in [0,1] \qquad Q_3 = \Phi(A) - m + 1$$

and the optimization (eq. 23): among $N$ available GCPs, pick the $K$-subset maximizing $Q_j$ — with the subset scored by its **worse axis**, $\min(Q_j(A_x), Q_j(A_y))$, since both systems must be solvable. Two facts about this problem:

**It's combinatorial.** $\binom{30}{12} = 86{,}493{,}225$ — the paper's full search is honest only "if the number of given points N is small." We do both: **exhaustive** on a 16-point pool ($\binom{16}{12} = 1820$ subsets, 0.3 s) and **greedy backward elimination** for the full 30 (start from all points, repeatedly drop the least-damaging one — every intermediate matrix stays well-posed, and it costs $(N-K) \cdot N$ scores instead of $\binom{N}{K}$). Measured: greedy hits **100% of the exhaustive optimum** on the pool where both are computable.

**It has a name.** Maximizing $\lambda_{\min}(A)$ over experiment configurations is **E-optimal design** from the statistics literature (siblings: D-optimal = max det $A$, A-optimal = min tr $A^{-1}$). The paper independently arrives at E-optimality plus a spectral-flatness surrogate — knowing the family name unlocks 60 years of literature on exchange algorithms and convex relaxations for exactly this selection problem.

### The payoff

![selection](figs/10_selection.png)

Top: what each criterion selects — all three spread the 12 points across the area *and* across the height range (the criterion sees the $H$ column of $M$; points at one elevation don't inform $a_3, b_3$ — the paper's remark about plains and lakes, and its advantage over Zhang's purely spatial gridding). Bottom, against 3000 random subsets (median 0.91 px, worst 10.2 px): **Q₁ → 0.63 px (beats 84.6% of random), Q₂ → 0.59 px (beats 88.6%), Q₃ → 0.48 px (beats 96.7%)** — with the pathological baselines (12.1, 213.9 px) off the chart to the right. The cheapest criterion — Φ, no eigensolver — performed best in this run; all three land in the good tail, which is the paper's conclusion verified.

---

## 11. Closure: conditioning *predicts* accuracy

The whole chain — eq. (9) says noise is amplified by $\lambda_{\min}^{-1/2}$; therefore ranking subsets by conditioning should rank them by final accuracy. The direct test, all 3000 subsets:

![predictive](figs/11_predictive.png)

Within a pool of already-scattered points the correlation is moderate (Spearman ρ ≈ 0.49 — most random subsets of well-spread points are fine, and the residual scatter is the noise realization itself); across the *pathological* configurations of §6 it is decisive (300× RMSE tracked by 5 orders of magnitude of $k$). That's the operationally correct reading of the paper: **conditioning-based selection is insurance against the disasters, not a fine-tuner among good options.**

### Connections for your work

- **This paper is the third stage of your auto-GCP pipeline.** SIFT gives candidate correspondences between the satellite image and a reference basemap; RANSAC removes the wrong ones; *this method chooses which survivors to actually fit the RPCs with* — replacing the human judgment of "are these points well distributed?" with $\max Q_j$. The greedy backward algorithm drops in directly after RANSAC's inlier mask.
- **The height dimension is the one your intuition will miss.** Spatially beautiful GCPs at uniform elevation leave the $H$ coefficients unobservable ($\Phi$ catches this; a spatial-coverage heuristic does not). Over flat scenes, first-order RFM with height terms is close to unidentifiable — detect it with these criteria and *reduce the model* rather than fight the conditioning.
- **The refinement regime matters more in practice:** vendors usually *do* ship RPCs, and operational work refines them with 1–5 GCPs (bias compensation, Grodecki & Dial 2003 — an affine correction in image space on top of vendor RPCs). The conditioning logic transfers unchanged: with 5 points, one badly placed point is 20% of your information matrix.
- The full lineage of this doc series, in one sentence: *Hartley normalization (RANSAC §7), the influence of point distribution on $M^\top M$, and E-optimal selection are one idea* — least squares is only as good as the geometry of the questions your data asks.

---

## Files

- `rpc_scratch.py` — scene + true sensor + RFM fitting + all conditioning metrics + exhaustive/greedy selection.
- `rpc_walkthrough.py` — generates every figure (~1 min; the 3000-subset sweep dominates).
- `figs/*.png` — all intermediate outputs.

Run: `python rpc_walkthrough.py` (needs `numpy`, `matplotlib`, `scipy`).

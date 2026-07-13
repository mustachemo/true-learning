"""
rpc_walkthrough_extended.py -- five additional figures for the expanded doc.

  12  eigen-anatomy of A: spectra, reciprocals, and WHICH direction is unobserved
  13  noise scaling: RMSE = (spectral multiplier) x (input noise), measured
  14  gauge freedom + the denominator field (why b0 = 1 is safe)
  15  Phi intuition: three spectra, same trace, different participation ratios
  16  the GCP budget sweep: a well-chosen 12 vs a random K
"""
import numpy as np
import matplotlib.pyplot as plt
import time

from rpc_scratch import (AREA, terrain, project, make_norms, normalize_ground,
                         build_M, solve_J, rfm_eval, fit_rpc, rfm_predict_px,
                         eigs, lam_min, cond_k, phi, q3, rmse_px,
                         select_greedy_backward)

plt.rcParams.update({
    "figure.dpi": 130, "font.size": 10,
    "axes.titlesize": 11, "axes.titleweight": "bold",
    "axes.spines.top": False, "axes.spines.right": False,
})
FIG = "figs"
BLUE, ORANGE, GREEN, RED, PURPLE, GRAY = "#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3", "#9a9a9a"
COLS = ["1", "L", "P", "H", "-YL", "-YP", "-YH"]

def save(fig, name):
    fig.savefig(f"{FIG}/{name}.png", bbox_inches="tight"); plt.close(fig)
    print("wrote", name)

# shared data (identical seeds to the main walkthrough)
rng = np.random.default_rng(42)
gE = rng.uniform(500, AREA-500, 30); gN = rng.uniform(500, AREA-500, 30)
gcp_ground = np.c_[gE, gN, terrain(gE, gN)]
gcp_img_true = project(gcp_ground)
SIG_PX = 0.5
gcp_img = gcp_img_true + rng.normal(0, SIG_PX, gcp_img_true.shape)
gg = np.meshgrid(np.linspace(0, AREA, 41), np.linspace(0, AREA, 41))
ref_ground = np.c_[gg[0].ravel(), gg[1].ravel(), terrain(gg[0].ravel(), gg[1].ravel())]
ref_img = project(ref_ground)
nm = make_norms(ref_ground, ref_img)

def make_config(kind, r):
    if kind == "even":
        ee, nn = np.meshgrid(np.linspace(2500, AREA-2500, 4), np.linspace(2500, AREA-2500, 3))
        E, N = ee.ravel(), nn.ravel()
    elif kind == "diagonal":
        t = np.linspace(0.08, 0.92, 12)
        E = t*AREA + r.normal(0, 350, 12); N = t*AREA + r.normal(0, 350, 12)
    elif kind == "vertical":
        N = np.linspace(1500, AREA-1500, 12)
        E = np.full(12, AREA*0.45) + r.normal(0, 300, 12)
    E, N = np.clip(E, 100, AREA-100), np.clip(N, 100, AREA-100)
    return np.c_[E, N, terrain(E, N)]

CONFIGS = {k: make_config(k, np.random.default_rng(1)) for k in ["even", "diagonal", "vertical"]}

def x_gram(g):
    img = project(g)
    L, P, H = normalize_ground(g, nm).T
    X = nm["x"].fwd(img[:, 0])
    M = build_M(X, L, P, H)
    return M.T @ M

# ============================================================================
# FIG 12 -- eigen-anatomy: A = V Lambda V^T, its inverse, and the blind direction
# ============================================================================
fig, axes = plt.subplots(2, 2, figsize=(13.5, 8.6))
for row, (kind, c) in enumerate([("even", GREEN), ("vertical", RED)]):
    A = x_gram(CONFIGS[kind])
    lam, V = np.linalg.eigh(A)                       # ascending
    ax = axes[row, 0]
    xp = np.arange(7)
    ax.bar(xp-0.2, lam[::-1], 0.4, color=c, label="λᵢ(A): information per direction")
    ax.bar(xp+0.2, 1/lam[::-1], 0.4, color=GRAY, label="1/λᵢ = eigenvalues of A⁻¹:\nnoise amplification per direction")
    ax.set_yscale("log"); ax.set_xticks(xp); ax.set_xticklabels([f"#{i+1}" for i in xp])
    ax.set_xlabel("eigen-direction (sorted by λ, descending)")
    ax.legend(fontsize=8); ax.set_title(f"{kind}: spectrum of A and of A⁻¹ (x-system)\n"
                                        f"k(A) = {cond_k(A):,.0f}")
    ax = axes[row, 1]
    vmin = V[:, 0]                                   # eigenvector of lambda_min
    ax.bar(xp, vmin * np.sign(vmin[np.argmax(np.abs(vmin))]), color=c)
    ax.set_xticks(xp); ax.set_xticklabels(COLS)
    ax.set_ylim(-1, 1); ax.axhline(0, color="k", lw=.6)
    ax.set_ylabel("component of v_min")
    ax.set_title(f"{kind}: the BLIND direction (eigenvector of λ_min = {lam[0]:.2e})\n"
                 "the coefficient combination the GCPs never measured")
fig.suptitle("A⁻¹ inherits reciprocal eigenvalues: whatever direction A knows least (λ_min), "
             "A⁻¹ amplifies most (1/λ_min)", fontsize=11.5, y=1.0)
fig.tight_layout(); save(fig, "12_eigen_anatomy")

# ============================================================================
# FIG 13 -- noise scaling: error = multiplier x noise, measured
# ============================================================================
sigmas = np.geomspace(0.05, 5, 8)
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
ax = axes[0]
amp = {}
for kind, c in [("even", GREEN), ("diagonal", ORANGE), ("vertical", RED)]:
    g = CONFIGS[kind]; img_t = project(g)
    med = []
    for s in sigmas:
        rs = []
        for t in range(40):
            imgn = img_t + np.random.default_rng(7000 + t).normal(0, s, (12, 2))
            Jx, Jy, *_ = fit_rpc(g, imgn, nm)
            rs.append(rmse_px(Jx, Jy, ref_ground, ref_img, nm)[2])
        med.append(np.median(rs))
    med = np.array(med)
    amp[kind] = med / sigmas                          # per-config multiplier
    ax.loglog(sigmas, med, marker="o", ms=5, lw=2.2, color=c,
              label=f"{kind}  (RMSE/σ ≈ {np.median(amp[kind]):.1f})")
ax.plot(sigmas, sigmas, "k:", lw=1.2, label="slope 1 reference (RMSE = σ)")
ax.set_xlabel("GCP measurement noise σ (px)"); ax.set_ylabel("check-grid RMSE (px)")
ax.legend(fontsize=8.5)
ax.set_title("error = (geometric multiplier) × (input noise):\nall configs scale linearly in σ — "
             "only the MULTIPLIER differs")
ax = axes[1]
lm = {k: lam_min(x_gram(CONFIGS[k])) for k in CONFIGS}
meas = [np.median(amp[k]) / np.median(amp["even"]) for k in ["even", "diagonal", "vertical"]]
pred = [np.sqrt(lm["even"] / lm[k]) for k in ["even", "diagonal", "vertical"]]
xp = np.arange(3)
ax.bar(xp-0.2, meas, 0.4, color=BLUE, label="measured: multiplier / multiplier(even)")
ax.bar(xp+0.2, pred, 0.4, color=ORANGE, label=r"eq (9) prediction: $\sqrt{\lambda_{min}^{even}/\lambda_{min}}$")
ax.set_yscale("log"); ax.set_xticks(xp); ax.set_xticklabels(["even", "diagonal", "vertical"])
ax.legend(fontsize=8.5)
ax.set_title("the multiplier IS the eq (9) spectral factor\n(order of magnitude agreement, "
             "residual gap = worst-case vs typical noise direction)")
fig.tight_layout(); save(fig, "13_noise_scaling")

# ============================================================================
# FIG 14 -- gauge freedom + the denominator field
# ============================================================================
Jx, Jy, *_ = fit_rpc(gcp_ground, gcp_img_true, nm)
L, P, H = normalize_ground(ref_ground, nm).T
den = 1 + Jy[4]*L + Jy[5]*P + Jy[6]*H
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
ax = axes[0]
sc = ax.scatter(ref_ground[:, 0]/1000, ref_ground[:, 1]/1000, c=den, s=16, cmap="coolwarm",
                vmin=1-2*np.abs(den-1).max(), vmax=1+2*np.abs(den-1).max())
fig.colorbar(sc, ax=ax, fraction=.046, label="denominator  1 + b₁L + b₂P + b₃H")
ax.set_xlabel("East (km)"); ax.set_ylabel("North (km)")
ax.set_title(f"the fitted denominator over the scene: range [{den.min():.3f}, {den.max():.3f}]\n"
             "never near 0 → the b₀ = 1 gauge is safe; division stays benign")
ax = axes[1]
tE = np.linspace(0, AREA, 300)
tg = np.c_[tE, np.full_like(tE, AREA/2), terrain(tE, np.full_like(tE, AREA/2))]
Lt, Pt, Ht = normalize_ground(tg, nm).T
y1 = rfm_eval(Jy, Lt, Pt, Ht)
# the SAME function with all 8 coefficients scaled by c (gauge transformation):
c_scale = 3.0
num_s = c_scale*(Jy[0] + Jy[1]*Lt + Jy[2]*Pt + Jy[3]*Ht)
den_s = c_scale*(1 + Jy[4]*Lt + Jy[5]*Pt + Jy[6]*Ht)
y2 = num_s / den_s
ax.plot(tE/1000, y1, color=BLUE, lw=3.5, label="coefficients (a, b)")
ax.plot(tE/1000, y2, color=ORANGE, lw=1.6, ls="--", label=f"coefficients ({c_scale:g}a, {c_scale:g}b) — identical")
ax.set_xlabel("East along transect (km)"); ax.set_ylabel("normalized image Y")
ax.legend(fontsize=9)
ax.set_title(f"gauge freedom: scaling ALL 8 coefficients by c leaves the function\n"
             f"unchanged (max |difference| = {np.abs(y1-y2).max():.1e}) — hence fix b₀ = 1")
fig.tight_layout(); save(fig, "14_gauge_and_denominator")

# ============================================================================
# FIG 15 -- Phi intuition: same trace, different participation
# ============================================================================
spectra = {
    "all directions equal": np.full(7, 1.0),
    "3 strong, 4 weak": np.array([2.0, 1.8, 1.6, .5, .4, .4, .3]),
    "one direction dominates": np.array([6.4, .2, .15, .1, .06, .05, .04]),
}
fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.3))
for ax, (name, lam) in zip(axes, spectra.items()):
    lam = lam * 7 / lam.sum()                            # same trace = same total information
    ph = lam.sum()**2 / (lam**2).sum()
    ax.bar(range(1, 8), lam, color=BLUE)
    ax.set_ylim(0, 7); ax.set_xlabel("eigen-direction")
    ax.set_title(f"{name}\ntr A = 7 (same total info)\nΦ = {ph:.2f}   k = {lam.max()/lam.min():,.0f}")
axes[0].set_ylabel("λᵢ")
fig.suptitle("Φ = (Σλ)²/Σλ² is the participation ratio: 'how many eigen-directions meaningfully participate'.\n"
             "Same total information, radically different usable geometry — Φ sees it without an eigensolver.",
             fontsize=11.5, y=1.06)
fig.tight_layout(); save(fig, "15_phi_intuition")

# ============================================================================
# FIG 16 -- the GCP budget sweep: quality vs quantity
# ============================================================================
t0 = time.time()
Ks = np.arange(7, 31)
med, q25, q75 = [], [], []
for K in Ks:
    rs = []
    n_draw = 1 if K == 30 else 200
    for t in range(n_draw):
        idx = np.random.default_rng(8000*K + t).choice(30, K, replace=False)
        Jx, Jy, *_ = fit_rpc(gcp_ground[idx], gcp_img[idx], nm)
        rs.append(rmse_px(Jx, Jy, ref_ground, ref_img, nm)[2])
    med.append(np.median(rs)); q25.append(np.percentile(rs, 25)); q75.append(np.percentile(rs, 75))
idx_sel, _ = select_greedy_backward(gcp_ground, gcp_img, nm, 12, q3)
Jx, Jy, *_ = fit_rpc(gcp_ground[idx_sel], gcp_img[idx_sel], nm)
rm_sel = rmse_px(Jx, Jy, ref_ground, ref_img, nm)[2]
K_equiv = Ks[np.argmin(np.abs(np.array(med) - rm_sel))]

fig, ax = plt.subplots(figsize=(9.5, 5.4))
ax.fill_between(Ks, q25, q75, color=GRAY, alpha=.35, label="random subsets: 25–75%")
ax.plot(Ks, med, color=BLUE, lw=2.4, marker="o", ms=4, label="random subsets: median")
ax.axhline(rm_sel, color=GREEN, lw=2.4, ls="--",
           label=f"Q₃-selected 12 GCPs: {rm_sel:.2f} px  (≈ a random {K_equiv})")
ax.axvline(7, color=RED, lw=1.4, ls=":", label="K = 7: exactly determined (no redundancy)")
ax.set_xlabel("number of GCPs K used for fitting"); ax.set_ylabel("median check-grid RMSE (px)")
ax.legend(fontsize=8.5)
ax.set_title("quality vs quantity: a well-CHOSEN 12 matches a random "
             f"{K_equiv} —\nselection buys back the survey cost of {K_equiv-12} GCPs")
fig.tight_layout(); save(fig, "16_budget_sweep")
print(f"budget sweep {time.time()-t0:.0f}s;  selected-12 RMSE {rm_sel:.2f}px ≈ random K={K_equiv}")
print("DONE")

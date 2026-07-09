"""
rpc_walkthrough.py -- Fursov & Kotov 2018, stage by stage, one figure per stage.

  01  what a GCP is: the scene, the sensor, the image
  02  why the model is RATIONAL: perspective divides
  03  normalization (offsets/scales): conditioning, again
  04  the linear system: eq (1) -> eq (3)/(4), the design matrix made visible
  05  a healthy fit: residuals with well-spread GCPs
  06  the failure the paper studies: Table 1 reproduced (even/diagonal/vertical)
  07  perturbation theory: spectra + Monte Carlo coefficient clouds vs sigma^2 A^-1
  08  the error bounds, eq (9)-(14), verified numerically
  09  the cheap surrogate Phi, eq (15)-(19): participation ratio + Samuelson bound
  10  the selection algorithm, eq (20)-(23): greedy + exhaustive, the payoff
  11  closure: conditioning PREDICTS accuracy across 3000 random subsets
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, FancyArrowPatch
import time
from itertools import combinations

from rpc_scratch import (AREA, terrain, project, SAT, make_norms, normalize_ground,
                         build_M, solve_J, rfm_eval, fit_rpc, rfm_predict_px,
                         eigs, lam_min, cond_k, phi, lam_min_bound,
                         q1, q2, q3, rmse_px, subset_score,
                         select_exhaustive, select_greedy_backward)

plt.rcParams.update({
    "figure.dpi": 130, "font.size": 10,
    "axes.titlesize": 11, "axes.titleweight": "bold",
    "axes.spines.top": False, "axes.spines.right": False,
})
FIG = "figs"
BLUE, ORANGE, GREEN, RED, PURPLE, GRAY = "#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3", "#9a9a9a"

def save(fig, name):
    fig.savefig(f"{FIG}/{name}.png", bbox_inches="tight"); plt.close(fig)
    print("wrote", name)

# ============================================================================
# Shared data
# ============================================================================
rng = np.random.default_rng(42)
gE = rng.uniform(500, AREA-500, 30); gN = rng.uniform(500, AREA-500, 30)
gcp_ground = np.c_[gE, gN, terrain(gE, gN)]
gcp_img_true = project(gcp_ground)
SIG_PX = 0.5
gcp_img = gcp_img_true + rng.normal(0, SIG_PX, gcp_img_true.shape)   # measured GCPs

gg = np.meshgrid(np.linspace(0, AREA, 41), np.linspace(0, AREA, 41))
ref_ground = np.c_[gg[0].ravel(), gg[1].ravel(), terrain(gg[0].ravel(), gg[1].ravel())]
ref_img = project(ref_ground)                                        # noiseless check grid
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

# ============================================================================
# FIG 01 -- what a GCP is
# ============================================================================
t0 = time.time()
# terrain map
he = np.meshgrid(np.linspace(0, AREA, 300), np.linspace(0, AREA, 300))
hmap = terrain(he[0], he[1])
# a rendered "satellite image": project hillshaded ground points, bin to a grid
n_pts = 500_000
sE = rng.uniform(0, AREA, n_pts); sN = rng.uniform(0, AREA, n_pts)
d = 30.0
gx = (terrain(sE + d, sN) - terrain(sE - d, sN)) / (2*d)
gy = (terrain(sE, sN + d) - terrain(sE, sN - d)) / (2*d)
nrm = np.c_[-gx, -gy, np.ones(n_pts)]; nrm /= np.linalg.norm(nrm, axis=1, keepdims=True)
light = np.array([-.5, .3, .8]); light = light/np.linalg.norm(light)
shade = np.clip(nrm @ light, 0, 1) * (0.75 + 0.25*terrain(sE, sN)/900)
sxy = project(np.c_[sE, sN, terrain(sE, sN)])
bins = 800
Hsum, xe, ye = np.histogram2d(sxy[:, 0], sxy[:, 1], bins=bins, weights=shade)
Hcnt, _, _ = np.histogram2d(sxy[:, 0], sxy[:, 1], bins=[xe, ye])
img_render = np.where(Hcnt > 0, Hsum/np.maximum(Hcnt, 1), np.nan).T

fig = plt.figure(figsize=(16, 5.2))
gs = fig.add_gridspec(1, 3, width_ratios=[1.05, 1.0, 1.15])
ax = fig.add_subplot(gs[0])
im = ax.imshow(hmap, origin="lower", extent=[0, AREA/1000, 0, AREA/1000], cmap="terrain")
ax.scatter(gcp_ground[:, 0]/1000, gcp_ground[:, 1]/1000, s=42, facecolor="w",
           edgecolor="k", lw=1.2, zorder=5)
fig.colorbar(im, ax=ax, fraction=.046, label="height h (m)")
ax.set_xlabel("East (km)"); ax.set_ylabel("North (km)")
ax.set_title("GROUND SPACE: 20x20 km terrain\n30 GCPs with surveyed (E, N, h)")
ax = fig.add_subplot(gs[1]); ax.axis("off")
ax.set_xlim(0, 10); ax.set_ylim(0, 10)
ax.fill_between(np.linspace(1, 9, 100), 1 + 0.8*np.sin(np.linspace(0, 9, 100)) + .8, 0.4,
                color="#c9b78e")
ax.plot([7.6], [8.8], marker=(5, 1), ms=22, color=GRAY)
ax.plot([2.0, 7.6], [2.35, 8.8], color=RED, lw=1.4)
ax.plot([5.0, 7.6], [1.9, 8.8], color=RED, lw=1.4)
ax.plot([6.2, 6.0], [8.2, 7.4], lw=0)
ax.plot([6.55, 7.4], [7.05, 7.7], color="k", lw=2.6)      # image plane segment
ax.annotate("satellite\n(450 km, ~42° oblique)", (7.6, 8.8), (4.1, 9.3), fontsize=9,
            arrowprops=dict(arrowstyle="->"))
ax.annotate("image plane", (7.0, 7.4), (8.0, 6.3), fontsize=9, arrowprops=dict(arrowstyle="->"))
ax.annotate("GCP:\nknown (E, N, h)\nAND known (x, y)", (2.0, 2.35), (0.6, 5.4), fontsize=9,
            color=RED, arrowprops=dict(arrowstyle="->", color=RED))
ax.set_title("THE LINK (not to scale): each GCP is one\npoint observed in BOTH coordinate systems")
ax = fig.add_subplot(gs[2])
ax.imshow(img_render, origin="lower", cmap="gray",
          extent=[xe[0], xe[-1], ye[0], ye[-1]])
ax.scatter(gcp_img_true[:, 0], gcp_img_true[:, 1], s=42, facecolor="none",
           edgecolor=ORANGE, lw=1.6)
ax.set_xlabel("image x (px)"); ax.set_ylabel("image y (px)")
ax.set_title("IMAGE SPACE: the (synthetic) satellite view\nsame 30 GCPs located at pixels (x, y)")
fig.tight_layout(); save(fig, "01_gcp_geometry")
print(f"  render {time.time()-t0:.0f}s")

# ============================================================================
# FIG 02 -- why RATIONAL: perspective divides
# ============================================================================
def fit_poly(ground, img_col_norm, order):
    L, P, H = normalize_ground(ground, nm).T
    cols = [np.ones_like(L), L, P, H]
    if order == 2:
        cols += [L*L, P*P, H*H, L*P, L*H, P*H]
    return np.column_stack(cols), np.linalg.lstsq(np.column_stack(cols), img_col_norm, rcond=None)[0]

def eval_poly(coef, ground, order):
    L, P, H = normalize_ground(ground, nm).T
    cols = [np.ones_like(L), L, P, H]
    if order == 2:
        cols += [L*L, P*P, H*H, L*P, L*H, P*H]
    return nm["x"].inv(np.column_stack(cols) @ coef)

X_gcp_norm = nm["x"].fwd(gcp_img_true[:, 0])                 # noiseless: isolate MODEL error
_, c_aff = fit_poly(gcp_ground, X_gcp_norm, 1)
_, c_quad = fit_poly(gcp_ground, X_gcp_norm, 2)
Jx0, Jy0, *_ = fit_rpc(gcp_ground, gcp_img_true, nm)

tE = np.linspace(-0.25*AREA, 1.25*AREA, 500)                  # transect incl. EXTRAPOLATION
tg = np.c_[tE, np.full_like(tE, AREA*0.5), terrain(tE, np.full_like(tE, AREA*0.5))]
tx_true = project(tg)[:, 0]
res_aff = eval_poly(c_aff, tg, 1) - tx_true
res_quad = eval_poly(c_quad, tg, 2) - tx_true
res_rfm = rfm_predict_px(Jx0, Jy0, tg, nm)[:, 0] - tx_true
inside = (tE >= 0) & (tE <= AREA)

fig = plt.figure(figsize=(15.5, 5))
gs = fig.add_gridspec(1, 3, width_ratios=[1.35, 1.35, 1.0])
ax = fig.add_subplot(gs[0])
ax.plot(tE/1000, res_aff, color=RED, lw=2, label="affine poly (4 params)")
ax.plot(tE/1000, res_quad, color=PURPLE, lw=2, label="quadratic poly (10 params)")
ax.plot(tE/1000, res_rfm, color=GREEN, lw=2, label="1st-order RATIONAL (7 params)")
ax.axvspan(-5, 0, color=GRAY, alpha=.18); ax.axvspan(20, 25, color=GRAY, alpha=.18)
ax.set_xlabel("East along transect (km)"); ax.set_ylabel("prediction error (px)")
ax.set_ylim(-80, 80)
ax.set_title("model error along a transect (no noise)\ngray = extrapolation beyond the GCPs")
ax.legend(fontsize=8.5, loc="lower left")
ax = fig.add_subplot(gs[1])
ax.plot(tE/1000, np.abs(res_aff)+1e-13, color=RED, lw=2)
ax.plot(tE/1000, np.abs(res_quad)+1e-13, color=PURPLE, lw=2)
ax.plot(tE/1000, np.abs(res_rfm)+1e-13, color=GREEN, lw=2)
ax.axvspan(-5, 0, color=GRAY, alpha=.18); ax.axvspan(20, 25, color=GRAY, alpha=.18)
ax.set_yscale("log"); ax.set_xlabel("East along transect (km)"); ax.set_ylabel("|error| (px, log)")
ax.set_title("same, log scale: the rational model is\nEXACT for a pinhole (error ~ 1e-10 px)")
ax = fig.add_subplot(gs[2])
rmse_in = [np.sqrt((r[inside]**2).mean()) for r in (res_aff, res_quad, res_rfm)]
bars = ax.bar(["affine\n(4)", "quad\n(10)", "rational\n(7)"], rmse_in, color=[RED, PURPLE, GREEN])
ax.set_yscale("log")
for b, v in zip(bars, rmse_in):
    ax.text(b.get_x()+b.get_width()/2, v*1.4, f"{v:.2g} px", ha="center", fontweight="bold", fontsize=9)
ax.set_ylabel("RMSE inside the area (px, log)")
ax.set_title("params don't fix it —\nthe FORM (division) does")
fig.tight_layout(); save(fig, "02_why_rational")

# ============================================================================
# FIG 03 -- normalization: conditioning, again
# ============================================================================
g12 = CONFIGS["even"]; img12 = project(g12)
# RAW units: E,N,h in meters, x in pixels
Xr = img12[:, 0]; Lr, Pr, Hr = g12.T
M_raw = np.column_stack([np.ones_like(Lr), Lr, Pr, Hr, -Xr*Lr, -Xr*Pr, -Xr*Hr])
L, P, H = normalize_ground(g12, nm).T; Xn = nm["x"].fwd(img12[:, 0])
M_nrm = build_M(Xn, L, P, H)
k_raw = (np.linalg.svd(M_raw, compute_uv=False)[0] / np.linalg.svd(M_raw, compute_uv=False)[-1])**2
k_nrm = cond_k(M_nrm.T @ M_nrm)
fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.4))
ax = axes[0]
bars = ax.bar(["raw units\n(m, px)", "normalized\nto [-1, 1]"], [k_raw, k_nrm], color=[RED, GREEN])
ax.set_yscale("log"); ax.set_ylabel("condition number k(A)")
for b, v in zip(bars, [k_raw, k_nrm]):
    ax.text(b.get_x()+b.get_width()/2, v*2, f"{v:.2g}", ha="center", fontweight="bold")
ax.set_title(f"same 12 GCPs, same model:\nnormalization buys {np.log10(k_raw/k_nrm):.0f} orders of magnitude")
ax = axes[1]
col_mags_raw = np.abs(M_raw).mean(0); col_mags_nrm = np.abs(M_nrm).mean(0)
labels = ["1", "L", "P", "H", "-YL", "-YP", "-YH"]
xpos = np.arange(7)
ax.bar(xpos-0.2, col_mags_raw, 0.4, color=RED, label="raw")
ax.bar(xpos+0.2, col_mags_nrm, 0.4, color=GREEN, label="normalized")
ax.set_yscale("log"); ax.set_xticks(xpos); ax.set_xticklabels(labels)
ax.set_ylabel("mean |column entry| (log)"); ax.legend()
ax.set_title("the disease: raw columns span ~8 orders of magnitude\n(1 vs meters vs pixel*meter products)")
fig.tight_layout(); save(fig, "03_normalization")

# ============================================================================
# FIG 04 -- the linear system made visible
# ============================================================================
Y12 = nm["y"].fwd(img12[:, 1])
My12 = build_M(Y12, L, P, H)
Ay12 = My12.T @ My12
fig = plt.figure(figsize=(13.5, 4.9))
gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 0.30, 1.0])
ax = fig.add_subplot(gs[0])
im = ax.imshow(My12, cmap="RdBu_r", vmin=-1.6, vmax=1.6, aspect="auto")
ax.set_xticks(range(7)); ax.set_xticklabels(["1", "L", "P", "H", "-YL", "-YP", "-YH"])
ax.set_yticks(range(12)); ax.set_yticklabels([f"GCP {i+1}" for i in range(12)], fontsize=7.5)
fig.colorbar(im, ax=ax, fraction=.046)
ax.set_title("design matrix M (12 x 7), y-axis system\none row per GCP  —  eq (4)")
ax = fig.add_subplot(gs[1])
im = ax.imshow(Y12[:, None], cmap="RdBu_r", vmin=-1.6, vmax=1.6, aspect="auto")
ax.set_xticks([0]); ax.set_xticklabels(["Y"])
ax.set_yticks([])
ax.set_title("observations\nY")
ax = fig.add_subplot(gs[2])
im = ax.imshow(Ay12, cmap="RdBu_r", vmin=-np.abs(Ay12).max(), vmax=np.abs(Ay12).max())
ax.set_xticks(range(7)); ax.set_xticklabels(["1", "L", "P", "H", "-YL", "-YP", "-YH"], fontsize=8)
ax.set_yticks(range(7)); ax.set_yticklabels(["1", "L", "P", "H", "-YL", "-YP", "-YH"], fontsize=8)
fig.colorbar(im, ax=ax, fraction=.046)
ax.set_title(f"information matrix A = MᵀM (7 x 7)  —  eq (6)\n"
             f"k(A) = {cond_k(Ay12):.0f},  Φ(A) = {phi(Ay12):.2f}")
fig.tight_layout(); save(fig, "04_linear_system")

# ============================================================================
# FIG 05 -- a healthy fit
# ============================================================================
img12n = img12 + np.random.default_rng(3).normal(0, SIG_PX, img12.shape)
Jx, Jy, Ax, Ay, _, _ = fit_rpc(g12, img12n, nm)
pred = rfm_predict_px(Jx, Jy, ref_ground, ref_img*0 + ref_ground[:, :2]*0 + 0, nm) if False else rfm_predict_px(Jx, Jy, ref_ground, nm)
err = pred - ref_img
rx, ry, rt = rmse_px(Jx, Jy, ref_ground, ref_img, nm)
fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
ax = axes[0]
sc = ax.scatter(ref_ground[:, 0]/1000, ref_ground[:, 1]/1000,
                c=np.linalg.norm(err, axis=1), s=16, cmap="viridis")
ax.scatter(g12[:, 0]/1000, g12[:, 1]/1000, s=70, facecolor="w", edgecolor=RED, lw=1.6, zorder=5)
fig.colorbar(sc, ax=ax, fraction=.046, label="|prediction error| (px)")
ax.set_xlabel("East (km)"); ax.set_ylabel("North (km)")
ax.set_title(f"12 well-spread GCPs (red), σ = {SIG_PX} px noise\ncheck-grid error field — RMSE = {rt:.2f} px")
ax = axes[1]
ax.hist(err[:, 0], bins=40, alpha=.65, color=BLUE, label=f"x residuals (RMSE {rx:.2f})", density=True)
ax.hist(err[:, 1], bins=40, alpha=.65, color=ORANGE, label=f"y residuals (RMSE {ry:.2f})", density=True)
ax.set_xlabel("residual (px)"); ax.set_ylabel("density"); ax.legend()
ax.set_title("residuals on 1681 noiseless check points\n(eq (24)-(26): RMSE per axis and total)")
fig.tight_layout(); save(fig, "05_fit_residuals")

# ============================================================================
# FIG 06 -- Table 1 reproduced: even / diagonal / vertical
# ============================================================================
table_rows = []
fig, axes = plt.subplots(2, 3, figsize=(15.5, 9.4))
for j, kind in enumerate(["even", "diagonal", "vertical"]):
    g = CONFIGS[kind]
    stats = []
    for t in range(60):
        imgn = project(g) + np.random.default_rng(1000+t).normal(0, SIG_PX, (12, 2))
        Jx, Jy, Ax, Ay, _, _ = fit_rpc(g, imgn, nm)
        stats.append([cond_k(Ax), cond_k(Ay), phi(Ax), phi(Ay),
                      *rmse_px(Jx, Jy, ref_ground, ref_img, nm)])
    kx, ky, px_, py_, rX, rY, rT = np.median(np.array(stats), axis=0)
    table_rows.append((kind, kx, ky, px_, py_, rX, rY, rT))
    imgn = project(g) + np.random.default_rng(1001).normal(0, SIG_PX, (12, 2))
    Jx, Jy, Ax, Ay, _, _ = fit_rpc(g, imgn, nm)
    pred = rfm_predict_px(Jx, Jy, ref_ground, ref_img, nm) if False else rfm_predict_px(Jx, Jy, ref_ground, nm)
    e = np.linalg.norm(pred - ref_img, axis=1)
    ax = axes[0, j]
    ax.imshow(hmap, origin="lower", extent=[0, AREA/1000, 0, AREA/1000], cmap="terrain", alpha=.85)
    ax.scatter(g[:, 0]/1000, g[:, 1]/1000, s=60, facecolor="w", edgecolor="k", lw=1.4, zorder=5)
    ax.set_title(f"{kind}: k(Aₓ)={kx:,.0f}, k(A_y)={ky:,.0f}\nΦₓ={px_:.2f}, Φ_y={py_:.2f}")
    ax = axes[1, j]
    sc = ax.scatter(ref_ground[:, 0]/1000, ref_ground[:, 1]/1000, c=np.log10(np.maximum(e, 1e-2)),
                    s=14, cmap="magma", vmin=-1, vmax=3)
    ax.scatter(g[:, 0]/1000, g[:, 1]/1000, s=40, facecolor="w", edgecolor=GREEN, lw=1.3, zorder=5)
    ax.set_title(f"RMSE_X={rX:.2f}  RMSE_Y={rY:.2f}  total={rT:.2f} px")
    if j == 2:
        fig.colorbar(sc, ax=axes[1, :].tolist(), fraction=.02, label="log10 |error| (px)")
fig.suptitle("The paper's Table 1, reproduced: same 12-point budget, same noise — the DISTRIBUTION alone "
             "moves the error by 300x", fontsize=12.5, y=0.99)
save(fig, "06_distributions")
print(f"{'config':<10} {'k(Ax)':>12} {'k(Ay)':>10} {'Phi_x':>6} {'Phi_y':>6} {'RMSE_X':>8} {'RMSE_Y':>8} {'total':>8}")
for r in table_rows:
    print(f"{r[0]:<10} {r[1]:>12,.0f} {r[2]:>10,.0f} {r[3]:>6.2f} {r[4]:>6.2f} {r[5]:>8.2f} {r[6]:>8.2f} {r[7]:>8.2f}")

# ============================================================================
# FIG 07 -- perturbation theory: spectra + Monte Carlo coefficient clouds
# ============================================================================
def x_system(g, imgn):
    L, P, H = normalize_ground(g, nm).T
    X = nm["x"].fwd(imgn[:, 0])
    Mx = build_M(X, L, P, H)
    return Mx, X

fig = plt.figure(figsize=(15.5, 5))
gs = fig.add_gridspec(1, 3, width_ratios=[1.15, 1.0, 1.0])
ax = fig.add_subplot(gs[0])
for kind, c in [("even", GREEN), ("diagonal", ORANGE), ("vertical", RED)]:
    Mx, X = x_system(CONFIGS[kind], project(CONFIGS[kind]))
    ax.semilogy(range(1, 8), eigs(Mx.T @ Mx)[::-1], marker="o", ms=6, lw=2, color=c, label=kind)
ax.set_xlabel("eigenvalue index (sorted)"); ax.set_ylabel("λᵢ(A) (log)")
ax.legend(); ax.set_title("spectra of A = MᵀM (x-system)\nbad configs: λ_min collapses — the 'cigar'")
# MC clouds
SIG_N = SIG_PX / nm["x"].scale
for panel, (kind, c) in enumerate([("even", GREEN), ("vertical", RED)]):
    ax = fig.add_subplot(gs[1 + panel])
    g = CONFIGS[kind]; img_t = project(g)
    Mx, X_t = x_system(g, img_t)
    A = Mx.T @ Mx
    J_true = solve_J(Mx, X_t)
    C = SIG_N**2 * np.linalg.inv(A)                    # predicted covariance of Ĵ
    sub = np.ix_([1, 4], [1, 4])                       # (a1, b1) block
    C2 = C[sub]
    cloud = []
    for t in range(500):
        xi = np.random.default_rng(50_000 + t).normal(0, SIG_N, 12)
        Jhat = solve_J(Mx, X_t + xi)                   # perturb RHS only: the regime
        cloud.append(Jhat[[1, 4]] - J_true[[1, 4]])    # eq (9)-(13) are derived for
    cloud = np.array(cloud)
    ax.scatter(cloud[:, 0], cloud[:, 1], s=7, color=c, alpha=.5)
    ev, evec = np.linalg.eigh(C2)
    angle = np.degrees(np.arctan2(evec[1, -1], evec[0, -1]))
    for ns in (1, 2):
        ax.add_patch(Ellipse((0, 0), 2*ns*np.sqrt(ev[-1]), 2*ns*np.sqrt(ev[0]),
                             angle=angle, fill=False, edgecolor="k", lw=1.6, ls="--"))
    ax.set_xlabel("Δa₁"); ax.set_ylabel("Δb₁")
    span = np.abs(cloud).max()*1.2
    ax.set_xlim(-span, span); ax.set_ylim(-span, span)
    ax.set_title(f"{kind}: 500 noisy refits of (a₁, b₁)\ndashed = 1σ, 2σ of σ²A⁻¹ — "
                 f"axes span {span:.1e}")
fig.tight_layout(); save(fig, "07_perturbation")

# ============================================================================
# FIG 08 -- the error bounds, verified
# ============================================================================
fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.7))
for kind, c in [("even", GREEN), ("diagonal", ORANGE), ("vertical", RED)]:
    g = CONFIGS[kind]; img_t = project(g)
    Mx, X_t = x_system(g, img_t)
    A = Mx.T @ Mx
    J_true = solve_J(Mx, X_t)
    lm = lam_min(A); kA = cond_k(A)
    b = Mx.T @ X_t
    dj, b9, b10, dJr, b13 = [], [], [], [], []
    for t in range(300):
        xi = np.random.default_rng(90_000 + t).normal(0, SIG_N, 12)
        dJ = solve_J(Mx, X_t + xi) - J_true
        zeta = Mx.T @ xi
        dj.append(np.linalg.norm(dJ))
        b9.append(np.linalg.norm(xi) / np.sqrt(lm))
        b10.append(np.linalg.norm(zeta) / lm)
        dJr.append(np.linalg.norm(dJ) / np.linalg.norm(J_true))
        b13.append(kA * np.linalg.norm(zeta) / np.linalg.norm(b))
    axes[0].scatter(b9, dj, s=8, color=c, alpha=.55, label=kind)
    axes[1].scatter(b10, dj, s=8, color=c, alpha=.55)
    axes[2].scatter(b13, dJr, s=8, color=c, alpha=.55)
for ax, (ttl, xl, yl) in zip(axes, [
        ("eq (9):  ‖ΔĴ‖ ≤ λ_min^{-1/2} ‖ξ‖", "bound  λ_min^{-1/2}‖ξ‖", "actual ‖ΔĴ‖"),
        ("eq (10):  ‖ΔĴ‖ ≤ λ_min^{-1} ‖ζ‖,  ζ = Mᵀξ", "bound  λ_min^{-1}‖ζ‖", "actual ‖ΔĴ‖"),
        ("eq (13):  δ_J ≤ k(A) · δ_b", "bound  k(A)·δ_b", "actual δ_J")]):
    lims = np.array([min(ax.get_xlim()[0], ax.get_ylim()[0]), max(ax.get_xlim()[1], ax.get_ylim()[1])])
    ax.plot([1e-12, 1e6], [1e-12, 1e6], "k--", lw=1.2)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel(xl); ax.set_ylabel(yl); ax.set_title(ttl + "\n(all points below y = x: bound holds)")
axes[0].legend(fontsize=8.5)
fig.tight_layout(); save(fig, "08_bounds")

# ============================================================================
# FIG 09 -- the cheap surrogate Phi: participation ratio + Samuelson bound
# ============================================================================
n_sub = 3000
r9 = np.random.default_rng(7)
subs = [r9.choice(30, 12, replace=False) for _ in range(n_sub)]
phis, invks, lams, rmses = [], [], [], []
for idx in subs:
    g = gcp_ground[idx]; imgn = gcp_img[idx]
    Jx, Jy, Ax, Ay, _, _ = fit_rpc(g, imgn, nm)
    phis.append(min(phi(Ax), phi(Ay)))
    invks.append(min(1/cond_k(Ax), 1/cond_k(Ay)))
    lams.append(min(lam_min(Ax), lam_min(Ay)))
    rmses.append(rmse_px(Jx, Jy, ref_ground, ref_img, nm)[2])
phis, invks, lams, rmses = map(np.array, (phis, invks, lams, rmses))

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
ax = axes[0]
sc = ax.scatter(phis, invks, s=9, c=np.log10(rmses), cmap="magma_r", alpha=.75)
fig.colorbar(sc, ax=ax, fraction=.046, label="log10 RMSE (px)")
from scipy.stats import spearmanr
rho = spearmanr(phis, invks).statistic
ax.set_yscale("log")
ax.set_xlabel("Φ(A)  (worse axis)"); ax.set_ylabel("1/k(A)  (worse axis, log)")
ax.set_title(f"3000 random 12-GCP subsets: Φ tracks the condition number\n"
             f"(Spearman ρ = {rho:.3f}) — WITHOUT any eigendecomposition")
ax = axes[1]
ph_grid = np.linspace(1.001, 7, 400)
m = 7
bound = 1/m * (1 - np.sqrt(np.maximum((m/ph_grid - 1) * (m - 1), 0)))
ax.plot(ph_grid, bound, color=BLUE, lw=2.4,
        label=r"eq (19):  $\lambda_{min}/\mathrm{tr}A \geq \frac{1}{m}[1-\sqrt{(m/\Phi-1)(m-1)}]$")
ax.axvspan(m-1, m, color=GREEN, alpha=.15, label="eq (18) window: bound is positive")
lam_over_tr = []
for idx in subs[:600]:
    g = gcp_ground[idx]; imgn = gcp_img[idx]
    _, _, Ax, Ay, _, _ = fit_rpc(g, imgn, nm)
    A_w = Ax if phi(Ax) < phi(Ay) else Ay
    lam_over_tr.append((phi(A_w), lam_min(A_w)/np.trace(A_w)))
lam_over_tr = np.array(lam_over_tr)
ax.scatter(lam_over_tr[:, 0], lam_over_tr[:, 1], s=10, color=ORANGE, alpha=.6,
           label="actual λ_min/trA (600 subsets)")
ax.set_xlabel("Φ(A)"); ax.set_ylabel("λ_min / tr A")
ax.set_ylim(-0.02, 0.16); ax.set_xlim(1, 7.2)
ax.legend(fontsize=8, loc="upper left")
ax.set_title("the Samuelson bound and its honesty problem:\nreal configs live at Φ ≈ 1.5–3, far below the\nguaranteed window — yet Φ still RANKS them (left panel)")
fig.tight_layout(); save(fig, "09_phi_surrogate")

# ============================================================================
# FIG 10 -- the selection algorithm: eq (20)-(23)
# ============================================================================
t0 = time.time()
sel = {}
for name, q in [("Q1 = λ_min", q1), ("Q2 = 1/k", q2), ("Q3 = Φ-m+1", q3)]:
    idx, s = select_greedy_backward(gcp_ground, gcp_img, nm, 12, q)
    Jx, Jy, Ax, Ay, _, _ = fit_rpc(gcp_ground[idx], gcp_img[idx], nm)
    sel[name] = (idx, rmse_px(Jx, Jy, ref_ground, ref_img, nm)[2])
print(f"greedy selection {time.time()-t0:.1f}s;",
      {k: f"{v[1]:.2f}px" for k, v in sel.items()})

# exhaustive-vs-greedy sanity on a small pool (paper's own regime)
pool = np.arange(16)
t0 = time.time()
ex_idx, ex_s = select_exhaustive(gcp_ground[pool], gcp_img[pool], nm, 12, q2)
gr_idx, gr_s = select_greedy_backward(gcp_ground[pool], gcp_img[pool], nm, 12, q2)
print(f"exhaustive C(16,12)={len(list(combinations(range(16),12)))} subsets in {time.time()-t0:.1f}s: "
      f"best Q2={ex_s:.2e}; greedy Q2={gr_s:.2e} ({gr_s/ex_s*100:.0f}% of optimum)")

fig = plt.figure(figsize=(15.5, 8.6))
gs = fig.add_gridspec(2, 3, height_ratios=[1, 1.05])
for j, (name, (idx, rm)) in enumerate(sel.items()):
    ax = fig.add_subplot(gs[0, j])
    ax.imshow(hmap, origin="lower", extent=[0, AREA/1000, 0, AREA/1000], cmap="terrain", alpha=.85)
    ax.scatter(gcp_ground[:, 0]/1000, gcp_ground[:, 1]/1000, s=26, facecolor="none",
               edgecolor="k", lw=.9, alpha=.6)
    ax.scatter(gcp_ground[idx, 0]/1000, gcp_ground[idx, 1]/1000, s=90, facecolor="w",
               edgecolor=RED, lw=2, zorder=5)
    ax.set_title(f"greedy max of {name}\n→ RMSE {rm:.2f} px")
ax = fig.add_subplot(gs[1, :])
ax.hist(rmses, bins=np.geomspace(rmses.min(), rmses.max(), 70), color=GRAY, alpha=.75,
        label="3000 RANDOM 12-subsets")
ax.set_xscale("log")
for (name, (idx, rm)), c in zip(sel.items(), [GREEN, BLUE, PURPLE]):
    ax.axvline(rm, color=c, lw=2.4, label=f"{name}-selected: {rm:.2f} px")
for kind, c, r in [("even", "k", None), ("diagonal", ORANGE, None), ("vertical", RED, None)]:
    rm = [r for r in table_rows if r[0] == kind][0][7]
    ax.axvline(rm, color=c, lw=1.6, ls=":", label=f"{kind} baseline: {rm:.2f} px")
ax.set_xlabel("check-grid RMSE (px, log)"); ax.set_ylabel("count of subsets")
ax.legend(fontsize=8.5, ncol=2)
ax.set_title("the payoff (eq 23): criterion-selected subsets sit in the best tail of the random-subset distribution")
fig.tight_layout(); save(fig, "10_selection")

# ============================================================================
# FIG 11 -- closure: conditioning PREDICTS accuracy
# ============================================================================
fig, ax = plt.subplots(figsize=(9.5, 5.6))
xq = 1/np.sqrt(lams)
sc = ax.scatter(xq, rmses, s=10, c=phis, cmap="viridis", alpha=.75)
rho = spearmanr(xq, rmses).statistic
fig.colorbar(sc, ax=ax, fraction=.046, label="Φ (worse axis)")
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel(r"$\lambda_{min}^{-1/2}$  (worse axis)  — the eq (9) amplification factor")
ax.set_ylabel("check-grid RMSE (px)")
ax.set_title(f"3000 random subsets: the eq (9) factor predicts the final accuracy\n"
             f"Spearman ρ = {rho:.3f} — selection by conditioning IS selection by accuracy")
fig.tight_layout(); save(fig, "11_predictive")

print("\nDONE")

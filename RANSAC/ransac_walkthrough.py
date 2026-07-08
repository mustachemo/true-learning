"""
ransac_walkthrough.py -- runs ransac_scratch.py stage by stage, dumps a figure
per stage.

  01  the outlier problem: least squares has a 0% breakdown point
  02  the inversion: minimal samples + consensus counting
  03  best-so-far evolution over iterations
  04  the iteration math: N(outlier ratio, sample size)
  05  the threshold epsilon: too tight / right / too loose
  06  refit: minimal-sample model vs least squares on the consensus set
  07  adaptive termination trace
  08  the real task: homography from contaminated SIFT matches
  09  validation vs cv2.findHomography + why DLT needs normalization
  10  breakdown stress test: OLS vs Huber vs RANSAC across outlier ratios
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import cv2
from scipy.optimize import least_squares
from skimage import data

from ransac_scratch import (ransac, ransac_line, ransac_homography, n_iterations,
                            fit_line_2pts, fit_line_ols, line_residuals,
                            dlt_homography, apply_h, homography_residuals)

plt.rcParams.update({
    "figure.dpi": 130, "font.size": 10,
    "axes.titlesize": 11, "axes.titleweight": "bold",
    "axes.spines.top": False, "axes.spines.right": False,
})
FIG = "figs"
BLUE, ORANGE, GREEN, RED, PURPLE, GRAY = "#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3", "#9a9a9a"

def save(fig, name):
    fig.savefig(f"{FIG}/{name}.png", bbox_inches="tight")
    plt.close(fig)
    print("wrote", name)

def line_xy(l, xs):
    return (-l[0]*xs - l[2]) / l[1]

def ang_deg(l):
    return np.degrees(np.arctan2(-l[0], l[1]) % np.pi)

# ----------------------------------------------------------------------------
# Shared synthetic line data: y = 0.5x + 2, sigma = 0.3, 35% outliers
# ----------------------------------------------------------------------------
rng = np.random.default_rng(42)
N_PTS, OUT_FRAC, SIGMA = 200, 0.35, 0.3
n_out = int(N_PTS*OUT_FRAC); n_in = N_PTS - n_out
xi = rng.uniform(-10, 10, n_in)
inliers  = np.c_[xi, 0.5*xi + 2 + rng.normal(0, SIGMA, n_in)]
outliers = np.c_[rng.uniform(-10, 10, n_out), rng.uniform(-6, 10, n_out)]
pts = np.vstack([inliers, outliers])
is_true_inl = np.r_[np.ones(n_in, bool), np.zeros(n_out, bool)]
TRUE = np.array([-0.5, 1, -2.0]); TRUE /= np.linalg.norm(TRUE[:2])
xs = np.linspace(-10.5, 10.5, 2)
EPS = 2.5 * SIGMA

# ----------------------------------------------------------------------------
# FIG 01 -- the outlier problem
# ----------------------------------------------------------------------------
ols = fit_line_ols(pts)
ols_clean = fit_line_ols(inliers)
fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
ax = axes[0]
ax.scatter(*inliers.T, s=14, color=BLUE, label=f"inliers ({n_in})")
ax.scatter(*outliers.T, s=14, color=RED, marker="x", label=f"outliers ({n_out})")
ax.plot(xs, line_xy(TRUE, xs), "k--", lw=2, label="true line")
ax.plot(xs, line_xy(ols, xs), color=ORANGE, lw=2.5, label="least squares (all points)")
ax.legend(fontsize=8); ax.set_title(f"35% outliers drag the fit {abs(ang_deg(ols)-ang_deg(TRUE)):.1f}° off")
ax = axes[1]
r = np.linspace(-4, 4, 200)
ax.plot(r, r**2, color=ORANGE, lw=2.5, label="squared loss ρ(r) = r²")
ax.plot(r, np.where(np.abs(r) < 1, r**2, 2*np.abs(r)-1), color=PURPLE, lw=2.2, label="Huber (bounds slope,\nnot influence region)")
ax.plot(r, np.minimum(r**2, 1.5), color=GREEN, lw=2.2, label="RANSAC's implicit loss:\ncapped (top-hat consensus)")
ax.legend(fontsize=8); ax.set_xlabel("residual r"); ax.set_ylabel("ρ(r)")
ax.set_title("WHY it fails: an outlier at r=100 gets\n10,000x the vote of a point at r=1")
ax = axes[2]
fracs = np.arange(0, 0.51, 0.025)
errs = []
for f in fracs:
    k = int(N_PTS*f)
    sub = np.vstack([inliers, outliers[:k]]) if k else inliers
    errs.append(abs(ang_deg(fit_line_ols(sub)) - ang_deg(TRUE)))
ax.plot(fracs*100, errs, color=ORANGE, lw=2.5, marker="o", ms=4)
ax.set_xlabel("outlier fraction (%)"); ax.set_ylabel("angle error (deg)")
ax.set_title("Breakdown point of least squares = 0%:\nerror grows from the FIRST outlier")
fig.tight_layout(); save(fig, "01_outlier_problem")

# ----------------------------------------------------------------------------
# FIG 02 -- the inversion: minimal samples + consensus
# ----------------------------------------------------------------------------
rng2 = np.random.default_rng(7)
fig, axes = plt.subplots(2, 3, figsize=(15, 8.4))
for ax in axes.ravel():
    idx = rng2.choice(N_PTS, 2, replace=False)
    model = fit_line_2pts(pts[idx[0]], pts[idx[1]])
    res = line_residuals(model, pts)
    inl = res < EPS
    clean = is_true_inl[idx].all()
    ax.scatter(*pts[~inl].T, s=10, color=GRAY, alpha=.6)
    ax.scatter(*pts[inl].T, s=12, color=GREEN)
    ax.scatter(*pts[idx].T, s=110, facecolor=ORANGE, edgecolor="k", zorder=5)
    ys = line_xy(model, xs)
    ax.plot(xs, ys, color=ORANGE, lw=2)
    nvec = model[:2] * EPS
    ax.fill_between(xs, ys - EPS/abs(model[1]), ys + EPS/abs(model[1]), color=GREEN, alpha=.15)
    ax.set_xlim(-10.5, 10.5); ax.set_ylim(-7, 11)
    ax.set_title(f"{'CLEAN' if clean else 'CONTAMINATED'} sample → consensus = {inl.sum()}",
                 color=GREEN if clean else RED)
fig.suptitle("Six random 2-point hypotheses. Score = COUNT of points inside the ε-tube.\n"
             "Contaminated samples produce garbage lines that little of the data agrees with — "
             "consensus is the filter.", fontsize=11.5, y=1.0)
fig.tight_layout(); save(fig, "02_minimal_samples")

# ----------------------------------------------------------------------------
# FIG 03 -- best-so-far evolution
# ----------------------------------------------------------------------------
rng3 = np.random.default_rng(3)
snapshots, snap_at = {}, [1, 3, 10, 60]
best_cnt, best_model = -1, None
for it in range(1, 61):
    idx = rng3.choice(N_PTS, 2, replace=False)
    m = fit_line_2pts(pts[idx[0]], pts[idx[1]])
    if m is None: continue
    cnt = int((line_residuals(m, pts) < EPS).sum())
    if cnt > best_cnt:
        best_cnt, best_model = cnt, m
    if it in snap_at:
        snapshots[it] = (best_model.copy(), best_cnt)
fig, axes = plt.subplots(1, 4, figsize=(16.5, 4.1))
for ax, it in zip(axes, snap_at):
    m, cnt = snapshots[it]
    inl = line_residuals(m, pts) < EPS
    ax.scatter(*pts[~inl].T, s=9, color=GRAY, alpha=.6)
    ax.scatter(*pts[inl].T, s=11, color=GREEN)
    ax.plot(xs, line_xy(m, xs), color=ORANGE, lw=2.2)
    ax.plot(xs, line_xy(TRUE, xs), "k--", lw=1.2, alpha=.7)
    ax.set_xlim(-10.5, 10.5); ax.set_ylim(-7, 11)
    ax.set_title(f"best after {it} iter{'s' if it>1 else ''}: {cnt} inliers")
fig.suptitle("'Best so far' only ever improves — RANSAC needs ONE lucky clean sample, ever. (dashed = truth)",
             fontsize=11.5, y=1.02)
fig.tight_layout(); save(fig, "03_consensus_evolution")

# ----------------------------------------------------------------------------
# FIG 04 -- how many iterations: N(e, s)
# ----------------------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
e = np.linspace(0, 0.8, 200)
ax = axes[0]
for s, c in [(2, GREEN), (4, ORANGE), (8, RED)]:
    ax.semilogy(e*100, [n_iterations(0.99, 1-ee, s) for ee in e], lw=2.4, color=c,
                label=f"s = {s}" + {2:" (line)", 4:" (homography)", 8:" (fundamental, 8-pt)"}[s])
ax.legend(fontsize=9); ax.set_xlabel("outlier fraction e (%)"); ax.set_ylabel("iterations N (p=0.99)")
ax.set_title("N = log(1−p)/log(1−(1−e)ˢ)\ngrows EXPONENTIALLY in sample size s")
ax = axes[1]
Nax = np.arange(1, 300)
for e0, c in [(0.3, GREEN), (0.5, ORANGE), (0.7, RED)]:
    ax.plot(Nax, 1-(1-(1-e0)**4)**Nax, lw=2.4, color=c, label=f"e = {e0:.0%}")
ax.axhline(.99, color="k", ls="--", lw=1)
ax.legend(fontsize=9); ax.set_xlabel("iterations N"); ax.set_ylabel("P(≥1 clean sample)")
ax.set_title("Success probability vs N  (s = 4)\n— the guarantee is probabilistic, not certain")
ax = axes[2]
ss = np.arange(2, 13)
for e0, c in [(0.3, GREEN), (0.5, ORANGE)]:
    ax.semilogy(ss, [n_iterations(0.99, 1-e0, s_) for s_ in ss], marker="o", ms=5, lw=2.2,
                color=c, label=f"e = {e0:.0%}")
ax.legend(fontsize=9); ax.set_xlabel("sample size s"); ax.set_ylabel("iterations N")
ax.set_title("WHY minimal samples: every extra point\nmultiplies the risk the sample is dirty")
fig.tight_layout(); save(fig, "04_iterations_math")

# ----------------------------------------------------------------------------
# FIG 05 -- the threshold epsilon
# ----------------------------------------------------------------------------
fig, axes = plt.subplots(1, 4, figsize=(16.5, 4.1))
for ax, ep, lbl in [(axes[0], 0.05, "too tight (ε = 0.17σ)"),
                    (axes[1], EPS,  f"ε = 2.5σ = {EPS:.2f}"),
                    (axes[2], 4.0,  "too loose (ε = 13σ)")]:
    m, msk, _ = ransac_line(pts, eps=ep, rng=np.random.default_rng(0))
    ax.scatter(*pts[~msk].T, s=9, color=GRAY, alpha=.6)
    ax.scatter(*pts[msk].T, s=11, color=GREEN)
    ax.plot(xs, line_xy(m, xs), color=ORANGE, lw=2.2)
    ax.plot(xs, line_xy(TRUE, xs), "k--", lw=1.2)
    ax.set_xlim(-10.5, 10.5); ax.set_ylim(-7, 11)
    ax.set_title(f"{lbl}\n{msk.sum()} inliers, err {abs(ang_deg(m)-ang_deg(TRUE)):.2f}°")
ax = axes[3]
eps_sweep = np.geomspace(0.02, 8, 25)
med_err = []
for ep in eps_sweep:
    errs = []
    for t in range(30):
        m, _, _ = ransac_line(pts, eps=ep, rng=np.random.default_rng(100+t), max_iter=500)
        errs.append(abs(ang_deg(m)-ang_deg(TRUE)))
    med_err.append(np.median(errs))
ax.loglog(eps_sweep/SIGMA, med_err, color=BLUE, lw=2.4, marker="o", ms=4)
ax.axvline(2.5, color=GREEN, ls="--", lw=1.5)
ax.set_xlabel("ε / σ"); ax.set_ylabel("median angle error (deg)")
ax.set_title("Sweep: too tight starves the consensus\n(noise splits the inliers), too loose\nadmits outliers back — U-shape")
fig.tight_layout(); save(fig, "05_threshold")

# ----------------------------------------------------------------------------
# FIG 06 -- refit on the consensus set
# ----------------------------------------------------------------------------
m_raw, msk_raw, _ = ransac_line(pts, eps=EPS, rng=np.random.default_rng(5), refit=False)
m_ref = fit_line_ols(pts[msk_raw])
for _ in range(2):
    msk_ref = line_residuals(m_ref, pts) < EPS
    m_ref = fit_line_ols(pts[msk_ref])
fig, axes = plt.subplots(1, 2, figsize=(12.5, 5))
ax = axes[0]
ax.scatter(*pts[~msk_raw].T, s=9, color=GRAY, alpha=.6)
ax.scatter(*pts[msk_raw].T, s=11, color=GREEN, alpha=.8)
ax.plot(xs, line_xy(TRUE, xs), "k--", lw=1.6, label="truth")
ax.plot(xs, line_xy(m_raw, xs), color=RED, lw=2, label=f"winning 2-pt model ({abs(ang_deg(m_raw)-ang_deg(TRUE)):.2f}° err)")
ax.plot(xs, line_xy(m_ref, xs), color=GREEN, lw=2, label=f"refit on {msk_ref.sum()} inliers ({abs(ang_deg(m_ref)-ang_deg(TRUE)):.2f}° err)")
ax.legend(fontsize=8.5); ax.set_title("RANSAC's winner is built from just s points\n— it locates the structure, LS polishes it")
ax = axes[1]
xz = np.linspace(6, 10.5, 2)
ax.scatter(*pts[msk_raw].T, s=22, color=GREEN, alpha=.8)
ax.plot(xz, line_xy(TRUE, xz), "k--", lw=2)
ax.plot(xz, line_xy(m_raw, xz), color=RED, lw=2.4)
ax.plot(xz, line_xy(m_ref, xz), color=GREEN, lw=2.4)
ax.set_xlim(6, 10.5); ax.set_ylim(4.2, 7.8)
ax.set_title("zoomed: the 2-pt model inherits the noise of\nits 2 points; the refit averages over ~130")
fig.tight_layout(); save(fig, "06_refit")

# ----------------------------------------------------------------------------
# FIG 07 -- adaptive termination (harder problem: 60% outliers)
# ----------------------------------------------------------------------------
n_out2 = 120; n_in2 = 80
xh = rng.uniform(-10, 10, n_in2)
pts2 = np.vstack([np.c_[xh, 0.5*xh + 2 + rng.normal(0, SIGMA, n_in2)],
                  np.c_[rng.uniform(-10, 10, n_out2), rng.uniform(-6, 10, n_out2)]])
m2, msk2, trace2 = ransac_line(pts2, eps=EPS, rng=np.random.default_rng(11))
tr = np.array(trace2, float)
fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
ax = axes[0]
ax.step(tr[:, 0], tr[:, 1], where="post", color=GREEN, lw=2.2)
ax.set_xlabel("iteration"); ax.set_ylabel("best consensus so far", color=GREEN)
ax2 = ax.twinx()
ax2.step(tr[:, 0], tr[:, 2], where="post", color=PURPLE, lw=2.2)
ax2.set_ylabel("required N (from current best w)", color=PURPLE)
ax2.set_yscale("log"); ax2.spines['top'].set_visible(False)
ax.axvline(tr[-1, 0], color="k", ls="--", lw=1.2)
ax.set_title(f"60% outliers: each better model raises ŵ and slashes N\n"
             f"terminated at iteration {int(tr[-1,0])} (theoretical N for true w=0.4: "
             f"{n_iterations(0.99, 0.4, 2)})")
ax = axes[1]
ax.scatter(*pts2[~msk2].T, s=9, color=GRAY, alpha=.6)
ax.scatter(*pts2[msk2].T, s=11, color=GREEN)
ax.plot(xs, line_xy(m2, xs), color=ORANGE, lw=2.2)
ax.plot(xs, line_xy(TRUE, xs), "k--", lw=1.2)
ax.set_title(f"the fit it stopped with: {msk2.sum()} inliers, "
             f"{abs(ang_deg(m2)-ang_deg(TRUE)):.2f}° error")
fig.tight_layout(); save(fig, "07_adaptive")

# ----------------------------------------------------------------------------
# FIG 08 -- the real task: homography from contaminated SIFT matches
# ----------------------------------------------------------------------------
img1 = data.camera()
Hh, Ww = img1.shape
src_c = np.float32([[0, 0], [Ww, 0], [Ww, Hh], [0, Hh]])
dst_c = np.float32([[70, 45], [Ww-25, 15], [Ww-15, Hh-60], [30, Hh-25]])
H_gt = cv2.getPerspectiveTransform(src_c, dst_c)
img2 = cv2.warpPerspective(img1, H_gt, (Ww, Hh))

sift_cv = cv2.SIFT_create()
k1, d1_ = sift_cv.detectAndCompute(img1, None)
k2, d2_ = sift_cv.detectAndCompute(img2, None)
knn = cv2.BFMatcher().knnMatch(d1_, d2_, k=2)
good = [m for m, n2 in knn if m.distance < 0.9 * n2.distance]     # loose ratio ON PURPOSE
p1 = np.float64([k1[m.queryIdx].pt for m in good])
p2 = np.float64([k2[m.trainIdx].pt for m in good])
gt_err = homography_residuals(H_gt, p1, p2)
gt_inl = gt_err < 3.0
print(f"putative matches: {len(p1)}, true inliers {gt_inl.sum()} ({gt_inl.mean()*100:.0f}%)")

H_est, msk_h, trace_h = ransac_homography(p1, p2, eps=3.0, rng=np.random.default_rng(0))
corners = np.float64([[0, 0], [Ww, 0], [Ww, Hh], [0, Hh]])
corner_err = np.linalg.norm(apply_h(H_est, corners) - apply_h(H_gt, corners), axis=1)
print(f"our RANSAC: {msk_h.sum()} inliers in {trace_h[-1][0]} iters; corner err mean {corner_err.mean():.2f}px")

canvas = np.hstack([img1, img2])
fig, axes = plt.subplots(3, 1, figsize=(12.5, 17))
ax = axes[0]
ax.imshow(canvas, cmap="gray", alpha=.85); ax.axis("off")
for i in range(0, len(p1), 3):
    ax.plot([p1[i,0], p2[i,0]+Ww], [p1[i,1], p2[i,1]], color=BLUE, lw=.6, alpha=.6)
ax.set_title(f"putative SIFT matches (ratio 0.9, drawn 1-in-3): {len(p1)} pairs, "
             f"only {gt_inl.mean()*100:.0f}% correct — this contamination is RANSAC's input")
ax = axes[1]
ax.imshow(canvas, cmap="gray", alpha=.85); ax.axis("off")
for i in np.nonzero(~msk_h)[0]:
    ax.plot([p1[i,0], p2[i,0]+Ww], [p1[i,1], p2[i,1]], color=RED, lw=.8, alpha=.85)
for i in np.nonzero(msk_h)[0][::3]:
    ax.plot([p1[i,0], p2[i,0]+Ww], [p1[i,1], p2[i,1]], color=GREEN, lw=.6, alpha=.7)
agree = (msk_h == gt_inl).mean()
ax.set_title(f"RANSAC verdict (s=4, ε=3px): {msk_h.sum()} inliers (green), {(~msk_h).sum()} outliers (red) — "
             f"{agree*100:.1f}% agreement with ground truth labels")
ax = axes[2]
ax.imshow(img2, cmap="gray", alpha=.9); ax.axis("off")
quad_gt  = apply_h(H_gt, corners)
quad_est = apply_h(H_est, corners)
ax.plot(*np.vstack([quad_gt, quad_gt[:1]]).T, color="w", lw=5, alpha=.9)
ax.plot(*np.vstack([quad_gt, quad_gt[:1]]).T, color=ORANGE, lw=2.4, ls="--", label="image-1 border under TRUE H")
ax.plot(*np.vstack([quad_est, quad_est[:1]]).T, color=GREEN, lw=1.6, label="under ESTIMATED H")
ax.legend(fontsize=9, loc="lower right")
ax.set_title(f"recovered vs true homography: mean corner error {corner_err.mean():.2f}px "
             f"(the two quadrilaterals are indistinguishable)")
fig.tight_layout(); save(fig, "08_homography")

# ----------------------------------------------------------------------------
# FIG 09 -- vs OpenCV + why normalization
# ----------------------------------------------------------------------------
H_cv, msk_cv = cv2.findHomography(p1, p2, cv2.RANSAC, ransacReprojThreshold=3.0)
msk_cv = msk_cv.ravel().astype(bool)
corner_err_cv = np.linalg.norm(apply_h(H_cv, corners) - apply_h(H_gt, corners), axis=1)
jac = (msk_h & msk_cv).sum() / (msk_h | msk_cv).sum()
# conditioning demo: DLT design matrix with vs without normalization
_, sv_norm   = dlt_homography(p1[gt_inl], p2[gt_inl], normalize=True)
_, sv_nonorm = dlt_homography(p1[gt_inl], p2[gt_inl], normalize=False)
c_norm, c_nonorm = sv_norm[0]/sv_norm[-2], sv_nonorm[0]/sv_nonorm[-2]

fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
ax = axes[0]
bars = ax.bar(["ours", "cv2.findHomography"], [corner_err.mean(), corner_err_cv.mean()],
              color=[GREEN, BLUE])
for b, v in zip(bars, [corner_err.mean(), corner_err_cv.mean()]):
    ax.text(b.get_x()+b.get_width()/2, v*1.02, f"{v:.2f}px", ha="center", fontweight="bold")
ax.set_ylabel("mean corner reprojection error vs TRUE H (px)")
ax.set_title("accuracy: both sub-pixel")
ax = axes[1]
bars = ax.bar(["ours", "cv2", "overlap\n(∩)"], [msk_h.sum(), msk_cv.sum(), (msk_h & msk_cv).sum()],
              color=[GREEN, BLUE, PURPLE])
for b, v in zip(bars, [msk_h.sum(), msk_cv.sum(), (msk_h & msk_cv).sum()]):
    ax.text(b.get_x()+b.get_width()/2, v+3, str(int(v)), ha="center", fontweight="bold")
ax.set_ylabel("inlier count")
ax.set_title(f"inlier sets: Jaccard overlap {jac*100:.1f}%")
ax = axes[2]
bars = ax.bar(["raw pixel\ncoordinates", "Hartley-\nnormalized"], [c_nonorm, c_norm], color=[RED, GREEN])
ax.set_yscale("log")
for b, v in zip(bars, [c_nonorm, c_norm]):
    ax.text(b.get_x()+b.get_width()/2, v*1.15, f"{v:.1e}", ha="center", fontweight="bold")
ax.set_ylabel("condition number of DLT matrix A")
ax.set_title("why normalize before DLT: pixel coords\n(~500) squared in A → wildly mixed scales")
fig.tight_layout(); save(fig, "09_vs_opencv")

# ----------------------------------------------------------------------------
# FIG 10 -- breakdown stress test: OLS vs Huber vs RANSAC
# ----------------------------------------------------------------------------
def huber_fit(P):
    m0 = fit_line_ols(P)
    th0 = [-m0[0]/m0[1], -m0[2]/m0[1]]              # slope, intercept from OLS start
    f = lambda th: P[:, 1] - (th[0]*P[:, 0] + th[1])
    sol = least_squares(f, th0, loss="huber", f_scale=SIGMA)
    a, b = sol.x
    l = np.array([-a, 1, -b]); return l / np.linalg.norm(l[:2])

ratios = np.arange(0.0, 0.85, 0.1)
n_trials = 120
succ = {"least squares": [], "Huber M-estimator": [], "RANSAC": []}
for e0 in ratios:
    ok = {k: 0 for k in succ}
    for t in range(n_trials):
        r_ = np.random.default_rng(int(e0*1000)*1000 + t)
        no = int(N_PTS*e0); ni = N_PTS - no
        xx = r_.uniform(-10, 10, ni)
        P = np.vstack([np.c_[xx, 0.5*xx + 2 + r_.normal(0, SIGMA, ni)],
                       np.c_[r_.uniform(-10, 10, no), r_.uniform(-6, 10, no)]])
        for name, fitf in [("least squares", lambda P: fit_line_ols(P)),
                           ("Huber M-estimator", huber_fit),
                           ("RANSAC", lambda P: ransac_line(P, eps=EPS, rng=r_, max_iter=800)[0])]:
            m = fitf(P)
            if abs(ang_deg(m) - ang_deg(TRUE)) < 2.0:
                ok[name] += 1
    for k in succ: succ[k].append(ok[k] / n_trials)
    print(f"e={e0:.1f}  " + "  ".join(f"{k}:{succ[k][-1]:.2f}" for k in succ))

fig, ax = plt.subplots(figsize=(9, 5.2))
for name, c in [("least squares", ORANGE), ("Huber M-estimator", PURPLE), ("RANSAC", GREEN)]:
    ax.plot(ratios*100, np.array(succ[name])*100, marker="o", ms=5, lw=2.4, color=c, label=name)
ax.axhline(99, color="k", ls=":", lw=1)
ax.set_xlabel("outlier fraction (%)"); ax.set_ylabel("success rate (angle error < 2°), %")
ax.set_title(f"Breakdown comparison, {n_trials} trials per point\n"
             "OLS dies immediately; Huber survives moderate contamination\n"
             "(bounded influence but still uses every point); RANSAC holds to ~70–80%")
ax.legend()
fig.tight_layout(); save(fig, "10_breakdown")

print(f"\nSTATS: putative {len(p1)} matches @ {gt_inl.mean()*100:.0f}% precision | "
      f"ours {msk_h.sum()} inl, corner err {corner_err.mean():.2f}px | "
      f"cv2 {msk_cv.sum()} inl, {corner_err_cv.mean():.2f}px | Jaccard {jac*100:.1f}%")

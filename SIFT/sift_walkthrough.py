"""
sift_walkthrough.py -- runs sift_scratch.py stage by stage and dumps a figure
per stage for the learning doc.

  01  the scale problem (why fixed windows fail)
  02  Gaussian blur as a scale dial
  03  the scale-space pyramid (octaves x layers)
  04  DoG: cheap scale-normalized LoG
  05  26-neighbor extrema detection
  06  refinement funnel: subpixel fit, contrast test, edge test
  07  orientation assignment (36-bin histogram)
  08  the 128-d descriptor (4x4x8, rotated frame, trilinear binning)
  09  validation vs OpenCV
  10  the payoff: matching under rotation + scale
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle, FancyArrow
import cv2
from collections import Counter
from skimage import data
from scipy.spatial import cKDTree

import sift_scratch as ss
from sift_scratch import (build_scale_space, build_dog, find_extrema,
                          refine_keypoints, assign_orientations,
                          compute_descriptors, compute_descriptor,
                          orientation_histogram, layer_sigmas, sift, _grads,
                          S, SIGMA0, N_ORI_BINS, PEAK_RATIO, D_DESC, N_DESC_BINS)

plt.rcParams.update({
    "figure.dpi": 130, "font.size": 10,
    "axes.titlesize": 11, "axes.titleweight": "bold",
    "axes.spines.top": False, "axes.spines.right": False,
})
FIG = "figs"
BLUE, ORANGE, GREEN, RED, PURPLE = "#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3"
OCT_COLORS = ["#e41a1c", "#ff7f00", "#4daf4a", "#377eb8", "#984ea3", "#a65628", "#f781bf"]

img = data.camera()
H, W = img.shape

def save(fig, name):
    fig.savefig(f"{FIG}/{name}.png", bbox_inches="tight")
    plt.close(fig)
    print("wrote", name)

# ----------------------------------------------------------------------------
# FIG 01 -- the scale problem
# ----------------------------------------------------------------------------
small = cv2.resize(img, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
box = 48                       # a FIXED-size analysis window
cy1, cx1 = 110, 265            # the camera body in the full-res image
cy2, cx2 = cy1//2, cx1//2      # the same physical point after 0.5x

fig, axes = plt.subplots(1, 4, figsize=(15, 4.4))
axes[0].imshow(img, cmap="gray"); axes[0].axis("off")
axes[0].add_patch(Rectangle((cx1-box//2, cy1-box//2), box, box, fill=False, edgecolor=RED, lw=2))
axes[0].set_title("Image at scale 1.0 — fixed 48px window")
axes[1].imshow(img[cy1-box//2:cy1+box//2, cx1-box//2:cx1+box//2], cmap="gray"); axes[1].axis("off")
axes[1].set_title("Window content: part of the camera", color=RED)
axes[2].imshow(small, cmap="gray"); axes[2].axis("off")
axes[2].add_patch(Rectangle((cx2-box//2, cy2-box//2), box, box, fill=False, edgecolor=RED, lw=2))
axes[2].set_title("Same scene at 0.5x — SAME 48px window")
axes[3].imshow(small[max(cy2-box//2,0):cy2+box//2, max(cx2-box//2,0):cx2+box//2], cmap="gray"); axes[3].axis("off")
axes[3].set_title("Window content: camera + head + sky", color=RED)
fig.suptitle("A fixed-size window sees different content at different scales →\n"
             "any fixed-window descriptor of 'the same point' won't match. Scale must be DETECTED per point.",
             fontsize=11, y=1.06)
fig.tight_layout(); save(fig, "01_scale_problem")

# ----------------------------------------------------------------------------
# FIG 02 -- Gaussian blur as the scale dial
# ----------------------------------------------------------------------------
sigmas_demo = [0, 1.6, 3.2, 6.4, 12.8]
fig, axes = plt.subplots(1, len(sigmas_demo), figsize=(16, 3.6))
imf = img.astype(np.float64)/255
for ax, s in zip(axes, sigmas_demo):
    ax.imshow(imf if s == 0 else cv2.GaussianBlur(imf, (0,0), s), cmap="gray")
    ax.axis("off"); ax.set_title(f"σ = {s}" if s else "input")
fig.suptitle("σ is a physical scale parameter: structures smaller than ~σ are erased.\n"
             "'This blob survives until σ=6' IS a statement about its size.", fontsize=11, y=1.08)
fig.tight_layout(); save(fig, "02_gaussian_scale")

# ----------------------------------------------------------------------------
# Build the actual pipeline (used by all remaining figures)
# ----------------------------------------------------------------------------
gp = build_scale_space(img)                     # with 2x upsampling (octave -1)
dp = build_dog(gp)
cands = find_extrema(dp)
kept, rejected = refine_keypoints(dp, cands, first_octave=-1)
oriented = assign_orientations(kept, gp)
descs = compute_descriptors(oriented, gp)
rej_counts = Counter(r[-1] for r in rejected)
print("candidates", len(cands), "| kept", len(kept), "| oriented", len(oriented), "|", dict(rej_counts))

# ----------------------------------------------------------------------------
# FIG 03 -- the pyramid
# ----------------------------------------------------------------------------
sig = layer_sigmas()
n_show_oct = 4
fig, axes = plt.subplots(n_show_oct, S+3, figsize=(15, 2.6*n_show_oct))
for o in range(n_show_oct):
    for i in range(S+3):
        ax = axes[o, i]
        ax.imshow(gp[o][i], cmap="gray", vmin=0, vmax=1); ax.set_xticks([]); ax.set_yticks([])
        abs_sig = sig[i] * 2.0**(o-1)          # first_octave = -1
        ax.set_title(f"σ_abs={abs_sig:.2f}", fontsize=8)
        if i == 0:
            ax.set_ylabel(f"octave {o-1}\n{gp[o].shape[1]}px", fontsize=9)
        if o == 0 and i == S:
            for sp in ax.spines.values(): sp.set_edgecolor(RED); sp.set_linewidth(2); sp.set_visible(True)
fig.suptitle(f"Scale space: {S+3} layers per octave, k=2^(1/{S}) blur steps; each next octave = "
             "the red-framed layer (exactly 2x blur) downsampled 2x.\nSame blur ladder, half the pixels — "
             "downsampling buys a 4x compute saving per octave with no information loss (Nyquist).",
             fontsize=11, y=1.0)
fig.tight_layout(); save(fig, "03_pyramid")

# ----------------------------------------------------------------------------
# FIG 04 -- DoG ~ scale-normalized LoG
# ----------------------------------------------------------------------------
x = np.linspace(-8, 8, 400)
def G(x, s): return np.exp(-x**2/(2*s**2)) / (np.sqrt(2*np.pi)*s)
s0 = 1.6; k = 2**(1/S)
log_norm = s0**2 * np.gradient(np.gradient(G(x, s0), x), x)     # sigma^2 * d2G/dx2
dog_1d   = (G(x, k*s0) - G(x, s0)) / (k - 1)                    # DoG / (k-1)

fig = plt.figure(figsize=(15, 7.5))
gs = fig.add_gridspec(2, 4, height_ratios=[1, 1.15])
ax = fig.add_subplot(gs[0, :2])
ax.plot(x, log_norm, color=BLUE, lw=2.5, label="scale-normalized LoG:  σ²∇²G")
ax.plot(x, dog_1d, color=ORANGE, lw=2.5, ls="--", label="(G(kσ) − G(σ)) / (k−1)   [DoG]")
ax.legend(); ax.set_title("DoG is a ~free approximation of the blob detector σ²∇²G (1-D slice)")
ax.set_xlabel("x"); ax.axhline(0, color="k", lw=.5)
ax2 = fig.add_subplot(gs[0, 2:])
sig_c = np.linspace(0.5, 8, 300)
r = 3.0                                                          # blob radius
resp = np.abs(sig_c**2 * (-2/sig_c**2 + (2*r**2)/(sig_c**4)) * np.exp(-r**2/sig_c**2))  # |σ²∇²G * blob| shape
resp = resp / resp.max()
ax2.plot(sig_c, resp, color=GREEN, lw=2.5)
ax2.axvline(r/np.sqrt(2), color=RED, ls="--", lw=1.5)
ax2.annotate("response PEAKS when σ matches\nthe blob's size (σ ≈ r/√2)",
             xy=(r/np.sqrt(2), 1.0), xytext=(4.4, .75), arrowprops=dict(arrowstyle="->"), fontsize=9)
ax2.set_xlabel("σ of the filter"); ax2.set_ylabel("|response| at blob center")
ax2.set_title("Why extrema ACROSS σ = size detection: a blob of radius r\nfires maximally at exactly one σ")
o_show = 1
for j in range(4):
    axd = fig.add_subplot(gs[1, j])
    d = dp[o_show][j+1]
    m = np.abs(d).max()
    axd.imshow(d, cmap="RdBu_r", vmin=-m, vmax=m); axd.axis("off")
    axd.set_title(f"DoG layer {j+1}, octave 0\n(σ={sig[j+1]:.2f})", fontsize=9)
fig.suptitle("", y=1.0)
fig.tight_layout(); save(fig, "04_dog")

# ----------------------------------------------------------------------------
# FIG 05 -- extrema detection
# ----------------------------------------------------------------------------
fig = plt.figure(figsize=(14, 5.4))
gs = fig.add_gridspec(1, 3, width_ratios=[1.1, 1.5, 0.06])
ax = fig.add_subplot(gs[0, 0])
ax.set_xlim(-.7, 4.5); ax.set_ylim(-.9, 3.6); ax.axis("off")
for li, (ox, oy, lbl) in enumerate([(0, 0, "scale below"), (0.65, 0.65, "same scale"), (1.3, 1.3, "scale above")]):
    for r in range(3):
        for c in range(3):
            center = (li == 1 and r == 1 and c == 1)
            face = RED if center else ("#d9e3f2" if li != 1 else "#aec7e8")
            ax.add_patch(Rectangle((ox + c*.55, oy + r*.55), .5, .5, facecolor=face,
                                   edgecolor="k", lw=.7, zorder=3-abs(li-1)))
    ax.text(ox + 1.75, oy + .75, lbl, fontsize=9)
ax.set_title("A candidate (red) must beat all 26 neighbors:\n8 in its layer + 9 below + 9 above\n"
             "→ extremum in SPACE and in SCALE simultaneously")
ax = fig.add_subplot(gs[0, 1])
ax.imshow(img, cmap="gray", alpha=.75); ax.axis("off")
for (o, i, y, x) in cands:
    mult = 2.0**(o-1)
    ax.scatter(x*mult, y*mult, s=5+2.5**o, facecolor="none",
               edgecolor=OCT_COLORS[o % len(OCT_COLORS)], lw=.8)
ax.set_title(f"All {len(cands)} raw candidates — color/size = octave\n(small red = octave −1 fine detail, "
             "big = coarse blobs)")
fig.tight_layout(); save(fig, "05_extrema")

# ----------------------------------------------------------------------------
# FIG 06 -- refinement: subpixel fit + the two rejection tests
# ----------------------------------------------------------------------------
fig = plt.figure(figsize=(15, 8.6))
gs = fig.add_gridspec(2, 3)
# 6a: 1-D picture of the quadratic fit
axa = fig.add_subplot(gs[0, 0])
xs = np.array([-1, 0, 1]); ys = np.array([0.55, 0.9, 0.78])
A = np.polyfit(xs, ys, 2); xf = np.linspace(-1.4, 1.4, 100)
x_hat = -A[1] / (2*A[0])
axa.plot(xf, np.polyval(A, xf), color=BLUE, lw=2, label="fitted quadratic")
axa.scatter(xs, ys, s=70, color="k", zorder=3, label="sampled DoG values")
axa.axvline(0, color="gray", ls=":", lw=1)
axa.axvline(x_hat, color=RED, ls="--", lw=1.5)
axa.annotate("sample grid says\nextremum is HERE", xy=(0, .9), xytext=(-1.35, .62),
             arrowprops=dict(arrowstyle="->"), fontsize=8.5)
axa.annotate(f"true extremum:\noffset = {x_hat:+.2f} px\n(x̂ = −H⁻¹g in 3-D)", xy=(x_hat, np.polyval(A, x_hat)),
             xytext=(.5, .58), arrowprops=dict(arrowstyle="->", color=RED), fontsize=8.5, color=RED)
axa.set_title("Subpixel refinement (1-D view)"); axa.set_xlabel("position (samples)"); axa.set_ylabel("D")
axa.legend(fontsize=8, loc="lower left")
# 6b: the funnel
axb = fig.add_subplot(gs[0, 1])
stages = ["raw\nextrema", "− diverged", "− low\ncontrast", "− edge\nresponses", "kept"]
n0 = len(cands)
n1 = n0 - rej_counts["diverged"]; n2 = n1 - rej_counts["low_contrast"]
n3 = n2 - rej_counts["edge"]
vals = [n0, n1, n2, n3, len(kept)]
bars = axb.bar(stages, vals, color=[BLUE, PURPLE, ORANGE, RED, GREEN])
for b, v in zip(bars, vals):
    axb.text(b.get_x()+b.get_width()/2, v+18, str(v), ha="center", fontsize=9, fontweight="bold")
axb.set_title("The rejection funnel"); axb.set_ylabel("keypoints")
# 6c: edge-rejected points really do sit on edges
axc = fig.add_subplot(gs[0, 2])
axc.imshow(img, cmap="gray", alpha=.8); axc.axis("off")
for (o, i, y, x, why) in rejected:
    if why == "edge":
        axc.scatter(x*2.0**(o-1), y*2.0**(o-1), s=6, color=RED, alpha=.6)
axc.set_title(f"The {rej_counts['edge']} 'edge' rejects (red)\n— they trace contours, as predicted")
# 6d/e: DoG surface at an edge vs a corner (the WHY of the Hessian test)
mag_full, _ = _grads(gp[1][2])
axd = fig.add_subplot(gs[1, 0], projection="3d")
axe = fig.add_subplot(gs[1, 1], projection="3d")
edge_pt   = next((o,i,y,x) for (o,i,y,x,w2) in rejected if w2=="edge" and o==1)
corner_kp = max([k for k in kept if k["octave"]==1], key=lambda k: k["response"])
for ax3, (yy, xx), name, col in [(axd, (edge_pt[2], edge_pt[3]), "REJECTED: edge point", RED),
                                 (axe, (int(corner_kp["y_oct"]), int(corner_kp["x_oct"])), "KEPT: corner/blob", GREEN)]:
    r_ = 7
    patch = dp[1][2][yy-r_:yy+r_+1, xx-r_:xx+r_+1]
    Xg, Yg = np.meshgrid(range(patch.shape[1]), range(patch.shape[0]))
    ax3.plot_surface(Xg, Yg, patch, cmap="viridis", edgecolor="none")
    ax3.set_title(f"{name}\nDoG surface around the point", color=col, fontsize=10)
    ax3.set_xticks([]); ax3.set_yticks([]); ax3.set_zticks([])
axf = fig.add_subplot(gs[1, 2]); axf.axis("off")
axf.text(0, .5,
    "Edge test (2-D spatial Hessian):\n\n"
    "curvatures α, β along principal axes\n"
    "edge: α ≫ β (a ridge — position along\nthe ridge is not localized)\n\n"
    "test  tr(H)²/det(H) = (α+β)²/(αβ)\n"
    "reject if  > (r+1)²/r,   r = 10\n\n"
    "Same trace/det trick as the Harris\ncorner measure — no eigendecomposition\nneeded.",
    fontsize=10, va="center")
fig.tight_layout(); save(fig, "06_refinement")

# ----------------------------------------------------------------------------
# FIG 07 -- orientation assignment
# ----------------------------------------------------------------------------
# pick a strong keypoint at a mid octave with visible structure
kp_demo = max([k for k in kept if k["octave"] == 2], key=lambda k: k["response"])
hist, raw_hist, (wmag, ang_w, wgt) = orientation_histogram(kp_demo, gp, return_details=True)
gimg = gp[kp_demo["octave"]][kp_demo["layer"]]
yc, xc = int(round(kp_demo["y_oct"])), int(round(kp_demo["x_oct"]))
rad = int(round(3 * 1.5 * kp_demo["sigma_oct"]))

fig = plt.figure(figsize=(15, 4.8))
gs = fig.add_gridspec(1, 4, width_ratios=[1, 1, 1, 1.4])
ax = fig.add_subplot(gs[0]); ax.imshow(img, cmap="gray"); ax.axis("off")
ax.add_patch(Circle((kp_demo["x"], kp_demo["y"]), kp_demo["sigma"]*3, fill=False, edgecolor=GREEN, lw=2))
ax.set_title(f"the keypoint (octave {kp_demo['octave']-1},\nσ_abs={kp_demo['sigma']:.1f})")
ax = fig.add_subplot(gs[1])
patch = gimg[yc-rad:yc+rad+1, xc-rad:xc+rad+1]
ax.imshow(patch, cmap="gray"); ax.axis("off")
ax.set_title("its window in ITS pyramid level\n(radius = 3·1.5·σ_oct)")
ax = fig.add_subplot(gs[2])
mag_p, ang_p = _grads(gimg)
mp = mag_p[yc-rad:yc+rad+1, xc-rad:xc+rad+1]; ap = ang_p[yc-rad:yc+rad+1, xc-rad:xc+rad+1]
step = max(rad//8, 1)
Yq, Xq = np.mgrid[0:patch.shape[0]:step, 0:patch.shape[1]:step]
ax.imshow(patch, cmap="gray", alpha=.55)
ax.quiver(Xq, Yq, (mp*np.cos(ap))[::step, ::step], (-mp*np.sin(ap))[::step, ::step],
          color=ORANGE, scale=mp.max()*22)
ax.axis("off"); ax.set_title("gradient field (arrows), each vote\nGaussian-weighted by distance")
ax = fig.add_subplot(gs[3])
centers = (np.arange(N_ORI_BINS) + .5) / N_ORI_BINS * 360
ax.bar(centers, raw_hist, width=360/N_ORI_BINS*.9, color=BLUE, alpha=.4, label="raw votes")
ax.plot(centers, hist, color=ORANGE, lw=2.2, label="smoothed")
mx = hist.max()
ax.axhline(PEAK_RATIO*mx, color=RED, ls="--", lw=1.5, label="80% of max")
th_deg = np.degrees([k["theta"] for k in oriented
                     if abs(k["x"]-kp_demo["x"]) < .5 and abs(k["y"]-kp_demo["y"]) < .5])
for t in th_deg:
    ax.axvline(t, color=GREEN, lw=2)
ax.set_title(f"36-bin orientation histogram → {len(th_deg)} peak(s) ≥ 80% (green)\n"
             "each peak becomes its OWN keypoint")
ax.set_xlabel("gradient direction (deg)"); ax.set_ylabel("weighted magnitude"); ax.legend(fontsize=8)
fig.tight_layout(); save(fig, "07_orientation")

# ----------------------------------------------------------------------------
# FIG 08 -- the descriptor
# ----------------------------------------------------------------------------
kp_d = next(k for k in oriented
            if abs(k["x"]-kp_demo["x"]) < .5 and abs(k["y"]-kp_demo["y"]) < .5)
desc, det = compute_descriptor(kp_d, gp, return_details=True)
(y0, y1), (x0, x1) = det["window"]
gimg = gp[kp_d["octave"]][kp_d["layer"]]

fig = plt.figure(figsize=(15, 5.2))
gs = fig.add_gridspec(1, 3, width_ratios=[1.1, 1.1, 1.6])
# 8a: window + rotated grid
ax = fig.add_subplot(gs[0])
ax.imshow(gimg, cmap="gray"); ax.axis("off")
hw = 3.0 * kp_d["sigma_oct"]; th_ = kp_d["theta"]
ct, st = np.cos(th_), np.sin(th_)
for gxx in np.arange(-2, 2.01, 1):
    for a, b in [((gxx, -2), (gxx, 2)), ((-2, gxx), (2, gxx))]:
        p = np.array([a, b]) * hw
        pr = np.c_[ct*p[:,0]-st*p[:,1], st*p[:,0]+ct*p[:,1]]
        ax.plot(kp_d["x_oct"]+pr[:,0], kp_d["y_oct"]+pr[:,1], color=ORANGE, lw=1.4)
L_ = 2.4*hw
ax.annotate("", xy=(kp_d["x_oct"]+ct*L_, kp_d["y_oct"]+st*L_),
            xytext=(kp_d["x_oct"], kp_d["y_oct"]),
            arrowprops=dict(arrowstyle="->", color=GREEN, lw=2.5))
ax.set_xlim(x0-8, x1+8); ax.set_ylim(y1+8, y0-8)
ax.set_title("4×4 grid, ROTATED to the keypoint's θ\n(green = θ). Cell side = 3·σ_oct px\n→ scale + rotation covariant window")
# 8b: the classic star plot
ax = fig.add_subplot(gs[1]); ax.set_aspect("equal"); ax.axis("off")
dgrid = desc.reshape(D_DESC, D_DESC, N_DESC_BINS)
for r_ in range(D_DESC):
    for c_ in range(D_DESC):
        cxx, cyy = c_+.5, D_DESC-1-r_+.5
        ax.add_patch(Rectangle((c_, D_DESC-1-r_), 1, 1, fill=False, edgecolor="gray", lw=.8))
        for b in range(N_DESC_BINS):
            aa = b/N_DESC_BINS*2*np.pi
            ln = dgrid[r_, c_, b]*2.2
            ax.plot([cxx, cxx+ln*np.cos(aa)], [cyy, cyy+ln*np.sin(aa)], color=BLUE, lw=1.6)
ax.set_xlim(-.3, D_DESC+.3); ax.set_ylim(-.3, D_DESC+.3)
ax.set_title("the descriptor as 16 'stars':\n8 orientation strengths per cell")
# 8c: the raw 128 numbers
ax = fig.add_subplot(gs[2])
ax.bar(np.arange(128), desc, color=BLUE, width=1.0)
ax.axhline(0.2, color=RED, ls="--", lw=1.2)
ax.annotate("0.2 clip: no single gradient may dominate\n(illumination / specular robustness)",
            xy=(64, .2), xytext=(48, .26), fontsize=8.5, color=RED, arrowprops=dict(arrowstyle="->", color=RED))
for cell in range(1, 16):
    ax.axvline(cell*8-.5, color="gray", lw=.4, alpha=.6)
ax.set_title("the same descriptor as the actual 128-vector (‖·‖₂ = 1)\ngray lines = cell boundaries (16 cells × 8 bins)")
ax.set_xlabel("dimension"); ax.set_ylabel("value")
fig.tight_layout(); save(fig, "08_descriptor")

# ----------------------------------------------------------------------------
# FIG 09 -- validation vs OpenCV
# ----------------------------------------------------------------------------
cv_sift = cv2.SIFT_create(contrastThreshold=0.04, edgeThreshold=10, sigma=1.6)
kp_cv, des_cv = cv_sift.detectAndCompute(img, None)
ours = np.array([[k["x"], k["y"], k["sigma"]] for k in oriented])
cvs  = np.array([[k.pt[0], k.pt[1], k.size/2] for k in kp_cv])
tree = cKDTree(cvs[:, :2])
dd_, idx = tree.query(ours[:, :2])
scale_ok = np.abs(np.log2(ours[:, 2]/np.maximum(cvs[idx, 2], 1e-6))) < 0.6
rep = ((dd_ < 2.0) & scale_ok).mean()

fig, axes = plt.subplots(1, 2, figsize=(13, 6.4))
for ax, pts, name, n in [(axes[0], ours, "ours (from scratch)", len(oriented)),
                         (axes[1], cvs, "cv2.SIFT_create", len(kp_cv))]:
    ax.imshow(img, cmap="gray", alpha=.8); ax.axis("off")
    for (xx, yy, ss_) in pts:
        ax.add_patch(Circle((xx, yy), ss_*2, fill=False, edgecolor=GREEN, lw=.7, alpha=.8))
    ax.set_title(f"{name}: {n} keypoints (circle radius ∝ scale)")
fig.suptitle(f"{rep*100:.0f}% of our keypoints coincide with an OpenCV keypoint (<2px, matching scale)",
             fontsize=12, y=0.99)
fig.tight_layout(); save(fig, "09_vs_opencv")

# ----------------------------------------------------------------------------
# FIG 10 -- the payoff: matching under rotation + scale
# ----------------------------------------------------------------------------
M = cv2.getRotationMatrix2D((W/2, H/2), 30, 0.7)
img2 = cv2.warpAffine(img, M, (W, H))
kp2, d2, _ = sift(img2)
d1 = descs; kp1 = oriented
sim = d1 @ d2.T
dmat = np.sqrt(np.maximum((d1**2).sum(1)[:, None] + (d2**2).sum(1)[None, :] - 2*sim, 0))
nn = np.argsort(dmat, axis=1)[:, :2]
best = dmat[np.arange(len(d1)), nn[:, 0]]
second = dmat[np.arange(len(d1)), nn[:, 1]]
ratio_ok = best < 0.75 * second
p1 = np.array([[k["x"], k["y"]] for k in kp1])
p2 = np.array([[k["x"], k["y"]] for k in kp2])
p1_gt = np.c_[p1, np.ones(len(p1))] @ M.T
err = np.linalg.norm(p1_gt - p2[nn[:, 0]], axis=1)
correct = ratio_ok & (err < 3)
wrong = ratio_ok & (err >= 3)
prec = correct.sum()/ratio_ok.sum()

canvas = np.hstack([img, img2])
fig, axes = plt.subplots(2, 1, figsize=(13.5, 12.5))
axes[0].imshow(canvas, cmap="gray", alpha=.85); axes[0].axis("off")
ii = np.nonzero(correct)[0][::3]
for i_ in ii:
    axes[0].plot([p1[i_,0], p2[nn[i_,0],0]+W], [p1[i_,1], p2[nn[i_,0],1]], color=GREEN, lw=.7, alpha=.8)
for i_ in np.nonzero(wrong)[0]:
    axes[0].plot([p1[i_,0], p2[nn[i_,0],0]+W], [p1[i_,1], p2[nn[i_,0],1]], color=RED, lw=.9, alpha=.9)
axes[0].set_title(f"original ↔ rotated 30° + scaled 0.7x   |   ratio test (0.75): {ratio_ok.sum()} matches, "
                  f"{correct.sum()} correct = {prec*100:.0f}% precision (every 3rd green line drawn)")
ax = axes[1]
ax.hist(best[~ratio_ok]/second[~ratio_ok], bins=40, alpha=.6, color=RED, label="rejected by ratio test", density=True)
ax.hist(best[ratio_ok]/second[ratio_ok], bins=40, alpha=.6, color=GREEN, label="accepted", density=True)
ax.axvline(.75, color="k", ls="--", lw=1.5)
ax.set_xlabel("nearest / second-nearest descriptor distance")
ax.set_ylabel("density")
ax.set_title("Lowe's ratio test: correct matches have a UNIQUELY close nearest neighbor;\n"
             "ambiguous descriptors (ratio→1) are discarded rather than guessed")
ax.legend()
fig.tight_layout(); save(fig, "10_matching")

med = np.median(err[correct])
print(f"\nSTATS: rep vs cv2 {rep*100:.1f}% | ours {len(oriented)} vs cv2 {len(kp_cv)} kps")
print(f"matching: {ratio_ok.sum()} matches, precision {prec*100:.1f}%, median reproj err {med:.2f}px")

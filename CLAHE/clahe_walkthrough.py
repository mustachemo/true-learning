"""
CLAHE from first principles, with every intermediate stage dumped as a figure.

Pipeline of stages (each maps to a section in the doc):
  01  the input image + its histogram/CDF        -> why contrast is "hiding"
  02  global histogram equalization (HE)          -> the baseline fix + its failure
  03  tiling the image (the "A" in AHE)           -> local statistics
  04  per-tile histograms                         -> why tiles differ
  05  naive per-tile HE (AHE)                     -> noise amplification + blocks
  06  clipping + redistribution (the "CL")        -> bounding the slope
  07  per-tile mapping functions (LUTs)           -> clipped vs unclipped CDFs
  08  bilinear interpolation of LUTs              -> killing block artifacts
  09  final CLAHE vs OpenCV                       -> validation
  10  parameter sweep                             -> clip limit x grid size
"""

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from skimage import data

plt.rcParams.update({
    "figure.dpi": 130,
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.titleweight": "bold",
    "axes.spines.top": False,
    "axes.spines.right": False,
})
FIG = "figs"
BLUE, ORANGE, GREEN, RED = "#4C72B0", "#DD8452", "#55A868", "#C44E52"

# ----------------------------------------------------------------------------
# STAGE 0 -- input image
# ----------------------------------------------------------------------------
img = data.moon()  # 512x512 uint8, classic low-contrast image
H, W = img.shape
L = 256  # number of gray levels


def save(fig, name):
    fig.savefig(f"{FIG}/{name}.png", bbox_inches="tight")
    plt.close(fig)
    print("wrote", name)


# ----------------------------------------------------------------------------
# STAGE 1 -- histogram + CDF of the whole image
# ----------------------------------------------------------------------------
hist = np.bincount(img.ravel(), minlength=L)  # h(v): pixel counts per level
pdf = hist / hist.sum()  # p(v): probability of level v
cdf = np.cumsum(pdf)  # C(v): P(pixel <= v)

fig, axes = plt.subplots(1, 3, figsize=(14, 4))
axes[0].imshow(img, cmap="gray", vmin=0, vmax=255)
axes[0].set_title("Input: moon (512x512, uint8)")
axes[0].axis("off")
axes[1].fill_between(np.arange(L), hist, color=BLUE, alpha=0.85)
axes[1].set_title("Histogram h(v)")
axes[1].set_xlabel("gray level v")
axes[1].set_ylabel("pixel count")
axes[1].axvspan(0, 60, color=RED, alpha=0.12)
axes[1].axvspan(200, 255, color=RED, alpha=0.12)
axes[1].annotate(
    "almost no pixels\nout here",
    xy=(228, hist.max() * 0.55),
    ha="center",
    fontsize=9,
    color=RED,
)
axes[2].plot(cdf, color=ORANGE, lw=2)
axes[2].set_title("CDF  C(v) = cumulative sum of p(v)")
axes[2].set_xlabel("gray level v")
axes[2].set_ylabel("fraction of pixels ≤ v")
axes[2].annotate(
    "steep = crowded levels",
    xy=(120, 0.5),
    xytext=(160, 0.25),
    arrowprops=dict(arrowstyle="->"),
    fontsize=9,
)
fig.tight_layout()
save(fig, "01_input_hist_cdf")

# ----------------------------------------------------------------------------
# STAGE 2 -- global histogram equalization
#   T(v) = round( (L-1) * C(v) )   (using the CDF as the transfer function)
# ----------------------------------------------------------------------------
T_global = np.round((L - 1) * cdf).astype(np.uint8)  # the lookup table (LUT)
img_he = T_global[img]  # apply LUT: every pixel remapped

hist_he = np.bincount(img_he.ravel(), minlength=L)
cdf_he = np.cumsum(hist_he / hist_he.sum())

fig, axes = plt.subplots(2, 3, figsize=(14, 8))
axes[0, 0].imshow(img, cmap="gray", vmin=0, vmax=255)
axes[0, 0].set_title("Before")
axes[0, 0].axis("off")
axes[0, 1].imshow(img_he, cmap="gray", vmin=0, vmax=255)
axes[0, 1].set_title("After global HE")
axes[0, 1].axis("off")
axes[0, 2].plot(T_global, color=GREEN, lw=2)
axes[0, 2].plot([0, 255], [0, 255], "k--", lw=1, alpha=0.5)
axes[0, 2].set_title("The mapping  T(v) = 255·C(v)")
axes[0, 2].set_xlabel("input level v")
axes[0, 2].set_ylabel("output level")
axes[0, 2].annotate(
    "identity (no change)",
    xy=(190, 190),
    xytext=(120, 60),
    arrowprops=dict(arrowstyle="->"),
    fontsize=9,
)
axes[1, 0].fill_between(np.arange(L), hist, color=BLUE, alpha=0.85)
axes[1, 0].set_title("Histogram before")
axes[1, 1].fill_between(np.arange(L), hist_he, color=GREEN, alpha=0.85)
axes[1, 1].set_title("Histogram after (spread out)")
axes[1, 2].plot(cdf, color=BLUE, lw=2, label="before")
axes[1, 2].plot(cdf_he, color=GREEN, lw=2, label="after")
axes[1, 2].plot([0, 255], [0, 1], "k--", lw=1, alpha=0.5, label="ideal (uniform)")
axes[1, 2].legend()
axes[1, 2].set_title("CDF: after ≈ straight line = uniform histogram")
for ax in axes[1]:
    ax.set_xlabel("gray level")
fig.tight_layout()
save(fig, "02_global_he")

# ----------------------------------------------------------------------------
# STAGE 2b -- where global HE fails: one LUT for the whole image
# Zoom into a dark crater region and a bright region.
# ----------------------------------------------------------------------------
dark_box = (380, 180, 120)  # (row, col, size) -- dark lower-left area
bright_box = (40, 80, 80)  # bright upper area


def crop(im, box):
    r, c, s = box
    return im[r : r + s, c : c + s]


fig, axes = plt.subplots(2, 3, figsize=(13, 8))
for j, (im, name) in enumerate([(img, "input"), (img_he, "global HE")]):
    axes[j, 0].imshow(im, cmap="gray", vmin=0, vmax=255)
    axes[j, 0].axis("off")
    axes[j, 0].set_title(f"{name} (full)")
    for box, color in [(dark_box, RED), (bright_box, ORANGE)]:
        r, c, s = box
        axes[j, 0].add_patch(Rectangle((c, r), s, s, fill=False, edgecolor=color, lw=2))
    axes[j, 1].imshow(crop(im, dark_box), cmap="gray", vmin=0, vmax=255)
    axes[j, 1].axis("off")
    axes[j, 1].set_title(f"{name}: dark region", color=RED)
    axes[j, 2].imshow(crop(im, bright_box), cmap="gray", vmin=0, vmax=255)
    axes[j, 2].axis("off")
    axes[j, 2].set_title(f"{name}: bright region", color=ORANGE)
fig.suptitle(
    "One global LUT can't serve both regions: dark detail barely improves,\n"
    "because T(v) is built from statistics dominated by the mid-gray majority",
    fontsize=11,
)
fig.tight_layout()
save(fig, "02b_global_he_failure")

# ----------------------------------------------------------------------------
# STAGE 3 -- tiling (8x8 grid of 64x64 tiles)
# ----------------------------------------------------------------------------
GRID = 8
th, tw = H // GRID, W // GRID  # tile height/width = 64x64

fig, ax = plt.subplots(figsize=(6.5, 6.5))
ax.imshow(img, cmap="gray", vmin=0, vmax=255)
ax.axis("off")
for i in range(1, GRID):
    ax.axhline(i * th, color=ORANGE, lw=1)
    ax.axvline(i * tw, color=ORANGE, lw=1)
# mark tile centers (these become the anchor points for interpolation later)
cy = np.arange(GRID) * th + th // 2
cx = np.arange(GRID) * tw + tw // 2
gy, gx = np.meshgrid(cy, cx, indexing="ij")
ax.scatter(gx, gy, s=14, color=GREEN, zorder=3)
ax.set_title(
    f"{GRID}x{GRID} tiles of {th}x{tw}px — green dots are tile CENTERS\n"
    "(each center will own one histogram + one mapping function)"
)
fig.tight_layout()
save(fig, "03_tiling")


# ----------------------------------------------------------------------------
# STAGE 4 -- per-tile histograms: pick a flat tile and a detailed tile
# ----------------------------------------------------------------------------
def tile(im, ti, tj):
    return im[ti * th : (ti + 1) * th, tj * tw : (tj + 1) * tw]


flat_ij = (0, 3)  # top-left corner: nearly featureless dark sky
detail_ij = (6, 3)  # middle: craters

fig, axes = plt.subplots(2, 2, figsize=(11, 7))
for row, (ij, name, color) in enumerate([
    (flat_ij, "flat tile (0,0)", RED),
    (detail_ij, "detailed tile (4,4)", GREEN),
]):
    t = tile(img, *ij)
    h_t = np.bincount(t.ravel(), minlength=L)
    axes[row, 0].imshow(t, cmap="gray", vmin=0, vmax=255)
    axes[row, 0].axis("off")
    axes[row, 0].set_title(name, color=color)
    axes[row, 1].fill_between(np.arange(L), h_t, color=color, alpha=0.85)
    axes[row, 1].set_title(f"tile histogram — {t.size} px total, peak = {h_t.max()} px")
    axes[row, 1].set_xlabel("gray level")
    axes[row, 1].set_ylabel("count")
fig.suptitle(
    "Local statistics differ wildly — this is why one global LUT loses", fontsize=11
)
fig.tight_layout()
save(fig, "04_tile_histograms")

# ----------------------------------------------------------------------------
# STAGE 5 -- AHE: equalize every tile independently (no clip, no interpolation)
# Shows BOTH failure modes we still need to fix: noise blowup + block seams.
# ----------------------------------------------------------------------------
img_ahe_blocks = np.empty_like(img)
for ti in range(GRID):
    for tj in range(GRID):
        t = tile(img, ti, tj)
        c = np.cumsum(np.bincount(t.ravel(), minlength=L)) / t.size
        img_ahe_blocks[ti * th : (ti + 1) * th, tj * tw : (tj + 1) * tw] = np.round(
            255 * c
        ).astype(np.uint8)[t]

fig, axes = plt.subplots(1, 3, figsize=(14, 5))
axes[0].imshow(img_ahe_blocks, cmap="gray", vmin=0, vmax=255)
axes[0].axis("off")
axes[0].set_title("Per-tile HE (AHE), no clip, no interpolation")
r, c, s = 0, 0, 128
axes[1].imshow(img_ahe_blocks[r : r + s, c : c + s], cmap="gray", vmin=0, vmax=255)
axes[1].axis("off")
axes[1].set_title(
    "Zoom: flat sky tile — sensor noise\nblown up to full contrast", color=RED
)
axes[2].imshow(img_ahe_blocks[192:320, 192:320], cmap="gray", vmin=0, vmax=255)
axes[2].axis("off")
axes[2].set_title("Zoom: visible seams at tile borders", color=ORANGE)
fig.tight_layout()
save(fig, "05_ahe_failures")


# ----------------------------------------------------------------------------
# STAGE 6 -- clipping + redistribution on ONE tile's histogram
# clip_limit (OpenCV convention): actual ceiling = clipLimit * tile_px / 256
# ----------------------------------------------------------------------------
def clip_histogram(h, clip_limit_mult, n_px):
    """Clip histogram at ceiling; redistribute excess (OpenCV-exact one pass).
    excess = total mass above the ceiling.
    Every bin gets excess // 256; the integer remainder is sprinkled one
    pixel at a time over evenly spaced bins so the pixel count is conserved."""
    ceiling = max(int(clip_limit_mult * n_px / L), 1)
    h = h.astype(np.int64).copy()
    excess = np.maximum(h - ceiling, 0).sum()  # mass to redistribute
    h = np.minimum(h, ceiling)  # cut the peaks
    batch = excess // L  # equal share per bin
    h += batch
    residual = excess - batch * L  # leftover after integer division
    if residual:
        step = max(L // residual, 1)
        h[np.arange(0, L, step)[:residual]] += 1  # sprinkle remainder evenly
    return h, ceiling


CLIP = 2.0
t = tile(img, *flat_ij)
h_raw = np.bincount(t.ravel(), minlength=L).astype(float)
h_clipped, ceiling = clip_histogram(h_raw, CLIP, t.size)
excess_total = np.maximum(h_raw - ceiling, 0).sum()

fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.2))
axes[0].fill_between(np.arange(L), h_raw, color=BLUE, alpha=0.85)
axes[0].axhline(ceiling, color=RED, lw=2)
axes[0].fill_between(
    np.arange(L),
    np.minimum(h_raw, ceiling),
    h_raw,
    where=h_raw > ceiling,
    color=RED,
    alpha=0.5,
)
axes[0].set_title(
    f"1) Raw tile histogram + ceiling\nceiling = {CLIP}·{t.size}/256 = {ceiling} px/bin"
)
axes[0].annotate(
    f"excess mass\n= {int(excess_total)} px",
    xy=(30, ceiling * 2.2),
    color=RED,
    fontsize=9,
)
axes[1].fill_between(np.arange(L), np.minimum(h_raw, ceiling), color=BLUE, alpha=0.85)
axes[1].axhline(ceiling, color=RED, lw=2)
axes[1].set_title("2) Peaks cut at the ceiling")
axes[2].fill_between(np.arange(L), h_clipped, color=GREEN, alpha=0.85)
axes[2].axhline(ceiling, color=RED, lw=2)
axes[2].set_title(
    f"3) Excess spread evenly: every bin +{int(excess_total) // L}"
    f" (+1 to {int(excess_total) % L} spaced bins)\n(total pixel count is conserved)"
)
for ax in axes:
    ax.set_xlabel("gray level")
    ax.set_ylabel("count")
    ax.set_ylim(0, h_raw.max() * 1.05)
axes[1].set_ylim(0, ceiling * 4)
axes[2].set_ylim(0, ceiling * 4)
fig.tight_layout()
save(fig, "06_clipping")


# ----------------------------------------------------------------------------
# STAGE 7 -- what clipping does to the MAPPING function (the CDF/LUT)
# ----------------------------------------------------------------------------
def lut_from_hist(h):
    c = np.cumsum(h)  # un-normalized CDF (max = tile pixel count)
    return np.floor(
        c * (L - 1) / c[-1] + 0.5
    )  # scale to [0,255], round half up (OpenCV-exact)


fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
for ax, ij, name in [
    (axes[0], flat_ij, "flat tile (0,0)"),
    (axes[1], detail_ij, "detailed tile (4,4)"),
]:
    t = tile(img, *ij)
    h_raw = np.bincount(t.ravel(), minlength=L).astype(float)
    for clip_mult, color, lbl in [
        (None, RED, "no clip (AHE)"),
        (4.0, ORANGE, "clip=4"),
        (2.0, GREEN, "clip=2"),
    ]:
        h_use = (
            h_raw if clip_mult is None else clip_histogram(h_raw, clip_mult, t.size)[0]
        )
        ax.plot(lut_from_hist(h_use), color=color, lw=2, label=lbl)
    ax.plot([0, 255], [0, 255], "k--", lw=1, alpha=0.5, label="identity")
    ax.set_title(f"Mapping T(v) — {name}")
    ax.set_xlabel("input level")
    ax.set_ylabel("output level")
    ax.legend(fontsize=8)
axes[0].annotate(
    "near-vertical jump:\ntiny input noise → huge\noutput contrast",
    xy=(105, 200),
    xytext=(150, 60),
    fontsize=9,
    color=RED,
    arrowprops=dict(arrowstyle="->", color=RED),
)
fig.tight_layout()
save(fig, "07_luts_clipped_vs_not")


# ----------------------------------------------------------------------------
# STAGE 8 -- full from-scratch CLAHE with bilinear interpolation of tile LUTs
# ----------------------------------------------------------------------------
def clahe_scratch(im, grid=8, clip_mult=2.0, return_luts=False):
    Hh, Ww = im.shape
    th_, tw_ = Hh // grid, Ww // grid
    # 1) one clipped LUT per tile
    luts = np.empty((grid, grid, L))
    for ti in range(grid):
        for tj in range(grid):
            t = im[ti * th_ : (ti + 1) * th_, tj * tw_ : (tj + 1) * tw_]
            h = np.bincount(t.ravel(), minlength=L).astype(float)
            h, _ = clip_histogram(h, clip_mult, t.size)
            luts[ti, tj] = lut_from_hist(h)
    # 2) bilinear interpolation between the 4 nearest tile-center LUTs
    ys = np.arange(Hh)[:, None].repeat(Ww, 1).astype(float)  # pixel row coords
    xs = np.arange(Ww)[None, :].repeat(Hh, 0).astype(float)  # pixel col coords
    # position in "tile-center space": center of tile k sits at (k + 0.5) * tile_size
    fy = ys / th_ - 0.5
    fx = xs / tw_ - 0.5
    y0 = np.clip(np.floor(fy).astype(int), 0, grid - 1)  # tile above-left
    x0 = np.clip(np.floor(fx).astype(int), 0, grid - 1)
    y1 = np.clip(y0 + 1, 0, grid - 1)  # tile below-right
    x1 = np.clip(x0 + 1, 0, grid - 1)
    wy = np.clip(fy - y0, 0, 1)  # 0 at center y0, 1 at center y1
    wx = np.clip(fx - x0, 0, 1)
    v = im.astype(int)
    out = (
        (1 - wy) * (1 - wx) * luts[y0, x0, v]  # blend 4 candidate outputs
        + (1 - wy) * wx * luts[y0, x1, v]
        + wy * (1 - wx) * luts[y1, x0, v]
        + wy * wx * luts[y1, x1, v]
    )
    out = np.round(out).astype(np.uint8)
    return (out, luts) if return_luts else out


img_clahe, luts = clahe_scratch(img, GRID, CLIP, return_luts=True)

# 8a: interpolation geometry diagram
fig, ax = plt.subplots(figsize=(7, 7))
ax.imshow(img, cmap="gray", vmin=0, vmax=255, alpha=0.55)
ax.axis("off")
for i in range(1, GRID):
    ax.axhline(i * th, color="w", lw=0.6, alpha=0.6)
    ax.axvline(i * tw, color="w", lw=0.6, alpha=0.6)
ax.scatter(gx, gy, s=18, color=GREEN, zorder=3)
py, px = 150, 230  # an example pixel
ax.scatter([px], [py], s=90, color=RED, marker="x", zorder=4, linewidths=3)
# its 4 surrounding tile centers
fy0, fx0 = int(py / th - 0.5), int(px / tw - 0.5)
corners = [
    (cy[fy0], cx[fx0]),
    (cy[fy0], cx[fx0 + 1]),
    (cy[fy0 + 1], cx[fx0]),
    (cy[fy0 + 1], cx[fx0 + 1]),
]
for yy, xx in corners:
    ax.plot([px, xx], [py, yy], color=ORANGE, lw=2)
    ax.scatter(
        [xx], [yy], s=90, facecolor="none", edgecolor=ORANGE, linewidths=2, zorder=5
    )
ax.add_patch(
    Rectangle((cx[fx0], cy[fy0]), tw, th, fill=False, edgecolor=ORANGE, lw=1.5, ls="--")
)
ax.set_title(
    "Every pixel (red X) blends the LUT outputs of its 4 nearest\n"
    "tile centers (orange), weighted by distance — bilinear interpolation"
)
fig.tight_layout()
save(fig, "08_interpolation_geometry")

# 8b: the 4 LUTs + blended LUT for that example pixel
wy_ = py / th - 0.5 - fy0
wx_ = px / tw - 0.5 - fx0
blend = (
    (1 - wy_) * (1 - wx_) * luts[fy0, fx0]
    + (1 - wy_) * wx_ * luts[fy0, fx0 + 1]
    + wy_ * (1 - wx_) * luts[fy0 + 1, fx0]
    + wy_ * wx_ * luts[fy0 + 1, fx0 + 1]
)
fig, ax = plt.subplots(figsize=(6.5, 4.6))
for (ti, tj), lbl, wgt in [
    ((fy0, fx0), "top-left", (1 - wy_) * (1 - wx_)),
    ((fy0, fx0 + 1), "top-right", (1 - wy_) * wx_),
    ((fy0 + 1, fx0), "bottom-left", wy_ * (1 - wx_)),
    ((fy0 + 1, fx0 + 1), "bottom-right", wy_ * wx_),
]:
    ax.plot(luts[ti, tj], lw=1.4, alpha=0.65, label=f"{lbl} LUT (w={wgt:.2f})")
ax.plot(blend, color="k", lw=2.6, label="blended LUT this pixel actually uses")
ax.set_title(f"LUT blending at pixel ({py},{px}):  weights wy={wy_:.2f}, wx={wx_:.2f}")
ax.set_xlabel("input level")
ax.set_ylabel("output level")
ax.legend(fontsize=8)
fig.tight_layout()
save(fig, "08b_lut_blending")

# 8c: with vs without interpolation
fig, axes = plt.subplots(1, 2, figsize=(11, 5.5))
# per-tile clipped HE WITHOUT interpolation
img_clipped_blocks = np.empty_like(img)
for ti in range(GRID):
    for tj in range(GRID):
        t = tile(img, ti, tj)
        img_clipped_blocks[ti * th : (ti + 1) * th, tj * tw : (tj + 1) * tw] = luts[
            ti, tj
        ].astype(np.uint8)[t]
axes[0].imshow(img_clipped_blocks, cmap="gray", vmin=0, vmax=255)
axes[0].axis("off")
axes[0].set_title("Clipped per-tile LUTs, NO interpolation\n(seams remain)")
axes[1].imshow(img_clahe, cmap="gray", vmin=0, vmax=255)
axes[1].axis("off")
axes[1].set_title("Clipped LUTs + bilinear interpolation\n= CLAHE (seams gone)")
fig.tight_layout()
save(fig, "08c_interp_vs_not")

# ----------------------------------------------------------------------------
# STAGE 9 -- validation against OpenCV + full journey summary
# ----------------------------------------------------------------------------
cv = cv2.createCLAHE(clipLimit=CLIP, tileGridSize=(GRID, GRID))
img_cv = cv.apply(img)
diff = np.abs(img_clahe.astype(int) - img_cv.astype(int))

fig, axes = plt.subplots(1, 3, figsize=(14, 4.8))
axes[0].imshow(img_clahe, cmap="gray", vmin=0, vmax=255)
axes[0].axis("off")
axes[0].set_title("Ours (from scratch)")
axes[1].imshow(img_cv, cmap="gray", vmin=0, vmax=255)
axes[1].axis("off")
axes[1].set_title("cv2.createCLAHE")
im2 = axes[2].imshow(diff, cmap="magma", vmin=0, vmax=max(10, diff.max()))
axes[2].axis("off")
axes[2].set_title(f"|difference| — mean {diff.mean():.2f}, max {diff.max()}")
fig.colorbar(im2, ax=axes[2], fraction=0.046)
fig.tight_layout()
save(fig, "09_vs_opencv")

fig, axes = plt.subplots(1, 4, figsize=(16, 4.4))
for ax, im, name in [
    (axes[0], img, "1. input"),
    (axes[1], img_he, "2. global HE"),
    (axes[2], img_ahe_blocks, "3. AHE (no clip/interp)"),
    (axes[3], img_clahe, "4. CLAHE"),
]:
    ax.imshow(im, cmap="gray", vmin=0, vmax=255)
    ax.axis("off")
    ax.set_title(name)
fig.tight_layout()
save(fig, "09b_journey")

# ----------------------------------------------------------------------------
# STAGE 10 -- parameter sweep: clip limit x grid size
# ----------------------------------------------------------------------------
clips = [1.0, 2.0, 4.0, 40.0]
grids = [2, 8, 16]
fig, axes = plt.subplots(
    len(grids), len(clips), figsize=(4 * len(clips), 4 * len(grids))
)
for i, g in enumerate(grids):
    for j, cl in enumerate(clips):
        out = cv2.createCLAHE(clipLimit=cl, tileGridSize=(g, g)).apply(img)
        axes[i, j].imshow(out, cmap="gray", vmin=0, vmax=255)
        axes[i, j].axis("off")
        axes[i, j].set_title(f"grid {g}x{g}, clip {cl:g}")
fig.suptitle(
    "clip → contrast strength (40 ≈ unclipped AHE) | grid → locality scale",
    fontsize=13,
    y=1.0,
)
fig.tight_layout()
save(fig, "10_param_sweep")

print("\nvalidation: mean|ours - cv2| =", diff.mean(), " max =", diff.max())
print("pixels differing by >2 levels:", (diff > 2).mean() * 100, "%")

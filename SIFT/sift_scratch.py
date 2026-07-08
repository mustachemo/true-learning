"""
sift_scratch.py -- SIFT from first principles (Lowe 2004), numpy implementation.

Every stage is a standalone function so the walkthrough script can dump
intermediate outputs:

  build_scale_space()   -> Gaussian pyramid (octaves x layers)
  build_dog()           -> Difference-of-Gaussians pyramid
  find_extrema()        -> raw 26-neighbor extrema candidates
  refine_keypoints()    -> subpixel fit + contrast + edge tests (returns rejects too)
  assign_orientations() -> 36-bin gradient histogram, 80% multi-peak rule
  compute_descriptors() -> 4x4x8 = 128-d descriptors with trilinear soft-binning

Conventions:
  images float64 in [0,1]; coordinates are (row=y, col=x) in OCTAVE pixels;
  absolute image coords = octave coords * 2**octave (we skip Lowe's initial 2x
  upsampling, i.e. firstOctave = 0).
"""

import numpy as np
import cv2

# ---- Lowe's parameters ------------------------------------------------------
S            = 3        # intervals per octave (layers where extrema are usable)
SIGMA0       = 1.6      # blur of the first layer of octave 0
ASSUMED_BLUR = 0.5      # blur assumed already present in the input photo
CONTRAST_TH  = 0.04     # reject |D(x_hat)| < CONTRAST_TH / S
EDGE_R       = 10.0     # max ratio of principal curvatures
BORDER       = 5        # ignore keypoints this close to the image edge
N_ORI_BINS   = 36
PEAK_RATIO   = 0.8      # secondary orientation peaks >= 80% of max
D_DESC       = 4        # descriptor grid is D_DESC x D_DESC subregions
N_DESC_BINS  = 8        # orientation bins per subregion -> 4*4*8 = 128


def gauss(im, sigma):
    """Gaussian blur helper (cv2 for the primitive; SIFT is what we build)."""
    return cv2.GaussianBlur(im, (0, 0), sigmaX=sigma, sigmaY=sigma)


# ============================================================================
# STAGE 1: scale space
# ============================================================================
def layer_sigmas():
    """Absolute blur of each layer within an octave: sigma0 * k^i, k = 2^(1/S).
    S+3 layers so that after taking differences (S+2 DoGs) the middle S DoG
    layers each have a scale above AND below them for 3D extrema detection."""
    k = 2.0 ** (1.0 / S)
    return np.array([SIGMA0 * k**i for i in range(S + 3)])


def build_scale_space(img, n_octaves=None, upsample=True):
    """upsample=True implements Lowe's '-1 octave': double the image first.
    Why: the finest detectable blob scale is tied to the finest sampling grid;
    doubling the grid roughly quadruples the number of stable keypoints
    (Lowe 2004, sec. 3.3). Doubling also doubles the pre-existing camera blur,
    so the assumed blur becomes 2*ASSUMED_BLUR."""
    img = img.astype(np.float64)
    if img.max() > 1.0:
        img = img / 255.0
    assumed = ASSUMED_BLUR
    if upsample:
        img = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
        assumed = 2 * ASSUMED_BLUR
    if n_octaves is None:
        n_octaves = int(np.log2(min(img.shape))) - 3      # stop while tiles are meaningful
    sig = layer_sigmas()
    # bring the input up to sigma0 (blurs add in quadrature: sig_a2 + sig_b2 = sig_c2)
    base = gauss(img, np.sqrt(max(SIGMA0**2 - assumed**2, 0.01)))
    gauss_pyr = []
    for o in range(n_octaves):
        octave = [base]
        for i in range(1, S + 3):
            # incremental blur: going from sig[i-1] to sig[i] needs sqrt(diff of squares)
            octave.append(gauss(octave[-1], np.sqrt(sig[i]**2 - sig[i-1]**2)))
        gauss_pyr.append(np.stack(octave))
        # layer index S has exactly 2*sigma0 blur -> halving it seeds the next octave
        nxt = octave[S]
        base = nxt[::2, ::2]
    return gauss_pyr


# ============================================================================
# STAGE 2: Difference of Gaussians
# ============================================================================
def build_dog(gauss_pyr):
    return [g[1:] - g[:-1] for g in gauss_pyr]


# ============================================================================
# STAGE 3: raw extrema (26-neighbor test)
# ============================================================================
def find_extrema(dog_pyr):
    """Return list of (octave, layer, y, x) where |D| is a local max/min of its
    3x3x3 neighborhood and passes a cheap pre-threshold."""
    pre_th = 0.5 * CONTRAST_TH / S
    cands = []
    for o, dogs in enumerate(dog_pyr):
        n_lay, h, w = dogs.shape
        for i in range(1, n_lay - 1):
            cube = dogs[i-1:i+2]                            # (3, h, w)
            # stack all 27 shifted views of the 3-layer cube -> (27, h-2, w-2)
            views = np.stack([cube[l, 1+dy:h-1+dy, 1+dx:w-1+dx]
                              for l in range(3)
                              for dy in (-1, 0, 1)
                              for dx in (-1, 0, 1)])
            center = dogs[i, 1:-1, 1:-1]
            is_max = (center >= views.max(0)) & (center > 0)
            is_min = (center <= views.min(0)) & (center < 0)
            mask = (np.abs(center) > pre_th) & (is_max | is_min)
            ys, xs = np.nonzero(mask)
            ys += 1; xs += 1
            keep = (ys >= BORDER) & (ys < h - BORDER) & (xs >= BORDER) & (xs < w - BORDER)
            cands += [(o, i, y, x) for y, x in zip(ys[keep], xs[keep])]
    return cands


# ============================================================================
# STAGE 4: subpixel refinement + contrast + edge rejection
# ============================================================================
def refine_keypoints(dog_pyr, candidates, max_iter=5, first_octave=-1):
    """Fit a 3D quadratic to D around each candidate, slide to the true extremum,
    then apply the contrast and edge tests. Returns (kept, rejected) where each
    kept item is a dict and rejected items are tagged with the reason.
    first_octave=-1 when the scale space was built on a 2x-upsampled image."""
    sig = layer_sigmas()
    kept, rejected = [], []
    for (o, i, y, x) in candidates:
        dogs = dog_pyr[o]
        n_lay, h, w = dogs.shape
        converged = False
        for _ in range(max_iter):
            c = dogs[i]
            # 3D gradient and Hessian of D by central finite differences
            g = np.array([(dogs[i+1][y, x] - dogs[i-1][y, x]) / 2,      # dD/dlayer
                          (c[y+1, x] - c[y-1, x]) / 2,                  # dD/dy
                          (c[y, x+1] - c[y, x-1]) / 2])                 # dD/dx
            dss = dogs[i+1][y, x] - 2*c[y, x] + dogs[i-1][y, x]
            dyy = c[y+1, x] - 2*c[y, x] + c[y-1, x]
            dxx = c[y, x+1] - 2*c[y, x] + c[y, x-1]
            dsy = (dogs[i+1][y+1, x] - dogs[i+1][y-1, x] - dogs[i-1][y+1, x] + dogs[i-1][y-1, x]) / 4
            dsx = (dogs[i+1][y, x+1] - dogs[i+1][y, x-1] - dogs[i-1][y, x+1] + dogs[i-1][y, x-1]) / 4
            dyx = (c[y+1, x+1] - c[y+1, x-1] - c[y-1, x+1] + c[y-1, x-1]) / 4
            Hm = np.array([[dss, dsy, dsx], [dsy, dyy, dyx], [dsx, dyx, dxx]])
            try:
                offset = -np.linalg.solve(Hm, g)     # x_hat = -H^-1 g  (layer, y, x)
            except np.linalg.LinAlgError:
                break
            if np.all(np.abs(offset) < 0.5):
                converged = True
                break
            # extremum lies in a neighboring sample: step there and re-fit
            i += int(np.round(offset[0])); y += int(np.round(offset[1])); x += int(np.round(offset[2]))
            if not (1 <= i < n_lay-1 and BORDER <= y < h-BORDER and BORDER <= x < w-BORDER):
                break
        if not converged:
            rejected.append((o, i, y, x, "diverged")); continue
        # value of the fitted quadratic at the extremum
        D_hat = dogs[i][y, x] + 0.5 * g @ offset
        if np.abs(D_hat) < CONTRAST_TH / S:
            rejected.append((o, i, y, x, "low_contrast")); continue
        # edge test: 2x2 SPATIAL Hessian; edges have one big + one tiny curvature
        tr, det = dyy + dxx, dyy*dxx - dyx**2
        if det <= 0 or tr**2 * EDGE_R >= det * (EDGE_R + 1)**2:
            rejected.append((o, i, y, x, "edge")); continue
        li, yy, xx = i + offset[0], y + offset[1], x + offset[2]
        mult = 2.0 ** (o + first_octave)                    # octave px -> original-image px
        kept.append(dict(
            octave=o, layer=i, y_oct=yy, x_oct=xx,
            y=yy * mult, x=xx * mult,                       # absolute image coords
            sigma_oct=SIGMA0 * (2.0 ** (li / S)),           # scale within octave
            sigma=SIGMA0 * (2.0 ** (li / S)) * mult,        # absolute scale
            response=np.abs(D_hat)))
    return kept, rejected


# ============================================================================
# STAGE 5: orientation assignment
# ============================================================================
def _grads(gimg):
    dy, dx = np.gradient(gimg)
    return np.sqrt(dx**2 + dy**2), np.arctan2(dy, dx)      # magnitude, angle


def orientation_histogram(kp, gauss_pyr, return_details=False):
    """36-bin gradient-orientation histogram in a Gaussian-weighted window."""
    o = kp["octave"]
    gimg = gauss_pyr[o][kp["layer"]]
    h, w = gimg.shape
    mag, ang = _grads(gimg)
    sigma_w = 1.5 * kp["sigma_oct"]                        # window std, prop. to scale
    radius = int(np.round(3 * sigma_w))
    yc, xc = int(np.round(kp["y_oct"])), int(np.round(kp["x_oct"]))
    y0, y1 = max(yc-radius, 0), min(yc+radius+1, h)
    x0, x1 = max(xc-radius, 0), min(xc+radius+1, w)
    m = mag[y0:y1, x0:x1]; a = ang[y0:y1, x0:x1]
    dy = np.arange(y0, y1)[:, None] - kp["y_oct"]
    dx = np.arange(x0, x1)[None, :] - kp["x_oct"]
    wgt = np.exp(-(dy**2 + dx**2) / (2 * sigma_w**2))      # closer pixels count more
    bins = (np.round(a / (2*np.pi) * N_ORI_BINS).astype(int)) % N_ORI_BINS
    hist = np.bincount(bins.ravel(), weights=(m*wgt).ravel(), minlength=N_ORI_BINS)
    raw = hist.copy()
    for _ in range(6):                                     # circular [1,4,6,4,1]/16 smoothing
        hist = (np.roll(hist, 2) + 4*np.roll(hist, 1) + 6*hist
                + 4*np.roll(hist, -1) + np.roll(hist, -2)) / 16.0
    if return_details:
        return hist, raw, (m*wgt, a, wgt)
    return hist


def assign_orientations(kps, gauss_pyr):
    """Each histogram peak >= 80% of the max spawns its own oriented keypoint."""
    out = []
    for kp in kps:
        hist = orientation_histogram(kp, gauss_pyr)
        mx = hist.max()
        if mx <= 0:
            continue
        for b in range(N_ORI_BINS):
            l, r = hist[(b-1) % N_ORI_BINS], hist[(b+1) % N_ORI_BINS]
            if hist[b] >= PEAK_RATIO * mx and hist[b] > l and hist[b] > r:
                # parabolic interpolation of the peak for sub-bin accuracy
                db = 0.5 * (l - r) / (l - 2*hist[b] + r)
                theta = ((b + db) / N_ORI_BINS) * 2*np.pi
                k2 = dict(kp); k2["theta"] = theta % (2*np.pi)
                out.append(k2)
    return out


# ============================================================================
# STAGE 6: the 128-d descriptor
# ============================================================================
def compute_descriptor(kp, gauss_pyr, return_details=False):
    o = kp["octave"]
    gimg = gauss_pyr[o][kp["layer"]]
    h, w = gimg.shape
    mag, ang = _grads(gimg)
    theta = kp["theta"]
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    hist_width = 3.0 * kp["sigma_oct"]                     # each subregion spans 3*sigma px
    radius = int(np.round(hist_width * np.sqrt(2) * (D_DESC + 1) * 0.5))
    yc, xc = int(np.round(kp["y_oct"])), int(np.round(kp["x_oct"]))
    y0, y1 = max(yc-radius, 0), min(yc+radius+1, h)
    x0, x1 = max(xc-radius, 0), min(xc+radius+1, w)
    dy = (np.arange(y0, y1)[:, None] - kp["y_oct"]) * np.ones((1, x1-x0))
    dx = (np.arange(x0, x1)[None, :] - kp["x_oct"]) * np.ones((y1-y0, 1))
    # rotate offsets INTO the keypoint's frame (this is what buys rotation invariance)
    x_rot = ( cos_t*dx + sin_t*dy) / hist_width
    y_rot = (-sin_t*dx + cos_t*dy) / hist_width
    rbin = y_rot + D_DESC/2 - 0.5                          # continuous row bin in [-1, D)
    cbin = x_rot + D_DESC/2 - 0.5
    m = mag[y0:y1, x0:x1]; a = ang[y0:y1, x0:x1]
    obin = ((a - theta) % (2*np.pi)) / (2*np.pi) * N_DESC_BINS  # gradient angle RELATIVE to theta
    wgt = np.exp(-(x_rot**2 + y_rot**2) / (2 * (0.5*D_DESC)**2))
    valid = (rbin > -1) & (rbin < D_DESC) & (cbin > -1) & (cbin < D_DESC)
    rb, cb, ob, mw = rbin[valid], cbin[valid], obin[valid], (m*wgt)[valid]
    # trilinear soft-binning: each sample spreads over its 2x2x2 neighboring bins
    hist = np.zeros((D_DESC+2, D_DESC+2, N_DESC_BINS))
    r0, c0, o0 = np.floor(rb).astype(int), np.floor(cb).astype(int), np.floor(ob).astype(int)
    fr, fc, fo = rb-r0, cb-c0, ob-o0
    for dr in (0, 1):
        for dc in (0, 1):
            for do in (0, 1):
                wv = mw * (fr if dr else 1-fr) * (fc if dc else 1-fc) * (fo if do else 1-fo)
                np.add.at(hist, (r0+dr+1, c0+dc+1, (o0+do) % N_DESC_BINS), wv)
    desc = hist[1:-1, 1:-1, :].ravel()
    # normalize -> clip at 0.2 -> renormalize (illumination robustness)
    n = np.linalg.norm(desc)
    if n > 1e-9: desc = desc / n
    desc = np.minimum(desc, 0.2)
    n = np.linalg.norm(desc)
    if n > 1e-9: desc = desc / n
    if return_details:
        return desc, dict(x_rot=x_rot, y_rot=y_rot, valid=valid, window=((y0, y1), (x0, x1)))
    return desc


def compute_descriptors(kps, gauss_pyr):
    return np.array([compute_descriptor(kp, gauss_pyr) for kp in kps])


# ============================================================================
# Full pipeline
# ============================================================================
def sift(img, upsample=True):
    gp = build_scale_space(img, upsample=upsample)
    dp = build_dog(gp)
    cands = find_extrema(dp)
    kept, rejected = refine_keypoints(dp, cands, first_octave=-1 if upsample else 0)
    oriented = assign_orientations(kept, gp)
    descs = compute_descriptors(oriented, gp)
    return oriented, descs, dict(gauss_pyr=gp, dog_pyr=dp, candidates=cands,
                                 kept=kept, rejected=rejected)

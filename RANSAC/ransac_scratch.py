"""
ransac_scratch.py -- RANSAC from first principles.

The core is ONE generic loop (ransac): hypothesize from a minimal sample,
verify by counting consensus, keep the best, stop when the adaptive iteration
bound says a clean sample has almost surely been drawn. Everything else in
this file is problem-specific plumbing for the two demo problems:

  LINE (s=2):        fit_line_2pts, line_residuals, fit_line_ols
  HOMOGRAPHY (s=4):  dlt_homography (normalized), homography_residuals

Both wrappers (ransac_line, ransac_homography) end with the standard
least-squares refit on the consensus set, iterated a few times ("LO-lite").
"""

import numpy as np


# ============================================================================
# The algorithm: one generic hypothesize-and-verify loop
# ============================================================================
def n_iterations(p, w, s):
    """Iterations N so that P(at least one all-inlier sample) >= p.

    One sample of size s is all-inlier with prob w**s (w = inlier ratio).
    P(N samples ALL contaminated) = (1 - w**s)**N  <= 1 - p
    =>  N >= log(1-p) / log(1 - w**s)
    """
    ws = np.clip(w ** s, 1e-12, 1 - 1e-12)
    return int(np.ceil(np.log(1 - p) / np.log(1 - ws)))


def ransac(n_data, fit_minimal, residuals, s, eps,
           p=0.99, max_iter=5000, rng=None, is_degenerate=None):
    """Generic RANSAC.

    n_data       : number of data points
    fit_minimal  : idx (s,) -> model (or None if the sample can't be fit)
    residuals    : model -> (n_data,) array of |residual| per point
    s            : minimal sample size (2 for a line, 4 for a homography)
    eps          : inlier threshold on |residual|
    p            : demanded probability that >=1 sample was all-inlier
    Returns (best_model, best_inlier_mask, trace); trace logs
    (iteration, best_inlier_count, current_N_bound) for plotting.
    """
    rng = np.random.default_rng() if rng is None else rng
    best_model, best_mask, best_cnt = None, None, -1
    N = max_iter                                    # unknown w -> start pessimistic
    trace = []
    it = 0
    while it < N:
        idx = rng.choice(n_data, size=s, replace=False)     # hypothesize...
        it += 1
        if is_degenerate is not None and is_degenerate(idx):
            trace.append((it, best_cnt, N)); continue
        model = fit_minimal(idx)
        if model is None:
            trace.append((it, best_cnt, N)); continue
        cnt = int((residuals(model) < eps).sum())           # ...verify by counting
        if cnt > best_cnt:
            best_model, best_cnt = model, cnt
            best_mask = residuals(model) < eps
            w_hat = max(cnt / n_data, 1e-6)                 # adaptive bound:
            N = min(max_iter, n_iterations(p, w_hat, s))    # better model -> fewer iters needed
        trace.append((it, best_cnt, N))
    return best_model, best_mask, trace


# ============================================================================
# Problem 1: 2-D line fitting
# Line as ax + by + c = 0 with a^2+b^2 = 1, so |ax+by+c| IS perpendicular distance.
# ============================================================================
def fit_line_2pts(p, q):
    d = q - p
    n = np.linalg.norm(d)
    if n < 1e-9:
        return None                                  # degenerate: identical points
    a, b = -d[1] / n, d[0] / n                       # unit normal
    return np.array([a, b, -(a * p[0] + b * p[1])])


def line_residuals(line, pts):
    return np.abs(pts @ line[:2] + line[2])


def fit_line_ols(pts):
    """Total least squares (perpendicular residuals) via PCA -- the honest
    'least squares' baseline for the same residual definition RANSAC uses."""
    c = pts.mean(0)
    _, _, Vt = np.linalg.svd(pts - c)
    a, b = Vt[-1]                                    # normal = least-variance direction
    return np.array([a, b, -(a * c[0] + b * c[1])])


def ransac_line(pts, eps, p=0.99, max_iter=5000, rng=None, refit=True):
    fit = lambda idx: fit_line_2pts(pts[idx[0]], pts[idx[1]])
    res = lambda m: line_residuals(m, pts)
    model, mask, trace = ransac(len(pts), fit, res, s=2, eps=eps,
                                p=p, max_iter=max_iter, rng=rng)
    if refit and mask is not None and mask.sum() >= 2:
        for _ in range(3):                           # refit -> re-gate -> repeat
            model = fit_line_ols(pts[mask])
            mask = line_residuals(model, pts) < eps
    return model, mask, trace


# ============================================================================
# Problem 2: homography via normalized DLT
# ============================================================================
def normalize_pts(pts):
    """Hartley normalization: centroid -> origin, mean distance -> sqrt(2).
    Returns (normalized pts, the 3x3 transform T that does it)."""
    c = pts.mean(0)
    d = np.sqrt(((pts - c) ** 2).sum(1)).mean()
    s = np.sqrt(2) / max(d, 1e-12)
    T = np.array([[s, 0, -s * c[0]],
                  [0, s, -s * c[1]],
                  [0, 0, 1.0]])
    return (pts - c) * s, T


def dlt_homography(p1, p2, normalize=True):
    """Direct Linear Transform from >=4 correspondences.
    Each pair (x,y)->(u,v) gives 2 rows of A h = 0; h = smallest right
    singular vector of A; H is recovered by undoing the normalizations."""
    if normalize:
        p1n, T1 = normalize_pts(p1)
        p2n, T2 = normalize_pts(p2)
    else:
        p1n, p2n = p1, p2
        T1 = T2 = np.eye(3)
    A = []
    for (x, y), (u, v) in zip(p1n, p2n):
        A.append([-x, -y, -1,  0,  0,  0, u * x, u * y, u])
        A.append([ 0,  0,  0, -x, -y, -1, v * x, v * y, v])
    _, sv, Vt = np.linalg.svd(np.asarray(A))
    H = Vt[-1].reshape(3, 3)
    H = np.linalg.inv(T2) @ H @ T1                   # undo normalization
    return H / H[2, 2], sv                           # sv returned for the conditioning demo


def apply_h(H, pts):
    ph = np.c_[pts, np.ones(len(pts))] @ H.T
    return ph[:, :2] / ph[:, 2:3]


def homography_residuals(H, p1, p2):
    """Forward transfer error d(p2, H p1) -- same definition cv2 uses for its
    ransacReprojThreshold, which keeps the comparison apples-to-apples."""
    return np.linalg.norm(apply_h(H, p1) - p2, axis=1)


def _any3_collinear(pts, tol=1e-6):
    from itertools import combinations
    for i, j, k in combinations(range(len(pts)), 3):
        a, b, c = pts[i], pts[j], pts[k]
        if abs(np.cross(b - a, c - a)) < tol * max(np.linalg.norm(b - a) * np.linalg.norm(c - a), 1e-12):
            return True
    return False


def ransac_homography(p1, p2, eps=3.0, p=0.99, max_iter=5000, rng=None, refit=True):
    fit = lambda idx: dlt_homography(p1[idx], p2[idx])[0]
    res = lambda H: homography_residuals(H, p1, p2)
    degen = lambda idx: _any3_collinear(p1[idx]) or _any3_collinear(p2[idx])
    H, mask, trace = ransac(len(p1), fit, res, s=4, eps=eps,
                            p=p, max_iter=max_iter, rng=rng, is_degenerate=degen)
    if refit and mask is not None and mask.sum() >= 4:
        for _ in range(3):                           # LS on ALL inliers beats 4-pt fit
            H = dlt_homography(p1[mask], p2[mask])[0]
            mask = homography_residuals(H, p1, p2) < eps
    return H, mask, trace

"""
rpc_scratch.py -- the Fursov & Kotov 2018 pipeline from first principles.

We build a fully synthetic but physically meaningful setup so that every claim
in the paper can be checked against ground truth:

  SCENE     terrain(E, N)     : smooth height field over a 20 x 20 km area
  CAMERA    project(pts)      : oblique pinhole "satellite" -- the TRUE sensor
  MODEL     first-order RFM   : eq (1)-(3) of the paper, solved per image axis
  METRICS   lam_min, k(A), Phi(A), the error bounds of eq (9)-(14), eq (19)
  SELECTION exhaustive + greedy maximization of Q1/Q2/Q3, eq (20)-(23)

Coordinates:
  ground: E (east, m), N (north, m), h (height, m)  -- "phi, lambda, h" stand-ins
  image : x, y in pixels (the camera's native output)
  All are affinely normalized to [-1, 1] before entering the RFM (P, L, H and
  X, Y in the paper's notation) -- Section "normalization" of the doc explains why.
"""

import numpy as np
from itertools import combinations

# ============================================================================
# SCENE: 20 x 20 km area, smooth terrain 0..~900 m
# ============================================================================
AREA = 20000.0

def terrain(E, N):
    e, n = E / AREA, N / AREA
    h = (350 * np.exp(-((e - .30) ** 2 + (n - .60) ** 2) / .050)
         + 500 * np.exp(-((e - .72) ** 2 + (n - .28) ** 2) / .035)
         + 220 * np.sin(3.1 * e) * np.cos(2.3 * n) + 240)
    return np.clip(h, 0, None)

# ============================================================================
# TRUE SENSOR: oblique pinhole. This is the "rigorous model" the paper says
# we do NOT have access to -- we use it only to manufacture ground truth.
# ============================================================================
SAT = np.array([AREA / 2, -4.0e5, 4.5e5])        # 450 km up, ~42 deg off-nadir
TARGET = np.array([AREA / 2, AREA / 2, 400.0])
F = 6.0e5                                        # focal length in pixels

def _look_at(sat, target):
    z = target - sat; z = z / np.linalg.norm(z)          # optical axis
    x = np.cross(z, np.array([0, 0, 1.0])); x = x / np.linalg.norm(x)
    y = np.cross(z, x)
    return np.stack([x, y, z])                           # world -> camera rows

R_CAM = _look_at(SAT, TARGET)

def project(pts):
    """TRUE camera: world (E,N,h) -> image pixels (x, y).
    Note the division by depth Zc: projection is INHERENTLY a ratio of
    functions that are affine in the world coordinates. This single division
    is the reason the RFM is a RATIONAL model."""
    c = (np.atleast_2d(pts) - SAT) @ R_CAM.T
    return np.c_[F * c[:, 0] / c[:, 2], F * c[:, 1] / c[:, 2]]

# ============================================================================
# NORMALIZATION: affine map of every coordinate to [-1, 1]
# ============================================================================
class Norm:
    def __init__(self, vals):
        self.off = (vals.max() + vals.min()) / 2         # OFFSET (RPC metadata term)
        self.scale = (vals.max() - vals.min()) / 2       # SCALE
    def fwd(self, v):  return (v - self.off) / self.scale
    def inv(self, v):  return v * self.scale + self.off

def make_norms(ground, img):
    """One Norm per coordinate, fitted on the working area."""
    return dict(E=Norm(ground[:, 0]), N=Norm(ground[:, 1]), h=Norm(ground[:, 2]),
                x=Norm(img[:, 0]), y=Norm(img[:, 1]))

def normalize_ground(g, nm):
    return np.c_[nm["E"].fwd(g[:, 0]), nm["N"].fwd(g[:, 1]), nm["h"].fwd(g[:, 2])]

# ============================================================================
# THE MODEL: first-order RFM, eq (1) -> linear system, eq (3)-(4)
#   img = (a0 + a1 L + a2 P + a3 H) / (1 + b1 L + b2 P + b3 H)
#   J = [a0, a1, a2, a3, b1, b2, b3]  (7 unknowns -> >= 7 GCPs per axis)
# Here L, P, H = normalized E, N, h  (keeping the paper's letters).
# ============================================================================
def build_M(img_norm, L, P, H):
    """One row per GCP:  [1, L, P, H, -Y L, -Y P, -Y H]  (paper eq (4))."""
    return np.column_stack([np.ones_like(L), L, P, H,
                            -img_norm * L, -img_norm * P, -img_norm * H])

def solve_J(M, Yv, use_normal_eq=False):
    if use_normal_eq:                                    # eq (5), verbatim
        A = M.T @ M
        return np.linalg.solve(A, M.T @ Yv)
    return np.linalg.lstsq(M, Yv, rcond=None)[0]         # QR/SVD: same estimate,
                                                         # kinder to roundoff

def rfm_eval(J, L, P, H):
    num = J[0] + J[1] * L + J[2] * P + J[3] * H
    den = 1.0 + J[4] * L + J[5] * P + J[6] * H
    return num / den

def fit_rpc(gcp_ground, gcp_img, nm):
    """Fit the two independent systems (one per image axis).
    Returns Jx, Jy and the two Gramians Ax, Ay = M^T M."""
    L, P, H = normalize_ground(gcp_ground, nm).T
    X = nm["x"].fwd(gcp_img[:, 0]); Y = nm["y"].fwd(gcp_img[:, 1])
    Mx, My = build_M(X, L, P, H), build_M(Y, L, P, H)
    return (solve_J(Mx, X), solve_J(My, Y),
            Mx.T @ Mx, My.T @ My, Mx, My)

def rfm_predict_px(Jx, Jy, ground, nm):
    L, P, H = normalize_ground(ground, nm).T
    return np.c_[nm["x"].inv(rfm_eval(Jx, L, P, H)),
                 nm["y"].inv(rfm_eval(Jy, L, P, H))]

# ============================================================================
# METRICS: the paper's conditioning measures, eq (7), (8), (15), (19)-(22)
# ============================================================================
def eigs(A):        return np.linalg.eigvalsh(A)               # ascending
def lam_min(A):     return eigs(A)[0]                          # eq (7)
def cond_k(A):      e = eigs(A); return e[-1] / max(e[0], 1e-300)   # eq (8)
def phi(A):         return np.trace(A) ** 2 / (A * A).sum()    # eq (15)
                    # for symmetric A: (sum lam)^2 / sum lam^2  (participation ratio)

def lam_min_bound(A):
    """Eq (19): lower bound on lam_min from trace and Phi alone (no eigensolver).
    Derivation in the doc: Samuelson's inequality applied to the spectrum.
    Valid (positive) only in the window of eq (18): m-1 < Phi <= m."""
    m = A.shape[0]
    ph = phi(A)
    return np.trace(A) / m * (1 - np.sqrt(max((m / ph - 1) * (m - 1), 0)))

def q1(A):  return lam_min(A)                                  # eq (20)
def q2(A):  return 1.0 / cond_k(A)                             # eq (21)
def q3(A):  return phi(A) - A.shape[0] + 1                     # eq (22)

def rmse_px(Jx, Jy, check_ground, check_img_true, nm):
    """Eq (24)-(26) on noiseless check points, reported in pixels."""
    pred = rfm_predict_px(Jx, Jy, check_ground, nm)
    ex = pred[:, 0] - check_img_true[:, 0]
    ey = pred[:, 1] - check_img_true[:, 1]
    return (np.sqrt((ex ** 2).mean()), np.sqrt((ey ** 2).mean()),
            np.sqrt((ex ** 2 + ey ** 2).mean()))

# ============================================================================
# SELECTION: eq (23) -- maximize Q_j over K-subsets of the N GCPs.
# Criterion of a subset = the WORSE of the two axes (both systems must be sane).
# ============================================================================
def subset_score(idx, gcp_ground, gcp_img, nm, q):
    L, P, H = normalize_ground(gcp_ground[idx], nm).T
    X = nm["x"].fwd(gcp_img[idx, 0]); Y = nm["y"].fwd(gcp_img[idx, 1])
    Ax = (Mx := build_M(X, L, P, H)).T @ Mx
    Ay = (My := build_M(Y, L, P, H)).T @ My
    return min(q(Ax), q(Ay))

def select_exhaustive(gcp_ground, gcp_img, nm, K, q):
    """Full search over C(N, K) subsets -- the paper's own proposal for small N."""
    best, best_s = None, -np.inf
    for idx in combinations(range(len(gcp_ground)), K):
        s = subset_score(np.array(idx), gcp_ground, gcp_img, nm, q)
        if s > best_s:
            best, best_s = np.array(idx), s
    return best, best_s

def select_greedy_backward(gcp_ground, gcp_img, nm, K, q):
    """Start from all N points; repeatedly drop the point whose removal hurts
    the criterion least. Keeps every intermediate system well-posed and costs
    O((N-K) * N) scores instead of C(N, K)."""
    idx = list(range(len(gcp_ground)))
    while len(idx) > K:
        scores = [subset_score(np.array(idx[:i] + idx[i+1:]),
                               gcp_ground, gcp_img, nm, q)
                  for i in range(len(idx))]
        idx.pop(int(np.argmax(scores)))
    return np.array(idx), subset_score(np.array(idx), gcp_ground, gcp_img, nm, q)

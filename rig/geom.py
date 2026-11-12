"""标定用的几何算法，全部闭式解，numpy 实现。"""
from __future__ import annotations

import numpy as np
import os

HERE = os.path.dirname(os.path.abspath(__file__))


def _normalizer(pts: np.ndarray) -> np.ndarray:
    """Hartley 归一化：平移到质心、缩放到平均距离 sqrt(2)。"""
    mu = pts.mean(axis=0)
    d = np.mean(np.hypot(*(pts - mu).T))
    s = np.sqrt(2.0) / max(d, 1e-12)
    return np.array([[s, 0, -s * mu[0]], [0, s, -s * mu[1]], [0, 0, 1.0]])


def homography(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """归一化 DLT：dst ~ H @ src（src/dst 都是 (N,2)）。"""
    Ts, sh = _normalizer(src), _normalizer(dst)
    s = (Ts @ np.hstack([src, np.ones((len(src), 1))]).T).T
    dh = (sh @ np.hstack([dst, np.ones((len(dst), 1))]).T).T
    M = []
    for (u, v, _), (x, y, _) in zip(s, dh):
        M.append([-u, -v, -1, 0, 0, 0, x * u, x * v, x])
        M.append([0, 0, 0, -u, -v, -1, y * u, y * v, y])
    _, _, Vt = np.linalg.svd(np.array(M))
    Hn = Vt[-1].reshape(3, 3)
    H = np.linalg.inv(sh) @ Hn @ Ts
    return H / H[2, 2]


def zhang_intrinsics(views: list[tuple[np.ndarray, np.ndarray]]):
    """views: [(obj_Nx3 平面 z=0, img_Nx2), ...]；至少 3 个不同姿态。"""
    Vs = []
    for obj, img in views:
        H = homography(obj[:, :2], img)
        h1, h2, h3 = H[:, 0], H[:, 1], H[:, 2]
        def v(a, b):
            return np.array([a[0] * b[0], a[0] * b[1] + a[1] * b[0], a[1] * b[1],
                             a[2] * b[0] + a[0] * b[2], a[2] * b[1] + a[1] * b[2], a[2] * b[2]])
        Vs.append(v(h1, h2))
        Vs.append(v(h1, h1) - v(h2, h2))
    V = np.array(Vs)
    _, _, Vt = np.linalg.svd(V)
    b = Vt[-1]
    B11, B12, B22, B13, B23, B33 = b
    v0 = (B12 * B13 - B11 * B23) / (B11 * B22 - B12 ** 2)
    lam = B33 - (B13 ** 2 + v0 * (B12 * B13 - B11 * B23)) / B11
    fx = np.sqrt(abs(lam / B11))
    fy = np.sqrt(abs(lam * B11 / (B11 * B22 - B12 ** 2)))
    cx = -B13 * fx ** 2 / lam
    cy = -B23 * fy ** 2 / lam
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]])

    rms = []
    for obj, img in views:
        R, t = pose_planar(K, obj, img)
        p = (K @ (R @ obj.T + t[:, None])).T
        p = p[:, :2] / p[:, 2:3]
        rms.append(float(np.sqrt(np.mean(np.sum((p - img) ** 2, axis=1)))))
    return K, np.array(rms)


def pose_planar(K: np.ndarray, obj: np.ndarray, img: np.ndarray):
    """平面目标（obj z=0）的位姿：先单应，再按 K^-1 H 分解出 R,t。"""
    H = homography(obj[:, :2], img)
    A = np.linalg.inv(K) @ H
    lam = 2.0 / (np.linalg.norm(A[:, 0]) + np.linalg.norm(A[:, 1]))
    A = A * lam
    r1, r2, t = A[:, 0], A[:, 1], A[:, 2]
    if t[2] < 0:
        r1, r2, t = -r1, -r2, -t
    r3 = np.cross(r1, r2)
    R = np.stack([r1, r2, r3], axis=1)
    U, _, Vt = np.linalg.svd(R)
    R = U @ np.diag([1, 1, np.linalg.det(U @ Vt)]) @ Vt
    return R, t


def umeyama(src: np.ndarray, dst: np.ndarray):
    """求 dst ≈ s R src + t 的最小二乘解（这里固定 s=1）。返回 R,t,每点残差。"""
    mu_s, mu_d = src.mean(axis=0), dst.mean(axis=0)
    S = src - mu_s
    D = dst - mu_d
    C = D.T @ S / len(src)
    U, _, Vt = np.linalg.svd(C)
    E = np.eye(3)
    if np.linalg.det(U @ Vt) < 0:
        E[2, 2] = -1
    R = U @ E @ Vt
    t = mu_d - R @ mu_s
    pred = (R @ src.T).T + t
    res = np.linalg.norm(pred - dst, axis=1)
    return R, t, res


def ray_plane(K: np.ndarray, R: np.ndarray, t: np.ndarray, uv: np.ndarray,
              plane_n: np.ndarray, plane_d: float):
    """像素 → 相机系射线 → 与平面 n·X=d 交于相机系 → 转到基座系。"""
    Ray = (np.linalg.inv(K) @ np.hstack([uv, np.ones((len(uv), 1))]).T).T
    n_cam = R @ plane_n                 # 平面法向在相机系
    d_cam = plane_d - float(n_cam @ t)  # 平面方程在相机系：n_cam·X_cam = d_cam
    denom = Ray @ n_cam
    s = d_cam / np.where(np.abs(denom) < 1e-9, np.nan, denom)
    X_cam = Ray * s[:, None]
    return (R.T @ (X_cam - t).T).T


def _normalizer3d(pts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mu = pts.mean(axis=0)
    d = np.mean(np.linalg.norm(pts - mu, axis=1))
    s = np.sqrt(3.0) / max(d, 1e-12)
    T = np.array([[s, 0, 0, -s * mu[0]], [0, s, 0, -s * mu[1]],
                  [0, 0, s, -s * mu[2]], [0, 0, 0, 1.0]])
    return T, (pts - mu) * s


def fit_projection(p3: np.ndarray, uv: np.ndarray) -> np.ndarray:
    """归一化 DLT 拟合 3x4 相机矩阵 P：uv ~ P @ [X,Y,Z,1]^T。至少 6 点。"""
    p3 = np.asarray(p3, float)
    uv = np.asarray(uv, float)
    T3, X = _normalizer3d(p3)
    T2 = _normalizer(uv)
    x = (T2 @ np.hstack([uv, np.ones((len(uv), 1))]).T).T
    Xh = np.hstack([X, np.ones((len(X), 1))])
    A = []
    for (X1, X2, X3, X4), (u, v, _) in zip(Xh, x):
        A.append([X1, X2, X3, X4, 0, 0, 0, 0, -u * X1, -u * X2, -u * X3, -u * X4])
        A.append([0, 0, 0, 0, X1, X2, X3, X4, -v * X1, -v * X2, -v * X3, -v * X4])
    _, _, Vt = np.linalg.svd(np.array(A))
    Pn = Vt[-1].reshape(3, 4)
    P = np.linalg.inv(T2) @ Pn @ T3
    return P / np.linalg.norm(P[:, :3])


def decompose_projection(P: np.ndarray):
    """P = K [R|t]（K 对角 > 0、R 正交、det(R)=1）。返回 K, R, t。"""
    P = np.asarray(P, float).copy()
    if np.linalg.det(P[:, :3]) < 0:      # P 与 -P 等价，选 det>0 那一支
        P = -P
    Q, Rm = np.linalg.qr(P[:, :3])       # P[:, :3] = Q @ Rm
    K, R = Rm.copy(), Q.copy()
    for i in range(3):                   # 让 K 的对角为正
        if K[i, i] < 0:
            K[:, i] *= -1.0
            R[i, :] *= -1.0
    K = K / K[2, 2]
    t = np.linalg.solve(K, P[:, 3])
    return K, R, t


def pixel_to_plane(K: np.ndarray, R: np.ndarray, t: np.ndarray,
                   uv, z: float) -> np.ndarray:
    """像素射线与 z=const 平面求交，返回基座系 (x,y)。"""
    C = -R.T @ t
    d = R.T @ np.linalg.solve(K, np.array([uv[0], uv[1], 1.0]))
    if abs(d[2]) < 1e-9:
        return np.array([np.nan, np.nan])
    lam = (z - C[2]) / d[2]
    return C[:2] + lam * d[:2]


def camera_center(P: np.ndarray) -> np.ndarray:
    """P 的零空间就是相机中心（齐次坐标）。"""
    _, _, Vt = np.linalg.svd(np.asarray(P, float))
    C = Vt[-1]
    return C[:3] / C[3]


def pixel_to_base(P: np.ndarray, uv, z: float) -> np.ndarray:
    """直接用投影矩阵把像素反投影到 z=const 平面，返回基座系 (x,y)。"""
    P = np.asarray(P, float)
    C = camera_center(P)
    d = np.linalg.solve(P[:, :3], np.array([uv[0], uv[1], 1.0]))
    if abs(d[2]) < 1e-12:
        return np.array([np.nan, np.nan])
    lam = (z - C[2]) / d[2]
    return C[:2] + lam * d[:2]


def _skew(k):
    return np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])


def rodrigues(r):
    th = float(np.linalg.norm(r))
    if th < 1e-12:
        return np.eye(3) + _skew(r)
    k = r / th
    Kx = _skew(k)
    return np.eye(3) + np.sin(th) * Kx + (1.0 - np.cos(th)) * (Kx @ Kx)


def rot_to_rvec(R):
    """旋转矩阵 → 轴角向量（与 rodrigues 互逆）。"""
    c = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
    th = float(np.arccos(c))
    v = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    if th < 1e-8:
        return 0.5 * v
    if th > np.pi - 1e-4:                      # 180°：轴从对角线取
        a = np.sqrt(np.maximum((np.diag(R) + 1.0) / 2.0, 0.0))
        i = int(np.argmax(a))
        a = a / max(a[i], 1e-12)
        for j in range(3):
            if j != i and R[i, j] + R[j, i] < 0:
                a[j] = -a[j]
        return th * a / np.linalg.norm(a)
    return th / (2.0 * np.sin(th)) * v


def _resid(th, p3, uv, cx, cy):
    f, rv, t = 1000.0 * th[0], th[1:4], th[4:7]
    R = rodrigues(rv)
    Xc = p3 @ R.T + t
    z = np.where(np.abs(Xc[:, 2]) < 1e-6, 1e-6, Xc[:, 2])
    return np.concatenate([f * Xc[:, 0] / z + cx - uv[:, 0],
                           f * Xc[:, 1] / z + cy - uv[:, 1]])


def _lm(resfun, x0, iters=200):
    """纯 numpy 的 Levenberg-Marquardt（数值雅可比），不依赖 scipy。"""
    x = np.asarray(x0, float)
    r = resfun(x)
    cost = float(r @ r)
    lam = 1e-3
    n = x.size
    for _ in range(iters):
        J = np.zeros((r.size, n))
        for i in range(n):
            h = 1e-7 * max(1.0, abs(x[i]))
            xp = x.copy()
            xp[i] += h
            J[:, i] = (resfun(xp) - r) / h
        A = J.T @ J
        g = J.T @ r
        d = np.diag(A).copy()
        d[d <= 0] = 1e-12
        improved = False
        for _ in range(12):
            try:
                step = np.linalg.solve(A + lam * np.diag(d), -g)
            except np.linalg.LinAlgError:
                lam *= 10.0
                continue
            xn = x + step
            rn = resfun(xn)
            cn = float(rn @ rn)
            if cn < cost:
                x, r, cost = xn, rn, cn
                lam = max(lam * 0.3, 1e-9)
                improved = True
                break
            lam *= 10.0
        if not improved or np.max(np.abs(step)) < 1e-12:
            break
    return x, float(np.sqrt(cost / max(1, r.size)))


def _reproj_rms(th, p3, uv, cx, cy) -> float:
    r = _resid(th, p3, uv, cx, cy)
    return float(np.sqrt(np.mean(r ** 2)))


def fit_camera(p3, uv, cx: float = 640.0, cy: float = 360.0, f0: float = 1100.0,
               p3_test=None, uv_test=None, f_min: float = 700.0, cz_min: float = 0.90):
    """已知主点、方形像素的相机外参标定：fit_camera[p3(米), uv(px)]。"""
    p3 = np.asarray(p3, float)
    uv = np.asarray(uv, float)
    C0 = p3.mean(axis=0) + np.array([0.0, 0.0, 1.0])
    f = lambda th: _resid(th, p3, uv, cx, cy)

    best = None
    for R0 in (np.diag([1.0, -1.0, -1.0]),
               np.array([[1.0, 0, 0], [0, 0, -1.0], [0, 1.0, 0]]),
               np.array([[1.0, 0, 0], [0, 0, -1.0], [0, -1.0, 0]])):
        x0 = np.concatenate([[f0 / 1000.0], rot_to_rvec(R0), -R0 @ C0])
        x, rms = _lm(f, x0)
        R = rodrigues(x[1:4])
        Cx, fv = -R.T @ x[4:7], 1000.0 * x[0]
        if not (f_min < fv < 3000.0) or Cx[2] < cz_min:
            continue          # 焦距太小 / 相机在桌面下：都是过拟合出来的假解，直接丢
        score = rms
        if p3_test is not None and len(p3_test) > 0:
            test_rms = _reproj_rms(x, p3_test, uv_test, cx, cy)
            score = test_rms                       # 有留出点时按留出误差挑，不看拟合误差
        if best is None or score < best[1]:
            best = (x, score)
    if best is None:
        raise RuntimeError("相机标定没收敛（焦距/高度都不合理），采点的对应关系可能不对")
    x, rms = best
    fv, rv, t = 1000.0 * float(x[0]), x[1:4], x[4:7]
    R = rodrigues(rv)
    if not (300.0 < fv < 4000.0):
        raise RuntimeError(f"标定出的焦距 {fv:.0f}px 不合理（正常 700–2000）")
    C = -R.T @ t
    if abs(C[2]) < 0.3:
        raise RuntimeError(f"标定出的相机高度 {C[2]:.2f} m 不合理")
    return {"f": fv, "R": R, "t": t, "C": C, "rms_px": rms,
            "K": np.array([[fv, 0, cx], [0, fv, cy], [0, 0, 1.0]])}


def _resid_tool(th, p3, uv, rots, cx, cy):
    """带「靶标相对末端偏移 o（工具系）」的相机模型：P_marker = P_tcp + R_ee @ o。"""
    f = 1000.0 * th[0]
    R = rodrigues(th[1:4])
    t = th[4:7]
    o = th[7:10]
    P = p3 + np.einsum("nij,j->ni", rots, o)
    Xc = P @ R.T + t
    z = np.where(np.abs(Xc[:, 2]) < 1e-6, 1e-6, Xc[:, 2])
    return np.concatenate([f * Xc[:, 0] / z + cx - uv[:, 0],
                           f * Xc[:, 1] / z + cy - uv[:, 1]])


def _resid_tool2(th, p3, uv, rots, cx, cy):
    """th = [f/1000, Cx, Cy, Cz, rvx, rvy, rvz, ox, oy, oz]"""
    f = 1000.0 * th[0]
    C = th[1:4]
    Rc = rodrigues(th[4:7])
    o = th[7:10]
    Pm = p3 + np.einsum("nij,j->ni", rots, o)
    Xc = (Pm - C) @ Rc.T
    z = np.where(np.abs(Xc[:, 2]) < 1e-6, 1e-6, Xc[:, 2])
    return np.concatenate([f * Xc[:, 0] / z + cx - uv[:, 0],
                           f * Xc[:, 1] / z + cy - uv[:, 1]])


try:
    from scipy.optimize import least_squares as _lsq
    from scipy.spatial.transform import Rotation as Rot
except Exception:            # 没有 scipy 就退回纯 numpy LM
    _lsq = None
    Rot = None


def _fit_tool_scipy(p3, uv, rots, cx, cy, f_min, cz_min, n_starts, seed=0):
    rng = np.random.default_rng(seed)
    lo = np.array([f_min / 1000.0, -1.5, -1.5, max(0.5, cz_min), -np.pi, -np.pi, -np.pi,
                   -0.30, -0.30, -0.30])
    hi = np.array([3.0, 1.5, 1.5, 2.5, np.pi, np.pi, np.pi,
                   0.30, 0.30, 0.30])
    best = None
    for _ in range(n_starts):
        x0 = np.array([rng.uniform(0.7, 2.0),
                       rng.uniform(0.2, 1.0), rng.uniform(-0.7, 0.5), rng.uniform(0.9, 1.8),
                       *Rot.from_rotvec(rng.normal(size=3) / np.sqrt(3) * rng.uniform(0, 2.6)
                                        ).as_rotvec(),
                       rng.uniform(-0.06, 0.06), rng.uniform(-0.06, 0.06), rng.uniform(-0.06, 0.06)])
        x0 = np.clip(x0, lo, hi)
        try:
            r = _lsq(lambda t: _resid_tool2(t, p3, uv, rots, cx, cy),
                     x0, bounds=(lo, hi), loss="soft_l1", f_scale=1.0, max_nfev=3000)
        except Exception:
            continue
        if best is None or r.cost < best.cost:
            best = r
    if best is None:
        return None
    e = _resid_tool2(best.x, p3, uv, rots, cx, cy).reshape(-1, 2)
    return best.x, float(np.sqrt((e ** 2).sum(axis=1).mean()))


def fit_camera_tool(p3, uv, rots, cx: float = 640.0, cy: float = 360.0,
                    f0: float = 1100.0, f_min: float = 700.0, cz_min: float = 0.90,
                    loo: bool = True, n_starts: int = 40):
    """解相机(7) + 靶标安装偏移 o(3)，共 10 个未知量。"""
    p3 = np.asarray(p3, float)
    uv = np.asarray(uv, float)
    rots = np.asarray(rots, float)
    if len(p3) < 8:
        raise RuntimeError(f"要解靶标偏移至少要 8 个点（现在 {len(p3)} 个）")

    res = None
    if _lsq is not None and Rot is not None:
        res = _fit_tool_scipy(p3, uv, rots, cx, cy, f_min, cz_min, n_starts)
    if res is None:          # 没有 scipy：沿用原来的多初值 LM
        C0 = p3.mean(axis=0) + np.array([0.0, 0.0, 1.0])
        fun = lambda th: _resid_tool(th, p3, uv, rots, cx, cy)
        rng = np.random.default_rng(7)
        inits = [np.diag([1.0, -1.0, -1.0]),
                 np.array([[1.0, 0, 0], [0, 0, -1.0], [0, 1.0, 0]]),
                 np.array([[1.0, 0, 0], [0, 0, -1.0], [0, -1.0, 0]])]
        for _ in range(6):
            A = rng.normal(size=(3, 3))
            Q, _ = np.linalg.qr(A)
            if np.linalg.det(Q) < 0:
                Q[:, 0] *= -1
            inits.append(Q)
        best = None
        for R0 in inits:
            for oz in (0.0, 0.04, 0.08, 0.12):
                x0 = np.concatenate([[f0 / 1000.0], rot_to_rvec(R0), -R0 @ C0, [0.0, 0.0, oz]])
                x, rms = _lm(fun, x0)
                R = rodrigues(x[1:4])
                C = -R.T @ x[4:7]
                fv = 1000.0 * x[0]
                if not (f_min < fv < 3000.0) or C[2] < cz_min or np.linalg.norm(x[7:10]) > 0.30:
                    continue
                if best is None or rms < best[1]:
                    best = (x, rms)
        if best is None:
            raise RuntimeError("相机 + 靶标偏移一起解也没收敛：检查 8 个点的对应关系（是不是同一台相机、同一个靶标）")
        x, rms = best
        tsol = np.array([1000.0 * x[0], *rot_to_rvec(rodrigues(x[1:4])), *x[4:10]])
        res = (tsol, float(rms))

    th, rms = res
    fv = 1000.0 * float(th[0])
    C = th[1:4]
    Rc = rodrigues(th[4:7])
    o = th[7:10]
    t = -Rc @ C
    if not (f_min < fv < 3000.0) or C[2] < cz_min:
        raise RuntimeError(f"解出的相机不合理：f={fv:.0f}px、高度 {C[2]:.2f}m")

    loo_px = float("nan")
    if loo and len(p3) >= 9:
        errs = []
        for i in range(len(p3)):
            m = np.ones(len(p3), bool)
            m[i] = False
            try:
                sub = _fit_tool_scipy(p3[m], uv[m], rots[m], cx, cy, f_min, cz_min, 12, seed=100 + i)
                if sub is None:
                    continue
                e = _resid_tool2(sub[0], p3[i:i + 1], uv[i:i + 1], rots[i:i + 1], cx, cy)
                errs.append(float(np.hypot(e[0], e[1])))
            except Exception:
                pass
        if errs:
            loo_px = float(np.mean(errs))
    return {"f": fv, "R": Rc, "t": t, "C": C, "rms_px": float(rms),
            "offset": o, "loo_px": loo_px, "K": np.array([[fv, 0, cx], [0, fv, cy], [0, 0, 1.0]])}

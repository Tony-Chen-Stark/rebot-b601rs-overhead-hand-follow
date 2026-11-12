#!/usr/bin/env python3
"""俯视相机标定：把标定纸的一角换成机械臂自己。"""
from __future__ import annotations

import argparse
import collections
import json
import os
import socket
import threading
import time

import numpy as np

import follow_hand_v2 as fv
from rig import geom

HERE = os.path.dirname(os.path.abspath(__file__))

OUT = os.path.join(HERE, "calib_base.json")


class UDPReader(threading.Thread):
    def __init__(self, port: int) -> None:
        super().__init__(daemon=True)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind(("127.0.0.1", port))
        self.sock.settimeout(0.05)
        self.lock = threading.Lock()
        self.packet: dict | None = None
        self.stamp = 0.0
        self._run = True

    def run(self) -> None:
        while self._run:
            try:
                raw, _ = self.sock.recvfrom(4096)
            except socket.timeout:
                continue
            try:
                p = json.loads(raw)
            except Exception:
                continue
            with self.lock:
                self.packet = p
                self.stamp = time.time()

    def latest(self):
        with self.lock:
            return self.packet, self.stamp


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ns", default="/rebot_arm_rs")
    ap.add_argument("--port", type=int, default=8899)
    ap.add_argument("--points", type=int, default=10)
    ap.add_argument("--min-gap", type=float, default=0.05, help="两个采样点至少相隔多少米")
    ap.add_argument("--still", type=float, default=0.8, help="静止多久才算一个点")
    ap.add_argument("--holdout", type=int, default=2, help="留几个点不参与拟合，用来验证")
    ap.add_argument("--height", type=float, default=0.10, help="标定时末端要保持的高度（米）")
    ap.add_argument("--target", choices=("marker", "hand"), default="marker",
                    help="用末端上的标记(marker)还是直接用手掌(hand)当靶标")
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()

    reader = UDPReader(args.port)
    reader.start()

    arm = fv.Arm()
    fv.rclpy.init()
    node = fv.Follow(args.ns)
    t0 = time.time()
    while node.q is None and time.time() - t0 < 8.0:
        fv.rclpy.spin_once(node, timeout_sec=0.1)
    if node.q is None:
        print(f"没收到 {args.ns}/joint_states —— 控制器没在跑？", flush=True)
        return 1

    key = "hand" if args.target == "hand" else "aruco"
    what = "手掌" if key == "hand" else "标记"
    print("=" * 60)
    print("两种采法，任选其一：")
    print("  A 稳（推荐）：10 个点都在同一高度（离桌面 10–15cm，差 <3cm），只平移 → 平面模型")
    print("  B 全：分三层高度（如 10/18/26cm）各 3–4 个点、xy 也铺开 → 解相机+靶标偏移")
    if key == "hand":
        print("  （手心摊平悬在夹爪正上方 1–2cm，拿掌心当靶标）")
    else:
        print("  （靶标绑在夹爪上，哪个位置都行，绑紧别晃，画面里要一直看得见）")
    print("每挪到一个位姿停稳等它记点；手腕怎么转都没关系。")
    print(f"点越分散越好（四角 + 中间），一共 {args.points} 个，最后 {args.holdout} 个留作验证。")
    print(f"找不到{what}的时候会有提示。Ctrl-C 退出。")
    print("=" * 60, flush=True)

    win: collections.deque = collections.deque(maxlen=120)
    last_seen = 0.0
    samples: list[tuple[float, float, float, float]] = []
    last_print = 0.0
    px_tol = 6.0
    arm_tol = 0.0015

    def _med(vals):
        return float(np.median(vals))

    try:
        while len(samples) < args.points:
            fv.rclpy.spin_once(node, timeout_sec=0.02)
            p, _R = arm.fk_rot(node.q)
            pkt, st = reader.latest()
            now = time.time()
            if pkt is None or not pkt.get(key):
                if now - last_seen > 0.4:   # 单帧丢失不清窗口，抖动时不至于一直重来
                    win.clear()
                if now - last_print > 0.5:
                    print(f"\r没看到{what}——预览窗口里它被标出来了吗？  "
                          f"已采 {len(samples)}/{args.points}   ", end="", flush=True)
                    last_print = now
                continue
            a = pkt[key]
            last_seen = now
            win.append((p[0], p[1], float(a["u"]), float(a["v"]), now, p[2]))

            recent = [w for w in win if now - w[4] <= args.still + 0.15]
            span = (recent[-1][4] - recent[0][4]) if len(recent) > 1 else 0.0
            if len(recent) < 10 or span < args.still * 0.75:
                if now - last_print > 0.4:
                    print(f"\r手 ({a['u']:.0f},{a['v']:.0f}) 高 {p[2]:.3f} m 静止 "
                          f"{span:.1f}/{args.still:.1f}s  已采 {len(samples)}/{args.points}   ",
                          end="", flush=True)
                    last_print = now
                continue

            h = len(recent) // 2
            du = abs(_med([r[2] for r in recent[h:]]) - _med([r[2] for r in recent[:h]]))
            dv = abs(_med([r[3] for r in recent[h:]]) - _med([r[3] for r in recent[:h]]))
            dx = abs(_med([r[0] for r in recent[h:]]) - _med([r[0] for r in recent[:h]]))
            dy = abs(_med([r[1] for r in recent[h:]]) - _med([r[1] for r in recent[:h]]))
            if du > px_tol or dv > px_tol or dx > arm_tol or dy > arm_tol:
                if now - last_print > 0.4:
                    bad = []
                    if du > px_tol or dv > px_tol:
                        bad.append(f"手在动 {du:.0f}/{dv:.0f}px")
                    if dx > arm_tol or dy > arm_tol:
                        bad.append(f"臂在动 {dx*1000:.1f}/{dy*1000:.1f}mm")
                    print(f"\r{'、'.join(bad)}  已采 {len(samples)}/{args.points}   ",
                          end="", flush=True)
                    last_print = now
                continue

            cx, cy = _med([r[0] for r in recent]), _med([r[1] for r in recent])
            cu, cv = _med([r[2] for r in recent]), _med([r[3] for r in recent])
            if samples:
                d = min(np.hypot(cx - s2[2], cy - s2[3]) for s2 in samples)
                if d < args.min_gap:
                    if now - last_print > 0.4:
                        print(f"\r离上一个点只有 {d*100:.0f}cm，再挪远些（要 >{args.min_gap*100:.0f}cm）"
                              f"  已采 {len(samples)}/{args.points}   ", end="", flush=True)
                        last_print = now
                    continue
            cz = _med([r[5] for r in recent])
            samples.append((cu, cv, cx, cy, cz) + tuple(np.asarray(_R, float).reshape(-1)))
            win.clear()
            print(f"\r采到第 {len(samples)}/{args.points} 点："
                  f"像素 ({cu:.0f},{cv:.0f}) → 基座 ({cx:.3f},{cy:.3f})   ", flush=True)
    except KeyboardInterrupt:
        pass

    if len(samples) < 6:
        print("\n点太少（<6），拟合不了。", flush=True)
        return 1

    # 把原始样本落盘：万一拟合失败，可以直接拿这组数离线复算，不用重采
    try:
        with open("/tmp/calib_samples.json", "w") as _f:
            json.dump({"ns": args.ns, "target": args.target, "stamp": time.strftime("%F %T"),
                       "samples": [list(map(float, s)) for s in samples]}, _f, indent=1)
        print(f"\n样本已存 /tmp/calib_samples.json（{len(samples)} 点），拟合失败时可复算。", flush=True)
    except Exception as _e:
        print(f"样本落盘失败：{_e}", flush=True)

    hold = max(0, min(args.holdout, len(samples) - 6))
    n = len(samples) - hold
    p3 = np.array([[s[2], s[3], s[4]] for s in samples[:n]])
    img = np.array([[s[0], s[1]] for s in samples[:n]])
    span = float(max(np.linalg.norm(a - b) for a in p3 for b in p3))
    if span < 0.12:
        print(f"\n采样点只铺开了 {span*100:0.0f} cm，太集中，映射不可信。")
        print("把末端挪到桌面的不同角落再采一遍（至少铺开 20cm）。")
        return 1

    zs = np.array([s[4] for s in samples])
    z_plane = float(np.median(zs))
    planar = float(np.ptp(zs)) < 0.03

    if planar:
        Hm = geom.homography(img, p3[:, :2])
        model = "homography"

        def _pred(u, v, z=None):
            q = Hm @ np.array([u, v, 1.0])
            return q[:2] / q[2]
    else:
        # 相机 + 靶标安装偏移一起解：靶标贴在夹爪上哪个位置都行，
        # 只要它在采点过程中相对夹爪没动（手腕转动不再破坏标定）。
        p3_all = np.array([[s[2], s[3], s[4]] for s in samples])
        img_all = np.array([[s[0], s[1]] for s in samples])
        rots_all = np.array([np.asarray(s[5:14], float).reshape(3, 3) for s in samples])
        if len(p3_all) >= 8:
            try:
                cam = geom.fit_camera_tool(p3_all, img_all, rots_all)
                n = len(p3_all)
                p3, img = p3_all, img_all
            except RuntimeError as exc:
                spread_z = float(np.ptp(p3_all[:, 2]))
                print(f"\n相机拟合失败：{exc}")
                print(f"采样点高度只差 {spread_z*100:.0f} cm —— "
                      f"{'几乎在一个平面上' if spread_z < 0.04 else '3D 分布仍然太扁'}。")
                print("如果高度差 < 4cm，会退回「平面单应」模型（要求 10 个点在同一高度）。")
                if spread_z >= 0.04:
                    print("→ 退回「不带偏移的相机模型」（靶标必须真的在末端点上，否则残差会大）")
                    p3_t2 = p3_all[-hold:] if hold else None
                    img_t2 = img_all[-hold:] if hold else None
                    cam = geom.fit_camera(p3_all, img_all,
                                          p3_test=p3_t2, uv_test=img_t2)
                    n = len(p3_all)
                    p3, img = p3_all, img_all
                Hm = geom.homography(img_all, p3_all[:, :2])
                cam = {"planar_fallback": True, "H": Hm}
                n = len(p3_all)
                p3, img = p3_all, img_all
        else:
            p3_t = np.array([[s[2], s[3], s[4]] for s in samples[n:]])
            img_t = np.array([[s[0], s[1]] for s in samples[n:]])
            cam = geom.fit_camera(p3, img, p3_test=p3_t if len(p3_t) else None,
                                  uv_test=img_t if len(img_t) else None)
        if cam.get("planar_fallback"):
            model = "homography"
            Hm = cam["H"]

            def _pred(u, v, z=None):
                q = Hm @ np.array([u, v, 1.0])
                return q[:2] / q[2]
        else:
            model = "camera"

            def _pred(u, v, z):
                return geom.pixel_to_plane(cam["K"], cam["R"], cam["t"], (u, v), z)

    # 有靶标偏移时，误差要拿「靶标实际所在点」去比，不能再用末端点
    off = cam.get("offset") if model == "camera" else None
    def _marker_xy(s):
        p_tcp = np.array([s[2], s[3], s[4]])
        if off is None:
            return p_tcp, p_tcp[2]
        R = np.asarray(s[5:14], float).reshape(3, 3)
        p_m = p_tcp + R @ np.asarray(off, float)
        return p_m, p_m[2]

    errs = []
    for s in samples[:n]:
        p_m, zm = _marker_xy(s)
        e = _pred(s[0], s[1], zm) - p_m[:2]
        errs.append(np.hypot(*e) * 1000.0)
    errs = np.array(errs)
    if model == "camera":
        print(f"\n相机：f={cam['f']:.0f} px，中心 ({cam['C'][0]:.3f},{cam['C'][1]:.3f},"
              f"{cam['C'][2]:.3f}) m，重投影 rms {cam['rms_px']:.2f} px")
        if off is not None:
            print(f"靶标安装偏移（工具系）：{off[0]*100:+.1f} {off[1]*100:+.1f} {off[2]*100:+.1f} cm"
                  f"（贴在末端上方 {off[2]*100:.0f} cm —— 已一并解出，不影响使用）")
    print(f"\n模型：{model}（采样高度 {zs.min():.3f}–{zs.max():.3f} m，{n} 点参与拟合）")
    hz = float(np.median(zs))
    if hz > 0.18:
        print(f"★ 注意：采点高度 {hz:.2f} m 偏高（手大约在 0.10 m）。映射要往下外推 "
              f"{hz - 0.10:.2f} m，越远越飘 —— 建议把末端降到 10–15cm 重采。")
    print(f"拟合残差：mean {errs.mean():.2f} mm   max {errs.max():.2f} mm")

    ok = errs.max() <= 8.0
    if model == "camera" and off is not None:
        loo = float(cam.get("loo_px", float("nan")))
        if loo == loo:                       # 不是 nan
            print(f"留一验证：{loo:.2f} px")
            ok = ok and loo <= 4.0
        ok = ok and cam["rms_px"] <= 2.5
    if hold:
        verrs = []
        for s in samples[n:]:
            p_m, zm = _marker_xy(s)
            e = _pred(s[0], s[1], zm) - p_m[:2]
            verrs.append(np.hypot(*e) * 1000.0)
        verrs = np.array(verrs)
        print(f"留出验证（{hold} 点，没参与拟合）：mean {verrs.mean():.2f} mm / "
              f"max {verrs.max():.2f} mm")
        ok = ok and verrs.max() <= 12.0

    xc, yc = _pred(640.0, 360.0, z_plane)
    print(f"画面中心 → 基座 ({xc:.3f},{yc:.3f})   （应在 0.2–0.5 / ±0.3 之间）")
    if not (0.05 < xc < 0.65 and abs(yc) < 0.35):
        print("★ 画面中心落点离谱 —— 多半是标记没贴牢、或采点时手不在末端上方。")
        ok = False

    if not ok:
        print("★ 误差太大，这次结果不可用，没有保存。")
        if model == "camera" and cam["rms_px"] > 4.0:
            print(f"  重投影 rms 已经 {cam['rms_px']:.1f} px —— 说明靶标相对夹爪在动：")
            print("  把靶标用胶带绑死在夹爪上（别只搭在上面），采点时尽量别转 joint6。")
        else:
            print("  检查：靶标是否绑死在夹爪上（不是手腕、不是桌面）、采点时代码和相机是不是同一台机器。")
        return 1

    data = {
        "model": model,
        "cam": ({"f": float(cam["f"]), "K": cam["K"].tolist(), "R": cam["R"].tolist(),
                 "t": [float(v) for v in cam["t"]], "C": [float(v) for v in cam["C"]],
                 "rms_px": float(cam["rms_px"])} if model == "camera" else None),
        "H": [[float(v) for v in row] for row in Hm] if model == "homography" else None,
        "z_plane": z_plane,
        "residual_mm": {"mean": float(errs.mean()), "max": float(errs.max())},
        "samples": [[round(s[0], 1), round(s[1], 1), round(s[2], 4), round(s[3], 4), round(s[4], 4)]
                    for s in samples],
        "ns": args.ns,
        "stamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(os.path.expanduser(args.out), "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=1)
    print(f"已保存 {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

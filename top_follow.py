#!/usr/bin/env python3
"""俯视相机追手：手在桌面上的位置 → 末端 1:1 跟过去。"""
from __future__ import annotations

import argparse
import signal
import csv
import json
import os
import sys
import time

import numpy as np

import follow_hand_v2 as fv
from rig import geom

HERE = os.path.dirname(os.path.abspath(__file__))

MAP = os.path.join(HERE, "calib_base.json")
MAX_DQ = 0.70          # 单个目标步的关节跳变上限（换解支就挡住）
MAX_IK_ERR = 0.004


def safe_ik(arm, p_target, q_seed, max_dq=MAX_DQ, max_err=MAX_IK_ERR):
    q, err = arm.ik(p_target, q_seed)
    if not np.all(np.isfinite(q)):
        return None
    if err > max_err or float(np.max(np.abs(q - q_seed))) > max_dq:
        return None
    if float(np.linalg.norm(arm.fk(q) - p_target)) > max_err:
        return None
    return q


def _on_term(signum, frame):          # noqa: ARG001
    raise KeyboardInterrupt


def _mit(pos: float, vlim: float):
    m = fv.JointMitCmd()
    m.pos = float(pos)
    m.vel = float(vlim)
    return m


class Grip:
    """手掌开合 → 夹爪：拇指+食指捏合=关闭，张开手掌=打开。"""

    def __init__(self, node, ns: str, args) -> None:
        self.pub = node.create_publisher(fv.JointMitCmd, f"{ns}/gripper/cmd/mit", 1)
        self.pos = float(args.grip_open)
        self.a = args
        self.want = None
        self.cnt = 0

    def step(self, pinch: float, dt: float) -> None:
        a = self.a
        if pinch <= 0.0:
            return
        if pinch > a.pinch_open:
            state = "open"
        elif pinch < a.pinch_close:
            state = "close"
        else:
            state = None
        if state is not None:
            self.cnt = self.cnt + 1 if state == self.want else 1
            self.want = state
        else:
            self.cnt = 0
        if self.cnt >= 3 and self.want == "open":
            tgt = float(a.grip_open)
        elif self.cnt >= 3 and self.want == "close":
            tgt = float(a.grip_close)
        else:
            tgt = self.pos
        step = float(a.grip_rate) * max(dt, 1e-3)
        new = self.pos + max(-step, min(step, tgt - self.pos))
        if abs(new - self.pos) > 1e-4:
            self.pos = new
            self.pub.publish(_mit(self.pos, 1.0))


def main() -> int:
    signal.signal(signal.SIGTERM, _on_term)
    ap = argparse.ArgumentParser()
    ap.add_argument("--map", default=MAP)
    ap.add_argument("--ns", default="/rebot_arm_rs")
    ap.add_argument("--port", type=int, default=8899)
    ap.add_argument("--target", default="hand",
                    choices=("hand", "color", "aruco", "blob"),
                    help="跟手还是跟靶标（要和 vision_track 的 --target 一致）")
    ap.add_argument("--z", type=float, default=-1.0, help="跟随高度（米），<0 用标定时的平面高度")
    ap.add_argument("--gain", type=float, default=1.0)
    ap.add_argument("--z-mode", choices=["fixed", "size"], default="size",
                    help="size=用手的像素大小反推高度跟随（默认）；fixed=固定高度")
    ap.add_argument("--no-grip", action="store_true", help="不控制夹爪")
    ap.add_argument("--grip-open", type=float, default=5.0)
    ap.add_argument("--grip-close", type=float, default=0.0)
    ap.add_argument("--grip-rate", type=float, default=8.0, help="夹爪每秒行程（0–5 全程约 0.6 s）")
    ap.add_argument("--pinch-open", type=float, default=0.55, help="大于它=张开手掌")
    ap.add_argument("--pinch-close", type=float, default=0.30, help="小于它=捏合")
    ap.add_argument("--size-d0", type=float, default=1.0,
                    help="免标定高度：手到相机的近似距离（米）")
    ap.add_argument("--z-deadband", type=float, default=0.010, help="高度死区（米）")
    ap.add_argument("--rel-gain", type=float, default=1.0,
                    help="相对模式下手的位移放大倍数（1.0 = 手走 10cm 末端走 10cm）")
    ap.add_argument("--relative", dest="relative", action="store_true", default=True,
                    help="相对模式：第一次看到手的位置当零点（默认开）")
    ap.add_argument("--absolute", dest="relative", action="store_false",
                    help="绝对模式：按标定把像素直接映射成基座坐标")
    ap.add_argument("--rate", type=float, default=60.0)
    ap.add_argument("--horizon", type=float, default=0.10)
    ap.add_argument("--alpha", type=float, default=0.6)
    ap.add_argument("--beta", type=float, default=0.15)
    ap.add_argument("--tau", type=float, default=0.06)
    ap.add_argument("--max-speed", type=float, default=0.25)
    ap.add_argument("--vlim", type=float, default=0.5)
    ap.add_argument("--pinch-pause", type=float, default=0.0)   # 0=不暂停；张开的手 pinch 本来就大，别误判
    ap.add_argument("--lost-hold", type=float, default=0.30)
    ap.add_argument("--box", type=float, nargs=4, default=[0.18, 0.46, -0.15, 0.15],
                    metavar=("XMIN", "XMAX", "YMIN", "YMAX"))
    ap.add_argument("--deadzone", type=float, default=0.004, help="小于这个距离不动（米）")
    ap.add_argument("--hand-z", type=float, default=None,
                    help="手离基座平面 z=0 的高度（米），不填按 0.06")
    ap.add_argument("--probe", action="store_true", help="只打印手的位置，不驱动")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--log", default=None)
    args = ap.parse_args()

    path = os.path.expanduser(args.map)
    if args.map.lower() == "none" or not os.path.isfile(path):
        if not os.path.isfile(path):
            print(f"没找到 {path}，改用「免标定」模式（近似线性映射，靠相对零点跟手）", flush=True)
        d = {"model": "none", "cam": None, "H": None, "z_plane": 0.12,
             "residual_mm": {"mean": 0.0, "max": 0.0}}
        args.relative = True
        # 没有相机模型时仍可用 size 模式：只是退化成「手变大→更近→更低」的近似
    else:
        d = json.load(open(path, encoding="utf-8"))
    model = d.get("model", "homography")
    Cm = d.get("cam") or {}
    if not Cm.get("K") and d.get("H") and d.get("samples"):
        # 只有平面单应时：用它 + 标定点反解相机，这样能按手的真实高度投影，
        # 而不是把手的像素硬投到标定平面（斜俯视时那会有十几厘米偏差）。
        try:
            Cm = geom.camera_from_homography(
                np.array(d["H"], float).reshape(3, 3), float(d["z_plane"]),
                d["samples"], 640.0, 360.0)   # 俯视相机 1280×720 的主点
            print(f"相机反解：f={Cm['f']:.0f}  高度 {Cm['C'][2]:.2f} m  "
                  f"（平面还原 {Cm['rms_mm']:.2f} mm）", flush=True)
        except Exception as exc:                     # noqa: BLE001
            print(f"相机反解失败（{exc}），退回单应直接投影", flush=True)
            Cm = {}
    Kc = np.array(Cm["K"], float) if Cm.get("K") is not None else None
    Rc = np.array(Cm["R"], float) if Cm.get("R") is not None else None
    Tc = np.array(Cm["t"], float) if Cm.get("t") is not None else None
    Hm = np.array(d["H"], float) if d.get("H") else None
    if args.hand_z is not None:
        args.z = args.hand_z
    if args.z < 0:
        # 投影模型下 z_plane 只是标定时末端的中位高度，跟手的高度无关；手面约 6 cm
        args.z = 0.10 if Kc is not None else float(d.get("z_plane", 0.12))
    print(f"标定：{model}，残差 mean {d['residual_mm']['mean']:.2f} mm / "
          f"max {d['residual_mm']['max']:.2f} mm，跟随高度 {args.z:.3f} m", flush=True)

    def base_of(u: float, v: float) -> np.ndarray:
        if d.get("model") == "none":
            # 免标定：只是把画面线性铺到工作区，方向/比例按俯视相机的常见安装估的，
            # 相对零点模式只用它的「位移方向 × 比例」，用 --rel-gain 就能微调幅度。
            return np.array([0.30 + (v - 360.0) * 0.00080,
                             0.0 + (u - 640.0) * 0.00100])
        if Kc is not None:
            return geom.pixel_to_plane(Kc, Rc, Tc, (u, v), args.z)
        q = Hm @ np.array([u, v, 1.0])
        if abs(q[2]) < 1e-9:
            return np.array([np.nan, np.nan])
        return q[:2] / q[2]

    def ray_of(u: float, v: float):
        if Kc is None:
            return None
        f = Kc[0, 0]
        p = np.array([(u - Kc[0, 2]) / f, (v - Kc[1, 2]) / f, 1.0])
        d = Rc.T @ p
        n = float(np.linalg.norm(d))
        return d / n if n > 1e-9 else None

    reader = fv.HandReader(args.port, args.target)
    reader.start()

    if args.probe:
        print("probe：手在画面里动，打印它对应的基座坐标。Ctrl-C 退出。", flush=True)
        try:
            while True:
                h, st = reader.get()
                if h:
                    b = base_of(h["u"], h["v"])
                    print(f"\r手 px({h['u']:6.1f},{h['v']:6.1f}) → 基座 ({b[0]:.3f},{b[1]:.3f})", end="", flush=True)
                time.sleep(0.1)
        except KeyboardInterrupt:
            return 0

    arm = fv.Arm()
    fv.rclpy.init()
    node = fv.Follow(args.ns)
    t0 = time.time()
    while node.q is None and time.time() - t0 < 6.0:
        fv.rclpy.spin_once(node, timeout_sec=0.1)
    if node.q is None:
        print(f"没收到 {args.ns}/joint_states —— 控制器没在跑？", flush=True)
        return 1
    if not node.enabled:
        print("★ 机械臂没使能。去网页点「使能」，我在这儿等（最多 3 分钟）…", flush=True)
        t0 = time.time()
        while time.time() - t0 < 180.0 and not node.enabled:
            fv.rclpy.spin_once(node, timeout_sec=0.1)
        if not node.enabled:
            print("★ 还是没使能，先退出。网页上点「使能」后再跑一次。", flush=True)
            return 1
        print("★ 已使能，继续。", flush=True)

    grip = None if args.no_grip else Grip(node, args.ns, args)
    q_cmd = node.q.copy()
    p_cmd = arm.fk(q_cmd)
    print(f"当前末端 {np.round(p_cmd, 3)}", flush=True)

    if not args.dry_run and sys.stdin and sys.stdin.isatty() and "sim" not in args.ns:
        input("手放到桌面上方，按 Enter 开始（Ctrl-C 只停脚本，臂会原地保持）…")

    ab = fv.AB(args.alpha, args.beta)
    period = 1.0 / args.rate
    h_stamp = 0.0
    sm = None
    n_fresh = 0
    n_clip = 0
    ref_hand = None
    ref_arm = None
    n_pkt = 0
    n_ok = 0
    t_stat = 0.0
    z0 = None          # 零点处的手高
    s0 = None          # 零点处的手掌像素尺度
    t_ray0 = None      # 零点处沿视线的距离
    z_smooth = None
    sizes = []
    t_warn = 0.0
    n_skip = 0
    t_lost = 0.0
    logf = None
    w = None
    if args.log:
        logf = open(os.path.expanduser(args.log), "w", newline="", encoding="utf-8")
        w = csv.writer(logf)
        w.writerow(["t", "tx", "ty", "tz", "ax", "ay", "az", "err_mm", "u", "v", "fresh"])

    print("跟随中。手往哪走，末端就往哪走；捏合（拇指+食指）暂停。", flush=True)
    try:
        while True:
            fv.rclpy.spin_once(node, timeout_sec=0.0)
            now = time.time()
            h, st = reader.get()
            if h is not None:
                n_pkt += 1
            fresh = 0
            if h is not None and now - st <= args.lost_hold:
                fresh = 1
                if st > h_stamp + 1e-9:
                    h_stamp = st
                    b_raw = base_of(h["u"], h["v"])
                    if np.all(np.isfinite(b_raw)) and abs(b_raw[0]) < 5 and abs(b_raw[1]) < 5:
                        ab.update(b_raw, st)
                if args.pinch_pause > 0 and h.get("pinch", 0.0) > args.pinch_pause:
                    fresh = 0
            else:
                t_lost = t_lost or now

            if now - t_stat > 2.0:
                t_stat = now
                print(f"[跟随] 收到 {n_pkt} 包（有效 {n_ok}） 目标 {np.round(p_cmd,3)}", flush=True)

            if fresh:
                n_ok += 1
                if t_lost and now - t_lost > 1.0:
                    # 丢手太久再回来：重新取零点，避免一下子跳很远
                    ref_hand = None
                    print("[跟随] 重新取零点", flush=True)
                t_lost = 0.0
                if ab.x is None:          # 还没量到第一帧真值，先别取零点
                    fresh = 0
                    b = None
                else:
                    b = ab.predict(args.horizon + max(now - h_stamp, 0.0))
                if b is None:
                    fresh = 0
                else:
                    if ref_hand is None:
                        ref_hand = b.copy()
                        ref_arm = arm.fk(node.q).copy()
                        s0 = float(h.get("size", 0.0) or 0.0)
                        dd0 = ray_of(h["u"], h["v"])
                        if s0 > 1e-4 and dd0 is not None and Tc is not None and abs(dd0[2]) > 1e-3:
                            t_ray0 = float((args.z - Tc[2]) / dd0[2])
                            z0 = float(args.z)
                            sizes.clear()
                            sizes.append(s0)
                            z_smooth = z0
                        else:
                            t_ray0 = None
                            z0 = None
                        print(f"[跟随] 零点：手 {np.round(ref_hand,3)}，末端 "
                              f"{np.round(ref_arm,3)}", flush=True)
                    if np.linalg.norm(b - ref_hand) > 0.6:
                        print("★ 手跳变过大，忽略这一帧", flush=True)
                        fresh = 0
                    gx = ref_arm[0] + args.rel_gain * (b[0] - ref_hand[0])
                    gy = ref_arm[1] + args.rel_gain * (b[1] - ref_hand[1])
                n_fresh += 1
                if not (args.box[0] <= gx <= args.box[1] and args.box[2] <= gy <= args.box[3]):
                    n_clip += 1
                if n_fresh > 30 and n_clip / n_fresh > 0.8 and now - t_warn > 2.0:
                    t_warn = now
                    print(f"★ 目标 {(n_clip*100)//n_fresh}% 被裁在工作区边界内 —— "
                          f"标定平面 z={float(d.get('z_plane', 0)):.2f} m 和手的高度差太多，"
                          f"或相机被挪过。重跑标定（末端保持在 10–15cm）会立刻恢复。", flush=True)
                gx = min(max(gx, args.box[0]), args.box[1])
                gy = min(max(gy, args.box[2]), args.box[3])
                if args.relative and ref_arm is not None:
                    gz = float(np.clip(ref_arm[2], 0.05, 0.35))
                else:
                    gz = args.z
                if args.z_mode == "size" and z0 is not None and t_ray0 is not None:
                    sz = float(h.get("size", 0.0) or 0.0)
                    if sz > 1e-4:
                        sizes.append(sz)
                        if len(sizes) > 5:
                            sizes.pop(0)
                        sm_sz = float(np.median(sizes))
                        dd = ray_of(h["u"], h["v"])
                        if dd is not None:
                            t_now = t_ray0 * (s0 / sm_sz)
                            z_now = float(Tc[2] + t_now * dd[2]) if Tc is not None else z0
                            z_smooth = z_now if z_smooth is None else (0.85 * z_smooth + 0.15 * z_now)
                            dz = z_smooth - z0
                            if abs(dz) > args.z_deadband:
                                gz = float(np.clip(gz + dz, 0.04, 0.45))
                if grip is not None:
                    grip.step(float(h.get("pinch", 0.0) or 0.0), period)
                if args.z_mode == "size" and t_ray0 is None:
                    sz = float(h.get("size", 0.0) or 0.0)
                    if s0 > 1e-4 and sz > 1e-4:
                        sizes.append(sz)
                        if len(sizes) > 5:
                            sizes.pop(0)
                        sm_sz = float(np.median(sizes))
                        dz = -float(args.size_d0) * (s0 / sm_sz - 1.0)   # 手变小=离相机远=更低
                        if abs(dz) > args.z_deadband:
                            gz = float(np.clip(gz + float(np.clip(dz, -0.15, 0.15)), 0.04, 0.45))
                p_goal = np.array([gx, gy, gz])
                if sm is None:
                    sm = p_goal.copy()
                else:
                    k = float(1.0 - np.exp(-period / max(args.tau, 1e-3)))
                    sm = sm + (p_goal - sm) * k
                dp = sm - p_cmd
                dist = float(np.linalg.norm(dp))
                if dist > 1e-6:
                    step = min(dist, args.max_speed * period)
                    p_new = p_cmd + dp / dist * step
                    if np.linalg.norm(p_new - p_cmd) > 1e-6:
                        q_new = safe_ik(arm, p_new, q_cmd)
                        if q_new is None:
                            n_skip += 1
                        else:
                            dq = float(np.max(np.abs(q_new - q_cmd)))
                            if dq > 0.06:
                                q_new = q_cmd + (q_new - q_cmd) * (0.06 / dq)
                            if not args.dry_run:
                                node.send(q_new, args.vlim)
                            q_cmd, p_cmd = q_new, p_new
            if w is not None:
                cur = arm.fk(q_cmd)
                e = float(np.linalg.norm(cur - np.array([p_cmd[0], p_cmd[1], args.z]))) * 1000.0
                w.writerow([round(now - t0, 3), *np.round(p_cmd, 4), *np.round(cur, 4),
                            round(e, 2), (h or {}).get("u", -1), (h or {}).get("v", -1), fresh])
            time.sleep(period)
    except KeyboardInterrupt:
        print("\n停。臂保持当前姿态（控制器还通着电）。", flush=True)
    except Exception as exc:                     # noqa: BLE001
        # 那不是故障，不该给用户一屏 traceback。
        if "context is not valid" in str(exc) or "rcl_shutdown" in str(exc):
            print("\n停。", flush=True)
        else:
            raise
    finally:
        if logf:
            logf.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

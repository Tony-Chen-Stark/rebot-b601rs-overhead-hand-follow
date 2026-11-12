#!/usr/bin/env python3
"""点一下画面里的靶标，把它的色相存下来，供 vision_track.py --hue auto 使用。"""
import argparse
import json
import os
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))

OUT = os.path.join(HERE, "color.json")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", default=0)
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--center-test", action="store_true",
                    help="不开窗口，直接取画面正中的颜色（自检用）")
    args = ap.parse_args()

    idx = 0 if str(args.camera) in ("0", "auto", "None") else int(args.camera)
    cap = cv2.VideoCapture(idx, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if not cap.isOpened():
        print(f"打不开相机 {idx}", file=sys.stderr)
        return 1

    def save(hsv, x, y):
        y0, y1 = max(0, y - 8), min(hsv.shape[0], y + 9)
        x0, x1 = max(0, x - 8), min(hsv.shape[1], x + 9)
        patch = hsv[y0:y1, x0:x1].reshape(-1, 3)
        if len(patch) == 0:
            return
        s = patch[:, 1]
        keep = patch[np.argsort(s)[-max(4, len(s) // 5):]]     # 取饱和度最高的那一小撮
        hue_cv, sat, val = np.median(keep, axis=0)
        out = {
            "hue": float(hue_cv) * 2.0,                        # 度
            "sat_min": float(max(60.0, sat * 0.6)),
            "val_min": float(max(40.0, val * 0.4)),
            "click_px": [int(x), int(y)],
            "sample_rgb": [int(v) for v in patch[len(patch) // 2][::-1]],
        }
        with open(args.out, "w") as f:
            json.dump(out, f, indent=2)
        print(f"✓ 色相 {out['hue']:.0f}°（OpenCV H={out['hue']/2:.0f}）"
              f"  S≥{out['sat_min']:.0f} V≥{out['val_min']:.0f}  → {args.out}", flush=True)

    ok, frame = cap.read()
    if not ok:
        print("读不到画面", file=sys.stderr)
        return 1

    if args.center_test:
        save(cv2.cvtColor(frame, cv2.COLOR_BGR2HSV), frame.shape[1] // 2, frame.shape[0] // 2)
        cap.release()
        return 0

    win = "pick-color"
    state = {"frame": frame}
    cv2.imshow(win, frame)          # 先真的建出窗口，否则 Qt 后端 setMouseCallback 会报 NULL handler

    def _on_mouse(ev, x, y, flags, _param):
        if ev == cv2.EVENT_LBUTTONDOWN and state["frame"] is not None:
            try:
                save(cv2.cvtColor(state["frame"], cv2.COLOR_BGR2HSV), x, y)
            except Exception as exc:                     # 回调里抛异常会刷屏，压住
                print(f"取色失败：{exc}", flush=True)

    cv2.setMouseCallback(win, _on_mouse)
    print("点一下靶标（可多次点击，q 退出）", flush=True)
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        state["frame"] = frame
        cv2.imshow(win, frame)
        if (cv2.waitKey(1) & 0xFF) in (ord("q"), 27):
            break

    cap.release()
    cv2.destroyAllWindows()
    if not os.path.exists(args.out):
        print("没有点到任何位置，未保存", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
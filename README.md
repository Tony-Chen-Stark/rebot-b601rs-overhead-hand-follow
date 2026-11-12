# 俯视单目相机 · 手部跟随（reBot B601 RS）

用一台普通 USB 相机从斜上方看桌面，手怎么动，机械臂末端就怎么动。手停下它也停下，手离开画面就原地保持。

演示视频：`media/fushi1.mp4`、`media/fushi2.mp4`

---

## 硬件

| 部件 | 说明 |
|---|---|
| 机械臂 | reBot Arm B601 **RS**（灵足 Robstride 电机，24V） |
| 相机 | 普通 UVC 相机，1280×720，装在桌面上方约 1.2 m，斜俯视 |
| 主机 | Ubuntu 22.04 / 24.04，ROS 2 Jazzy，CAN 1 Mbit |

软件依赖：官方 `ReBotArmController_ROS2`（提供 `/rebot_arm_rs` 命名空间）、`ReBotArm_control_py`（pinocchio 运动学）、MediaPipe（手部关键点）、OpenCV。

## 方法

1. **手部位姿**：MediaPipe HandLandmarker 取 21 个关键点，用掌心点作为手的像素位置，用掌宽估计手的大小。
2. **像素 → 基座**：默认「相对零点」——第一次看到手时同时记录手的映射位置和机械臂当时末端位置，之后只跟手的**位移**。这样起点随意、不依赖标定。  
   可选「1:1 米制」模式：用机械臂自己量出的点做标定（平面单应，或相机位姿 + 靶标安装偏移），把像素映射成基座米制坐标。
3. **平滑与预测**：alpha-beta 滤波估计速度，按视觉链路延迟做位置预测，再一阶平滑 + 限速，避免抖动放大成关节抖动。
4. **逆运动学**：阻尼最小二乘（位置优先，姿态随动），每拍从实测关节角重新求解，带解支跳变检测与工作区裁剪。
5. **输出**：60 Hz 向 `/{ns}/joints/jointN/cmd/mit` 发位置指令。
6. **夹爪**（可选）：拇指与食指捏合 → 关闭，张开手掌 → 打开，带限速与迟滞。

## 参考

- [Seeed-Projects/reBot-DevArm](https://github.com/Seeed-Projects/reBot-DevArm) — 官方硬件与教程
- [Seeed-Projects/ReBotArm_control_py](https://github.com/Seeed-Projects/ReBotArm_control_py) — Python SDK（pinocchio 模型、IK/FK）
- 官方 `web_mujoco/src/tcp-ik.js` — TCP 拖拽用的阻尼最小二乘 IK，本项目的解算参数与之对齐
- [google-ai-edge/mediapipe](https://github.com/google-ai-edge/mediapipe) — 手部关键点
- AnyTeleop, arXiv:[2307.04577](https://arxiv.org/abs/2307.04577) — 手部关键点 → 机械臂末端的位置重定向思路
- Chaumette & Hutchinson, *Visual Servo Control, Part I*, IEEE RAM 2006 — 视觉伺服基础
- Zhang Z., *A Flexible New Technique for Camera Calibration*, PAMI 2000 — 相机标定
- Hartley & Zisserman, *Multiple View Geometry*, ch.4 / ch.13 — 归一化 DLT 与平面单应

## 运行

```bash
# 1) 机械臂上电，等 5 秒；确认 can0 已 UP（1 Mbit）

# 2) 起控制器
cd ~/ReBot_Arm_DigitalTwin_RS
REBOTARM_RS_HARDWARE_CONFIRM=I_UNDERSTAND_RS_WILL_MOVE ./rebotarm start rs

# 3) 起跟随
bash top.sh            # 真机
bash top.sh --sim      # MuJoCo 仿真
```

环境变量（可选）：`REBOT_WS` 指向官方工作区，`HAND_VISION_PY` 指定带 mediapipe 的解释器。

### 常用参数

| 目的 | 参数 |
|---|---|
| 幅度放大/缩小 | `--rel-gain 1.6` / `--rel-gain 0.7` |
| 动作更稳 | `--max-speed 0.10` |
| 上下更灵敏 | `--z-deadband 0.005` |
| 夹爪速度 | `--grip-rate 15` |
| 不要夹爪 | `--no-grip` |
| 1:1 米制跟随 | `--recal` 先标定一次 |

### 停止

跟随终端 `Ctrl-C` 后：

```bash
ros2 service call /rebot_arm_rs/disable std_srvs/srv/Trigger
./rebotarm stop
```

**不要对控制器终端按 Ctrl-C**，那会直接切断电机使能、机械臂掉下来。

## 文件

| 文件 | 作用 |
|---|---|
| `top.sh` | 启动视觉进程 + 跟随进程 |
| `top_follow.py` | 跟随主程序 |
| `vision_track.py` | 视觉：手部关键点 / 彩色靶标 → UDP |
| `follow_hand_v2.py` | ROS 接口（关节状态、指令、夹爪） |
| `robot_map.py` | 标定：手动挪机械臂采点，拟合映射 |
| `rig/geom.py` | 单应、相机反解、像素射线 |
| `pick_color.py` | 用彩色靶标时点选颜色 |
| `models/` | MediaPipe 手部模型 |

## 许可

MIT
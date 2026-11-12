#!/usr/bin/env bash
# 俯视相机追手：一条命令
#   bash "$0"              # 真机（第一次会自动先标定）
#   bash "$0" --recal      # 重新用机械臂标定映射（靶标）
#   蓝色靶标：HUE=224 bash "$0" --recal   （度：红0 绿120 蓝224）
#   bash "$0" --recal --hand  # 用悬在末端上方的手掌当靶标
#   bash "$0" --sim        # 仿真命名空间
#   bash "$0" --color      # 改用彩色靶标（默认是跟手）
#   TARGET=blob  bash "$0" # 白点靶标（桌面别是白的）
#   TARGET=aruco bash "$0" # 打印的 ArUco 靶标
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="${HAND_VISION_PY:-$HOME/hand_vision_venv/bin/python}"
WS="${REBOT_WS:-$HOME/ReBot_Arm_DigitalTwin_RS}"
ENVSH="$WS/scripts/rs_env.sh"
LOG="$WS/rebotarm_ros2_RS/.venv/bin/python"
PORT=8899
NS=/rebot_arm_rs
RECAL=0
HAND=""
SIMARG=""
TARGET="${TARGET:-hand}"
ARGS=()

for a in "$@"; do
  case "$a" in
    --sim) NS=/rebot_arm_rs_sim; SIMARG=" --sim" ;;
    --recal) RECAL=1 ;;
    --hand) HAND=" --target hand"; TARGET=hand ;;
    --color) TARGET=color ;;
    *) ARGS+=("$a") ;;
  esac
done

die() {
  echo
  echo "✗ $*"
  echo "  日志：/tmp/vision_top.log"
  if [ -t 0 ]; then read -r -p "按回车关闭…" _ || true; fi
  exit 1
}

for f in vision_track.py robot_map.py top_follow.py follow_hand_v2.py models/hand_landmarker.task; do
  [ -f "$ROOT/$f" ] || die "缺文件 $ROOT/$f"
done
[ -x "$VENV" ] || die "缺视觉环境 $VENV"
[ -x "$LOG" ] || die "缺 ROS 环境 $LOG"

# 视觉进程不继承 ROS 的 PYTHONPATH，避免两套依赖打架
pkill -f "vision_track.py .*127.0.0.1:$PORT" 2>/dev/null
sleep 0.5
echo "视觉：target=$TARGET  hue=${HUE:-auto}"
env -u PYTHONPATH -u VIRTUAL_ENV -u ROS_DISTRO -u AMENT_PREFIX_PATH \
    "$VENV" "$ROOT/vision_track.py" --camera 0 --udp "127.0.0.1:$PORT" \
    --target "$TARGET" \
    --hue "${HUE:-auto}" --hue-tol "${HUE_TOL:-18}" \
    --sat-min "${SAT_MIN:-120}" --val-min "${VAL_MIN:-60}" \
    --preview > /tmp/vision_top.log 2>&1 &
VPID=$!
trap 'kill "$VPID" 2>/dev/null' EXIT INT TERM

for _ in $(seq 1 40); do
  sleep 0.25
  kill -0 "$VPID" 2>/dev/null || { echo; tail -12 /tmp/vision_top.log; die "视觉进程退出了"; }
  grep -q "UDP ->" /tmp/vision_top.log 2>/dev/null && break
done
grep -q "UDP ->" /tmp/vision_top.log || { echo; tail -12 /tmp/vision_top.log; die "相机没出帧"; }

# ROS 侧环境（venv + jazzy + 工作区）
# shellcheck disable=SC1090
set +u
source "$ENVSH" >/dev/null 2>&1 || die "加载 $ENVSH 失败"
set -u
"$LOG" -c "import rclpy, pinocchio" 2>/dev/null || die "rclpy/pinocchio 不可用，先把工作区编译好"

if [ ! -f "$ROOT/calib_base.json" ] && [ "$RECAL" != "1" ]; then
  echo "没有 calib_base.json —— 用免标定模式（相对零点跟手，不需要标定）。"
  echo "想要 1:1 的绝对位移，再跑一次：bash "$0" --recal"
  ARGS=(--map none ${ARGS[@]+"${ARGS[@]}"})
fi

if [ "$RECAL" = "1" ]; then
  echo
  if [ -n "$HAND" ]; then
    echo "先标定：手掌摊平悬在夹爪正上方 1–2cm，用网页把末端挪到 8 个位置（离桌面约 10cm），"
    echo "每到一个位置手掌停住 1 秒；预览窗口里手掌中心会被绿点标出来。"
  else
    echo "先标定：靶标贴在夹爪朝上的面，用网页把末端挪到 8 个位置（离桌面约 10cm），"
    echo "每到一个点停稳 1 秒；预览窗口里靶标被圈住 = 相机看到了。"
  fi
  echo
  "$LOG" "$ROOT/robot_map.py" --ns "$NS" $HAND || die "标定没完成"
  echo
  echo "标定好了，现在跑："
  echo "  bash "$0"${SIMARG}"
  exit 0
fi

"$LOG" "$ROOT/top_follow.py" --ns "$NS" --target "$TARGET" ${ARGS[@]+"${ARGS[@]}"} &
FPID=$!
trap 'kill "$FPID" "$VPID" 2>/dev/null' EXIT INT TERM
wait "$FPID"

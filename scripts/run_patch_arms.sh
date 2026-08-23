#!/bin/bash
# Patch runs: complete the task blocks the killed fleet didn't finish.
# Original fleet (outputs/rebuttal_final) died at 06:27; per-arm completed episodes:
#   arm0 185 -> tasks 0-8 complete, rerun task 9
#   arm1 140 -> tasks 0-6 complete, rerun 7,8,9
#   arm2 184 -> rerun 9
#   arm3 150 -> tasks 0-6 complete, rerun 7,8,9
#   arm4 188 -> rerun 9
#   arm5 200 -> DONE
#   arm6 178 -> tasks 0-7 complete, rerun 8,9
#   arm7 162 -> rerun 8,9
# Partial-task episodes in the original dirs are EXCLUDED at merge time; each rerun
# task gets its full 20 episodes here (same deterministic init states, 0-19 per task).
set -u
BASE=/mnt/nas/ishneet/vls-rebuttal
source "$BASE/venv/bin/activate"
set -a; source "$HOME/brain/ishneet/.env"; set +a
export GOOGLE_API_KEY="${GOOGLE_API_KEY:-$GEMINI_API_KEY}"
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl TOKENIZERS_PARALLELISM=false
export HF_HOME=$BASE/hf_home TORCH_HOME=$BASE/torch_home

REPO=$BASE/code
OUT=$REPO/outputs/rebuttal_patch
mkdir -p "$OUT"

COMMON="backend=libero backend.libero.suite_name=libero_goal_task \
  main.use_guidance=true main.sample_batch_size=20 \
  policy.pi05.num_inference_steps=50 main.sigmoid_x0=0.8 main.vlm_query_limit=10"

launch() {  # name gpu tasks episodes extra...
  local NAME=$1 GPU=$2 TASKS=$3 EPS=$4; shift 4
  echo "[GPU $GPU] $NAME tasks=$TASKS eps=$EPS $*"
  CUDA_VISIBLE_DEVICES=$GPU setsid nohup python "$REPO/main.py" $COMMON \
    "backend.libero.task_ids_filter=[$TASKS]" main.episode_num=$EPS "$@" \
    hydra.run.dir="$OUT/$NAME" > "$OUT/$NAME.log" 2>&1 < /dev/null &
  echo $! > "$OUT/$NAME.pid"
  sleep "${STAGGER:-90}"
}

launch arm0_full_vls       0 "9"     20
launch arm1_best_of_b      1 "7,8,9" 60 main.steering_mode=best_of_b
launch arm2_reward_only    2 "9"     20 main.guide_scale=0.0
launch arm3_generic_prompt 3 "7,8,9" 60 perception.vlm_agent.patterns_file=none
launch arm4_const_lambda   4 "9"     20 main.sigmoid_k=0.0 main.guide_scale=160.0
launch arm6_kp_noise_2cm   5 "8,9"   40 main.keypoint_noise_sigma=0.02
launch arm7_kp_noise_5cm   6 "8,9"   40 main.keypoint_noise_sigma=0.05

echo "All patch arms launched (setsid-detached)."

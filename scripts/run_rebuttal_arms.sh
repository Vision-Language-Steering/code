#!/bin/bash
# ============================================================================
# CoRL 2026 #930 rebuttal — 8-arm ablation slate on LIBERO-PRO task perturbation
#
# One arm per GPU, all on libero_goal_task (fixed init states per episode index,
# so arms are directly comparable). ALL arms share the same VLM config
# (gpt-5.6-terra reward gen, gemini-robotics-er-2-preview grounding/stage), so the
# figure is internally consistent and anchored on arm 0's full-VLS re-run.
#
# Usage:
#   ./scripts/run_rebuttal_arms.sh            # full runs (200 eps/arm = 20/task)
#   SMOKE=1 ./scripts/run_rebuttal_arms.sh    # smoke test (10 eps/arm = 1/task)
#   ARMS="0 1 3" ./scripts/run_rebuttal_arms.sh   # run a subset of arms
#
# Protocol pinned to the paper (Table 4, LIBERO-PRO column + Appendix B.3):
#   50 denoising steps, sigmoid offset 0.8, stage-query cap 10, lambda_max 80,
#   alpha 25, RBF 20, thresholds 0.8/0.6, B=20, chunk horizon 10.
# The shipped configs differ (10 steps / 0.75 / 50) — overridden in COMMON below.
# ============================================================================
set -u

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_FILE="${ENV_FILE:-$HOME/brain/ishneet/.env}"
SUITE="${SUITE:-libero_goal_task}"
EPISODES="${EPISODES:-200}"          # 20 per task x 10 tasks (Table 1 protocol)
[ "${SMOKE:-0}" = "1" ] && EPISODES=10
OUTROOT="${OUTROOT:-$REPO/outputs/rebuttal_$(date +%m%d_%H%M)}"
STAGGER="${STAGGER:-120}"            # seconds between arm launches (API rate limits)

# --- Environment ---
VENV="${VENV:-/mnt/nas/ishneet/vls-rebuttal/venv}"
if [ -f "$VENV/bin/activate" ]; then
  source "$VENV/bin/activate"
fi
if [ -f "$ENV_FILE" ]; then
  set -a; source "$ENV_FILE"; set +a
fi
export GOOGLE_API_KEY="${GOOGLE_API_KEY:-${GEMINI_API_KEY:-}}"
export AZURE_OPENAI_BASE_URL="${AZURE_OPENAI_BASE_URL:-https://ishi.cognitiveservices.azure.com/openai/v1/}"
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export TOKENIZERS_PARALLELISM=false
export HF_HOME="${HF_HOME:-/mnt/nas/ishneet/vls-rebuttal/hf_home}"
export TORCH_HOME="${TORCH_HOME:-/mnt/nas/ishneet/vls-rebuttal/torch_home}"

for var in OPENAI_API_KEY GOOGLE_API_KEY; do
  if [ -z "${!var:-}" ]; then echo "FATAL: $var is not set"; exit 1; fi
done

COMMON="backend=libero backend.libero.suite_name=$SUITE \
  main.use_guidance=true main.sample_batch_size=20 main.episode_num=$EPISODES \
  policy.pi05.num_inference_steps=50 main.sigmoid_x0=0.8 main.vlm_query_limit=10"

# --- Arm definitions: "name|extra hydra overrides" ---
declare -a ARM_DEFS=(
  "arm0_full_vls|"
  "arm1_best_of_b|main.steering_mode=best_of_b"
  "arm2_reward_only|main.guide_scale=0.0"
  "arm3_generic_prompt|perception.vlm_agent.patterns_file=none"
  "arm4_const_lambda|main.sigmoid_k=0.0 main.guide_scale=160.0"
  "arm5_det_stage|main.stage_switch_mode=gripper"
  "arm6_kp_noise_2cm|main.keypoint_noise_sigma=0.02"
  "arm7_kp_noise_5cm|main.keypoint_noise_sigma=0.05"
)
# arm2: sigma_k unchanged; guide_scale=0 keeps RBF diversity + FKD resampling but
#       injects zero reward gradient = "reward-only steering" (paper's w/o-grad on LIBERO)
# arm4: sigmoid_k=0 -> constant strength 0.5; guide_scale doubled (80->160) so the
#       effective constant scale equals the adaptive arm's lambda_max

ARMS="${ARMS:-0 1 2 3 4 5 6 7}"
mkdir -p "$OUTROOT"
echo "Output root: $OUTROOT   suite=$SUITE  episodes/arm=$EPISODES"

for i in $ARMS; do
  DEF="${ARM_DEFS[$i]}"
  NAME="${DEF%%|*}"
  EXTRA="${DEF#*|}"
  LOG="$OUTROOT/$NAME.log"
  echo "[GPU $i] launching $NAME  ${EXTRA:+overrides: $EXTRA}"
  CUDA_VISIBLE_DEVICES=$i nohup python "$REPO/main.py" \
      $COMMON $EXTRA \
      hydra.run.dir="$OUTROOT/$NAME" \
      > "$LOG" 2>&1 &
  echo "$!" > "$OUTROOT/$NAME.pid"
  sleep "$STAGGER"
done

echo ""
echo "All arms launched. Monitor with:"
echo "  tail -f $OUTROOT/arm0_full_vls.log"
echo "  grep -h 'success rate' $OUTROOT/*/results.txt"

# WAIT=1: block until every arm exits (lets a supervisor get notified on completion)
if [ "${WAIT:-0}" = "1" ]; then
  echo "Waiting for all arms to finish..."
  FAIL=0
  for i in $ARMS; do
    DEF="${ARM_DEFS[$i]}"; NAME="${DEF%%|*}"
    PID=$(cat "$OUTROOT/$NAME.pid")
    if wait "$PID"; then
      echo "[done] $NAME exited OK"
    else
      echo "[done] $NAME exited NONZERO"
      FAIL=1
    fi
  done
  echo "ALL ARMS FINISHED (fail_flag=$FAIL, outputs in $OUTROOT)"
  exit $FAIL
fi

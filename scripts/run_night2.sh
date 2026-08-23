#!/bin/bash
# Night 2: E16 frequency slice (GPUs 1-4) + E11 end-to-end replicates (GPUs 5-7).
# All arms = full-VLS config except the single FKD-frequency override. Config-only.
set -u
BASE=/mnt/nas/ishneet/vls-rebuttal
source "$BASE/venv/bin/activate"
set -a; source "$HOME/brain/ishneet/.env"; set +a
export GOOGLE_API_KEY="${GOOGLE_API_KEY:-$GEMINI_API_KEY}"
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl TOKENIZERS_PARALLELISM=false
export HF_HOME=$BASE/hf_home TORCH_HOME=$BASE/torch_home

REPO=$BASE/code
OUT=$REPO/outputs/night2
mkdir -p "$OUT"

COMMON="backend=libero backend.libero.suite_name=libero_goal_task \
  main.use_guidance=true main.sample_batch_size=20 main.episode_num=200 \
  policy.pi05.num_inference_steps=50 main.sigmoid_x0=0.8 main.vlm_query_limit=10"

launch() {  # name gpu extra...
  local NAME=$1 GPU=$2; shift 2
  echo "[GPU $GPU] $NAME $*"
  CUDA_VISIBLE_DEVICES=$GPU setsid nohup python "$REPO/main.py" $COMMON "$@" \
    hydra.run.dir="$OUT/$NAME" > "$OUT/$NAME.log" 2>&1 < /dev/null &
  echo $! > "$OUT/$NAME.pid"
  sleep "${STAGGER:-90}"
}

# E16 frequency slice (anchors already measured: freq=5 -> 32.5, terminal-only best-of-B -> 11.5)
launch e16_freq2   1 main.fkd.resample_frequency=2
launch e16_freq10  2 main.fkd.resample_frequency=10
launch e16_freq25  3 main.fkd.resample_frequency=25
launch e16_nofkd   4 main.use_fkd=false   # gradient+RBF, no resampling at all

# E11 zero-code variant: independent end-to-end replicates of full VLS (fresh program per episode)
launch e11_rep2    5
launch e11_rep3    6
launch e11_rep4    7

echo "Night-2 arms launched (setsid-detached)."

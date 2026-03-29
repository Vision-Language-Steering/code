# VLS: Vision Language Steering for Robot Manipulation

VLS - a diffusion-based robot manipulation policy with Feynman-Kac resample and VLM-based task planning.

## Installation

### 1. Clone with Submodules

```bash
git clone --recursive https://github.com/Vision-Language-Steering/code.git
cd code
```

If you already cloned without submodules:

```bash
git submodule update --init --recursive
```

### 2. Install Dependencies

#### Option A: Use Conda Environment (Recommended)

```bash
# Create environment from exported yml
conda env create -f environment.yml
conda activate vls
```

#### Option B: Use pip

```bash
# Install from requirements.txt (exported from vls conda env)
pip install -r requirements.txt
```

#### CALVIN Environment

```bash
cd third_party/calvin/calvin_env
pip install -e .
cd ../calvin_models
pip install -e .
cd ../../..
```

#### LeRobot (Modified Fork)

```bash
cd third_party/lerobot
pip install -e .
cd ../..
```

#### LIBERO (Optional)

```bash
cd third_party/libero_pro
pip install -e .
cd ../..
```

### 3. Download Model Checkpoints

**Diffusion Policy Checkpoint:**

Download or train your diffusion policy checkpoint and update the path in `config.yaml`:

```yaml
policy:
  pretrained_path: "/path/to/your/checkpoint/"
```

We provide a pretrained base [Diffusion Policy checkpoint](https://huggingface.co/Vision-Language-Steering/vls_calvin_base) for CALVIN.

## Configuration

Main configuration is in `config.yaml`. Key sections:

### Main Settings

```yaml
main:
  episode_num: 1                    # Number of episodes to run
  instruction: "close the drawer"   # Task instruction
  use_guidance: true                # Enable steering
  guide_scale: 40.0                 # Guidance strength
  diversity_scale: 10.0             # Diversity weight for particle sampling
  sample_batch_size: 20             # Number of particles for FK steering
  action_horizon: 14                # Action sequence length
  start_step: 70                    # When to start guidance (diffusion step)
  MCMC_steps: 4                     # MCMC steps foe each denoising step
```

### Environment Backend

```yaml
backend:
  backend: "calvin"  # Options: "calvin", "libero", "realworld"
```

**CALVIN-specific:**
```yaml
backend:
  calvin:
    id: "PlayTableSimEnv"
    show_gui: false               # Set true for visualization
    use_egl: true                 # EGL rendering (headless)
    vlm_camera: "static"          # Camera for VLM queries
    cubes_table_only: true        # Only spawn cubes on table
```

**LIBERO-specific:**
```yaml
backend:
  libero:
    suite_name: "libero_spatial"  # Options: libero_spatial, libero_object, libero_goal, libero_10
    vlm_camera: "agentview"
```


### VLM Agent

```yaml
vlm_agent:
  model: "gpt-4"                    # or "gpt-4o", "claude-3.5-sonnet"
  temperature: 0.7
  max_completion_tokens: 2000
```

### Keypoint Detection

```yaml
keypoint_detector:
  num_candidates_per_mask: 5       # Keypoints per detected object
  min_dist_bt_keypoints: 0.02      # Minimum distance between keypoints
  max_mask_ratio: 0.5              # Ignore masks larger than this ratio
  bounds_min: [-1.0, -0.75, -0.1]  # Workspace bounds
  bounds_max: [0.10, 0.75, 1.2]
```

## Running the Pipeline

### Basic Usage

```bash
python main.py --config config.yaml
```

### Pipeline Overview

The execution follows this pipeline:

```
1. Environment Setup
   └─> Load environment adapter (CALVIN/LIBERO/RealWorld)
   └─> Initialize observation space

2. VLM Query Stage
   └─> Capture scene image from vlm_camera
   └─> Send to VLM with task instruction
   └─> Extract guidance keypoints and stage information

3. Keypoint Detection & Tracking
   └─> Get VLM image and segmentation image from adapter
   └─> Extract keypoint candidates for each mask by clustering from DINO feature
   └─> Initialize KeypointTracker for online tracking

4. Policy Rollout Loop (each step):
   a) Get current observation from environment
   b) Update keypoint positions via tracker
   c) Compute guidance (if use_guidance=true):
      - Sample multiple action sequences (particles)
      - Transform delta_ee to 3D trajectories
      - Compute reward based on reward functions
      - (will) FK resampling: weight and resample particles
      - Guided MCMC sampling
   d) Select best action from guided samples
   e) Execute action in environment
   f) Log trajectory and visualizations

5. Episode Termination
   └─> Save trajectory video
   └─> Save keypoint tracking video
   └─> Generate behavior heatmap
   └─> Log success metrics
```

### Key Components

**Environment Adapter** (`core/env_adapters/`):
- Unified interface across different backends
- Handles observation processing, action execution, camera access
- Each adapter implements: `reset()`, `step()`, `get_obs()`, `get_camera_image()`

**Keypoint Detector** (`core/keypoint_detector.py`):
- Grounding DINO for text-conditional object detection
- SAM for precise segmentation
- Extracts 3D keypoints from depth + segmentation masks

**Keypoint Tracker** (`core/keypoint_tracker.py`):
- Tracks keypoints across frames using optical flow
- Handles occlusion and reinitialization

**FK Steering** (`core/fkd_class.py`):
- Maintains particle swarm during diffusion sampling
- Resamples based on reward (keypoint proximity)
- Non-gradient particle filter approach

**Diffusion Policy** (`third_party/lerobot/.../modeling_diffusion_steer.py`):
- Modified diffusion policy that supports particle-based sampling
- Integrates FK steering into the denoising loop
- Returns multiple samples for reward evaluation

## Important Notes

### 1. API Keys for VLM

Set your API key as environment variable:

```bash
export OPENAI_API_KEY="your-key-here"
# or
export ANTHROPIC_API_KEY="your-key-here"
```

Or set in code before running.

### 2. Checkpoint Compatibility

Make sure your policy checkpoint matches the observation space and action space:
- CALVIN: RGB (200x200) + Proprioception
- Action: 7-DOF delta pose + gripper

### 3. Guidance Parameters Tuning

- `guide_scale`: Higher = stronger guidance, but may reduce diversity
- `diversity_scale`: Controls particle diversity during resampling
- `sample_batch_size`: More particles = better coverage but slower
- `start_step`: When to apply guidance in diffusion steps (0-100)
- `MCMC_steps`: More steps = better refinement but slower

Typical ranges:
- `guide_scale`: 10-100
- `diversity_scale`: 1-20
- `sample_batch_size`: 10-50
- `start_step`: 50-80

### 4. Output Directory Structure

```
results/
└── TIMESTAMP/
    ├── episode_1/
    │   ├── vlm_agent/
    │   │   ├── query_img.png          # Scene image sent to VLM
    │   │   ├── prompt.txt              # Full prompt
    │   │   ├── output_raw.txt          # VLM response
    │   │   └── stage1_guidance.txt     # Parsed guidance
    │   ├── trajectory_*.png            # Trajectory visualization per step
    │   ├── episode_1_success.mp4       # Execution video
    │   └── keypoints_tracking.mp4      # Keypoint tracking video
    ├── episode_2/
    │   └── ...
    └── behavior_static.png             # Heatmap of end-effector positions
```

### 5. Debugging

Enable visualizations for debugging:

```yaml
main:
  visualize_trajectory: true
  debug_draw_trajectory: true
  render: true  # Show GUI if supported
```

View logs:
```bash
tail -f results/TIMESTAMP/run.log
```

## Troubleshooting

**Issue: `ModuleNotFoundError: No module named 'calvin_env'`**
- Make sure you installed calvin_env: `cd third_party/calvin/calvin_env && pip install -e .`

**Issue: VLM queries failing**
- Check API key is set: `echo $OPENAI_API_KEY`
- Check internet connection
- Try with a different model in config

**Issue: Keypoint detection finds nothing**
- Check VLM output in `results/.../vlm_agent/output_raw.txt`
- Make sure object names match what's in the scene
- Try adjusting `max_mask_ratio` in config

**Issue: Policy output is random/bad**
- Verify checkpoint path is correct
- Check if checkpoint is compatible with environment
- Try without guidance first (`use_guidance: false`)

**Issue: Slow execution**
- Reduce `sample_batch_size`
- Reduce `MCMC_steps`
- Set `visualize_trajectory: false`
- Use smaller image sizes in env config


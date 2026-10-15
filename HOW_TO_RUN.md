# How to Run

Tested on 3× NVIDIA A40 (46 GB) inside a Docker container: two GPUs serve the VLM and one runs SceneDiff. Back to the [README](README.md).

## 1. Get the code and start a container

```bash
git clone --recursive <this-repo-url> SpatialClaw
cd SpatialClaw

docker run -it --rm --gpus '"device=0,1,3"' --shm-size=256g \
  -v "$(pwd)":/workspace/SpatialClaw \
  -v /path/to/hf_cache:/root/.cache/huggingface \
  -e HF_HOME=/root/.cache/huggingface \
  -w /workspace/SpatialClaw \
  <your-image> /bin/bash
```

A base image is provided in [docker/Dockerfile](docker/Dockerfile).

## 2. One-time setup (inside the container)

```bash
bash spatial_agent/scripts/setup.sh                   # SpatialClaw environments

export SCENEDIFF_DINOV3_URL='<signed dinov3_vith16plus URL>'
bash spatial_agent/scripts/setup.sh --scene_diff      # SceneDiff deps + SAM, π³, DINOv3 weights
```

DINOv3 weights are gated: request access at <https://ai.meta.com/dinov3/>. Weights are stored in `$HF_HOME/scene_diff/`.

## 3. Start the services (one terminal each, from `/workspace/SpatialClaw`)

**Terminal 1: VLM server** (Qwen3.6-27B, FP8, 2 GPUs). Wait for `Application startup complete`.

```bash
bash spatial_agent/launch_managers/vllm_manager/run_vllm.sh \
    Qwen/Qwen3.6-27B qwen3.6-27b 262144 4 2 2 auto fp8
```

**Terminal 2: GPU tool server** (SceneDiff on its own GPU). Wait for `[GPU Server] READY`.

```bash
source /opt/conda/etc/profile.d/conda.sh && conda activate spatialagent
export PYTHONPATH=$PWD/tools/third_party/sam3:$PWD/tools/third_party/Pi3:$PWD/tools/third_party/Depth-Anything-3/src:$PWD/tools/third_party/map-anything
CUDA_VISIBLE_DEVICES=2 python -m spatial_agent.entrypoints.launch_gpu_server --tools SceneDiff
```

**Terminal 3: agent.** Video 1 is "before", video 2 is "after".

```bash
source /opt/conda/etc/profile.d/conda.sh && conda activate spatialagent
python -m spatial_agent.entrypoints.run_videos \
  --videos /workspace/SpatialClaw/video_data/video1_indoor.mp4 \
           /workspace/SpatialClaw/video_data/video2_indoor.mp4 \
  --tools_to_use SceneDiff \
  --question "Find the difference between video 1 and video 2."
```

## Outputs

Each run is saved to `spatial_agent/work_dir/<model>/session-<id>/`:

```
session-<id>/
├── result.json, trace.jsonl, session_report.html   # final answer, step trace, HTML report
└── scene_diff/video1_vs_video2/
    ├── scene_shared.ply        # both videos in one shared 3D frame
    ├── changes_highlight.ply   # scene in gray, changes colored with 3D boxes
    ├── render_*.png, crops/    # top-down / 3D views, per-object crops
    ├── changes.json, report.md # detected changes with the agent's labels
    └── scenediff_raw/          # SceneDiff's own outputs (masks, cost maps, frames)
```

Open the `.ply` files in [MeshLab](https://www.meshlab.net/) or [CloudCompare](https://www.cloudcompare.org/).

## Notes

- **`--tools_to_use` must only list tools that a running GPU server hosts.** A call to a tool with no server waits up to 4 hours. To also host SpatialClaw's own tools, start the server with `--tools Reconstruct SAM3 SceneDiff`; on a single 46 GB GPU this may run out of memory.
- **Runtime:** one SceneDiff comparison takes about 3 minutes for 8 + 9 frames and about 12 minutes for 24 + 23 frames on one A40.
- **If Ctrl-C hangs** on a busy GPU server, run `pkill -9 -f launch_gpu_server`.

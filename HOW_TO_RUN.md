# How to Run

Back to the [README](README.md).

The system runs as three services:

1. **VLM server:** the model the agent thinks with.
2. **GPU tool server:** hosts SceneDiff.
3. **Agent:** takes your two videos and question.

Start them in that order, each in its own terminal, from the repository root.

## Requirements

- **Linux with NVIDIA GPUs and Docker** (or a machine with conda and the CUDA toolkit).
- **GPU for the VLM.** It depends on the model; see [Choosing a VLM](#choosing-a-vlm). For example, Qwen3.6-27B (FP8) needs 2× 46 GB GPUs, and Qwen3.6-35B-A3B (FP8) needs 1.
- **GPU for SceneDiff.** One GPU with about 30 GB free (peak for ~25 frames per video).
- **Disk.** About 10 GB of SceneDiff weights plus the VLM weights, stored in your Hugging Face cache (`$HF_HOME`).

## 1. Install

```bash
git clone --recursive https://github.com/siri-rouser/spatialclaw_scenediff.git SpatialClaw
cd SpatialClaw
```

**With Docker (recommended).** Build the base image, start a container with the repo and your Hugging Face cache mounted, and run setup inside it:

```bash
docker build -t spatialclaw:base docker/
docker run -it --gpus all --shm-size=64g \
  -v "$(pwd)":/workspace/SpatialClaw \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -e HF_HOME=/root/.cache/huggingface \
  -w /workspace/SpatialClaw \
  spatialclaw:base /bin/bash
```

Inside the container (or directly on your machine):

```bash
bash spatial_agent/scripts/setup.sh                 # SpatialClaw: agent env, vLLM env, CUDA libs

export SCENEDIFF_DINOV3_URL='<signed URL for dinov3_vith16plus_pretrain_lvd1689m>'
bash spatial_agent/scripts/setup.sh --scene_diff    # SceneDiff: Python deps + SAM, π³, DINOv3 weights
```

- **More on the base environment** (conda/vLLM environments, model weights, troubleshooting): see the original [SpatialClaw README](SPATIALCLAW_README.md) and its [installation guide](docs/installation.md).
- **DINOv3 weights are gated.** Request access at <https://ai.meta.com/dinov3/>, which emails you the download links.
- **Where weights go:** `$HF_HOME/scene_diff/`. To use files stored elsewhere, set `SCENEDIFF_DINOV3_WEIGHTS` and `SCENEDIFF_SAM_CHECKPOINT` to their paths.
- **Saving the setup:** with Docker, `docker commit` the container after setup so you don't repeat it.

## 2. Start the VLM server

```bash
bash spatial_agent/scripts/vllm/serve_local.sh [hf_model] [served_name] [tp] [max_model_len]
# default: Qwen/Qwen3.6-27B qwen3.6-27b 2 262144
```

- **GPUs:** it automatically takes the first `tp` free GPUs.
- **Ready:** wait for `Application startup complete`. The server registers itself in `spatial_agent/logs/serve.json`, which is how the agent finds it.
- **Quantization:** `serve_local.sh` always applies FP8 quantization. For other settings (quantization, KV-cache dtype, batch size), use `spatial_agent/launch_managers/vllm_manager/run_vllm.sh` instead; its arguments are documented at the top of the script.

### Choosing a VLM

Any model with a config in [spatial_agent/config/model/](spatial_agent/config/model/) works, for example `qwen3.6-27b`, `qwen3.6-35b-a3b`, `qwen3.5-122b-a10b` and `gemma-4-31b`. Recommended Hugging Face checkpoints and GPU counts for each are listed in [models.json](spatial_agent/launch_managers/vllm_manager/models.json).

**`served_name` must match the config name you later pass to the agent with `--model`.**

To use a hosted OpenAI-compatible API instead of a local server:
1. Copy one of the model configs.
2. Set `llm_base_url` to the API endpoint, `llm_model` to the API's model name, and `llm_api_key` to `"${OPENAI_API_KEY}"`.
3. Put the key in `.env` (see [.env.example](.env.example)).
4. Pass the config's path to the agent with `--model`.

The model must accept images.

## 3. Start the GPU tool server

```bash
source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate spatialagent
export PYTHONPATH=$PWD/tools/third_party/sam3:$PWD/tools/third_party/Pi3:$PWD/tools/third_party/Depth-Anything-3/src:$PWD/tools/third_party/map-anything
CUDA_VISIBLE_DEVICES=<free_gpu> python -m spatial_agent.entrypoints.launch_gpu_server --tools SceneDiff
```

- **Pick a GPU** that the VLM server isn't using. Wait for `[GPU Server] READY`.
- **SpatialClaw's own perception tools:** to host them as well, use `--tools Reconstruct SAM3 SceneDiff`. These need a lot of extra GPU memory (the default DA3 reconstruction model alone reserves about 20 GB), so on a single 46 GB GPU, run them as a second server on another GPU.
- **Remote GPU servers:** add `--host 0.0.0.0` when the agent runs on another machine that shares this filesystem.

## 4. Run the agent

Record two videos of the same space, then:

```bash
source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate spatialagent
python -m spatial_agent.entrypoints.run_videos \
  --videos /path/to/before.mp4 /path/to/after.mp4 \
  --model qwen3.6-27b \
  --tools_to_use SceneDiff \
  --question "What changed between video 1 and video 2? Use SceneDiff and describe each moved, added or removed object."
```

The first video is treated as "before" and the second as "after". When the run finishes, the script prints the answer and the session folder.

| Option | Meaning |
|---|---|
| `--model` | Model config name under `spatial_agent/config/model/` (default `qwen3.6-27b`), or a path to a config JSON. |
| `--tools_to_use` | GPU tools the agent may call. List only tools a running GPU server hosts. |
| `--question` | Free-form question. Add `--options "first" "second" ...` for multiple choice (labelled A, B, ...). |
| `--max_steps` | Maximum agent steps (default 30). |
| `--timeout_sec` | Per-step code timeout (default 600 s; 3600 s when SceneDiff is enabled). |
| `--session_id`, `--work_dir` | Name and location of the output folder. |
| `--llm_base_url` | Point at a specific VLM endpoint instead of auto-discovery. |

**Recording tips**
- Film the same area in both videos with plenty of overlap; different viewpoints are fine.
- Move the phone slowly.
- 10–40 seconds per video is enough. SceneDiff samples 1 frame per second, up to 40 frames per video.

## Outputs

Each run is saved to `spatial_agent/work_dir/custom_<model>/session-<id>/`. Pass `--work_dir` to change this. See [example_output/](example_output/) for a complete example.

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

## Troubleshooting

- **The agent is stuck before its first step, or a tool call never returns.** `--tools_to_use` lists a tool that no running GPU server hosts; such calls wait up to 4 hours for a server to appear. Match `--tools_to_use` to the server's `--tools`.
- **SceneDiff is slow.** One comparison takes roughly 3–15 minutes depending on frame count. High CPU use during the SAM stage is expected.
- **Out of GPU memory.** Run SceneDiff on its own GPU, or shorten the videos.
- **The GPU server ignores Ctrl-C while busy.** Run `pkill -9 -f launch_gpu_server`.
- **A tool call fails.** The error is printed in the agent's output and, with a full traceback, in the GPU server's terminal.

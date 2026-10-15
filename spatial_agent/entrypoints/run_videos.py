"""Run the agent on your own videos + a free-form question (no benchmark).

Each video is exposed in the kernel as its own ``InputImages_<N>`` (1-indexed,
in the order given on the command line), so the agent can compare them. With
a single video the kernel gets a plain ``InputImages`` instead.

Usage::

    python -m spatial_agent.entrypoints.run_videos \\
        --videos /path/to/before.mp4 /path/to/after.mp4 \\
        --question "What changed in the scene between the two videos?"

    # Other model (name under config/model/ or a path to a model JSON),
    # multiple-choice options, and an explicit vLLM endpoint:
    python -m spatial_agent.entrypoints.run_videos \\
        --videos a.mp4 b.mp4 \\
        --question "Which video shows the chair closer to the door?" \\
        --options "Video 1" "Video 2" "Same distance" \\
        --model qwen3.6-35b-a3b \\
        --llm_base_url http://localhost:8000/v1
"""

import argparse
import asyncio
import json
import os
from typing import List

_PKG_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_MODEL_CONFIG_DIR = os.path.join(_PKG_DIR, "config", "model")
DEFAULT_MODEL = "qwen3.6-27b"


def parse_args():
    parser = argparse.ArgumentParser(description="Spatial Agent on custom videos")
    parser.add_argument("--videos", nargs="+", required=True,
                        help="One or more video files. Order defines InputImages_1, _2, ...")
    parser.add_argument("--question", type=str, required=True,
                        help="Question to ask about the video(s)")
    parser.add_argument("--options", nargs="+", default=None,
                        help="Optional multiple-choice options; labelled A, B, C, ... "
                             "and the agent is asked to answer with the letter")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL,
                        help=f"Model name under config/model/ (default: {DEFAULT_MODEL}) "
                             "or a path to a model config JSON")
    parser.add_argument("--dataset", type=str, default=None,
                        help="Optional dataset config JSON to borrow agent/tool settings from "
                             "(e.g. config/dataset/cvbench.json)")
    parser.add_argument("--session_id", type=str, default=None,
                        help="Name for this run's session directory (default: random)")
    parser.add_argument("--work_dir", type=str, default=None)
    parser.add_argument("--llm_model", type=str, default=None,
                        help="Override the served model name from the model config")
    parser.add_argument("--llm_base_url", type=str, default=None,
                        help="Override the endpoint, e.g. http://localhost:8000/v1 "
                             "(default from model config: 'vllm' = serve.json auto-discovery)")
    parser.add_argument("--gpu_server", type=str, default=None,
                        help="GPU tool server: 'auto' (default) or http://host:port")
    parser.add_argument("--tools_to_use", nargs="*", default=None,
                        help="Perception tools (default: Reconstruct SAM3). "
                             "Pass with no values to run without the GPU tool server")
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--num_key_frames", type=int, default=None,
                        help="Key frames shown to the LLM per video (default: 32)")
    parser.add_argument("--video_max_fps", type=float, default=None,
                        help="Frame extraction rate cap (default: 1.0)")
    parser.add_argument("--video_frame_resize_short_edge", type=int, default=None)
    parser.add_argument("--executor_type", type=str, default=None,
                        choices=["code", "react", "single_pass"])
    parser.add_argument("--enable_reflection", action="store_true", default=None)
    parser.add_argument("--timeout_sec", type=int, default=None,
                        help="Per-code-cell timeout in seconds (default: 600, or 3600 "
                             "when SceneDiff is in --tools_to_use)")
    return parser.parse_args()


def resolve_model_config(model: str) -> str:
    """Accept either a path to a model JSON or a name under config/model/."""
    if os.path.isfile(model):
        return model
    candidate = os.path.join(_MODEL_CONFIG_DIR, f"{model}.json")
    if os.path.isfile(candidate):
        return candidate
    available = sorted(f[:-5] for f in os.listdir(_MODEL_CONFIG_DIR) if f.endswith(".json"))
    raise SystemExit(f"Unknown model '{model}'. Available: {available}")


def build_instruction(question: str, video_names: List[str], options) -> str:
    n = len(video_names)
    if n > 1:
        listing = "\n".join(
            f"- `InputImages_{i}`: {name}" for i, name in enumerate(video_names, 1)
        )
        context = (
            f"This sample contains {n} separate videos, each exposed in the kernel "
            f"as its own `InputImages_<N>` (1-indexed, in this order):\n{listing}\n"
            "To compare across videos, pass frames from multiple videos to "
            "`vlm.ask_with_thinking(...)`."
        )
    else:
        context = f"This sample contains one video ({video_names[0]}), exposed as `InputImages`."

    instruction = f"{question}\n\n{context}"
    if options:
        instruction += "\nAnswer with the option letter."
        for i, text in enumerate(options):
            instruction += f"\n{chr(65 + i)}. {text}"
    else:
        instruction += (
            "\nAnswer in free-form text, concretely describing what you found "
            "and citing the evidence (video, frame) that supports it."
        )
    return instruction


async def main():
    args = parse_args()

    videos = [os.path.abspath(v) for v in args.videos]
    missing = [v for v in videos if not os.path.isfile(v)]
    if missing:
        raise SystemExit(f"Video(s) not found: {missing}")

    from spatial_agent.config import SpatialAgentConfig, set_config
    config = SpatialAgentConfig()
    config.benchmark = "custom"
    config.concurrency = 1
    config.video_max_fps = 1.0

    # Same layering as run.py: dataset JSON → model JSON → CLI args.
    if args.dataset:
        config.update_from_dataset_json(args.dataset)
    config.update_from_model_json(resolve_model_config(args.model))
    config.update_from_args(args)
    if args.timeout_sec is None and "SceneDiff" in (config.tools_to_use or []):
        config.timeout_sec = max(config.timeout_sec, 3600)  # a SceneDiff run takes minutes

    if not config.work_dir:
        model_short = config.llm_model.split("/")[-1][:30] if config.llm_model else "unknown"
        config.work_dir = os.path.join(_PKG_DIR, "work_dir", f"custom_{model_short}")
    os.makedirs(config.work_dir, exist_ok=True)

    set_config(config)

    import shortuuid
    from spatial_agent.evals.base import extract_video_frames

    session_id = args.session_id or shortuuid.uuid()[:8]

    # Cache frames under work_dir, keyed by position + extraction settings:
    # the stock cache (next to the video) is keyed by basename only, so it
    # would serve stale frames after changing --video_max_fps, and two
    # videos with the same filename in different folders would collide.
    fps_tag = f"fps{config.video_max_fps}" if config.video_max_fps else "native"
    size_tag = (f"se{config.video_frame_resize_short_edge}"
                if config.video_frame_resize_short_edge else "full")
    groups, indices_groups, fps_list, totals, durations, names = [], [], [], [], [], []
    for i, vp in enumerate(videos, 1):
        name = os.path.basename(vp)
        cache_dir = os.path.join(
            config.work_dir, "frame_cache", f"{session_id}_{i}_{name}_{fps_tag}_{size_tag}"
        )
        frames, indices, fps, total = extract_video_frames(
            vp, cache_dir,
            video_max_fps=config.video_max_fps,
            video_frame_resize_short_edge=config.video_frame_resize_short_edge,
        )
        if not frames:
            raise SystemExit(f"No frames could be extracted from {vp}")
        print(f"Video {i}: {name} — {len(frames)} frames @ {fps:.2f} fps source")
        groups.append(frames)
        indices_groups.append(indices)
        fps_list.append(fps)
        totals.append(total)
        durations.append(total / fps if fps else 0.0)
        names.append(name)

    instruction = build_instruction(args.question, names, args.options)

    with open(os.path.join(config.work_dir, f"config-{session_id}.json"), "w") as f:
        json.dump(config.to_dict(), f, indent=2, default=str)

    from spatial_agent.workflow import SpatialAgentWorkflow
    workflow = SpatialAgentWorkflow(config)

    try:
        if len(videos) > 1:
            run_result = await workflow.arun(
                instruction=instruction,
                images=[f for g in groups for f in g],
                session_id=session_id,
                image_groups=groups,
                frame_indices_groups=indices_groups,
                fps_per_video=fps_list,
                total_frames_per_video=totals,
                duration_per_video=durations,
                video_names=names,
                video_sources_per_video=videos,
            )
        else:
            run_result = await workflow.arun(
                instruction=instruction,
                images=groups[0],
                session_id=session_id,
                frame_indices=indices_groups[0],
                video_source=videos[0],
                fps=fps_list[0],
                total_video_frames=totals[0],
                duration_sec=durations[0],
            )
    finally:
        workflow.shutdown()

    final_answer = run_result.get("final_answer") or {}
    answer_text = final_answer.get("text", "")
    session_dir = run_result["_report_context"]["session_dir"]

    result = {
        "session_id": session_id,
        "model": config.llm_model,
        "videos": videos,
        "question": args.question,
        "options": args.options,
        "answer": answer_text,
        "termination_reason": run_result.get("termination_reason"),
        "step_count": run_result.get("step_count"),
        "total_tool_calls": run_result.get("total_tool_calls"),
        "usage": run_result.get("usage"),
    }
    with open(os.path.join(session_dir, "result.json"), "w") as f:
        json.dump(result, f, indent=2, default=str)

    print(f"\n{'=' * 60}")
    print(f"Question: {args.question}")
    print(f"Answer:   {answer_text or '(none)'}")
    print(f"Ended:    {result['termination_reason']} after {result['step_count']} steps")
    print(f"Session:  {session_dir}")
    scene_diff_dir = os.path.join(session_dir, "scene_diff")
    if os.path.isdir(scene_diff_dir):
        print(f"SceneDiff outputs (point clouds, renders, report): {scene_diff_dir}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    asyncio.run(main())

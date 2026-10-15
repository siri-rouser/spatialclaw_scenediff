# Example Output

The complete output of one run on the indoor desk example from the [main README](../README.md): [video1_indoor.mp4](../video_data/video1_indoor.mp4) (before) vs. [video2_indoor.mp4](../video_data/video2_indoor.mp4) (after), with Qwen3.6-27B as the agent.

```
session-beKFiVa3/
├── result.json                 # final answer, steps, token usage
├── trace.jsonl                 # every agent step: plan, code, output
├── session_report.html         # browsable report of the whole run
├── show_images/                # images the agent looked at
└── scene_diff/video1_vs_video2/
    ├── scene_shared.ply        # both videos in one shared 3D frame (open in MeshLab / CloudCompare)
    ├── scene_video1.ply, scene_video2.ply
    ├── changes_highlight.ply   # scene in gray, changes colored with 3D boxes
    ├── render_topdown.png, render_oblique.png, render_crops.png
    ├── crops/                  # per-object before/after crops
    ├── changes.json, report.md # detections with the agent's labels
    ├── scene_vis.npz           # data used to re-render the figures
    └── scenediff_raw/          # SceneDiff's own outputs: frames used, object_masks.pkl, per-frame overlays
```

To keep the repository small, six of SceneDiff's intermediate files (137 MB) are not included: `scenediff_raw/cost_map_{1,2,merged}.ply`, `object_instances_{1,2}.ply` and `sam_masks.pkl`. They are still listed in `report.md` and `changes.json`. Re-run the example ([HOW_TO_RUN.md](../HOW_TO_RUN.md)) to regenerate them. Paths inside the JSON files are absolute paths from the original run's container (`/workspace/SpatialClaw/...`).

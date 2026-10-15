"""SceneDiff GPU tool: what changed between two videos of the same scene?

Wraps ``SceneDiffModel`` on the GPU server. The server reads the original
video files and writes all artifacts (point clouds, renders, crops,
changes.json, report.md) into ``<session_dir>/scene_diff/<run>/``; this
client exposes the result to the kernel and lets the agent attach semantic
labels afterwards with ``Annotate``.
"""

import os
import time
from typing import Dict, List, Optional, Union

import numpy as np
from PIL import Image

from spatial_agent.gpu_models.types import SceneDiffObject, SceneDiffOutput
from spatial_agent.tools import scene_diff_vis as vis
from spatial_agent.tools.base import GPUTool


class SceneDiffResult:
    """Kernel-side view of a SceneDiff comparison.

    Attributes:
        objects: list of ``SceneDiffObject`` (``.id``, ``.change_type``,
            ``.centroid_1/2``, ``.bbox_1/2``, ``.displacement``,
            ``.best_time_1/2``, ``.best_frame_index_1/2``, ``.crop_1/2``,
            ``.context_1/2``, ``.label``, ``.description``).
        render_topdown / render_oblique: PIL images of the shared 3D scene.
        output_dir: folder holding every saved artifact.
        files: {name: path} of saved artifacts.
        scene_extent: robust scene diameter (coordinates are up-to-scale).
    """

    def __init__(self, out: SceneDiffOutput, video_indices, fps_per_video):
        self._out = out
        self.objects: List[SceneDiffObject] = list(out.objects or [])
        self.output_dir = out.output_dir
        self.files = dict(out.files or {})
        self.render_topdown = out.render_topdown
        self.render_oblique = out.render_oblique
        self.num_frames = out.num_frames
        self.scene_extent = out.scene_extent
        self.video_indices = video_indices
        for o in self.objects:
            for s, fps in (("1", fps_per_video[0]), ("2", fps_per_video[1])):
                t = getattr(o, f"best_time_{s}")
                setattr(o, f"best_frame_index_{s}",
                        int(round(t * fps)) if (t is not None and fps) else None)

    def get(self, obj_id: int) -> SceneDiffObject:
        for o in self.objects:
            if o.id == obj_id:
                return o
        raise KeyError(f"No object with id {obj_id}; ids: {[o.id for o in self.objects]}")

    def object_panel(self, obj_id: int) -> Image.Image:
        """[frame v1 | crop v1 | crop v2 | frame v2] strip for one object (pass to show())."""
        o = self.get(obj_id)
        return vis.object_panel(
            {"id": o.id, "change_type": o.change_type, "label": o.label},
            {"context_1": o.context_1, "crop_1": o.crop_1,
             "crop_2": o.crop_2, "context_2": o.context_2},
        )

    def renders(self) -> List[Image.Image]:
        """[top-down, oblique 3D] renders of the shared scene (pass to show())."""
        return [self.render_topdown, self.render_oblique]

    def summary(self) -> str:
        a, b = self.video_indices
        lines = [
            f"SceneDiff video {a} (before) vs video {b} (after): {len(self.objects)} changed object(s); "
            f"frames used {self.num_frames}; scene extent {self.scene_extent:.2f} units "
            "(x right, y forward, z up; up-to-scale).",
        ]
        for o in self.objects:
            moved = ""
            if o.displacement is not None:
                d = float(np.linalg.norm(o.displacement))
                moved = (f" moved {d:.2f} ({100 * d / max(self.scene_extent, 1e-6):.0f}% of scene), "
                         f"Δ={np.round(o.displacement, 2).tolist()}")
            label = f" [{o.label}]" if o.label else ""
            lines.append(
                f"  #{o.id} {o.change_type}{label}: "
                f"before@{_fmt(o.centroid_1)} (t={o.best_time_1}s, frame={o.best_frame_index_1}) "
                f"after@{_fmt(o.centroid_2)} (t={o.best_time_2}s, frame={o.best_frame_index_2})"
                f"{moved} score={o.score:.2f}"
            )
        lines.append(f"Artifacts saved in {self.output_dir}")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return self.summary()


def _fmt(v) -> str:
    return "None" if v is None else str(np.round(v, 2).tolist())


class SceneDiffTool(GPUTool):
    """Client wrapper for the SceneDiff GPU deployment."""

    TOOL_ABLATION_PREFIX = "tool_scenediff"
    TOOL_PROMPT_DESCRIPTION = """
### tools.SceneDiff - 3D change detection between two videos (GPU, SLOW ~5-15 min)

Fuses two videos of the same space into ONE shared 3D reconstruction and detects objects that were
**Added**, **Removed** or **Moved** between them (robust to viewpoint change / occlusion). It finds
*where* things changed but gives NO semantic labels — you must look at the crops and name them.
All artifacts (shared point cloud .ply files, renders, crops, changes.json, report.md) are saved to disk
automatically.

- `tools.SceneDiff.Compare(video_a=1, video_b=2, max_frames_per_video=40, change_threshold=None)` → `SceneDiffResult`
  - `video_a` = "before" video index, `video_b` = "after" video index (1-indexed, as in `InputImages_<N>`).
  - Call it ONCE and reuse the result variable; it is expensive. Only re-run with a different
    `change_threshold` (float, e.g. 0.3 = more detections, 0.6 = fewer) if results are clearly wrong.
- `res.summary()` → str: one line per change with id, type, 3D position before/after, displacement,
  and the time (s) / frame index of the clearest view in each video.
- `res.renders()` → `[topdown, oblique]` PIL images of the shared 3D scene with colored changes
  (green = Added, red = Removed, orange = Moved; dashed box = old position).
- `res.object_panel(id)` → PIL strip `[frame v1 | crop v1 | crop v2 | frame v2]` for one object.
- `res.objects` → list with `.id`, `.change_type`, `.centroid_1/2`, `.bbox_1/2`, `.displacement`,
  `.best_frame_index_1/2`, `.crop_1/2`, `.context_1/2`.
- `tools.SceneDiff.Annotate(res, labels, summary="")` → str path of the saved report
  - `labels`: `{{id: {{"label": "blue chair", "description": "moved ~1 m toward the window"}}}}`
    (or `{{id: "blue chair"}}`). Use `{{id: {{"label": "false positive"}}}}` for spurious detections.
  - Saves labels into changes.json, re-renders the figures with names, and writes report.md.

**Workflow:**
```
res = tools.SceneDiff.Compare(1, 2)
print(res.summary())
show(*res.renders())
for o in res.objects:
    show(res.object_panel(o.id))       # inspect each change visually, then name it
report = tools.SceneDiff.Annotate(res, {{0: {{"label": "...", "description": "..."}}}}, summary="...")
```
Verify doubtful detections against the original frames (e.g. `InputImages_2[...]`) before labelling.
Mention moved/added/removed objects with labels and approximate 3D locations in your final answer.
"""

    HTTP_TIMEOUT = 3600  # a full SceneDiff run takes minutes

    def __init__(self, session_dir: Optional[str] = None, work_dir: Optional[str] = None,
                 video_source: Optional[str] = None,
                 video_sources_list: Optional[List[str]] = None,
                 fps_per_video: Optional[List[float]] = None,
                 gpu_tool_max_retries: int = 1):
        # Never auto-retry a long-running comparison.
        super().__init__(deployment_name="spatial_SceneDiff", gpu_tool_max_retries=1)
        self._session_dir = session_dir
        self._work_dir = work_dir
        self._video_sources = list(video_sources_list or ([video_source] if video_source else []))
        self._fps = list(fps_per_video or [])
        self._run_counter = 0

    def _output_dir(self, a: int, b: int) -> str:
        base = self._session_dir or os.path.join(self._work_dir or os.getcwd(), "scene_diff_runs",
                                                 time.strftime("%Y%m%d-%H%M%S"))
        self._run_counter += 1
        suffix = "" if self._run_counter == 1 else f"_{self._run_counter}"
        return os.path.abspath(os.path.join(base, "scene_diff", f"video{a}_vs_video{b}{suffix}"))

    def Compare(self, video_a: int = 1, video_b: int = 2, max_frames_per_video: int = 40,
                change_threshold: Optional[float] = None, resample_rate: int = 30) -> SceneDiffResult:
        """Detect moved / added / removed objects between two input videos."""
        n = len(self._video_sources)
        if n < 2:
            raise RuntimeError(
                "SceneDiff.Compare needs two input videos (backed by video files); "
                f"this sample has {n}."
            )
        for name, v in (("video_a", video_a), ("video_b", video_b)):
            if not isinstance(v, int) or not 1 <= v <= n:
                raise ValueError(f"{name}={v!r} must be an integer in [1, {n}].")
        if video_a == video_b:
            raise ValueError("video_a and video_b must differ.")

        out_dir = self._output_dir(video_a, video_b)
        print(f"[SceneDiff] Comparing video {video_a} → video {video_b} (this takes several minutes)...")
        raw = self._call_remote(
            "compare",
            video_1=self._video_sources[video_a - 1],
            video_2=self._video_sources[video_b - 1],
            output_dir=out_dir,
            resample_rate=resample_rate,
            max_frames_per_video=max_frames_per_video,
            change_threshold=change_threshold,
        )
        if getattr(raw, "err", None):
            raise RuntimeError(f"SceneDiff failed: {raw.err['msg']}")
        out = raw.result if hasattr(raw, "result") else raw
        fps = [self._fps[i - 1] if i - 1 < len(self._fps) else None for i in (video_a, video_b)]
        res = SceneDiffResult(out, (video_a, video_b), fps)
        print(f"[SceneDiff] Done in {out.runtime_sec:.0f}s: {len(res.objects)} change(s). "
              f"Saved to {res.output_dir}")
        return res

    def Annotate(self, result: SceneDiffResult,
                 labels: Dict[int, Union[str, Dict[str, str]]],
                 summary: str = "") -> str:
        """Attach semantic labels/descriptions, re-render figures, write report.md."""
        if not isinstance(result, SceneDiffResult):
            raise TypeError("Annotate expects the SceneDiffResult returned by Compare().")
        if not isinstance(labels, dict):
            raise TypeError("labels must be a dict {object_id: label or {'label':..., 'description':...}}")

        known = {o.id for o in result.objects}
        for raw_id, value in labels.items():
            obj_id = int(raw_id)
            if obj_id not in known:
                raise ValueError(f"Unknown object id {raw_id!r}; ids are {sorted(known)}")
            o = result.get(obj_id)
            if isinstance(value, str):
                o.label = value
            elif isinstance(value, dict):
                o.label = str(value.get("label", o.label) or "")
                o.description = str(value.get("description", o.description) or "")
            else:
                raise TypeError(f"labels[{raw_id!r}] must be a str or dict")

        payload = vis.read_changes_json(result.output_dir)
        by_id = {o.id: o for o in result.objects}
        for d in payload["objects"]:
            o = by_id.get(d["id"])
            if o is not None:
                d["label"], d["description"] = o.label, o.description
        if summary:
            payload["summary"] = summary

        result.files.update(vis.render_all(result.output_dir, payload["objects"]))
        result.render_topdown = Image.open(result.files["render_topdown"]).convert("RGB")
        result.render_oblique = Image.open(result.files["render_oblique"]).convert("RGB")
        payload["files"] = {**payload.get("files", {}), **result.files}
        vis.write_changes_json(result.output_dir, payload)
        report = vis.write_report(result.output_dir, payload)
        print(f"[SceneDiff] Labels saved; report: {report}")
        return report

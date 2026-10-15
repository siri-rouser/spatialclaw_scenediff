"""SceneDiff: two-video 3D change detection (moved / added / removed objects).

Wraps ``tools/third_party/scene_diff`` (Pi3 geometry + SAM ViT-H regions +
DINOv3 appearance). The server receives two video *paths* plus an output
directory on the shared filesystem, runs the full SceneDiff pipeline, and
writes every artifact there:

    <output_dir>/
        scenediff_raw/          frames, sam_masks.pkl, object_masks.pkl,
                                cost_map_*.ply, object_instances_*.ply,
                                video_*_detection/ overlays (upstream outputs)
        scene_shared.ply        both videos fused in one gravity-aligned frame
        scene_video1.ply / scene_video2.ply
        changes_highlight.ply   scene in gray + changed points / boxes colored
        render_topdown.png / render_oblique.png / render_crops.png
        crops/                  per-object crops and context frames
        scene_vis.npz           downsampled data used to re-render
        changes.json / report.md

Models are built per call and released afterwards, so the server only holds
SceneDiff's GPU memory while a comparison is running.
"""

import asyncio
import copy
import gc
import os
import shutil
import subprocess
import sys
import time
import types
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from PIL import Image, ImageDraw
import torch

from spatial_agent.gpu_models.base import AgentTool, AgentToolOutput, gpu_inference_lock
from spatial_agent.gpu_models.types import SceneDiffObject, SceneDiffOutput
from spatial_agent.tools import scene_diff_vis as vis

ImageLoader = None  # stub: not used

__all__ = ["SceneDiffModel"]

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_SCENEDIFF_ROOT = os.path.join(_PROJECT_ROOT, "tools", "third_party", "scene_diff")
_SAM_LS_PATH = os.path.join(_SCENEDIFF_ROOT, "submodules", "segment-anything-langsplat-modified")
_PI3_PATH = os.path.join(_PROJECT_ROOT, "tools", "third_party", "Pi3")
_EXTRACT_FPS = 30.0  # SceneDiff's resample_rate assumes 30-fps frame numbering


def _find_checkpoint(env_var: str, filename: str) -> str:
    """Resolve a checkpoint: $env_var → $HF_HOME/scene_diff/ → scene_diff/checkpoints/."""
    if os.environ.get(env_var):
        return os.environ[env_var]
    hf_home = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
    candidates = [os.path.join(hf_home, "scene_diff", filename),
                  os.path.join(_SCENEDIFF_ROOT, "checkpoints", filename)]
    return next((p for p in candidates if os.path.exists(p)), candidates[0])


def _install_torch_scatter_shim() -> None:
    """Provide ``torch_scatter.scatter_mean`` (dim=0 only) in pure torch.

    SceneDiff only uses ``scatter_mean(src, index, dim=0)``; the compiled
    torch_scatter wheel is tied to an exact torch build, so we avoid it.
    """
    try:
        import torch_scatter  # noqa: F401
        return
    except ImportError:
        pass

    def scatter_mean(src, index, dim=0, dim_size=None):
        if dim != 0:
            raise NotImplementedError("scatter_mean shim supports dim=0 only")
        index = index.long()
        size = int(index.max().item()) + 1 if index.numel() else 0
        if dim_size is not None:
            size = max(size, dim_size)
        out = torch.zeros((size,) + tuple(src.shape[1:]), dtype=src.dtype, device=src.device)
        out.index_add_(0, index, src)
        count = torch.zeros(size, dtype=src.dtype, device=src.device)
        count.index_add_(0, index, torch.ones_like(index, dtype=src.dtype))
        count = count.clamp_min(1).view((size,) + (1,) * (src.dim() - 1))
        return out / count

    mod = types.ModuleType("torch_scatter")
    mod.scatter_mean = scatter_mean
    sys.modules["torch_scatter"] = mod


def _install_open3d_shim() -> None:
    """Provide the tiny slice of open3d SceneDiff uses (writing colored PLYs).

    open3d's wheel needs system GL/EGL libraries (libEGL.so.1) that headless
    containers often lack; SceneDiff only uses it to write point clouds.
    """
    try:
        import open3d  # noqa: F401
        return
    except (ImportError, OSError):
        pass

    class PointCloud:
        def __init__(self):
            self.points = np.zeros((0, 3), np.float32)
            self.colors = np.zeros((0, 3), np.float32)

    def write_point_cloud(path, pcd, *args, **kwargs):
        colors = pcd.colors if len(pcd.colors) else np.full((len(pcd.points), 3), 0.5)
        vis.write_ply(path, pcd.points, colors)
        return True

    mod = types.ModuleType("open3d")
    mod.geometry = types.SimpleNamespace(PointCloud=PointCloud)
    mod.utility = types.SimpleNamespace(Vector3dVector=lambda a: np.asarray(a, dtype=np.float64))
    mod.io = types.SimpleNamespace(write_point_cloud=write_point_cloud)
    sys.modules["open3d"] = mod
    print("[SceneDiff] open3d unavailable; using built-in PLY writer", flush=True)


def _pairwise_mask_stats(masks_ord: torch.Tensor):
    """Return (intersection, area) for all mask pairs via one GPU matmul."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    n = masks_ord.shape[0]
    flat = masks_ord.reshape(n, -1).to(device=device, dtype=torch.float32)
    inter = flat @ flat.T  # exact: pixel counts < 2**24
    area = flat.sum(dim=1)
    return inter, area


def fast_mask_nms(masks, scores, iou_thr=0.7, score_thr=0.1, inner_thr=0.2, **kwargs):
    """Vectorized drop-in for SceneDiff's ``utils.mask_nms`` (same outputs).

    The upstream version loops over all O(N^2) mask pairs in Python on CPU
    tensors at full image resolution, which dominates SceneDiff's runtime.
    Here the pairwise intersections come from a single matmul (on GPU when
    available); the selection logic below is unchanged.
    """
    scores, idx = scores.sort(0, descending=True)
    masks_ord = masks[idx.view(-1), :]
    inter, area = _pairwise_mask_stats(masks_ord)
    union = area[:, None] + area[None, :] - inter
    ri = inter / area[:, None]   # intersection / area of mask i (row)
    rj = inter / area[None, :]   # intersection / area of mask j (col)
    upper = torch.ones_like(inter, dtype=torch.bool).triu_()  # pairs with j >= i
    zero = torch.zeros_like(inter)
    iou_matrix = torch.where(upper, inter / union, zero)
    inner_val = 1 - rj * ri
    cond_ij = upper & (ri < 0.5) & (rj >= 0.85)
    cond_ji = upper & (ri >= 0.85) & (rj < 0.5)
    inner_iou_matrix = torch.where(cond_ij, inner_val, zero) + torch.where(cond_ji, inner_val, zero).T
    iou_matrix = iou_matrix.cpu()
    inner_iou_matrix = inner_iou_matrix.cpu()

    iou_matrix.triu_(diagonal=1)
    iou_max, _ = iou_matrix.max(dim=0)
    inner_iou_matrix_u = torch.triu(inner_iou_matrix, diagonal=1)
    inner_iou_max_u, _ = inner_iou_matrix_u.max(dim=0)
    inner_iou_matrix_l = torch.tril(inner_iou_matrix, diagonal=1)
    inner_iou_max_l, _ = inner_iou_matrix_l.max(dim=0)

    keep = iou_max <= iou_thr
    keep_conf = scores > score_thr
    keep_inner_u = inner_iou_max_u <= 1 - inner_thr
    keep_inner_l = inner_iou_max_l <= 1 - inner_thr
    num_top_k = scores.shape[0]
    if keep_conf.sum() == 0:
        keep_conf[scores.topk(num_top_k).indices] = True
    if keep_inner_u.sum() == 0:
        keep_inner_u[scores.topk(num_top_k).indices] = True
    if keep_inner_l.sum() == 0:
        keep_inner_l[scores.topk(num_top_k).indices] = True
    keep *= keep_conf
    keep *= keep_inner_u
    keep *= keep_inner_l
    return idx[keep]


def _import_scenediff():
    """Import SceneDiff's top-level ``modules`` / ``utils`` packages."""
    _install_torch_scatter_shim()
    _install_open3d_shim()
    for p in (_PI3_PATH, _SAM_LS_PATH, _SCENEDIFF_ROOT):
        if p not in sys.path:
            sys.path.insert(0, p)
    from modules import SceneDiff  # noqa: E402
    import utils as sd_utils  # noqa: E402
    # masks_update() looks mask_nms up in utils' globals at call time.
    sd_utils.mask_nms = fast_mask_nms
    return SceneDiff, sd_utils


def _extract_frames(video: str, out_dir: Path, resample_rate: int,
                    max_frames: int) -> tuple:
    """Extract every ``resample_rate``-th frame (30-fps numbering) via ffmpeg.

    Returns ``(paths, times_sec)`` after uniform subsampling to ``max_frames``.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    rate = _EXTRACT_FPS / max(1, resample_rate)
    cmd = ["ffmpeg", "-loglevel", "error", "-i", video, "-vf", f"fps={rate}",
           "-q:v", "2", "-start_number", "0", str(out_dir / "%05d.jpg")]
    subprocess.run(cmd, check=True)
    paths = sorted(out_dir.glob("*.jpg"))
    if not paths:
        raise ValueError(f"No frames could be extracted from {video}")
    keep = np.unique(np.linspace(0, len(paths) - 1, min(max_frames, len(paths))).round().astype(int))
    keep_set = set(keep.tolist())
    for i, p in enumerate(paths):
        if i not in keep_set:
            p.unlink()
    times = [float(i) / rate for i in keep]
    return [paths[i] for i in keep], times


def _decode_rle(rle) -> np.ndarray:
    from pycocotools import mask as mask_utils
    return mask_utils.decode(rle).astype(bool)


def _outline(mask: np.ndarray, width: int = 3) -> np.ndarray:
    """Boundary pixels of a boolean mask."""
    from scipy.ndimage import binary_erosion
    return mask & ~binary_erosion(mask, iterations=width)


def _crop_and_context(frame_path: Path, mask_lowres: np.ndarray, color) -> tuple:
    """Crop the object from the full-res frame and make a downscaled context view."""
    img = Image.open(frame_path).convert("RGB")
    W, H = img.size
    mask = np.asarray(Image.fromarray(mask_lowres.astype(np.uint8) * 255)
                      .resize((W, H), Image.NEAREST)) > 127
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None, None
    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    pad_x, pad_y = int(0.25 * (x1 - x0 + 1)) + 8, int(0.25 * (y1 - y0 + 1)) + 8
    box = (max(0, x0 - pad_x), max(0, y0 - pad_y), min(W, x1 + pad_x + 1), min(H, y1 + pad_y + 1))

    arr = np.asarray(img).copy()
    edge = _outline(mask, width=max(2, W // 300))
    arr[edge] = color
    crop = Image.fromarray(arr).crop(box)
    if max(crop.size) > 512:
        crop.thumbnail((512, 512))

    tinted = np.asarray(img).astype(np.float32)
    tinted[mask] = 0.55 * tinted[mask] + 0.45 * np.array(color, np.float32)
    ctx = Image.fromarray(tinted.astype(np.uint8))
    ImageDraw.Draw(ctx).rectangle(box, outline=tuple(color), width=max(3, W // 200))
    ctx.thumbnail((640, 640))
    return crop, ctx


class SceneDiffModel(AgentTool):
    """Two-video change detection via SceneDiff."""

    CONFIG_PATH = os.environ.get(
        "SCENEDIFF_CONFIG", os.path.join(_SCENEDIFF_ROOT, "configs", "scenediff_config.yml"))
    DINOV3_WEIGHTS = _find_checkpoint(
        "SCENEDIFF_DINOV3_WEIGHTS", "dinov3_vith16plus_pretrain_lvd1689m-7c1da9a5.pth")
    SAM_CHECKPOINT = _find_checkpoint("SCENEDIFF_SAM_CHECKPOINT", "sam_vit_h_4b8939.pth")

    def __init__(self, image_loader: ImageLoader = None) -> None:
        super().__init__()
        import yaml

        with open(self.CONFIG_PATH) as f:
            cfg = yaml.safe_load(f)
        cfg["models"]["sam"]["checkpoint"] = self.SAM_CHECKPOINT
        cfg["models"]["dinov3"]["weight_url"] = self.DINOV3_WEIGHTS
        cfg["visualization"].update(vis_pc=True, save_detections=True, save_gt_visualization=False)
        cfg["cache"].update(use_cache=False, save_cache=False)
        self.base_config = cfg

        for path, what in ((self.SAM_CHECKPOINT, "SAM ViT-H checkpoint"),
                           (self.DINOV3_WEIGHTS, "DINOv3 ViT-H+/16 weights"),
                           (os.path.join(_SCENEDIFF_ROOT, "submodules", "dinov3", "hubconf.py"),
                            "dinov3 submodule")):
            if not os.path.exists(path):
                print(f"[SceneDiff] WARNING: {what} not found at {path}", flush=True)
        import importlib.util
        missing = [m for m in ("faiss", "pycocotools", "skimage", "scipy")
                   if importlib.util.find_spec(m) is None]
        if missing:
            print(f"[SceneDiff] WARNING: missing Python packages {missing} "
                  "(run: bash spatial_agent/scripts/setup.sh --scene_diff)", flush=True)
        self.infer_lock = asyncio.Lock()

    # ------------------------------------------------------------------
    # Core pipeline
    # ------------------------------------------------------------------

    def _run(self, video_1: str, video_2: str, output_dir: str, resample_rate: int,
             max_frames_per_video: int, change_threshold: Optional[float]) -> SceneDiffOutput:
        t_start = time.time()
        SceneDiff, sd_utils = _import_scenediff()

        out = Path(output_dir)
        raw = out / "scenediff_raw"  # SceneDiff names its result dir after this folder
        if raw.exists():
            shutil.rmtree(raw)  # also drops a stale sam_masks.pkl cache
        frames_1, times_1 = _extract_frames(video_1, raw / "video_1", resample_rate, max_frames_per_video)
        frames_2, times_2 = _extract_frames(video_2, raw / "video_2", resample_rate, max_frames_per_video)
        n1, n2 = len(frames_1), len(frames_2)
        file_list = [str(p) for p in frames_1 + frames_2]
        print(f"[SceneDiff] {n1} + {n2} frames → {out}", flush=True)

        config = copy.deepcopy(self.base_config)
        config["dataset"]["resample_rate"] = resample_rate
        if change_threshold is not None:
            config["detection"]["change_region_threshold"] = float(change_threshold)

        sd = SceneDiff(config, device="cuda")
        captured: Dict[str, object] = {}
        gm = sd.geometry_model
        orig_normalize, orig_valid = gm.normalize_scene_scale, gm.get_valid_depth_masks

        def _normalize(*args, **kwargs):
            res = orig_normalize(*args, **kwargs)
            captured["point_map"] = np.asarray(res[1], dtype=np.float32)
            captured["poses"] = res[2][0].detach().float().cpu().numpy()
            return res

        def _valid(*args, **kwargs):
            res = orig_valid(*args, **kwargs)
            captured["valid"] = res.detach().cpu().numpy().astype(bool)
            return res

        gm.normalize_scene_scale, gm.get_valid_depth_masks = _normalize, _valid
        try:
            object_masks = sd.process_scene(file_list, n1, n2, str(out))
        finally:
            sd.cleanup()
            del sd, gm
            gc.collect()
            torch.cuda.empty_cache()

        images = sd_utils.load_images_as_tensor_from_list(file_list).permute(0, 2, 3, 1).numpy()
        return self._postprocess(
            out, raw, object_masks, captured, images, frames_1, frames_2,
            times_1, times_2, [video_1, video_2], time.time() - t_start,
        )

    # ------------------------------------------------------------------
    # Post-processing: shared scene, per-object 3D info, crops, renders
    # ------------------------------------------------------------------

    def _postprocess(self, out: Path, raw: Path, object_masks: dict, captured: dict,
                     images: np.ndarray, frames_1: List[Path], frames_2: List[Path],
                     times_1: List[float], times_2: List[float], videos: List[str],
                     runtime: float) -> SceneDiffOutput:
        n1 = len(frames_1)
        T = vis.display_frame_from_poses(captured["poses"])
        pts = vis.transform_points(T, captured["point_map"].reshape(-1, 3)).reshape(
            captured["point_map"].shape)
        valid = captured["valid"]
        rgb = (np.clip(images, 0, 1) * 255).astype(np.uint8)

        lo, hi = np.percentile(pts[valid], 2, axis=0), np.percentile(pts[valid], 98, axis=0)
        extent = float(np.linalg.norm(hi - lo))
        files: Dict[str, str] = {}

        # --- shared + per-video scene point clouds ---
        for name, sl in (("scene_shared", slice(None)), ("scene_video1", slice(0, n1)),
                         ("scene_video2", slice(n1, None))):
            xyz, col = vis.voxel_downsample(pts[sl][valid[sl]], rgb[sl][valid[sl]], extent / 1500)
            path = str(out / f"{name}.ply")
            vis.write_ply(path, xyz, col)
            files[name] = path
        scene_xyz, scene_rgb = vis.voxel_downsample(pts[valid], rgb[valid], extent / 400)

        # --- per-object geometry and crops ---
        crops_dir = out / "crops"
        crops_dir.mkdir(exist_ok=True)
        obj_ids = sorted(k for k in object_masks if k not in ("H", "W"))
        objects: List[SceneDiffObject] = []
        obj_pts_all, obj_id_all = [], []
        for new_id, key in enumerate(obj_ids):
            entry = object_masks[key]
            has1, has2 = bool(entry.get("video_1")), bool(entry.get("video_2"))
            change = "Moved" if (has1 and has2) else ("Removed" if has1 else "Added")
            color = vis.CHANGE_COLORS[change]
            obj = SceneDiffObject(id=new_id, change_type=change, frames_1=[], frames_2=[])
            costs = []
            for vid, offset, frame_paths, times in (("video_1", 0, frames_1, times_1),
                                                    ("video_2", n1, frames_2, times_2)):
                per_frame = entry.get(vid) or {}
                if not per_frame:
                    continue
                s = vid[-1]
                obj_pts = []
                for fi, rec in per_frame.items():
                    m = _decode_rle(rec["mask"]) & valid[offset + fi]
                    obj_pts.append(pts[offset + fi][m])
                    costs.append(rec.get("cost", 0.0))
                obj_pts = np.concatenate(obj_pts) if obj_pts else np.zeros((0, 3), np.float32)
                setattr(obj, f"frames_{s}", sorted(int(f) for f in per_frame))
                best = max(per_frame, key=lambda f: per_frame[f]["mask_size"])
                setattr(obj, f"best_time_{s}", round(times[best], 2))
                if len(obj_pts):
                    setattr(obj, f"centroid_{s}", np.median(obj_pts, 0).round(4).tolist())
                    setattr(obj, f"bbox_{s}", [np.percentile(obj_pts, 3, 0).round(4).tolist(),
                                               np.percentile(obj_pts, 97, 0).round(4).tolist()])
                    sel = np.random.default_rng(new_id).choice(
                        len(obj_pts), min(4000, len(obj_pts)), replace=False)
                    obj_pts_all.append(obj_pts[sel])
                    obj_id_all.append(np.full(len(sel), new_id))
                crop, ctx = _crop_and_context(frame_paths[best], _decode_rle(per_frame[best]["mask"]), color)
                setattr(obj, f"crop_{s}", crop)
                setattr(obj, f"context_{s}", ctx)
                for k, img in ((f"crop_{s}", crop), (f"context_{s}", ctx)):
                    if img is not None:
                        img.save(crops_dir / f"obj{new_id}_{k}.png")
            if obj.centroid_1 is not None and obj.centroid_2 is not None and change == "Moved":
                obj.displacement = (np.array(obj.centroid_2) - np.array(obj.centroid_1)).round(4).tolist()
            obj.score = float(np.mean(costs)) if costs else 0.0
            objects.append(obj)

        obj_xyz = np.concatenate(obj_pts_all).astype(np.float32) if obj_pts_all else np.zeros((0, 3), np.float32)
        obj_idx = np.concatenate(obj_id_all) if obj_id_all else np.zeros((0,), np.int64)
        np.savez_compressed(out / "scene_vis.npz", scene_xyz=scene_xyz.astype(np.float32),
                            scene_rgb=scene_rgb, obj_xyz=obj_xyz, obj_ids=obj_idx)
        files["scene_vis"] = str(out / "scene_vis.npz")

        # --- highlight cloud: gray scene + colored change points + box edges ---
        gray = (0.35 * scene_rgb + 0.65 * 160).astype(np.uint8)
        hl_xyz, hl_rgb = [scene_xyz], [gray]
        for o in objects:
            color = np.array(vis.CHANGE_COLORS[o.change_type], np.uint8)
            sel = obj_xyz[obj_idx == o.id]
            hl_xyz.append(sel); hl_rgb.append(np.tile(color, (len(sel), 1)))
            for bb in (o.bbox_1, o.bbox_2):
                if bb is not None:
                    e = vis.box_edge_points(bb)
                    hl_xyz.append(e); hl_rgb.append(np.tile(color, (len(e), 1)))
        vis.write_ply(str(out / "changes_highlight.ply"), np.concatenate(hl_xyz), np.concatenate(hl_rgb))
        files["changes_highlight"] = str(out / "changes_highlight.ply")

        # --- upstream SceneDiff artifacts ---
        for name in ("object_masks.pkl", "cost_map_merged.ply", "cost_map_1.ply", "cost_map_2.ply",
                     "object_instances_1.ply", "object_instances_2.ply"):
            if (raw / name).exists():
                files[f"raw_{name.rsplit('.', 1)[0]}"] = str(raw / name)

        # --- renders, changes.json, report ---
        obj_dicts = [_object_to_dict(o) for o in objects]
        files.update(vis.render_all(str(out), obj_dicts))
        payload = {
            "videos": videos,
            "num_frames": [n1, len(frames_2)],
            "frame_times_sec": [times_1, times_2],
            "scene_extent": extent,
            "coordinate_frame": "gravity-aligned, x right, y forward, z up, origin = video 1 first camera; up-to-scale units",
            "objects": obj_dicts,
            "files": files,
            "summary": "",
        }
        files["changes_json"] = str(out / "changes.json")
        files["report"] = str(out / "report.md")
        vis.write_changes_json(str(out), payload)
        vis.write_report(str(out), payload)

        return SceneDiffOutput(
            objects=objects,
            output_dir=str(out),
            files=files,
            render_topdown=Image.open(files["render_topdown"]).convert("RGB"),
            render_oblique=Image.open(files["render_oblique"]).convert("RGB"),
            num_frames=[n1, len(frames_2)],
            scene_extent=extent,
            runtime_sec=round(runtime, 1),
        )

    # ------------------------------------------------------------------
    # RPC entry point
    # ------------------------------------------------------------------

    def _run_locked(self, *args) -> SceneDiffOutput:
        with gpu_inference_lock():
            return self._run(*args)

    @AgentTool.document_output_class(SceneDiffOutput)
    async def compare(
        self,
        video_1: str,
        video_2: str,
        output_dir: str,
        resample_rate: int = 30,
        max_frames_per_video: int = 40,
        change_threshold: Optional[float] = None,
    ) -> AgentToolOutput:
        """Detect objects moved / added / removed between two videos of one scene.

        Args:
            video_1: Path to the "before" video (shared filesystem).
            video_2: Path to the "after" video.
            output_dir: Absolute directory where all artifacts are written.
            resample_rate: Keep 1 frame every N (30-fps numbering).
            max_frames_per_video: Uniform cap on frames per video.
            change_threshold: Override the automatic change-region threshold.
        """
        for v in (video_1, video_2):
            if not isinstance(v, str) or not os.path.isfile(v):
                raise ValueError(f"Video not found on the GPU server: {v!r}")
        if not isinstance(output_dir, str) or not os.path.isabs(output_dir):
            raise ValueError("output_dir must be an absolute path")
        if not (1 <= int(resample_rate) <= 300):
            raise ValueError("resample_rate must be in [1, 300]")
        if not (2 <= int(max_frames_per_video) <= 120):
            raise ValueError("max_frames_per_video must be in [2, 120]")
        os.makedirs(output_dir, exist_ok=True)

        async with self.infer_lock:
            output = await asyncio.to_thread(
                self._run_locked, video_1, video_2, output_dir, int(resample_rate),
                int(max_frames_per_video), change_threshold,
            )
        return self.success(result=output)


def _object_to_dict(o: SceneDiffObject) -> dict:
    return {
        "id": o.id, "change_type": o.change_type, "label": o.label, "description": o.description,
        "frames_1": o.frames_1, "frames_2": o.frames_2,
        "best_time_1": o.best_time_1, "best_time_2": o.best_time_2,
        "centroid_1": o.centroid_1, "centroid_2": o.centroid_2,
        "bbox_1": o.bbox_1, "bbox_2": o.bbox_2,
        "displacement": o.displacement, "score": round(o.score, 4),
    }

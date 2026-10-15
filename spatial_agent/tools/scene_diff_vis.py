"""CPU-only helpers for SceneDiff outputs: PLY writing, renders, reports.

Shared by the GPU-server model (first render after detection) and the
kernel-side tool (re-render with semantic labels after ``Annotate``).
Only numpy / PIL / matplotlib — no torch, no open3d.

All 3D coordinates handled here are in the gravity-aligned *display frame*
built by ``display_frame_from_poses``: x = right, y = forward, z = up,
origin at the first camera of video 1.
"""

import json
import os
from typing import Dict, List, Optional, Sequence

import numpy as np
from PIL import Image, ImageDraw

CHANGE_COLORS = {
    "Added": (44, 160, 44),     # green
    "Removed": (214, 39, 40),   # red
    "Moved": (255, 127, 14),    # orange
}


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------

def display_frame_from_poses(c2w: np.ndarray) -> np.ndarray:
    """Return a 4x4 world→display transform from OpenCV camera-to-world poses.

    Up is the mean camera "up" (-y axis) — phones are held roughly upright, so
    this is a good gravity estimate. Forward is the mean viewing direction
    projected onto the ground plane.
    """
    R = c2w[:, :3, :3]
    up = -R[:, :, 1].mean(0)
    up /= np.linalg.norm(up) + 1e-9
    fwd = R[:, :, 2].mean(0)
    fwd = fwd - up * np.dot(fwd, up)
    if np.linalg.norm(fwd) < 1e-6:
        fwd = np.cross(up, [1.0, 0.0, 0.0])
    fwd /= np.linalg.norm(fwd) + 1e-9
    right = np.cross(fwd, up)
    rot = np.stack([right, fwd, up], axis=0)  # rows: display axes in world coords
    T = np.eye(4)
    T[:3, :3] = rot
    T[:3, 3] = -rot @ c2w[0, :3, 3]
    return T


def transform_points(T: np.ndarray, pts: np.ndarray) -> np.ndarray:
    return pts @ T[:3, :3].T + T[:3, 3]


def voxel_downsample(xyz: np.ndarray, rgb: np.ndarray, voxel: float):
    """Keep one point per voxel (first occurrence)."""
    if len(xyz) == 0 or voxel <= 0:
        return xyz, rgb
    keys = np.floor(xyz / voxel).astype(np.int64)
    keys -= keys.min(0)
    dims = keys.max(0) + 1
    flat = (keys[:, 0] * dims[1] + keys[:, 1]) * dims[2] + keys[:, 2]  # 1-D unique is much faster
    _, idx = np.unique(flat, return_index=True)
    return xyz[idx], rgb[idx]


def box_edge_points(bbox: Sequence[Sequence[float]], n: int = 40) -> np.ndarray:
    """Densely sample the 12 edges of an axis-aligned box (for PLY viewers)."""
    lo, hi = np.asarray(bbox[0], float), np.asarray(bbox[1], float)
    corners = np.array([[x, y, z] for x in (lo[0], hi[0])
                        for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
    edges = [(0, 1), (2, 3), (4, 5), (6, 7), (0, 2), (1, 3),
             (4, 6), (5, 7), (0, 4), (1, 5), (2, 6), (3, 7)]
    t = np.linspace(0, 1, n)[:, None]
    return np.concatenate([corners[a] * (1 - t) + corners[b] * t for a, b in edges])


def write_ply(path: str, xyz: np.ndarray, rgb: np.ndarray) -> None:
    """Write a binary little-endian PLY point cloud (rgb as uint8)."""
    xyz = np.asarray(xyz, dtype="<f4").reshape(-1, 3)
    rgb = np.asarray(rgb).reshape(-1, 3)
    if rgb.dtype != np.uint8:
        rgb = np.clip(rgb * (255.0 if rgb.max(initial=0) <= 1.0 else 1.0), 0, 255).astype(np.uint8)
    vertex = np.empty(len(xyz), dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
                                       ("red", "u1"), ("green", "u1"), ("blue", "u1")])
    vertex["x"], vertex["y"], vertex["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    vertex["red"], vertex["green"], vertex["blue"] = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    header = (
        "ply\nformat binary_little_endian 1.0\n"
        f"element vertex {len(xyz)}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        "end_header\n"
    )
    with open(path, "wb") as f:
        f.write(header.encode("ascii"))
        f.write(vertex.tobytes())


# ---------------------------------------------------------------------------
# Renders
# ---------------------------------------------------------------------------

def _object_title(obj: dict) -> str:
    label = obj.get("label") or ""
    base = f"#{obj['id']} {obj['change_type']}"
    return f"{base}: {label}" if label else base


def _fig_to_pil(fig) -> Image.Image:
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    return Image.fromarray(np.asarray(canvas.buffer_rgba())[..., :3].copy())


def _subsample(xyz, rgb, max_points, seed=0):
    if len(xyz) <= max_points:
        return xyz, rgb
    idx = np.random.default_rng(seed).choice(len(xyz), max_points, replace=False)
    return xyz[idx], rgb[idx]


def _legend_handles():
    from matplotlib.lines import Line2D
    return [Line2D([0], [0], marker="s", color="w", markersize=9,
                   markerfacecolor=np.array(c) / 255.0, label=k)
            for k, c in CHANGE_COLORS.items()]


def render_topdown(scene_xyz, scene_rgb, obj_xyz, obj_ids, objects: List[dict],
                   max_points: int = 120_000) -> Image.Image:
    """Bird's-eye view (x right, y forward) with change points, boxes, arrows."""
    from matplotlib.figure import Figure
    from matplotlib.patches import FancyArrowPatch, Rectangle

    xyz, rgb = _subsample(scene_xyz, scene_rgb, max_points)
    order = np.argsort(xyz[:, 2])  # draw low points first so tops stay visible
    xyz, rgb = xyz[order], rgb[order]

    fig = Figure(figsize=(10, 10), dpi=110)
    ax = fig.add_subplot(111)
    ax.scatter(xyz[:, 0], xyz[:, 1], c=rgb / 255.0, s=0.6, linewidths=0, alpha=0.55)

    for obj in objects:
        color = np.array(CHANGE_COLORS.get(obj["change_type"], (0, 0, 255))) / 255.0
        pts = obj_xyz[obj_ids == obj["id"]]
        if len(pts):
            ax.scatter(pts[:, 0], pts[:, 1], color=color, s=2.0, linewidths=0, alpha=0.9)
        for key, style in (("bbox_1", "--"), ("bbox_2", "-")):
            bb = obj.get(key)
            if bb is None:
                continue
            if obj["change_type"] != "Moved":
                style = "-"
            (x0, y0, _), (x1, y1, _) = bb
            ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False,
                                   edgecolor=color, linestyle=style, linewidth=2))
        c1, c2 = obj.get("centroid_1"), obj.get("centroid_2")
        if obj["change_type"] == "Moved" and c1 is not None and c2 is not None:
            ax.add_patch(FancyArrowPatch((c1[0], c1[1]), (c2[0], c2[1]),
                                         arrowstyle="-|>", mutation_scale=20,
                                         color=color, linewidth=2))
        anchor = c2 if c2 is not None else c1
        if anchor is not None:
            ax.text(anchor[0], anchor[1], _object_title(obj), fontsize=10, weight="bold",
                    color="black", ha="center", va="bottom",
                    bbox=dict(facecolor="white", alpha=0.8, edgecolor=color, pad=2))

    ax.plot([0], [0], marker="^", color="blue", markersize=10)
    ax.text(0, 0, " cam1 start", color="blue", fontsize=8, va="top")
    ax.set_aspect("equal")
    ax.set_xlabel("x (right)")
    ax.set_ylabel("y (forward)")
    ax.set_title("SceneDiff — top-down view (dashed = old position of moved object)")
    ax.legend(handles=_legend_handles(), loc="upper right")
    ax.grid(alpha=0.2)
    return _fig_to_pil(fig)


def render_oblique(scene_xyz, scene_rgb, obj_xyz, obj_ids, objects: List[dict],
                   max_points: int = 60_000, elev: float = 35, azim: float = -65) -> Image.Image:
    """3D perspective view with wireframe boxes and labels."""
    from matplotlib.figure import Figure
    import mpl_toolkits.mplot3d  # noqa: F401  (registers the 3d projection)

    xyz, rgb = _subsample(scene_xyz, scene_rgb, max_points)
    fig = Figure(figsize=(11, 9), dpi=110)
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], c=rgb / 255.0, s=0.4, linewidths=0,
               alpha=0.45, depthshade=False)

    for obj in objects:
        color = np.array(CHANGE_COLORS.get(obj["change_type"], (0, 0, 255))) / 255.0
        pts = obj_xyz[obj_ids == obj["id"]]
        if len(pts):
            ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], color=color, s=2.0,
                       linewidths=0, depthshade=False)
        for key in ("bbox_1", "bbox_2"):
            bb = obj.get(key)
            if bb is None:
                continue
            style = "--" if (key == "bbox_1" and obj["change_type"] == "Moved") else "-"
            lo, hi = np.asarray(bb[0]), np.asarray(bb[1])
            c = np.array([[x, y, z] for x in (lo[0], hi[0])
                          for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
            for a, b in [(0, 1), (2, 3), (4, 5), (6, 7), (0, 2), (1, 3),
                         (4, 6), (5, 7), (0, 4), (1, 5), (2, 6), (3, 7)]:
                ax.plot(*zip(c[a], c[b]), color=color, linestyle=style, linewidth=1.5)
        c1, c2 = obj.get("centroid_1"), obj.get("centroid_2")
        if obj["change_type"] == "Moved" and c1 is not None and c2 is not None:
            ax.plot(*zip(c1, c2), color=color, linewidth=2.5)
        anchor = c2 if c2 is not None else c1
        if anchor is not None:
            top = obj.get("bbox_2") or obj.get("bbox_1")
            z = top[1][2] if top else anchor[2]
            ax.text(anchor[0], anchor[1], z, _object_title(obj), fontsize=9,
                    weight="bold", color="black")

    lo, hi = np.percentile(xyz, 1, axis=0), np.percentile(xyz, 99, axis=0)
    ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1]); ax.set_zlim(lo[2], hi[2])
    ax.set_box_aspect(np.maximum(hi - lo, 1e-3))
    ax.view_init(elev=elev, azim=azim)
    ax.set_xlabel("x (right)"); ax.set_ylabel("y (forward)"); ax.set_zlabel("z (up)")
    ax.set_title("SceneDiff — shared 3D scene with detected changes")
    ax.legend(handles=_legend_handles(), loc="upper left")
    return _fig_to_pil(fig)


def object_panel(obj: dict, crops: Dict[str, Optional[Image.Image]],
                 height: int = 300) -> Image.Image:
    """[context v1 | crop v1 | crop v2 | context v2] strip with a title bar."""
    tiles = []
    for key, cap in (("context_1", "video 1 (frame)"), ("crop_1", "video 1 (crop)"),
                     ("crop_2", "video 2 (crop)"), ("context_2", "video 2 (frame)")):
        img = crops.get(key)
        if img is None:
            img = Image.new("RGB", (height, height), (235, 235, 235))
            ImageDraw.Draw(img).text((10, height // 2), "not visible / absent", fill=(90, 90, 90))
        else:
            img = img.convert("RGB")
            img = img.resize((max(1, int(img.width * height / img.height)), height))
        tile = Image.new("RGB", (img.width, height + 18), (255, 255, 255))
        tile.paste(img, (0, 18))
        ImageDraw.Draw(tile).text((4, 3), cap, fill=(0, 0, 0))
        tiles.append(tile)
    width = sum(t.width for t in tiles) + 6 * (len(tiles) - 1)
    color = CHANGE_COLORS.get(obj["change_type"], (0, 0, 255))
    panel = Image.new("RGB", (width, height + 18 + 26), (255, 255, 255))
    ImageDraw.Draw(panel).rectangle([0, 0, width, 24], fill=color)
    ImageDraw.Draw(panel).text((6, 6), _object_title(obj), fill=(255, 255, 255))
    x = 0
    for t in tiles:
        panel.paste(t, (x, 26))
        x += t.width + 6
    return panel


def render_crop_grid(objects: List[dict], crop_lookup) -> Optional[Image.Image]:
    """Stack ``object_panel`` strips for all objects; ``crop_lookup(id)`` → dict."""
    panels = [object_panel(o, crop_lookup(o["id"])) for o in objects]
    if not panels:
        return None
    width = max(p.width for p in panels)
    grid = Image.new("RGB", (width, sum(p.height + 8 for p in panels)), (255, 255, 255))
    y = 0
    for p in panels:
        grid.paste(p, (0, y))
        y += p.height + 8
    return grid


# ---------------------------------------------------------------------------
# Persisted artifacts (changes.json / scene_vis.npz / report.md)
# ---------------------------------------------------------------------------

def load_vis_data(output_dir: str):
    data = np.load(os.path.join(output_dir, "scene_vis.npz"))
    return data["scene_xyz"], data["scene_rgb"], data["obj_xyz"], data["obj_ids"]


def load_crops(output_dir: str, obj_id: int) -> Dict[str, Optional[Image.Image]]:
    out = {}
    for key in ("context_1", "crop_1", "crop_2", "context_2"):
        p = os.path.join(output_dir, "crops", f"obj{obj_id}_{key}.png")
        out[key] = Image.open(p).convert("RGB") if os.path.exists(p) else None
    return out


def render_all(output_dir: str, objects: List[dict]) -> Dict[str, str]:
    """(Re)render the top-down / oblique / crop-grid PNGs into ``output_dir``."""
    scene_xyz, scene_rgb, obj_xyz, obj_ids = load_vis_data(output_dir)
    paths = {
        "render_topdown": os.path.join(output_dir, "render_topdown.png"),
        "render_oblique": os.path.join(output_dir, "render_oblique.png"),
        "render_crops": os.path.join(output_dir, "render_crops.png"),
    }
    render_topdown(scene_xyz, scene_rgb, obj_xyz, obj_ids, objects).save(paths["render_topdown"])
    render_oblique(scene_xyz, scene_rgb, obj_xyz, obj_ids, objects).save(paths["render_oblique"])
    grid = render_crop_grid(objects, lambda i: load_crops(output_dir, i))
    if grid is not None:
        grid.save(paths["render_crops"])
    else:
        paths.pop("render_crops")
    return paths


def _fmt_vec(v) -> str:
    return "—" if v is None else "(" + ", ".join(f"{x:+.2f}" for x in v) + ")"


def write_report(output_dir: str, payload: dict) -> str:
    """Write ``report.md`` summarizing the changes and linking every artifact."""
    objects = payload.get("objects", [])
    extent = payload.get("scene_extent") or 1.0
    lines = ["# SceneDiff change report", ""]
    videos = payload.get("videos") or []
    if videos:
        lines += [f"- Video 1 (before): `{videos[0]}`", f"- Video 2 (after): `{videos[1]}`"]
    lines += [
        f"- Frames used: {payload.get('num_frames')}",
        f"- Scene extent (robust diameter): {extent:.2f} scene units. Coordinates are "
        "gravity-aligned (x right, y forward, z up) relative to video 1's first camera; "
        "the reconstruction is up-to-scale, so distances are reported in scene units "
        "and as a % of the scene extent.",
        "",
    ]
    if payload.get("summary"):
        lines += ["## Summary", "", payload["summary"], ""]
    lines += ["## Changes", ""]
    if not objects:
        lines.append("No changed objects were detected.")
    else:
        lines += ["| id | change | label | position (before) | position (after) | moved by | description |",
                  "|---|---|---|---|---|---|---|"]
        for o in objects:
            moved = ""
            if o.get("displacement") is not None:
                d = float(np.linalg.norm(o["displacement"]))
                moved = f"{d:.2f} ({100 * d / extent:.0f}% of scene)"
            lines.append(
                f"| {o['id']} | {o['change_type']} | {o.get('label') or ''} | "
                f"{_fmt_vec(o.get('centroid_1'))} | {_fmt_vec(o.get('centroid_2'))} | "
                f"{moved} | {(o.get('description') or '').replace('|', '/')} |"
            )
    lines += ["", "## Files", ""]
    for name, path in sorted((payload.get("files") or {}).items()):
        lines.append(f"- **{name}**: `{os.path.relpath(path, output_dir)}`")
    lines += ["", "![top-down](render_topdown.png)", "", "![3D](render_oblique.png)", "",
              "![crops](render_crops.png)", ""]
    path = os.path.join(output_dir, "report.md")
    with open(path, "w") as f:
        f.write("\n".join(lines))
    return path


def write_changes_json(output_dir: str, payload: dict) -> str:
    path = os.path.join(output_dir, "changes.json")
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    return path


def read_changes_json(output_dir: str) -> dict:
    with open(os.path.join(output_dir, "changes.json")) as f:
        return json.load(f)

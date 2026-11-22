"""Render qualitative OOF comparisons from saved images and masks on CPU."""

import json
import math
from itertools import pairwise
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

from prostate.data import CLASS_NAMES, NUM_CLASSES, open_image

DISPLAY_COLORS = np.array(
    [[230, 230, 230], [0, 0, 0], [0, 0, 255], [0, 255, 0], [255, 0, 0]],
    dtype=np.uint8,
)
LEGEND_NAMES = ("non-gland", "non-cancer", "GP3", "GP4", "GP5")
MARGIN = 16
GAP = 12


def _score(value) -> str:
    return "n/a" if value is None else f"{value:.3f}"


def _mask_preview(labels: np.ndarray, size: tuple[int, int]) -> Image.Image:
    image = Image.fromarray(np.asarray(labels, dtype=np.uint8))
    small = np.asarray(image.resize(size, Image.Resampling.NEAREST))
    return Image.fromarray(DISPLAY_COLORS[small])


def _error_density(
    target: np.ndarray, prediction: np.ndarray, size: tuple[int, int]
) -> Image.Image:
    """Average full-resolution mistakes into display bins using bounded-memory strips."""
    if target.ndim != 2 or target.shape != prediction.shape:
        raise ValueError("Target and prediction must have matching 2-D shapes")
    if not all(np.issubdtype(array.dtype, np.integer) for array in (target, prediction)):
        raise ValueError("Masks must contain integer class IDs")
    height, width = target.shape
    display_width, display_height = size
    if not (0 < display_width <= width and 0 < display_height <= height):
        raise ValueError("Preview must fit within the source image")
    x_edges = np.linspace(0, width, display_width + 1, dtype=np.int64)
    y_edges = np.linspace(0, height, display_height + 1, dtype=np.int64)
    density = np.empty((display_height, display_width), dtype=np.float32)
    for row, (top, bottom) in enumerate(pairwise(y_edges)):
        truth = np.asarray(target[top:bottom])
        predicted = np.asarray(prediction[top:bottom])
        if (
            min(truth.min(), predicted.min()) < 0
            or max(truth.max(), predicted.max()) >= NUM_CLASSES
        ):
            raise ValueError("Masks must contain class IDs 0..4")
        column_errors = np.count_nonzero(truth != predicted, axis=0)
        errors = np.add.reduceat(column_errors, x_edges[:-1])
        area = (bottom - top) * np.diff(x_edges)
        density[row] = errors / area
    shade = np.rint(255 * (1 - density)).astype(np.uint8)
    return Image.fromarray(np.stack((np.full_like(shade, 255), shade, shade), axis=-1))


def _overlay(rgb: Image.Image, mask: Image.Image, labels: np.ndarray) -> Image.Image:
    foreground = (
        np.asarray(
            Image.fromarray(np.asarray(labels, dtype=np.uint8)).resize(
                rgb.size, Image.Resampling.NEAREST
            )
        )
        != 0
    )
    alpha = Image.fromarray((foreground * 115).astype(np.uint8))
    return Image.composite(mask, rgb, alpha)


def _views(sample: dict, prediction_path: Path, side: int) -> list[Image.Image]:
    target = np.load(sample["label_cache"], mmap_mode="r")
    prediction = np.load(prediction_path, mmap_mode="r")
    expected = (sample["height"], sample["width"])
    if target.shape != expected or prediction.shape != expected:
        raise ValueError(f"Mask geometry mismatch: {sample['image_id']}")
    with open_image(sample["image_path"]) as source:
        if source.size != expected[::-1]:
            raise ValueError(f"Source geometry mismatch: {sample['image_id']}")
        # Resize before converting so JPEG's reduced-resolution decoder can be used.
        source.thumbnail((side, side), Image.Resampling.LANCZOS, reducing_gap=3.0)
        rgb = source.convert("RGB")
    error = _error_density(target, prediction, rgb.size)
    truth = _mask_preview(target, rgb.size)
    predicted = _mask_preview(prediction, rgb.size)
    return [
        rgb,
        truth,
        predicted,
        _overlay(rgb, truth, target),
        _overlay(rgb, predicted, prediction),
        error,
    ]


def _legend(draw: ImageDraw.ImageDraw, top: int, scale: float = 1.0):
    font = ImageFont.load_default(size=round(13 * scale))
    left = margin = round(MARGIN * scale)
    swatch = round(12 * scale)
    for color, name in zip(DISPLAY_COLORS, LEGEND_NAMES, strict=True):
        draw.rectangle(
            (left, top, left + swatch, top + swatch),
            fill=tuple(int(c) for c in color),
            outline=(100, 100, 100),
        )
        draw.text((left + round(18 * scale), top - 1), name, fill="black", font=font)
        left += int(draw.textlength(name, font=font)) + round(38 * scale)
    draw.text(
        (margin, top + round(22 * scale)),
        "Error density: white = 0%, red = 100% wrong pixels",
        fill=(70, 70, 70),
        font=font,
    )


def _paste_panel(
    canvas: Image.Image,
    image: Image.Image,
    left: int,
    top: int,
    side: int,
    height: int | None = None,
):
    height = side if height is None else height
    canvas.paste(image, (left + (side - image.width) // 2, top + (height - image.height) // 2))


def _save_jpeg(image: Image.Image, path: Path, quality: int):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".partial.jpg")
    image.convert("RGB").save(
        temporary, format="JPEG", quality=quality, subsampling=0, optimize=True
    )
    temporary.replace(path)


def _save_card(
    views: list[Image.Image], record: dict, run_name: str, path: Path, side: int, quality: int
):
    scale = max(1.0, side / 384)
    margin, gap = round(MARGIN * scale), round(GAP * scale)
    panel_height = max(view.height for view in views)
    width = 2 * margin + 3 * side + 2 * gap
    height = 2 * panel_height + round(182 * scale)
    card = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(card)
    title_font = ImageFont.load_default(size=round(17 * scale))
    font = ImageFont.load_default(size=round(13 * scale))
    draw.text(
        (margin, round(12 * scale)),
        f"{run_name} | {record['image_id']} | fold {record['fold']}",
        fill="black",
        font=title_font,
    )
    draw.text(
        (margin, round(37 * scale)),
        f"Foreground macro Dice: {_score(record.get('foreground_macro_dice'))}",
        fill=(60, 60, 60),
        font=font,
    )
    names = (
        "Input JPEG",
        "Ground truth",
        "OOF prediction",
        "GT overlay",
        "OOF overlay",
        "Error density",
    )
    for index, (view, name) in enumerate(zip(views, names, strict=True)):
        row, col = divmod(index, 3)
        left = margin + col * (side + gap)
        top = round(82 * scale) + row * (panel_height + round(31 * scale))
        draw.text((left, top - round(21 * scale)), name, fill="black", font=font)
        _paste_panel(card, view, left, top, side, panel_height)
    _legend(draw, height - round(56 * scale), scale)
    _save_jpeg(card, path, quality)


def _save_sheet(
    rows: list[tuple[dict, list[Image.Image]]],
    run_name: str,
    path: Path,
    side: int,
    page: int,
    quality: int,
):
    scale = max(1.0, side / 192)
    margin, gap = round(MARGIN * scale), round(GAP * scale)
    width = 2 * margin + 4 * side + 3 * gap
    panel_heights = [max(view.height for view in views) for _, views in rows]
    height = round(136 * scale) + sum(h + round(47 * scale) for h in panel_heights) + margin
    sheet = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default(size=round(13 * scale))
    draw.text(
        (margin, round(12 * scale)),
        f"{run_name} | OOF thumbnails | page {page}",
        fill="black",
        font=ImageFont.load_default(size=round(17 * scale)),
    )
    _legend(draw, round(40 * scale), scale)
    for col, name in enumerate(("Input JPEG", "Ground truth", "OOF prediction", "Error density")):
        draw.text((margin + col * (side + gap), round(94 * scale)), name, fill="black", font=font)
    top = round(136 * scale)
    for (record, views), panel_height in zip(rows, panel_heights, strict=True):
        score = _score(record.get("foreground_macro_dice"))
        draw.text(
            (margin, top - round(20 * scale)),
            f"{record['image_id']} | fold {record['fold']} | FG Dice {score}",
            fill="black",
            font=font,
        )
        for col, view in enumerate(views):
            _paste_panel(sheet, view, margin + col * (side + gap), top, side, panel_height)
        top += panel_height + round(47 * scale)
    _save_jpeg(sheet, path, quality)


def _select_records(
    records: list[dict], sort: str, limit: int | None, image_ids: list[str] | None
) -> list[dict]:
    available = {record["image_id"] for record in records}
    if len(available) != len(records):
        raise ValueError("Duplicate image IDs in OOF evaluation")
    if image_ids:
        unknown = set(image_ids) - available
        if unknown:
            raise ValueError(f"No saved OOF evaluation for: {', '.join(sorted(unknown))}")
        records = [record for record in records if record["image_id"] in image_ids]

    def key(record):
        value = record.get("foreground_macro_dice")
        missing = value is None or not math.isfinite(value)
        score = 0.0 if missing else value
        return missing, score if sort == "worst" else -score, record["image_id"]

    records = sorted(records, key=(lambda r: r["image_id"]) if sort == "id" else key)
    return records if limit is None else records[:limit]


def generate_thumbnails(
    run_dir: Path,
    output_dir: Path | None = None,
    max_side: int = 768,
    per_page: int = 8,
    sort: str = "worst",
    limit: int | None = None,
    image_ids: list[str] | None = None,
    contact_side: int = 320,
    jpeg_quality: int = 95,
) -> dict:
    """Export JPEG comparisons with 4:4:4 chroma and a JSON index."""
    if not 128 <= max_side <= 2048:
        raise ValueError("max_side must be between 128 and 2048")
    if not 1 <= per_page <= 20 or (limit is not None and limit < 1):
        raise ValueError("per_page must be in 1..20 and limit must be positive")
    if sort not in ("worst", "best", "id"):
        raise ValueError("sort must be worst, best or id")
    if not 128 <= contact_side <= 2048 or not 1 <= jpeg_quality <= 100:
        raise ValueError("contact_side must be in 128..2048 and jpeg_quality in 1..100")
    manifest = json.loads((run_dir / "manifest.json").read_text())
    if tuple(manifest["classes"]) != CLASS_NAMES:
        raise ValueError("Manifest class order does not match the display legend")
    samples = {sample["image_id"]: sample for sample in manifest["samples"]}
    records = _select_records(
        json.loads((run_dir / "oof_images.json").read_text()), sort, limit, image_ids
    )
    if not records:
        raise ValueError("No completed OOF images are available")
    for record in records:
        image_id = record["image_id"]
        if Path(image_id).name != image_id or image_id not in samples:
            raise ValueError(f"Invalid OOF image ID: {image_id}")
        if not (run_dir / "oof" / f"{image_id}.npy").is_file():
            raise FileNotFoundError(f"Missing OOF prediction: {image_id}")
    destination = output_dir if output_dir is not None else run_dir / "thumbnails"
    destination.mkdir(parents=True, exist_ok=True)
    index = {
        "run_dir": str(run_dir.resolve()),
        "sort": sort,
        "images_generated": len(records),
        "max_side": max_side,
        "contact_side": min(contact_side, max_side),
        "format": "JPEG",
        "jpeg_quality": jpeg_quality,
        "chroma_subsampling": "4:4:4",
        "legend": {
            name: color.tolist() for name, color in zip(CLASS_NAMES, DISPLAY_COLORS, strict=True)
        },
        "error_view": "Full-resolution error fraction in each display bin; white=0%, red=100%",
        "contact_sheets": [],
        "images": [],
    }
    rows = []
    contact_side = min(contact_side, max_side)
    for position, record in enumerate(tqdm(records, desc="Rendering OOF thumbnails"), start=1):
        image_id = record["image_id"]
        sample = samples[image_id]
        prediction = run_dir / "oof" / f"{image_id}.npy"
        views = _views(sample, prediction, max_side)
        card_path = Path("images") / f"{image_id}.jpg"
        _save_card(views, record, run_dir.name, destination / card_path, max_side, jpeg_quality)
        small_views = _views(sample, prediction, contact_side)
        rows.append((record, [small_views[i] for i in (0, 1, 2, 5)]))
        index["images"].append(
            {
                "image_id": image_id,
                "fold": record["fold"],
                "foreground_macro_dice": record.get("foreground_macro_dice"),
                "file": card_path.as_posix(),
            }
        )
        if len(rows) == per_page or position == len(records):
            page = len(index["contact_sheets"]) + 1
            path = f"sheet_{page:03d}.jpg"
            _save_sheet(rows, run_dir.name, destination / path, contact_side, page, jpeg_quality)
            index["contact_sheets"].append(path)
            rows = []
    temporary = destination / "index.partial.json"
    temporary.write_text(json.dumps(index, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(destination / "index.json")
    return {
        "output_dir": str(destination.resolve()),
        "images_generated": len(records),
        "contact_sheets": index["contact_sheets"],
    }

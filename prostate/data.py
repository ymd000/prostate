"""Image pairing, legacy color labels and disk-backed random training patches."""

import hashlib
import json
import warnings
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from tqdm import tqdm

CLASS_NAMES = ("non_gland", "non_cancer", "GP3", "GP4", "GP5")
NUM_CLASSES = len(CLASS_NAMES)
IGNORE_INDEX = 255
# RGB nonzero bits (R=1, G=2, B=4), equivalent to __archive/formula.py.
RGB_TO_CLASS = np.array([1, 4, 3, 3, 2, 1, 2, 0], dtype=np.uint8)
CACHE_VERSION = 1


@dataclass(frozen=True)
class Sample:
    image_id: str
    case_id: str
    image_path: str
    annotation_path: str
    rgb_cache: str
    label_cache: str
    width: int
    height: int
    class_counts: tuple[int, ...]
    fingerprint: dict

    def to_dict(self) -> dict:
        return asdict(self)


def open_image(path: str | Path) -> Image.Image:
    # The known dataset includes a legitimate 10000 x 10000 image. Keep Pillow's
    # hard error limit; suppress only its advisory warning within this opener.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", Image.DecompressionBombWarning)
        return Image.open(path)


def decode_annotation(rgba: np.ndarray) -> np.ndarray:
    """Decode painted RGB colors; photographic pixels with all channels >0 are class 0."""
    if rgba.ndim != 3 or rgba.shape[2] not in (3, 4):
        raise ValueError("Expected an RGB or RGBA color annotation")
    bits = (rgba[..., :3] > 0).astype(np.uint8)
    codes = bits[..., 0] + 2 * bits[..., 1] + 4 * bits[..., 2]
    labels = RGB_TO_CLASS[codes]
    if rgba.shape[2] == 4:
        labels[rgba[..., 3] == 0] = 0
    return labels


def discover_pairs(data_dir: Path) -> list[tuple[Path, Path]]:
    if not data_dir.is_dir():
        raise ValueError(f"Dataset directory is unavailable: {data_dir}")
    images = {}
    annotations = {}
    for path in sorted(data_dir.iterdir()):
        if not path.is_file() or path.name.startswith("."):
            continue
        if path.suffix.lower() in (".jpg", ".jpeg"):
            if path.stem in images:
                raise ValueError(f"Duplicate JPEG identifier: {path.stem}")
            images[path.stem] = path
        elif path.suffix.lower() == ".png":
            if path.stem in annotations:
                raise ValueError(f"Duplicate PNG identifier: {path.stem}")
            annotations[path.stem] = path
    if images.keys() != annotations.keys():
        raise ValueError(
            f"Unpaired files: JPEG only={sorted(images.keys() - annotations.keys())}; "
            f"PNG only={sorted(annotations.keys() - images.keys())}"
        )
    if not images:
        raise ValueError(f"No JPEG/PNG pairs in {data_dir}")
    return [(images[key], annotations[key]) for key in sorted(images)]


def _fingerprint(image_path: Path, annotation_path: Path) -> dict:
    result = {"cache_version": CACHE_VERSION}
    for name, path in (("image", image_path), ("annotation", annotation_path)):
        stat = path.stat()
        result[name] = {
            "path": str(path.resolve()),
            "bytes": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
        }
    return result


def _cache_pair(image_path: Path, annotation_path: Path, cache_dir: Path) -> Sample:
    image_id = image_path.stem
    case_id = image_id.split("_", maxsplit=1)[0]
    if not case_id or "_" not in image_id:
        raise ValueError(f"Cannot infer case ID from filename: {image_path.name}")
    fingerprint = _fingerprint(image_path, annotation_path)
    cache_key = hashlib.sha256(json.dumps(fingerprint, sort_keys=True).encode()).hexdigest()[:16]
    destination = cache_dir.resolve() / image_id / cache_key
    destination.mkdir(parents=True, exist_ok=True)
    metadata_path = destination / "sample.json"
    rgb_path = destination / "rgb.npy"
    label_path = destination / "labels.npy"
    if metadata_path.exists() and rgb_path.exists() and label_path.exists():
        metadata = json.loads(metadata_path.read_text())
        if metadata.get("fingerprint") == fingerprint:
            metadata["class_counts"] = tuple(metadata["class_counts"])
            return Sample(**metadata)
    with open_image(image_path) as image, open_image(annotation_path) as annotation:
        if image.size != annotation.size:
            raise ValueError(f"Image/mask size mismatch: {image_id}")
        if annotation.mode not in ("RGB", "RGBA", "P"):
            raise ValueError(f"Expected a color annotation for {image_id}: {annotation.mode}")
        width, height = image.size
        rgb_temp = destination / "rgb.partial.npy"
        label_temp = destination / "labels.partial.npy"
        rgb = np.lib.format.open_memmap(
            rgb_temp, mode="w+", dtype=np.uint8, shape=(height, width, 3)
        )
        labels = np.lib.format.open_memmap(
            label_temp, mode="w+", dtype=np.uint8, shape=(height, width)
        )
        counts = np.zeros(NUM_CLASSES, dtype=np.int64)
        for top in range(0, height, 256):
            bottom = min(top + 256, height)
            box = (0, top, width, bottom)
            rgb[top:bottom] = np.asarray(image.crop(box).convert("RGB"))
            strip = decode_annotation(np.asarray(annotation.crop(box).convert("RGBA")))
            labels[top:bottom] = strip
            counts += np.bincount(strip.ravel(), minlength=NUM_CLASSES)
        rgb.flush()
        labels.flush()
        del rgb, labels
        rgb_temp.replace(rgb_path)
        label_temp.replace(label_path)
    sample = Sample(
        image_id,
        case_id,
        str(image_path.resolve()),
        str(annotation_path.resolve()),
        str(rgb_path),
        str(label_path),
        width,
        height,
        tuple(int(value) for value in counts),
        fingerprint,
    )
    temporary = metadata_path.with_suffix(".tmp")
    temporary.write_text(json.dumps(sample.to_dict(), indent=2) + "\n")
    temporary.replace(metadata_path)
    return sample


def prepare_dataset(data_dir: Path, cache_dir: Path) -> list[Sample]:
    pairs = discover_pairs(data_dir)
    return [
        _cache_pair(image, annotation, cache_dir)
        for image, annotation in tqdm(pairs, desc="Preparing image cache")
    ]


@lru_cache(maxsize=8)
def cached_arrays(rgb_path: str, label_path: str) -> tuple[np.ndarray, np.ndarray]:
    return np.load(rgb_path, mmap_mode="r"), np.load(label_path, mmap_mode="r")


def image_tensor(rgb: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(rgb.transpose(2, 0, 1))).float().div_(255)


def read_tile(rgb: np.ndarray, top: int, left: int, size: int) -> np.ndarray:
    """Read a fixed-size tile, reflecting context at the image boundaries."""
    height, width = rgb.shape[:2]
    bottom, right = top + size, left + size
    y0, y1 = max(0, top), min(height, bottom)
    x0, x1 = max(0, left), min(width, right)
    if y0 >= y1 or x0 >= x1:
        raise ValueError("Tile does not intersect the image")
    tile = np.asarray(rgb[y0:y1, x0:x1])
    padding = ((y0 - top, bottom - y1), (x0 - left, right - x1), (0, 0))
    if any(before or after for before, after in padding):
        mode = "reflect" if min(tile.shape[:2]) > 1 else "edge"
        tile = np.pad(tile, padding, mode=mode)
    return tile


class PatchDataset(Dataset):
    """Sample cases uniformly, then images and positions, without loading all images."""

    def __init__(self, samples: list[Sample], patch_size: int, length: int, seed: int):
        self.groups = {}
        for sample in samples:
            self.groups.setdefault(sample.case_id, []).append(sample)
        self.case_ids = sorted(self.groups)
        if not self.case_ids:
            raise ValueError("Training requires at least one case")
        self.patch_size = patch_size
        self.length = length
        self.seed = seed
        self.epoch = 0

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, self.epoch, index]))
        case = self.case_ids[int(rng.integers(len(self.case_ids)))]
        samples = self.groups[case]
        sample = samples[int(rng.integers(len(samples)))]
        rgb, labels = cached_arrays(sample.rgb_cache, sample.label_cache)
        size = self.patch_size
        top = int(rng.integers(max(sample.height - size, 0) + 1))
        left = int(rng.integers(max(sample.width - size, 0) + 1))
        image = read_tile(rgb, top, left, size)
        target = np.asarray(labels[top : top + size, left : left + size])
        target = np.pad(
            target,
            ((0, size - target.shape[0]), (0, size - target.shape[1])),
            constant_values=IGNORE_INDEX,
        )
        rotation = int(rng.integers(4))
        image, target = np.rot90(image, rotation), np.rot90(target, rotation)
        if rng.random() < 0.5:
            image, target = image[:, ::-1], target[:, ::-1]
        return image_tensor(image), torch.from_numpy(np.ascontiguousarray(target)).long()

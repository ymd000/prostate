"""Historical area-weighted random crops and BGR/ImageNet input preprocessing."""

from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from prostate.config import ArchiveConfig
from prostate.data import Sample, cached_arrays, open_image

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def archive_image_tensor(rgb: np.ndarray) -> torch.Tensor:
    """Match cv2.imread's BGR order, including the archived normalization order."""
    bgr = np.ascontiguousarray(rgb[..., ::-1].transpose(2, 0, 1))
    image = torch.from_numpy(bgr).float().div_(255)
    mean = image.new_tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = image.new_tensor(IMAGENET_STD).view(3, 1, 1)
    return (image - mean) / std


def _cache_annotation_nonzero(sample: Sample) -> Path:
    # Label 0 also represents opaque white tissue, which the old sampler accepted.
    # Preserve the raw PNG test rather than treating class 0 as an empty crop.
    path = Path(sample.label_cache).with_name("archive_annotation_nonzero_v1.npy")
    if path.exists():
        return path
    temporary = path.with_suffix(".partial.npy")
    mask = np.lib.format.open_memmap(
        temporary, mode="w+", dtype=np.bool_, shape=(sample.height, sample.width)
    )
    with open_image(sample.annotation_path) as annotation:
        for top in range(0, sample.height, 256):
            bottom = min(top + 256, sample.height)
            rgba = np.asarray(annotation.crop((0, top, sample.width, bottom)).convert("RGBA"))
            mask[top:bottom] = np.any(rgba != 0, axis=2)
    mask.flush()
    del mask
    temporary.replace(path)
    return path


@lru_cache(maxsize=8)
def _nonzero_mask(path: str) -> np.ndarray:
    return np.load(path, mmap_mode="r")


class ArchivePatchDataset(Dataset):
    def __init__(self, samples: list[Sample], config: ArchiveConfig):
        if not samples:
            raise ValueError("Archive training requires at least one image")
        self.samples = samples
        self.config = config
        self.epoch = 0
        side = config.patch_size
        if any(min(s.height, s.width) <= side for s in samples):
            raise ValueError("Archive random crops require both image dimensions > patch_size")
        weights = np.array([np.sqrt((s.height - side) * (s.width - side)) for s in samples])
        self.probabilities = weights / weights.sum()
        self.mask_paths = [str(_cache_annotation_nonzero(sample)) for sample in samples]
        if not any(np.any(_nonzero_mask(path)) for path in self.mask_paths):
            raise ValueError("No nonzero annotations are available for archive random crops")
        self.length = config.samples_multiplier * sum(
            int((sample.height / side) * (sample.width / side)) for sample in samples
        )

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        rng = np.random.default_rng(np.random.SeedSequence([self.config.seed, self.epoch, index]))
        side = self.config.patch_size
        step = self.config.crop_start_step
        for _ in range(self.config.max_sampling_attempts):
            number = int(rng.choice(len(self.samples), p=self.probabilities))
            sample = self.samples[number]
            # Draw uniformly from grid-aligned starts within the old exclusive
            # bounds. Step 1 preserves the original RNG calls and crop sequence.
            left = int(rng.integers((sample.width - side - 1) // step + 1)) * step
            top = int(rng.integers((sample.height - side - 1) // step + 1)) * step
            if np.any(_nonzero_mask(self.mask_paths[number])[top : top + side, left : left + side]):
                break
        else:
            raise ValueError("Could not draw a nonzero archive crop; inspect the training masks")
        rgb, labels = cached_arrays(sample.rgb_cache, sample.label_cache)
        image = np.asarray(rgb[top : top + side, left : left + side])
        target = np.asarray(labels[top : top + side, left : left + side])
        operation = int(rng.integers(8))
        if operation > 3:
            image, target = np.flip(image, axis=0), np.flip(target, axis=0)
        image, target = np.rot90(image, operation % 4), np.rot90(target, operation % 4)
        return archive_image_tensor(image), torch.from_numpy(np.ascontiguousarray(target)).long()

"""h5-backed dataset of pre-extracted ViT tokens and label tiles."""

from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from prostate.data import Sample, cached_arrays, grid_positions, read_label_tile


class TokenPatchDataset(Dataset):
    """Visit every cached grid tile once per epoch, in the loader's shuffled order."""

    def __init__(
        self,
        samples: list[Sample],
        token_dir: Path,
        patch_size: int,
        seed: int,
    ):
        if not samples:
            raise ValueError("Training requires at least one case")
        self.patch_size = patch_size
        self.seed = seed
        self.epoch = 0

        # Only index metadata is loaded; feature tensors remain on disk.
        self.patches: list[tuple[Sample, Path, int, int, int]] = []
        for sample in samples:
            h5_path = token_dir / f"{sample.image_id}.h5"
            expected = np.array(
                list(grid_positions(sample.height, sample.width, patch_size)), dtype=np.int32
            )
            with h5py.File(h5_path, "r") as f:
                positions = f["positions"][:]  # (N, 2)
                if f.attrs.get("patch_size") != patch_size:
                    raise ValueError(f"Token cache patch_size mismatch: {h5_path}")
                if not np.array_equal(positions, expected) or len(f["tokens"]) != len(expected):
                    raise ValueError(
                        f"Token cache does not match the complete patch grid: {h5_path}"
                    )
            self.patches.extend(
                (sample, h5_path, index, int(top), int(left))
                for index, (top, left) in enumerate(positions)
            )

    def __len__(self) -> int:
        return len(self.patches)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, self.epoch, index]))
        sample, h5_path, patch_idx, top, left = self.patches[index]

        with h5py.File(h5_path, "r") as f:
            tokens = f["tokens"][patch_idx]  # (grid, grid, TOKEN_DIM) float16

        _, label_array = cached_arrays(sample.rgb_cache, sample.label_cache)
        size = self.patch_size
        target = read_label_tile(label_array, top, left, size)

        rotation = int(rng.integers(4))
        tokens = np.rot90(tokens, rotation, axes=(0, 1))
        target = np.rot90(target, rotation)
        if rng.random() < 0.5:
            tokens = tokens[:, ::-1, :]
            target = target[:, ::-1]

        # (grid, grid, TOKEN_DIM) → (TOKEN_DIM, grid, grid)
        token_tensor = torch.from_numpy(np.ascontiguousarray(tokens.transpose(2, 0, 1))).float()
        label_tensor = torch.from_numpy(np.ascontiguousarray(target)).long()
        return token_tensor, label_tensor

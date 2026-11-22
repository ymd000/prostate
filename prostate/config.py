"""Configuration shared by the training and OOF pipeline."""

from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class Config:
    data_dir: Path = Path("data/completed")
    cache_dir: Path = Path("outputs/cache")
    run_dir: Path = Path("outputs/unet-baseline")
    token_dir: Path = Path("outputs/tokens")
    encoder: str = "none"
    n_splits: int = 5
    seed: int = 42
    epochs: int = 20
    patch_sampling: str = field(default="full_grid_v1", init=False)
    patch_size: int = 256
    halo: int = 32
    batch_size: int = 1
    base_channels: int = 32
    normalization: str = "group"
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    device: str = "auto"
    amp: bool = True
    amp_dtype: str = "auto"
    num_workers: int = 0
    cpu_threads: int = 4

    def __post_init__(self):
        for name in ("epochs", "batch_size", "base_channels", "cpu_threads"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive")
        if self.n_splits < 2:
            raise ValueError("n_splits must be at least 2")
        if self.patch_size < 16 or self.patch_size % 16:
            raise ValueError("patch_size must be a positive multiple of 16")
        if not 0 <= self.halo < self.patch_size / 2:
            raise ValueError("halo must satisfy 0 <= halo < patch_size / 2")
        if self.learning_rate <= 0 or self.weight_decay < 0:
            raise ValueError("learning_rate must be positive and weight_decay nonnegative")
        if self.num_workers < 0 or self.seed < 0:
            raise ValueError("num_workers and seed must be nonnegative")
        if self.device not in ("auto", "cpu", "cuda"):
            raise ValueError("device must be auto, cpu or cuda")
        if self.encoder not in ("none", "gigapath-flash"):
            raise ValueError("encoder must be none or gigapath-flash")
        if self.normalization not in ("group", "none"):
            raise ValueError("normalization must be group or none")
        if self.amp_dtype not in ("auto", "bfloat16", "float16"):
            raise ValueError("amp_dtype must be auto, bfloat16 or float16")

    def to_dict(self) -> dict:
        result = asdict(self)
        for key in ("data_dir", "cache_dir", "run_dir", "token_dir"):
            result[key] = str(Path(result[key]).resolve())
        return result

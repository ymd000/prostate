"""Configuration shared by the training and OOF pipeline."""

from dataclasses import asdict, dataclass, field
from pathlib import Path

DEFAULT_HOLDOUT_CASE_ID = "15-1428"
DEFAULT_HOLDOUT_RUN_DIR = Path("outputs/holdout-15-1428")
ARCHIVE_MODELS = (
    "unet11",
    "unet11b",
    "unet11n",
    "unet16",
    "unet16b",
    "unet16n",
    "albunet",
    "albunet_b",
    "albunet_n",
)
ARCHIVE_BATCH_SIZES = {256: 32, 512: 8, 768: 4}


@dataclass(frozen=True)
class Config:
    data_dir: Path = Path("data/completed")
    cache_dir: Path = Path("outputs/cache")
    run_dir: Path = Path("outputs/unet-baseline")
    token_dir: Path = Path("outputs/tokens")
    encoder: str = "none"
    n_splits: int = 5
    holdout_case_id: str | None = None
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
    class_weighting: str = "none"
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
        if self.holdout_case_id is not None and (
            not self.holdout_case_id.strip() or "_" in self.holdout_case_id
        ):
            raise ValueError(
                "holdout_case_id must be a nonempty case ID before the first underscore"
            )
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
        if self.class_weighting not in ("none", "inverse-sqrt"):
            raise ValueError("class_weighting must be none or inverse-sqrt")
        if self.normalization not in ("group", "none"):
            raise ValueError("normalization must be group or none")
        if self.amp_dtype not in ("auto", "bfloat16", "float16"):
            raise ValueError("amp_dtype must be auto, bfloat16 or float16")

    def to_dict(self) -> dict:
        result = asdict(self)
        # Keep existing CV configurations compatible with saved experiments.
        if self.holdout_case_id is None:
            del result["holdout_case_id"]
        else:
            del result["n_splits"]
        for key in ("data_dir", "cache_dir", "run_dir", "token_dir"):
            result[key] = str(Path(result[key]).resolve())
        return result


@dataclass(frozen=True)
class ArchiveConfig:
    """Historical training conditions for case-level CV or fixed holdout verification."""

    data_dir: Path = Path("data/completed")
    cache_dir: Path = Path("outputs/cache")
    run_dir: Path = Path("outputs/archive-holdout-15-1428")
    holdout_case_id: str | None = DEFAULT_HOLDOUT_CASE_ID
    n_splits: int = 5
    model_name: str = "unet11"
    pretrained: bool = True
    patch_size: int = 768
    batch_size: int | None = None
    epochs: int = 30
    learning_rate: float = 0.01
    momentum: float = 0.9
    lr_decay: float = 0.95
    samples_multiplier: int = 8
    max_sampling_attempts: int = 10000
    crop_start_step: int = 1
    inference_tile_size: int = 3000
    validation_mode: str = "eval"
    seed: int = 42
    device: str = "auto"
    num_workers: int = 4
    cpu_threads: int = 4
    training_recipe: str = field(default="archive_random_v1", init=False)
    patch_sampling: str = field(default="archive_random_v1", init=False)
    optimizer: str = field(default="sgd", init=False)
    weight_decay: float = field(default=0.0, init=False)
    class_weighting: str = field(default="none", init=False)
    normalization: str = field(default="batch", init=False)
    input_color: str = field(default="BGR", init=False)
    input_normalization: str = field(default="imagenet", init=False)
    gradient_clip: float | None = field(default=None, init=False)
    encoder: str = field(default="none", init=False)
    amp: bool = field(default=False, init=False)
    amp_dtype: str = field(default="auto", init=False)
    halo: int = field(default=0, init=False)

    def __post_init__(self):
        if self.batch_size is None:
            object.__setattr__(self, "batch_size", ARCHIVE_BATCH_SIZES.get(self.patch_size, 1))
        for name in (
            "batch_size",
            "epochs",
            "samples_multiplier",
            "max_sampling_attempts",
            "crop_start_step",
            "inference_tile_size",
            "cpu_threads",
        ):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive")
        if self.model_name not in ARCHIVE_MODELS:
            raise ValueError(f"Unknown archive model: {self.model_name}")
        if self.patch_size < 64 or self.patch_size % 64:
            raise ValueError("Archive patch_size must be a positive multiple of 64")
        if self.n_splits < 2:
            raise ValueError("n_splits must be at least 2")
        if self.holdout_case_id is not None and (
            not self.holdout_case_id.strip() or "_" in self.holdout_case_id
        ):
            raise ValueError("holdout_case_id must be a case ID before the first underscore")
        if self.learning_rate <= 0 or not 0 <= self.momentum < 1 or not 0 < self.lr_decay <= 1:
            raise ValueError("Invalid archive optimizer settings")
        if self.num_workers < 0 or self.seed < 0:
            raise ValueError("num_workers and seed must be nonnegative")
        if self.device not in ("auto", "cpu", "cuda"):
            raise ValueError("device must be auto, cpu or cuda")
        if self.validation_mode not in ("eval", "archive-train"):
            raise ValueError("validation_mode must be eval or archive-train")

    def to_dict(self) -> dict:
        result = asdict(self)
        # Missing in historical configurations means the original 1px sampling.
        if self.crop_start_step == 1:
            del result["crop_start_step"]
        if self.holdout_case_id is None:
            del result["holdout_case_id"]
        else:
            # Preserve the saved configuration of existing holdout experiments.
            del result["n_splits"]
        for key in ("data_dir", "cache_dir", "run_dir"):
            result[key] = str(Path(result[key]).resolve())
        return result


@dataclass(frozen=True)
class ArchiveEvaluationConfig:
    """Inference overrides for a saved archive checkpoint; never training settings."""

    run_dir: Path
    device: str = "auto"
    inference_tile_size: int | None = None
    validation_mode: str | None = None
    output_dir: Path | None = None

    def __post_init__(self):
        if self.device not in ("auto", "cpu", "cuda"):
            raise ValueError("device must be auto, cpu or cuda")
        if self.inference_tile_size is not None and self.inference_tile_size < 1:
            raise ValueError("inference_tile_size must be positive")
        if self.validation_mode not in (None, "eval", "archive-train"):
            raise ValueError("validation_mode must be eval or archive-train")

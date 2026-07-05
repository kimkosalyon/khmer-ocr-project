from dataclasses import dataclass


@dataclass
class TrainConfig:
    """Configuration container for OCR training hyperparameters and system setup."""
    samples: int
    epochs: int
    batch_size: int
    lr: float
    compile: bool
    device: str
    num_workers: int
    profile: bool
    full: bool = False
    checkpoint_dir: str = "checkpoints"
    resume: str = None
    lr_scheduler: str = "none"
    dataset_dir: str = None
    val_split: float = 0.0
    test_split: float = 0.0




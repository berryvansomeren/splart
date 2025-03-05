from dataclasses import dataclass
import math

from PIL import Image
import torch


@dataclass
class Size:
    width: int
    height: int

    def as_w_h(self) -> tuple[int, int]:
        return self.width, self.height

    def as_h_w(self) -> tuple[int, int]:
        return self.height, self.width


@dataclass
class TrainingConfig:
    # ----------------
    # target image
    target_image_path: str
    target_image_load_size: Size

    # ----------------
    # texture image
    texture_image_path: str
    texture_load_size: Size

    # ----------------
    # training
    device: torch.device
    n_epochs_growth_phase: int
    n_epochs_finalization_phase: int
    learning_rate: float
    display_interval: int

    # ----------------
    # samples
    primary_samples: int
    backup_samples: int

    # ----------------
    # pruning
    pruning_interval: int
    pruning_gradient_threshold: float
    pruning_scale_threshold: float
    log_pruning: bool

    # ----------------
    # growth
    growth_interval: int
    n_growth_samples: int

    # ----------------
    # loss perturbation
    use_loss_perturbation: bool
    log_loss_perturbation: bool
    n_loss_perturbation_epochs: int
    expected_loss_after_perturbation_epochs: float


def get_scaled_size(image_path: str) -> Size:
    image = Image.open(image_path)
    width, height = image.size
    min_dim = min(width, height)
    scale_factor = 256 / min_dim
    new_width = int(width * scale_factor)
    new_height = int(height * scale_factor)
    # Note that we return H,W!
    return Size(width=new_width, height=new_height)


def make_full_config(
    target_image_path: str,
    texture_image_path: str,
    primary_samples: int,
    n_epochs_growth_phase: int,
    growth_interval: int,
    n_growth_samples: int,
) -> TrainingConfig:
    # ----------------
    # target image
    target_image_load_size = get_scaled_size(target_image_path)  # H, W

    # ----------------
    # texture image
    # H = W, Just match target image load size as closely as possible
    # The reason is that we cannot scale up textures easily in the current code, only down (need to look into it)
    texture_load_size_1: int = max(target_image_load_size.as_w_h())
    texture_load_size: Size = Size(width=texture_load_size_1, height=texture_load_size_1)

    # ----------------
    # training
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"DEVICE: {device}")
    learning_rate = 0.01
    display_interval = 1
    n_epochs_finalization_phase = n_epochs_growth_phase

    # ----------------
    # pruning
    pruning_interval = growth_interval
    pruning_gradient_threshold = 1e-5
    # Rule of thumb: choose the highest value equal to a loss so low you think it'll never be reached
    pruning_scale_threshold = 0.02
    log_pruning = True

    # ----------------
    # samples
    # This computation might overshoot the real required number.
    # Catching all edge cases is complicated, so we just accept it.
    backup_samples: int = math.ceil(n_epochs_growth_phase / growth_interval) * n_growth_samples

    # ----------------
    # loss perturbation
    use_loss_perturbation = True
    n_loss_perturbation_epochs = n_epochs_growth_phase // 2
    log_loss_perturbation = True
    expected_loss_after_perturbation_epochs = 0.3

    assert n_loss_perturbation_epochs <= n_epochs_growth_phase, "Perturbation epochs should be less than growth epochs"

    print(f"Expecting {primary_samples + backup_samples} samples in total")

    # ----------------
    training_config = TrainingConfig(
        target_image_path=target_image_path,
        target_image_load_size=target_image_load_size,
        texture_image_path=texture_image_path,
        texture_load_size=texture_load_size,
        device=device,
        n_epochs_growth_phase=n_epochs_growth_phase,
        n_epochs_finalization_phase=n_epochs_finalization_phase,
        learning_rate=learning_rate,
        display_interval=display_interval,
        primary_samples=primary_samples,
        backup_samples=backup_samples,
        pruning_interval=pruning_interval,
        pruning_gradient_threshold=pruning_gradient_threshold,
        pruning_scale_threshold=pruning_scale_threshold,
        log_pruning=log_pruning,
        growth_interval=growth_interval,
        n_growth_samples=n_growth_samples,
        use_loss_perturbation=use_loss_perturbation,
        log_loss_perturbation=log_loss_perturbation,
        n_loss_perturbation_epochs=n_loss_perturbation_epochs,
        expected_loss_after_perturbation_epochs=expected_loss_after_perturbation_epochs,
    )
    return training_config

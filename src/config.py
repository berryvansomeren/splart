from dataclasses import dataclass
import math

import torch


@dataclass
class TrainingConfig:
    # ----------------
    # target image
    target_image_file_path: str
    target_image_load_size: tuple[int, int]  # H, W

    # ----------------
    # texture image
    texture_image_file_path: str
    texture_load_size: int  # H = W

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
    pruning_scale_threshold: float
    pruning_gradient_threshold: float
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


def make_full_config(
    n_epochs_growth_phase: int, growth_interval: int, n_growth_samples: int, n_loss_perturbation_epochs: int
) -> TrainingConfig:
    # ----------------
    # target image
    target_image_file_path = "./input/2d_gaussian_splatting.png"
    target_image_load_size = (256, 256)  # H, W

    # ----------------
    # texture image
    texture_image_file_path: str = "./brushes/canvas/canvas_1.png"
    # H = W, Just match target image load size as closely as possible
    # The reason is that we cannot scale up textures easily in the current code, only down (need to look into it)
    texture_load_size: int = max(target_image_load_size)

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
    pruning_scale_threshold = 0.05
    pruning_gradient_threshold = 1e-5
    log_pruning = True

    # ----------------
    # samples
    primary_samples: int = 5
    # This computation might overshoot the real required number.
    # Catching all edge cases is complicated, so we just accept it.
    backup_samples: int = math.ceil(n_epochs_growth_phase / growth_interval) * n_growth_samples

    # ----------------
    # loss perturbation
    use_loss_perturbation = True
    log_loss_perturbation = True
    expected_loss_after_perturbation_epochs = 0.3

    assert n_loss_perturbation_epochs < n_epochs_growth_phase, "Perturbation epochs should be less than growth epochs"

    print(f"Expecting {primary_samples + backup_samples} samples in total")

    # ----------------
    training_config = TrainingConfig(
        target_image_file_path=target_image_file_path,
        target_image_load_size=target_image_load_size,
        texture_image_file_path=texture_image_file_path,
        texture_load_size=texture_load_size,
        device=device,
        n_epochs_growth_phase=n_epochs_growth_phase,
        n_epochs_finalization_phase=n_epochs_finalization_phase,
        learning_rate=learning_rate,
        display_interval=display_interval,
        primary_samples=primary_samples,
        backup_samples=backup_samples,
        pruning_interval=pruning_interval,
        pruning_scale_threshold=pruning_scale_threshold,
        pruning_gradient_threshold=pruning_gradient_threshold,
        log_pruning=log_pruning,
        growth_interval=growth_interval,
        n_growth_samples=n_growth_samples,
        use_loss_perturbation=use_loss_perturbation,
        log_loss_perturbation=log_loss_perturbation,
        n_loss_perturbation_epochs=n_loss_perturbation_epochs,
        expected_loss_after_perturbation_epochs=expected_loss_after_perturbation_epochs,
    )
    return training_config


def load_training_config() -> TrainingConfig:

    # Quick!
    # return make_full_config(
    #     n_epochs_growth_phase=100, growth_interval=1, n_growth_samples=1, n_loss_perturbation_epochs=50
    # )

    # Quality!
    return make_full_config(
        n_epochs_growth_phase=500, growth_interval=1, n_growth_samples=1, n_loss_perturbation_epochs=100
    )

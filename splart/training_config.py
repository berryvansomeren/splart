from dataclasses import dataclass
import math

from PIL import Image
import torch

from splart.size import Size


@dataclass
class TrainingConfig:

    universal_random_seed: int

    # ----------------
    # target image
    target_image_path: str
    target_image_load_size: Size

    # ----------------
    # texture image
    texture_directory_path: str
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
    n_loss_interpolation_epochs: int
    expected_loss_after_perturbation_epochs: float
    base_loss_multiplier: float
    minimum_brush_scale: float

    # ----------------
    # Importance Sampling
    use_importance_sampling_for_positions: bool
    use_importance_sampling_for_rotations: bool


def get_scaled_size(image_path: str, min_dimension_size: int) -> Size:
    image = Image.open(image_path)
    width, height = image.size
    min_dim = min(width, height)
    scale_factor = min_dimension_size / min_dim
    new_width = int(width * scale_factor)
    new_height = int(height * scale_factor)
    # Note that we return H,W!
    return Size(width=new_width, height=new_height)


def get_texture_load_size_for_target_image_load_size(target_image_load_size: Size) -> Size:
    # H = W, Just match target image load size as closely as possible
    # The reason is that we cannot scale up textures easily in the current code, only down (need to look into it)
    texture_load_size_1: int = min(target_image_load_size.as_w_h())
    texture_load_size: Size = Size(width=texture_load_size_1, height=texture_load_size_1)
    return texture_load_size


def make_full_config_with_defaults(
    target_image_path: str,
    texture_directory_path: str,
    primary_samples: int,
    n_epochs_growth_phase: int,
    growth_interval: int,
    n_growth_samples: int,
) -> TrainingConfig:

    universal_random_seed = 1337

    # ----------------
    # target image
    target_image_load_size = get_scaled_size(target_image_path, min_dimension_size=256)  # H, W

    # ----------------
    # texture image
    texture_load_size: Size = get_texture_load_size_for_target_image_load_size(target_image_load_size)

    # ----------------
    # training
    device: torch.device = torch.device("cpu")  # torch.device("cuda" if torch.cuda.is_available() else "cpu")  #
    learning_rate = 0.02
    display_interval = 1
    n_epochs_finalization_phase = int(n_epochs_growth_phase / 2)

    # ----------------
    # pruning
    pruning_interval = growth_interval
    pruning_gradient_threshold = 1e-5
    # Rule of thumb: choose the highest value equal to a loss so low you think it'll never be reached
    pruning_scale_threshold = 0.01
    log_pruning = True

    # ----------------
    # samples
    # This computation might overshoot the real required number.
    # Catching all edge cases is complicated, so we just accept it.
    backup_samples: int = (math.ceil(n_epochs_growth_phase / growth_interval) - 1) * n_growth_samples

    # ----------------
    # loss perturbation
    use_loss_perturbation = True
    n_loss_interpolation_epochs = n_epochs_growth_phase
    log_loss_perturbation = True
    expected_loss_after_perturbation_epochs = 0.01
    base_loss_multiplier = 2.5
    minimum_brush_scale = 0.1

    # ----------------
    # Importance Sampling
    use_importance_sampling_for_positions = True
    use_importance_sampling_for_rotations = True
    assert n_loss_interpolation_epochs <= n_epochs_growth_phase, "Perturbation epochs should be less than growth epochs"

    # ----------------
    training_config = TrainingConfig(
        universal_random_seed=universal_random_seed,
        target_image_path=target_image_path,
        target_image_load_size=target_image_load_size,
        texture_directory_path=texture_directory_path,
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
        n_loss_interpolation_epochs=n_loss_interpolation_epochs,
        expected_loss_after_perturbation_epochs=expected_loss_after_perturbation_epochs,
        base_loss_multiplier=base_loss_multiplier,
        minimum_brush_scale=minimum_brush_scale,
        use_importance_sampling_for_positions=use_importance_sampling_for_positions,
        use_importance_sampling_for_rotations=use_importance_sampling_for_rotations,
    )
    return training_config

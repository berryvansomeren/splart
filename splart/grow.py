import math
import torch

from splart.training_config import TrainingConfig
from splart.splat_components import in_place_encode_into_splat_tensor_starting_from
from splart.importance_sampling import sample_positions_from_difference_image, get_gradient_angles_at


def clamp(v, v_min, v_max):
    return min(max(v_min, v), v_max)


def get_brush_scale_for_l1_loss(config: TrainingConfig, current_l1_loss: float) -> float:
    # loss is assumed to be a l1 loss
    # by multiplying with N we turn the MSE into a sum of absolute differences
    # this can then be turned into a percentage, which is used to determine the brush size
    # the scale will simply be applied relative to the size the texture was loaded with
    n_pixels = math.prod(config.target_image_load_size.as_w_h())
    diff_score = current_l1_loss * n_pixels
    max_potential_diff_score = n_pixels  # pixel values are normalized otherwise we had to * 255
    normalized_diff_score = diff_score / max_potential_diff_score
    scale_clamped = clamp(normalized_diff_score, config.minimum_brush_scale, 1.0)
    return scale_clamped


def in_place_add_samples_starting_from(
    config: TrainingConfig,
    splat_weights: torch.nn.Parameter,
    start_index: int,
    n_samples: int,
    target_image: torch.Tensor,
    current_l1_loss: float,
    difference_image: torch.Tensor,
    target_gradients: torch.Tensor,
) -> None:
    target_hw = config.target_image_load_size.as_h_w()  # h, w

    # Sample new positions
    if config.use_importance_sampling_for_positions:
        new_positions_absolute_yx = sample_positions_from_difference_image(difference_image, n_samples)
    else:
        new_positions_absolute_yx = torch.stack(
            [
                torch.randint(0, target_hw[0], (n_samples,), device=config.device),  # y coordinates (height)
                torch.randint(0, target_hw[1], (n_samples,), device=config.device),  # x coordinates (width)
            ],
            dim=1,
        )
    new_positions_unit_normalized = new_positions_absolute_yx.float() / torch.tensor(target_hw, device=config.device)

    # Here we transform to the coordinate space used by torch.nn.functional.grid_sample
    # Top left = (-1,-1), bottom right = (1,1)
    new_positions_in_grid_sample_space = 2 * new_positions_unit_normalized[:, [1, 0]] - 1  # map to [-1, 1]

    # Fetching the color of the pixels at each coordinate pair
    new_colors_tensor = target_image[new_positions_absolute_yx[:, 0], new_positions_absolute_yx[:, 1]]

    # Scales based on current loss -> lower loss means adding more fine-grained details
    brush_scale_for_loss = get_brush_scale_for_l1_loss(config=config, current_l1_loss=current_l1_loss)
    new_scales = torch.full((n_samples, 1), brush_scale_for_loss, device=config.device)

    # Random rotations
    if config.use_importance_sampling_for_rotations:
        gradient_angles_at_positions = get_gradient_angles_at(target_gradients, new_positions_absolute_yx)
        half_pi = torch.tensor(torch.pi / 2, device=config.device)
        # only use angle component, and use a direction perpendicular to the gradient
        new_rotations = (gradient_angles_at_positions + half_pi).unsqueeze(1)
    else:
        new_rotations = 2 * torch.rand(n_samples, 1, device=config.device) - 1

    in_place_encode_into_splat_tensor_starting_from(
        splat_weights=splat_weights,
        start_index=start_index,
        positions=new_positions_in_grid_sample_space,
        colors=new_colors_tensor,
        scales=new_scales,
        rotations=new_rotations,
    )

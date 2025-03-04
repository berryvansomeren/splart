import numpy as np
import torch

from config import TrainingConfig
from splat_components import in_place_pack_into_splat_tensor_starting_from


def get_brush_scale_for_l1_loss(config: TrainingConfig, current_l1_loss: float, current_epoch: int) -> float:
    # loss is assumed to be a l1 loss
    # by multiplying with N we turn the MSE into a sum of absolute differences
    # this can then be turned into a percentage, which is used to determine the brush size
    n_pixels = np.prod(config.target_image_load_size)
    diff_score = current_l1_loss * n_pixels
    max_potential_diff_score = n_pixels  # pixel values are normalized otherwise we had to * 255
    normalized_diff_score = diff_score / max_potential_diff_score

    # use the normalized diff score to determine a desired brush size
    image_min_extend = min(config.target_image_load_size[:2])
    desired_brush_size = int(image_min_extend * normalized_diff_score)

    # determine the scale required to achieve the desired brush size
    scale = desired_brush_size / config.texture_load_size
    assert scale <= 1.0, f"Scale {scale} is larger than 1.0. This results in NaNs. Choose a higher texture_load_size!"
    return scale


def in_place_add_samples_starting_from(
    config: TrainingConfig,
    splat_weights: torch.nn.Parameter,
    start_index: int,
    n_samples: int,
    target_image_np: np.ndarray,
    current_epoch: int,
    current_l1_loss: float,
) -> None:
    # Normalising pixel coordinates [-1,1]
    # Normally the coordinate order in np is y, x
    # But we are going to use them in grid_sample, which expects x, y,
    #  and an NDC like coordinate space with y down
    # Let's already use the x, y order here.
    target_wh = config.target_image_load_size[1::-1]  # w, h
    new_positions_absolute = np.random.randint(0, target_wh, size=(n_samples, 2))  # x, y
    new_positions_unit_normalized = torch.tensor(new_positions_absolute / target_wh, device=config.device).float()
    new_positions_ndc = (new_positions_unit_normalized * 2.0) - 1.0

    # Fetching the colour of the pixels in each coordinates
    new_colors_np = np.array([target_image_np[coord[1], coord[0]] for coord in new_positions_absolute])
    new_colors_tensor = torch.tensor(new_colors_np, device=config.device).float()

    # Scales based on current loss -> lower loss means adding more fine-grained details
    brush_scale_for_loss = get_brush_scale_for_l1_loss(
        config=config, current_l1_loss=current_l1_loss, current_epoch=current_epoch
    )
    new_scales = torch.full((n_samples, 1), brush_scale_for_loss, device=config.device)

    # Random rotations
    new_rotations = 2 * torch.rand(n_samples, 1, device=config.device) - 1

    in_place_pack_into_splat_tensor_starting_from(
        splat_weights=splat_weights,
        start_index=start_index,
        positions=new_positions_ndc,
        colors=new_colors_tensor,
        scales=new_scales,
        rotations=new_rotations,
    )

from splart.size import Size
import torch
from splart.splat_components import SplatComponents


def transform_splat_textures(
    splats: SplatComponents,
    splat_textures: torch.Tensor,
) -> torch.Tensor:

    batch_size = splats.colors.shape[0]
    _, _, height, width = splat_textures.shape
    device = splat_textures.device

    # Step 1: Rotation and Scaling
    cos_rot = torch.cos(splats.rotations)
    sin_rot = torch.sin(splats.rotations)
    # Note that we are transforming the sampling grid/camera
    # We therefore need to invert the rotation of the matrix!
    # Instead of building a rotation matrix and inverting it, we immediately define it in the inverted way
    rotation_matrix = torch.stack([cos_rot, sin_rot, -sin_rot, cos_rot], dim=-1).view(batch_size, 2, 2)

    # In case of rectangular images, the normalized coordinate system would stretch our textures
    # We correct the aspect ratio through heterogeneous scaling
    # We scale relative to the smallest image dimension
    min_dim = min(width, height)
    aspect_correction = torch.tensor([width / min_dim, height / min_dim], device=device).view(1, 2)
    scale_matrix = torch.diag_embed(1.0 / splats.scales.view(-1, 1).expand(-1, 2) * aspect_correction)

    # Combining rotation and scale in a single affine matrix
    scaling_rotation_matrix = torch.zeros(batch_size, 2, 3, dtype=torch.float32, device=device)
    scaling_rotation_matrix[:, :2, :2] = rotation_matrix @ scale_matrix  # Apply rotation and scaling
    scaling_rotation_grid = torch.nn.functional.affine_grid(
        theta=scaling_rotation_matrix, size=list(splat_textures.shape), align_corners=True
    )
    scaled_rotated_textures = torch.nn.functional.grid_sample(
        input=splat_textures, grid=scaling_rotation_grid, align_corners=True
    )

    # Step 2: Translation
    translation_matrix = torch.zeros(batch_size, 2, 3, dtype=torch.float32, device=device)
    translation_matrix[:, 0, 0] = 1.0
    translation_matrix[:, 1, 1] = 1.0
    translation_matrix[:, :, 2] = -splats.positions
    translating_grid = torch.nn.functional.affine_grid(
        theta=translation_matrix, size=list(splat_textures.shape), align_corners=True
    )
    translated_textures = torch.nn.functional.grid_sample(
        input=scaled_rotated_textures, grid=translating_grid, align_corners=True
    )
    return translated_textures


def blend_alpha_colored_on_top(
    splat_layers: torch.Tensor, colors: torch.Tensor, background: torch.Tensor
) -> torch.Tensor:
    batch_size, _, height, width = splat_layers.shape

    # Multiply the grayscale texture by the color (apply foreground color)
    fg_colors = colors.unsqueeze(-1).unsqueeze(-1) * splat_layers  # Multiply color by texture intensity
    fg_colors = fg_colors.permute(0, 2, 3, 1)  # Convert from (B, C, H, W) to (B, H, W, C)

    # Interpret the gray scale values as alpha
    alpha = splat_layers  # Grayscale values used as alpha
    alpha = torch.clamp(alpha, 0, 1)  # Ensure alpha is between 0 and 1

    # Alpha blending over consecutive layers
    for i in range(batch_size):
        fg_layer = fg_colors[i]

        # We need to expand alpha to match (H, W, 3) for blending
        fg_alpha = alpha[i].squeeze(0).unsqueeze(-1).expand(-1, -1, 3)

        # Accumulate layers: use blending formula for each layer (front-to-back blending)
        prev_alpha = 1 - fg_alpha
        background = fg_layer + prev_alpha * background

    # Ensure the final image is in [0, 1] range
    final_image = torch.clamp(background, 0, 1)
    return final_image


def pad_extend_textures(image_size: Size, textures: torch.Tensor, batch_size: int) -> torch.Tensor:
    """
    Pad the texture to match the target image size while keeping it centered.
    """
    pad_h = image_size.height - textures.shape[2]
    pad_w = image_size.width - textures.shape[3]

    assert pad_h >= 0, "Texture load size is larger than image load size, you will be clipping your textures! - Height"
    assert pad_w >= 0, "Texture load size is larger than image load size, you will be clipping your textures! - Width"

    # Pad on all sides while also compensating for rounding errors
    padding = (pad_w // 2, pad_w // 2 + pad_w % 2, pad_h // 2, pad_h // 2 + pad_h % 2)
    textures_padded = torch.nn.functional.pad(textures, padding, "constant", 0)  # Pad

    # Since the positions of the splats are already random,
    # we won't randomize the texture, but just cycle through them.

    # Determine how many full cycles we need
    full_cycles = batch_size // len(textures)
    remaining = batch_size % len(textures)
    # Repeat full cycles and add remaining textures
    cyclic_textures_batch = torch.cat(
        [textures_padded.repeat(full_cycles, 1, 1, 1), textures_padded[:remaining]], dim=0
    )
    return cyclic_textures_batch

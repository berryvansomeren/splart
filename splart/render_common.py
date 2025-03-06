from splart.size import Size
import torch
from splart.splat_components import SplatComponents


def transform_splat_textures(
    splats: SplatComponents,
    splat_textures: torch.Tensor,
) -> torch.Tensor:

    batch_size = splats.colors.shape[0]
    _, n_channels, height, width = splat_textures.shape
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
    batch_size, n_channels, height, width = splat_layers.shape

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


def pad_extend_texture(image_size: Size, texture: torch.Tensor, batch_size: int) -> torch.Tensor:
    """
    Pad the texture to match the target image size while keeping it centered.
    """
    pad_h = image_size.height - texture.shape[1]
    pad_w = image_size.width - texture.shape[2]
    # Pad on all sides while also compensating for rounding errors
    padding = (pad_w // 2, pad_w // 2 + pad_w % 2, pad_h // 2, pad_h // 2 + pad_h % 2)
    texture_padded = torch.nn.functional.pad(texture, padding, "constant", 0)  # Pad

    # Return the single-channel texture expanded for the batch size
    texture_batch = texture_padded.unsqueeze(0).expand(batch_size, -1, -1, -1)  # Shape: (batch_size, 1, H, W)
    return texture_batch

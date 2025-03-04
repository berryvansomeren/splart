import torch
import numpy as np
from splat_components import unpack, Columns


def extend_texture(image_size, batch_size, texture):
    # Ensure padding to match image size
    pad_h = image_size[0] - texture.shape[1]  # image_size = H, W
    pad_w = image_size[1] - texture.shape[2]  # texture.shape = C, H, W
    padding = (pad_w // 2, pad_w // 2 + pad_w % 2, pad_h // 2, pad_h // 2 + pad_h % 2)
    texture_padded = torch.nn.functional.pad(texture, padding, "constant", 0)  # Pad

    # Return the single-channel texture expanded for the batch size
    texture_batch = texture_padded.unsqueeze(0).expand(batch_size, -1, -1, -1)  # Shape: (batch_size, 1, H, W)
    return texture_batch


def combine_layers_using_color_and_alpha(splat_layers: torch.Tensor, colors: torch.Tensor) -> torch.Tensor:
    batch_size, n_channels, height, width = splat_layers.shape

    # Multiply the grayscale texture by the color (apply foreground color)
    fg_colors = colors.unsqueeze(-1).unsqueeze(-1) * splat_layers  # Multiply color by texture intensity
    fg_colors = fg_colors.permute(0, 2, 3, 1)  # Convert from (B, C, H, W) to (B, H, W, C)

    # Interpret the gray scale values as alpha
    alpha = splat_layers  # Grayscale values used as alpha
    alpha = torch.clamp(alpha, 0, 1)  # Ensure alpha is between 0 and 1

    # Initialize final image with zeros, but it should have 3 channels (RGB)
    final_image = torch.zeros((height, width, 3), dtype=torch.float32, device=splat_layers.device)

    # Alpha blending over consecutive layers
    for i in range(batch_size):
        fg_layer = fg_colors[i]

        # We need to expand alpha to match (H, W, 3) for blending
        fg_alpha = alpha[i].squeeze(0).unsqueeze(-1).expand(-1, -1, 3)

        # Accumulate layers: use blending formula for each layer (front-to-back blending)
        prev_alpha = 1 - fg_alpha
        final_image = fg_layer + prev_alpha * final_image

    # Ensure the final image is in [0, 1] range
    final_image = torch.clamp(final_image, 0, 1)

    return final_image


def render_2d_texture_splatting(
    splat_weights,
    texture: torch.Tensor,
    image_size,
    device: torch.device,
):
    scale = unpack(splat_weights, Columns.scales)
    rotation = np.pi / 2 * unpack(splat_weights, Columns.rotations)
    colors = unpack(splat_weights, Columns.colors)
    coords = unpack(splat_weights, Columns.positions)

    batch_size = colors.shape[0]
    splat_textures = extend_texture(image_size, batch_size, texture)
    _, n_channels, height, width = splat_textures.shape

    # Create the scaling and rotation matrix
    cos_rot = torch.cos(rotation)
    sin_rot = torch.sin(rotation)
    rotation_matrix = torch.stack([cos_rot, -sin_rot, sin_rot, cos_rot], dim=-1).view(batch_size, 2, 2)
    scale_matrix = torch.diag_embed(1.0 / scale.view(-1, 1).expand(-1, 2))
    affine_matrix = rotation_matrix @ scale_matrix

    # Pre-scale the translation to counteract scaling
    adjusted_translation = (-coords) / scale.view(-1, 1)

    # Construct the final affine transformation matrix
    final_affine_matrix = torch.zeros(batch_size, 2, 3, dtype=torch.float32, device=device)
    final_affine_matrix[:, :2, :2] = affine_matrix
    final_affine_matrix[:, :, 2] = adjusted_translation

    # Apply the transformation in a single step
    grid = torch.nn.functional.affine_grid(final_affine_matrix, list(splat_textures.shape), align_corners=True)
    transformed_textures = torch.nn.functional.grid_sample(splat_textures, grid, align_corners=True)

    final_image = combine_layers_using_color_and_alpha(
        splat_layers=transformed_textures,
        colors=colors,
    )
    return final_image

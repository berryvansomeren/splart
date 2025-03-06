import torch
from splart.size import Size
from splart.splat_components import unpack_all_components, slice_single_splat
from splart.render_common import pad_extend_texture, transform_splat_textures, blend_alpha_colored_on_top


def render_2d_texture_splats_sequential(
    splat_weights: torch.Tensor,
    texture: torch.Tensor,
    image_size: Size,
):
    """
    Sequential rendering is slower but less memory intensive.
    Suitable for rendering 4k images!
    During optimization, you'd probably prefer batched rendering.
    """

    n_splats = splat_weights.shape[0]
    splats = unpack_all_components(splat_weights)

    # We have a texture tensor of batch size 1
    splat_texture = pad_extend_texture(image_size, texture, 1)

    # Process each splat sequentially
    final_image = torch.zeros((image_size.height, image_size.width, 3), dtype=torch.float32, device=texture.device)
    for i in range(n_splats):
        current_splat = slice_single_splat(splats, i)
        current_color = splats.colors[i]
        transformed_texture = transform_splat_textures(splats=current_splat, splat_textures=splat_texture)
        final_image = blend_alpha_colored_on_top(
            splat_layers=transformed_texture, colors=current_color, background=final_image
        )

    final_image = final_image.clamp(0, 1)
    return final_image

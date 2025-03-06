import torch
from splart.size import Size
from splart.splat_components import unpack_all_components
from splart.render_common import blend_alpha_colored_on_top, transform_splat_textures, pad_extend_texture


def render_2d_texture_splats_batched(
    splat_weights,
    texture: torch.Tensor,
    image_size: Size,
):
    """
    Batched rendering is faster but more memory intensive.
    NOT suitable for rendering 4k images!
    Preferred during optimization as long as things fit in memory (especially VRAM when using CUDA).
    """

    n_splats = splat_weights.shape[0]
    splats = unpack_all_components(splat_weights)

    # Prepare all splats textures as one big tensor!
    splat_textures = pad_extend_texture(image_size, texture, n_splats)
    transformed_textures = transform_splat_textures(splats=splats, splat_textures=splat_textures)

    # process all splats in one big batch!
    background = torch.zeros((image_size.height, image_size.width, 3), dtype=torch.float32, device=texture.device)
    final_image = blend_alpha_colored_on_top(
        splat_layers=transformed_textures, colors=splats.colors, background=background
    )
    final_image = torch.clamp(final_image, 0, 1)
    return final_image

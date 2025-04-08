from dataclasses import dataclass
from enum import Enum
from typing import Callable

import torch

"""
This module has two main purposes:

1) Define the components of our splats and the associated Parameter space for optimization
2) Make it safer/easier to deal with these components without requiring the developer to remember all their details. 

# 1) Component definitions
The ranges specified describe the ranges BEFORE packing!

- Scales: 
    Size multiplication in (1,0) range.  
    Initial scales are determined based on the current loss. 
    When the loss is 1, a scale of 1 is used. When the loss is 0, a scale of 0 is used. 
    If loss perturbation is enabled, the loss is artificially changed before using it to determine scale.
    The purpose of this is to have more control over how scale change throughout the optimization process.
    Specifically; we don't want splats to become too small too fast.
    If pruning is enabled: 
    When the scale becomes too small, splats will be removed.
    
- Rotations:
    Raw: Unbounded Radians, 
    Encoded using sin/cos encoding.
    After decoding you will have the equivalent angle but in a (-pi,pi) range
    0 points to the right. A positive angle rotates the image counter-clockwise -> 0.5*pi points up   
    Initial rotation is random OR
    If guidance through image gradients is enabled: 
    The image gradient is used to define an initial rotation. 
    Since only a single pixel is used to determine initial rotation,
    while the splat might cover many pixels, 
    it's clear that there is room for further optimization. 
    
- Colors:
    RGB in a (0,1) range.  
    When adding a new splat, the color is based on the target image.
    Since only a single pixel is used to determine initial color,
    while the splat might cover many pixels, 
    it's clear that there is room for further optimization. 
    
- Positions:
    WARNING! 
    The positions are confusing, 
    because we optimize them directly in the coordinate space used by torch.nn.functional.grid_sample.
    See: https://pytorch.org/docs/stable/generated/torch.nn.functional.grid_sample.html 
    x,y in (-1,1) range
    Top left = (-1,-1), bottom right = (1,1)
    So, his is x,y ordering, with y increasing downward. It's a bit similar to, but not the same as NDC.
    Positions are randomly sampled over the image space OR
    If guidance through difference image is enabled:
    positions are randomly sampled over the image space, but weighted by how much their color deviates from the target.
    
    
# 2) Mechanisms to safely/easily work with the components:

This is simultaneously an approximation and combination of Named Tensors and TorchTyping.
Instead of having to hardcode column indices everywhere, we should be able to refer to them by some meaningful name.
We should also not be required remember when to use what function/inverse to pack/unpack a certain column, 
like when to use logit/sigmoid and when to use tanh/atanh.
This is achieved by defining named slices, and associate packing/unpacking functions with them in a central location.
"""


# Using an enum specifically to store the slices to make it easy to loop over the components
class Columns(Enum):
    scales = slice(0, 1)  # (1,0)
    rotations = slice(1, 3)  # unbounded radians (single element angle) encoded to sin/cos (2 elements) in (-1,1)
    colors = slice(3, 6)  # RGB (0,1)
    positions = slice(6, 8)  # x, y in grid_sample space: top left = (-1,-1), bottom right = (1,1)


# Some aliases for terser code
SCALES = Columns.scales.value
ROTATIONS = Columns.rotations.value
COLORS = Columns.colors.value
POSITIONS = Columns.positions.value


def encode_angles(angles_radians_raw: torch.Tensor) -> torch.Tensor:
    # We use a sin/cos encoding to retain smoothness across the -pi/pi boundary
    # If we would just use tanh, then the circular nature of angles is lost,
    # with tanh you can not increase an angle beyond 1 to get something equal to -1
    angles_radians = angles_radians_raw.squeeze()
    angles_sin_cos_encoded = torch.stack([torch.sin(angles_radians), torch.cos(angles_radians)], dim=-1)
    return angles_sin_cos_encoded


def decode_angles(angles_radians_encoded: torch.Tensor) -> torch.Tensor:
    angles_radians_decoded = torch.atan2(angles_radians_encoded[:, 0], angles_radians_encoded[:, 1]).unsqueeze(dim=1)
    return angles_radians_decoded


_COLUMN_TO_ENCODER_MAP: dict[Columns, Callable] = {
    Columns.scales: torch.logit,
    Columns.rotations: encode_angles,
    Columns.colors: torch.logit,
    Columns.positions: torch.atanh,
}

_ENCODER_TO_DECODER_MAP: dict[Callable, Callable] = {
    torch.logit: torch.sigmoid,
    torch.atanh: torch.tanh,
    encode_angles: decode_angles,
}

_COLUMN_TO_DECODER_MAP: dict[Columns, Callable] = {
    k: _ENCODER_TO_DECODER_MAP[v] for k, v in _COLUMN_TO_ENCODER_MAP.items()
}


def init_empty_splat_weights(device: torch.device, n_samples: int) -> torch.nn.Parameter:
    n_columns = sum([s.value.stop - s.value.start for s in Columns])
    splat_tensor = torch.Tensor(torch.empty(n_samples, n_columns, device=device))
    splat_weights = torch.nn.Parameter(splat_tensor)
    return splat_weights


def in_place_encode_into_splat_tensor_starting_from(
    splat_weights: torch.nn.Parameter,
    start_index: int,
    positions: torch.Tensor,
    colors: torch.Tensor,
    scales: torch.Tensor,
    rotations: torch.Tensor,
) -> None:
    # For the individual components,
    # see the long description at the to of this file.
    n_samples = positions.shape[0]
    column_component_tuples = [
        (Columns.scales, scales),
        (Columns.rotations, rotations),
        (Columns.colors, colors),
        (Columns.positions, positions),
    ]
    for columns, component in column_component_tuples:
        encoder = get_encoder_for_columns(columns)
        splat_weights.data[start_index : start_index + n_samples :, columns.value] = encoder(component)


def get_encoder_for_columns(columns: Columns) -> Callable:
    return _COLUMN_TO_ENCODER_MAP[columns]


def get_decoder_for_columns(columns: Columns) -> Callable:
    return _COLUMN_TO_DECODER_MAP[columns]


def decode(source_tensor: torch.Tensor, columns: Columns) -> torch.Tensor:
    decoder = get_decoder_for_columns(columns)
    result = decoder(source_tensor[:, columns.value])
    return result


@dataclass
class SplatComponents:
    # Values inside SplatComponents should always be pre-decoded!
    scales: torch.Tensor
    rotations: torch.Tensor
    colors: torch.Tensor
    positions: torch.Tensor


def slice_single_splat(splats: SplatComponents, i: int) -> SplatComponents:
    single_splat = SplatComponents(
        scales=splats.scales[i].unsqueeze(0),
        rotations=splats.rotations[i].unsqueeze(0),
        colors=splats.colors[i].unsqueeze(0),
        positions=splats.positions[i].unsqueeze(0),
    )
    return single_splat


def unpack_all_components(splat_weights: torch.Tensor) -> SplatComponents:
    splat_components = SplatComponents(
        scales=decode(splat_weights, Columns.scales),
        rotations=decode(splat_weights, Columns.rotations),
        colors=decode(splat_weights, Columns.colors),
        positions=decode(splat_weights, Columns.positions),
    )
    return splat_components

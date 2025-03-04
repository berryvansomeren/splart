from enum import Enum
from typing import Callable
import torch
from torch import Tensor


"""
The point of this module is to make it a bit safer/easier to refer to columns in a tensor that represents a splat.
Instead of having to hardcode column indices everywhere, we should be able to refer to them by some meaningful name.
We should also not have to remember when to use what function/inverse to pack/unpack a certain column, 
like when to use logit/sigmoid and when to use tanh/atanh.
"""


# This enum also makes it easy to loop over the components
class Columns(Enum):
    scales = slice(0, 1)  # scale
    rotations = slice(1, 2)  # rot
    colors = slice(2, 5)  # r, g, b
    positions = slice(5, 7)  # x, y


SCALES = Columns.scales.value
ROTATIONS = Columns.rotations.value
COLORS = Columns.colors.value
POSITIONS = Columns.positions.value


_COLUMN_TO_PACKER_FUNCTION_MAP: dict[Columns, Callable] = {
    Columns.scales: torch.logit,
    Columns.rotations: torch.atanh,
    Columns.colors: torch.logit,
    Columns.positions: torch.atanh,
}

_INVERSE_FUNCTION_MAP: dict[Callable, Callable] = {
    torch.logit: torch.sigmoid,
    torch.atanh: torch.tanh,
}

_COLUMN_TO_UNPACKER_FUNCTION_MAP: dict[Columns, Callable] = {
    k: _INVERSE_FUNCTION_MAP[v] for k, v in _COLUMN_TO_PACKER_FUNCTION_MAP.items()
}


def init_empty_splat_weights(device: torch.device, n_samples: int) -> torch.nn.Parameter:
    n_columns = sum([s.value.stop - s.value.start for s in Columns])
    splat_tensor = torch.Tensor(torch.empty(n_samples, n_columns, device=device))
    splat_weights = torch.nn.Parameter(splat_tensor)
    return splat_weights


def in_place_pack_into_splat_tensor_starting_from(
    splat_weights: torch.nn.Parameter,
    start_index: int,
    positions: Tensor,
    colors: Tensor,
    scales: Tensor,
    rotations: Tensor,
) -> None:
    n_samples = positions.shape[0]
    column_component_tuples = [
        (Columns.scales, scales),
        (Columns.rotations, rotations),
        (Columns.colors, colors),
        (Columns.positions, positions),
    ]
    for columns, component in column_component_tuples:
        packer = get_packer_for_columns(columns)
        splat_weights.data[start_index : start_index + n_samples :, columns.value] = packer(component)


def get_packer_for_columns(columns: Columns) -> Callable:
    return _COLUMN_TO_PACKER_FUNCTION_MAP[columns]


def get_unpacker_for_columns(columns: Columns) -> Callable:
    return _COLUMN_TO_UNPACKER_FUNCTION_MAP[columns]


def unpack(source_tensor: Tensor, columns: Columns) -> Tensor:
    unpacker = get_unpacker_for_columns(columns)
    return unpacker(source_tensor[:, columns.value])

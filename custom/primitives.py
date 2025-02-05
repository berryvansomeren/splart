from dataclasses import dataclass
import numpy as np
import torch


NPImage = np.ndarray
TorchImage = torch.Tensor


@dataclass
class Splats:
    positions: torch.nn.Parameter
    radii: torch.nn.Parameter
    sigmas: torch.nn.Parameter


def torch_to_np_image(torch_image: TorchImage) -> NPImage:
    return torch_image.detach().numpy()

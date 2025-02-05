import torch

from primitives import Splats, TorchImage


def gaussian_kernel_torch(size, sigma) -> torch.Tensor:
    """Create a 2D Gaussian kernel in PyTorch."""
    x = torch.arange(size, dtype=torch.float32)
    y = torch.arange(size, dtype=torch.float32)
    x_grid, y_grid = torch.meshgrid(x, y, indexing="ij")
    center = (size - 1) / 2
    kernel = torch.exp(
        -((x_grid - center) ** 2 + (y_grid - center) ** 2) / (2 * sigma**2)
    )
    kernel = kernel / kernel.sum()  # Normalize the kernel
    return kernel


def render_splat_in_place(image, position, radius, sigma) -> None:
    """Splat a Gaussian kernel onto the image at a given position."""
    size = 2 * int(radius) + 1  # Kernel size based on radius
    kernel = gaussian_kernel_torch(size, sigma)

    x, y = position
    x_min = max(0, int(x - radius))
    x_max = min(image.shape[1], int(x + radius + 1))
    y_min = max(0, int(y - radius))
    y_max = min(image.shape[0], int(y + radius + 1))

    kernel_x_min = max(0, -int(x - radius))
    kernel_x_max = kernel.shape[1] - max(0, int(x + radius + 1) - image.shape[1])
    kernel_y_min = max(0, -int(y - radius))
    kernel_y_max = kernel.shape[0] - max(0, int(y + radius + 1) - image.shape[0])

    # Add the kernel to the image in a differentiable manner
    image[y_min:y_max, x_min:x_max] += kernel[
        kernel_y_min:kernel_y_max, kernel_x_min:kernel_x_max
    ]


def render_image(size, splats: Splats) -> TorchImage:
    """Render an image using a set of splats"""
    result = torch.zeros(size, dtype=torch.float32)
    for i, pos in enumerate(splats.positions):
        render_splat_in_place(result, pos, splats.radii[i], splats.sigmas[i])
    return result

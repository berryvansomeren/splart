import torch


def get_difference_image(current: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    diff_tensor = torch.norm(current - target, dim=2)
    return diff_tensor


def sample_positions_from_difference_image(diff_image: torch.Tensor, n_samples: int) -> torch.Tensor:
    # Create a flat array of probability per index
    flat_weights = diff_image.flatten()
    flat_probabilities = flat_weights / flat_weights.sum()

    # Sample indices based on probabilities
    random_flat_index = torch.multinomial(flat_probabilities, n_samples)

    # Convert the flat indices back to 2D coordinates
    positions_tuple = torch.unravel_index(random_flat_index, diff_image.shape)
    positions_tensor = torch.stack(positions_tuple, dim=1)
    return positions_tensor


def get_scharr_kernels(device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    scharr_x_hw = torch.tensor([[-3.0, 0.0, 3.0], [-10.0, 0.0, 10.0], [-3.0, 0.0, 3.0]], device=device) / 16.0
    scharr_x = scharr_x_hw.unsqueeze(0).unsqueeze(0)
    scharr_y = scharr_x.transpose(2, 3)
    return scharr_x, scharr_y


def get_gaussian_kernel(size: int, sigma: float) -> torch.Tensor:
    ax = torch.arange(size) - size // 2
    xx, yy = torch.meshgrid(ax, ax, indexing="ij")
    kernel = torch.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    kernel /= kernel.sum()  # Normalize to sum to 1
    kernel = kernel.unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, size, size)
    return kernel


def get_gradients(image_hwc: torch.Tensor) -> torch.Tensor:
    image_chw = image_hwc.permute(2, 0, 1)
    gray = (0.2989 * image_chw[0] + 0.5870 * image_chw[1] + 0.1140 * image_chw[2]).unsqueeze(0).unsqueeze(0)
    # We blur the input instead of the gradients themselves
    # This basically clean up the signal by removing high frequency noise before estimating gradients
    # This should result in cleaner gradients, that are less affected by high frequency noise
    gaussian_kernel = get_gaussian_kernel(size=5, sigma=2.0).to(image_hwc.device)
    gray_blurred = torch.nn.functional.conv2d(gray, gaussian_kernel, padding=2)  # padding=2 for 5x5 kernel

    # Apply Scharr filter
    kernel_scharr_x, kernel_scharr_y = get_scharr_kernels(device=image_hwc.device)
    dxs = torch.nn.functional.conv2d(gray_blurred, kernel_scharr_x, padding=1).squeeze()
    dys = torch.nn.functional.conv2d(gray_blurred, kernel_scharr_y, padding=1).squeeze()
    gradients = torch.stack((dys, dxs), dim=0)
    return gradients


def get_gradient_angles_at(gradients: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
    # Extract the y and x gradients from the stacked tensor
    position_dys = gradients[0, positions[:, 0], positions[:, 1]]
    position_dxs = gradients[1, positions[:, 0], positions[:, 1]]
    directions_radians = torch.atan2(position_dys, position_dxs)
    return directions_radians


def gradients_to_normal_map(gradients: torch.Tensor) -> torch.Tensor:
    # Normalize gradients to the range [-1, 1]
    max_val = torch.max(torch.abs(gradients))
    normalized_gradients = gradients / max_val

    # Convert gradients to RGB format
    normal_map = torch.zeros((gradients.shape[1], gradients.shape[2], 3), device=gradients.device)
    normal_map[..., 0] = (normalized_gradients[1] + 1) / 2  # X-gradient (Red)
    normal_map[..., 1] = (normalized_gradients[0] + 1) / 2  # Y-gradient (Green)
    normal_map[..., 2] = 1.0  # Blue channel set to 1

    return normal_map

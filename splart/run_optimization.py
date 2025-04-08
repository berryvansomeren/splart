import json
from dataclasses import asdict
import numpy as np
import torch

import time
from typing import cast
from torch.optim import Adam
from datetime import datetime
from PIL import Image
import random
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Any

from splart.size import Size
from splart.training_config import TrainingConfig, get_texture_load_size_for_target_image_load_size
from splart.grow import in_place_add_samples_starting_from
from splart.render_batched import render_2d_texture_splats_batched
from splart.render_sequential import render_2d_texture_splats_sequential
from splart.splat_components import init_empty_splat_weights, decode, Columns
from splart.make_gif import write_gif_for_folder
from splart.importance_sampling import get_difference_image, get_gradients, gradients_to_normal_map
from splart.loss_perturbation import get_perturbed_loss


def set_random_seeds(universal_seed: int) -> None:
    random.seed(universal_seed)
    np.random.seed(universal_seed)
    torch.manual_seed(universal_seed)
    torch.cuda.manual_seed_all(universal_seed)


def _load_target_image(config: TrainingConfig) -> tuple[torch.Tensor, np.ndarray]:
    # PIL uses W,H order!
    image_file = Image.open(config.target_image_path).resize(config.target_image_load_size.as_w_h()).convert("RGB")
    target_image_np = np.array(image_file) / 255.0
    target_image_tensor_hwc = torch.tensor(target_image_np, dtype=torch.float32, device=config.device)
    return target_image_tensor_hwc, target_image_np


def _load_textures(config: TrainingConfig) -> torch.Tensor:
    load_size = config.texture_load_size
    individual_textures = []
    for image_path in Path(config.texture_directory_path).glob("*.png"):
        # PIL uses W,H order!
        current_texture_file = Image.open(image_path).resize(load_size.as_w_h()).convert("RGB")
        current_texture_np = np.array(current_texture_file)  # Shape: (H, W, 3)
        current_texture_red_channel_np = current_texture_np[..., 0] / 255.0  # Extract Red channel (H, W)
        current_texture_tensor = torch.tensor(current_texture_red_channel_np, dtype=torch.float32, device=config.device)
        current_texture_tensor = current_texture_tensor.unsqueeze(0)  # Add channel dimension (1, H, W)
        individual_textures.append(current_texture_tensor)

    textures = torch.stack(individual_textures)
    return textures


def _write_image(image_array: np.ndarray, image_path: Path) -> None:
    img = Image.fromarray((image_array * 255).astype(np.uint8))
    img.save(str(image_path))


def _write_config_json(config: TrainingConfig, json_path: Path) -> None:
    class ConfigJSONEncoder(json.JSONEncoder):
        def default(self, obj: Any):
            if isinstance(obj, torch.device):
                return str(obj)  # Convert torch.device to string (e.g., "cuda" or "cpu")
            if isinstance(obj, Size):
                return {"width": obj.width, "height": obj.height}  # Convert SizeXY to a dictionary
            return super().default(obj)

    with open(str(json_path), "w") as f:
        json.dump(asdict(config), f, indent=4, cls=ConfigJSONEncoder)


def write_loss_graphs(loss_history: list[tuple[float, float]], output_directory: Path) -> None:
    epoch_durations, epoch_losses = zip(*loss_history)

    plt.figure()
    plt.plot(epoch_losses, label="Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss Over Epochs")
    plt.legend()
    plt.grid()
    file_path = output_directory / "loss_over_epochs.png"
    plt.savefig(str(file_path), dpi=300)
    print(f"Wrote 'Loss over Epochs'-Graph to {file_path}")

    plt.figure()
    epoch_times = np.cumsum(epoch_durations) - epoch_durations[0]
    plt.plot(epoch_times, epoch_losses, label="Loss")
    plt.xlabel("Time")
    plt.ylabel("Loss")
    plt.title("Training Loss Over Time")
    plt.legend()
    plt.grid()
    file_path = output_directory / "loss_over_time.png"
    plt.savefig(str(file_path), dpi=300)
    print(f"Wrote 'Loss over Time'-Graph to {file_path}")


def init_persistent_mask(config: TrainingConfig) -> torch.Tensor:
    persistent_mask = torch.cat(
        [
            torch.ones(config.primary_samples, dtype=torch.bool, device=config.device),
            torch.zeros(
                config.backup_samples,
                dtype=torch.bool,
                device=config.device,
            ),
        ],
        dim=0,
    )
    return persistent_mask


def run_optimization(config: TrainingConfig) -> None:
    print(f"DEVICE: {config.device}")
    run_start = datetime.now()
    set_random_seeds(config.universal_random_seed)

    output_directory = Path("output") / run_start.strftime("%Y_%m_%d-%H_%M_%S")
    print(f"Writing output to: {output_directory}")
    output_directory.mkdir(parents=True, exist_ok=True)

    config_json_path = output_directory / "config.json"
    print(f"Writing config json to {config_json_path}")
    _write_config_json(config=config, json_path=config_json_path)

    target_image_hwc, target_image_np = _load_target_image(config)
    scaled_input_image_path = output_directory / "input_image.png"
    print(f"Writing scaled input image to {scaled_input_image_path}")
    _write_image(image_array=target_image_hwc.clone().detach().cpu().numpy(), image_path=scaled_input_image_path)

    target_gradients = get_gradients(target_image_hwc)
    gradient_normal_map = gradients_to_normal_map(target_gradients)
    normal_map_image_path = output_directory / "input_normal_map.png"
    print(f"Writing gradient normal map to {normal_map_image_path}")
    _write_image(image_array=gradient_normal_map.cpu().numpy(), image_path=normal_map_image_path)

    textures = _load_textures(config)
    current_image = torch.zeros_like(target_image_hwc)

    current_l1_loss_value = 1.0
    perturbed_loss = 1.0
    current_sample_end_pointer = 0
    n_total_expected_samples = config.primary_samples + config.backup_samples
    splat_weights: torch.nn.Parameter = init_empty_splat_weights(
        device=config.device, n_samples=n_total_expected_samples
    )
    splats_weights_for_rendering: torch.Tensor = splat_weights

    # We have two masks to apply to our weights
    # Remember that splat_weights is a buffer that already allocated memory for samples to be added later
    # The render mask contains all splats that have been activated from the buffer, and have been optimized
    # The optimization mask is a subset of the render mask
    # When splats have low gradients they will no longer be optimized, but still will be rendered
    samples_rendering_mask = init_persistent_mask(config=config)
    samples_optimization_mask = samples_rendering_mask.clone()

    # Training loop
    optimizer = Adam([splat_weights], lr=config.learning_rate)
    current_l1_loss_tensor: torch.Tensor = torch.tensor(float("inf"))
    loss_history: list[tuple[float, float]] = []  # time, loss
    last_epoch_end = time.perf_counter()
    n_total_epochs = config.n_epochs_growth_phase + config.n_epochs_finalization_phase
    for current_epoch in range(n_total_epochs):

        if config.use_loss_perturbation and current_epoch == config.n_loss_interpolation_epochs:
            print(
                f"Epoch: {current_epoch}/{n_total_epochs-1} - "
                "Loss Interpolation phase DONE. Loss used for rendering will no longer be perturbed!"
            )

        if current_epoch == config.n_epochs_growth_phase:
            print(f"Epoch: {current_epoch}/{n_total_epochs-1} - Growth phase DONE. Finalization phase starts!")

        # Growth phase
        do_growth = current_epoch % config.growth_interval == 0 and current_epoch < config.n_epochs_growth_phase
        if do_growth or current_epoch == 0:
            perturbed_loss = get_perturbed_loss(
                config=config,
                current_loss=current_l1_loss_value,
                current_epoch=current_epoch,
            )
            difference_image = get_difference_image(current_image, target_image_hwc)
            n_growth_samples_used = config.primary_samples if current_epoch == 0 else config.n_growth_samples
            in_place_add_samples_starting_from(
                config=config,
                splat_weights=splat_weights,
                start_index=current_sample_end_pointer,
                n_samples=n_growth_samples_used,
                target_image=target_image_hwc,
                current_l1_loss=perturbed_loss,
                difference_image=difference_image,
                target_gradients=target_gradients,
            )

            # Update both the rendering and optimization masks with the newly added splats
            new_sample_end_pointer = current_sample_end_pointer + n_growth_samples_used
            samples_rendering_mask[current_sample_end_pointer:new_sample_end_pointer] = True
            samples_optimization_mask[current_sample_end_pointer:new_sample_end_pointer] = True
            current_sample_end_pointer = new_sample_end_pointer

        # Render the splats
        splats_weights_for_rendering = splat_weights[samples_rendering_mask]
        current_image = render_2d_texture_splats_batched(
            splat_weights=splats_weights_for_rendering,
            textures=textures,
            image_size=config.target_image_load_size,
        )

        # Compute loss and gradients
        current_l1_loss_tensor = torch.nn.L1Loss()(current_image, target_image_hwc)
        current_l1_loss_value = current_l1_loss_tensor.item()
        optimizer.zero_grad()
        current_l1_loss_tensor.backward()

        # Pruning
        do_pruning = current_epoch % config.pruning_interval == 0 and current_epoch > 0
        if do_pruning:
            # Prune splats with low gradients
            # They will still be rendered, but no longer optimized!
            # This is intended to stop optimizing splats that already have near-optimal components
            # Of course later splats could change the situation again,
            # but it's nice to allow some pruning for the sake of performance,
            # and to reinforce the "painting" effect, where older paint is no longer changed
            # once new paint is applied on top
            gradient_norms = torch.norm(splat_weights.grad, dim=1, p=2)
            gradient_norms_small_bools_non_masked = gradient_norms < config.pruning_gradient_threshold
            gradient_norms_small_bools = gradient_norms_small_bools_non_masked & samples_optimization_mask
            indices_to_remove_based_on_gradient = gradient_norms_small_bools.nonzero(as_tuple=True)[0]
            if len(indices_to_remove_based_on_gradient) > 0:
                print(f"Pruned {len( indices_to_remove_based_on_gradient )} points based on Gradient Norm")
            samples_optimization_mask[indices_to_remove_based_on_gradient] = False

            # Prune splats with low scales
            # They will no longer be optimized and no longer be rendered!
            # This is intended to remove points
            # where the optimizer tries to get rid of bad points,
            # by reducing their scale
            scale = decode(splat_weights, Columns.scales).squeeze()
            scale_small_bools_non_masked = scale < config.pruning_scale_threshold
            scale_small_bools = scale_small_bools_non_masked & samples_rendering_mask
            indices_to_remove_based_on_scale = scale_small_bools.nonzero(as_tuple=True)[0]
            if len(indices_to_remove_based_on_scale) > 0:
                print(f"Pruned {len( indices_to_remove_based_on_scale )} points based on Scale")
            samples_optimization_mask[indices_to_remove_based_on_scale] = False
            samples_rendering_mask[indices_to_remove_based_on_scale] = False

        # Every epoch, zero-out the gradients of pruned points.
        # This should speed up the optimization step, but you won't notice it much,
        # as rendering is a severe bottleneck, not much impacted by the pruning
        # Make sure this happens after computing gradients, and before optimization
        if current_epoch > 0:
            # We use a cast to satisfy mypy, we know grad is not None
            cast(torch.Tensor, splat_weights.grad).data[~samples_optimization_mask] = 0.0

        # Optimize
        if current_epoch == n_total_epochs - 1:
            print('Dont optimize after final epoch')
            optimizer.step()
        epoch_end = time.perf_counter()
        loss_history.append((epoch_end, current_l1_loss_tensor.item()))
        epoch_time = epoch_end - last_epoch_end
        last_epoch_end = epoch_end

        # Logging
        if current_epoch % config.display_interval == 0 or current_epoch == n_total_epochs - 1:
            _write_image(current_image.clone().detach().cpu().numpy(), output_directory / f"{current_epoch}_batched.jpg")
            n_rendering_samples = samples_rendering_mask.sum().item()
            n_optimization_samples = samples_optimization_mask.sum().item()
            message = (
                f"Epoch: {current_epoch}/{n_total_epochs-1}, "
                f"Time: {epoch_time:.2f}s, "
                f"Points: {n_optimization_samples}/{n_rendering_samples}/{n_total_expected_samples}, "
                f"Loss: {current_l1_loss_tensor:.3f}, "
            )
            if config.log_loss_perturbation and do_growth:
                message += f"- Previous Loss was perturbed to {perturbed_loss:.3f}."
            print(message)

    # Postprocessing
    print(f"Final Loss: {current_l1_loss_tensor}")
    write_loss_graphs(loss_history=loss_history, output_directory=output_directory)
    write_gif_for_folder(output_directory=output_directory)
    write_high_res_result(splats_weights=splats_weights_for_rendering, config=config, output_directory=output_directory)


def get_image_size_high_res(
    image_path: str,
    target_min_extent=2160,
    target_max_extent: int = 3840,
) -> Size:
    image = Image.open(image_path)
    original_width, original_height = image.size

    # Determine the current min and max extents of the original image
    original_min_extent = min(original_width, original_height)
    original_max_extent = max(original_width, original_height)

    # Calculate the scaling factor for both dimensions
    scale_min = target_min_extent / original_min_extent
    scale_max = target_max_extent / original_max_extent

    # Choose the larger scaling factor to ensure both dimensions meet the target resolution
    scale_factor = max(scale_min, scale_max)

    # Calculate the new dimensions based on the chosen scale factor
    new_width = int(original_width * scale_factor)
    new_height = int(original_height * scale_factor)
    new_size = Size(width=new_width, height=new_height)
    return new_size


def write_high_res_result(splats_weights: torch.Tensor, config: TrainingConfig, output_directory: Path) -> None:
    print("Creating High Res result.")
    image_size_high_res = get_image_size_high_res(config.target_image_path)
    texture_load_size_high_res = get_texture_load_size_for_target_image_load_size(image_size_high_res)

    # Render on the cpu this time to prevent out of memory errors!
    config.device = torch.device("cpu")
    config.texture_load_size = texture_load_size_high_res
    textures_high_res = _load_textures(config)
    sequential_rendering_result = render_2d_texture_splats_sequential(
        splat_weights=splats_weights.detach().cpu(),
        textures=textures_high_res,
        image_size=image_size_high_res,
    )

    # Write the result
    result_path = output_directory / "render__high_res.png"
    _write_image(sequential_rendering_result.cpu().detach().numpy(), result_path)
    print(f"Wrote High Res result to {result_path}")

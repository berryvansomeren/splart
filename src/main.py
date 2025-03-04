import numpy as np
import torch
import os
import time
from typing import cast
from torch.optim import Adam
from datetime import datetime
from PIL import Image
import random
import matplotlib.pyplot as plt

from config import load_training_config, TrainingConfig
from grow import in_place_add_samples_starting_from
from loss_perturbation import get_perturbed_loss
from renderer import render_2d_texture_splatting
from splat_components import init_empty_splat_weights
from make_gif import write_gif_for_folder


def set_random_seeds() -> None:
    universal_seed = 1337
    random.seed(universal_seed)
    np.random.seed(universal_seed)
    torch.manual_seed(universal_seed)
    torch.cuda.manual_seed_all(universal_seed)


def _load_target_image(config: TrainingConfig) -> tuple[torch.Tensor, np.ndarray]:
    image_file = (
        Image.open(config.target_image_file_path)
        .resize((config.target_image_load_size[1], config.target_image_load_size[0]))
        .convert("RGB")
    )
    target_image_np = np.array(image_file) / 255.0
    target_image_tensor = torch.tensor(target_image_np, dtype=torch.float32, device=config.device)
    return target_image_tensor, target_image_np


def _load_texture(config: TrainingConfig) -> torch.Tensor:
    texture_size_2d = (config.texture_load_size, config.texture_load_size)
    texture_file = Image.open(config.texture_image_file_path).convert("RGB").resize(texture_size_2d)
    texture_np = np.array(texture_file)  # Shape: (H, W, 3)
    red_channel_np = texture_np[..., 0] / 255.0  # Extract Red channel (H, W)
    texture_tensor = torch.tensor(red_channel_np, dtype=torch.float32, device=config.device)
    texture = texture_tensor.unsqueeze(0)  # Add channel dimension (1, H, W)
    return texture


def _write_image(image_array: np.ndarray, image_path: str) -> None:
    img = Image.fromarray((image_array * 255).astype(np.uint8))
    img.save(image_path)


def write_loss_graphs(loss_history: list[tuple[float, float]], output_folder: str) -> None:
    epoch_durations, epoch_losses = zip(*loss_history)

    plt.figure()
    plt.plot(epoch_losses, label="Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss Over Epochs")
    plt.legend()
    plt.grid()
    file_path = os.path.join(output_folder, "loss_over_epochs.png")
    plt.savefig(file_path, dpi=300)
    print(f"Wrote 'Loss over Epochs'-Graph to {file_path}")

    plt.figure()
    epoch_times = np.cumsum(epoch_durations) - epoch_durations[0]
    plt.plot(epoch_times, epoch_losses, label="Loss")
    plt.xlabel("Time")
    plt.ylabel("Loss")
    plt.title("Training Loss Over Time")
    plt.legend()
    plt.grid()
    file_path = os.path.join(output_folder, "loss_over_time.png")
    plt.savefig(file_path, dpi=300)
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


def main(config: TrainingConfig) -> None:
    run_start = datetime.now()
    set_random_seeds()

    output_folder = "output/" + run_start.strftime("%Y_%m_%d-%H_%M_%S")
    print(f"Writing output to: {output_folder}")
    os.makedirs(output_folder, exist_ok=True)

    target_image, target_image_np = _load_target_image(config)
    texture = _load_texture(config)

    current_l1_loss_value = 1.0
    perturbed_loss = 1.0
    current_sample_end_pointer = 0
    n_total_expected_samples = config.primary_samples + config.backup_samples
    splat_weights = init_empty_splat_weights(device=config.device, n_samples=n_total_expected_samples)

    # Initial splats
    in_place_add_samples_starting_from(
        config=config,
        splat_weights=splat_weights,
        start_index=current_sample_end_pointer,
        n_samples=config.primary_samples,
        target_image_np=target_image_np,
        current_l1_loss=current_l1_loss_value,
        current_epoch=0,
    )
    current_sample_end_pointer += config.primary_samples

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

        if config.use_loss_perturbation and current_epoch == config.n_loss_perturbation_epochs:
            print("Perturbation phase DONE. Loss used for rendering will no longer be perturbed!")

        if current_epoch == config.n_epochs_growth_phase:
            print("Growth phase DONE. Finalization phase starts!")

        # Growth phase
        do_growth = current_epoch % config.growth_interval == 0 and 0 < current_epoch < config.n_epochs_growth_phase
        if do_growth:
            if current_epoch < config.n_loss_perturbation_epochs:
                perturbed_loss = get_perturbed_loss(
                    current_loss=current_l1_loss_value,
                    current_epoch=current_epoch,
                    n_loss_perturbation_epochs=config.n_loss_perturbation_epochs,
                    expected_loss_after_perturbation_epochs=config.expected_loss_after_perturbation_epochs,
                )
                current_l1_loss_value = perturbed_loss

            in_place_add_samples_starting_from(
                config=config,
                splat_weights=splat_weights,
                start_index=current_sample_end_pointer,
                n_samples=config.n_growth_samples,
                target_image_np=target_image_np,
                current_l1_loss=current_l1_loss_value,
                current_epoch=current_epoch,
            )

            # Update both the rendering and optimization masks with the newly added splats
            new_sample_end_pointer = current_sample_end_pointer + config.n_growth_samples
            samples_rendering_mask[current_sample_end_pointer:new_sample_end_pointer] = True
            samples_optimization_mask[current_sample_end_pointer:new_sample_end_pointer] = True
            current_sample_end_pointer = new_sample_end_pointer

        # Render the splats
        splats_weights_for_rendering = splat_weights[samples_rendering_mask]
        current_image = render_2d_texture_splatting(
            splat_weights=splats_weights_for_rendering,
            texture=texture,
            image_size=config.target_image_load_size,
            device=config.device,
        )

        # Compute loss and gradients
        current_l1_loss_tensor = torch.nn.L1Loss()(current_image, target_image)
        current_l1_loss_value = current_l1_loss_tensor.item()
        optimizer.zero_grad()
        current_l1_loss_tensor.backward()

        # Pruning
        do_pruning = current_epoch % config.pruning_interval == 0 and current_epoch > 0
        if do_pruning:
            # Prune splats with low gradients
            # They will still be rendered, but no longer optimized!
            gradient_norms = torch.norm(splat_weights.grad, dim=1, p=2)
            gradient_norms_small_bools_non_masked = gradient_norms < config.pruning_gradient_threshold
            gradient_norms_small_bools = gradient_norms_small_bools_non_masked & samples_optimization_mask
            indices_to_remove_based_on_gradient = gradient_norms_small_bools.nonzero(as_tuple=True)[0]
            if len(indices_to_remove_based_on_gradient) > 0:
                print(f"Pruned {len( indices_to_remove_based_on_gradient )} points based on gradients")
            samples_optimization_mask[indices_to_remove_based_on_gradient] = False

        # Every epoch, zero-out the gradients of pruned points.
        # This should speed up the optimization step, but you won't notice it much, as rendering is the bottleneck
        if current_epoch > 0:
            # We use a cast to satisfy mypy, we know grad is not None
            cast(torch.Tensor, splat_weights.grad).data[~samples_optimization_mask] = 0.0

        # Optimize
        optimizer.step()
        epoch_end = time.perf_counter()
        loss_history.append((epoch_end, current_l1_loss_tensor.item()))
        epoch_time = epoch_end - last_epoch_end
        last_epoch_end = epoch_end

        # Logging
        if current_epoch % config.display_interval == 0 or current_epoch == n_total_epochs - 1:
            current_image_np = current_image.cpu().detach().numpy()
            _write_image(current_image_np, os.path.join(output_folder, f"{current_epoch}.jpg"))
            n_rendering_samples = samples_rendering_mask.sum().item()
            n_optimization_samples = samples_optimization_mask.sum().item()
            message = (
                f"Epoch: {current_epoch}/{n_total_epochs-1}, "
                f"Time: {epoch_time:.2f}s, "
                f"Points: {n_optimization_samples}/{n_rendering_samples}/{n_total_expected_samples}, "
                f"Loss: {current_l1_loss_tensor:.3f}, "
            )
            if config.log_loss_perturbation and do_growth and current_epoch < config.n_loss_perturbation_epochs:
                message += f" - Loss of previous epoch was perturbed to {perturbed_loss:.3f}."
            print(message)

    # Postprocessing
    print(f"Final Loss: {current_l1_loss_tensor}")
    write_loss_graphs(loss_history=loss_history, output_folder=output_folder)
    write_gif_for_folder(output_folder=output_folder)


if __name__ == "__main__":
    main_start_time = time.perf_counter()
    main(config=load_training_config())
    main_end_time = time.perf_counter()
    main_elapsed_time = main_end_time - main_start_time
    print(f"Done! Took: {main_elapsed_time:.6f} seconds")

import numpy as np
import torch.nn.functional as F
import torch.nn as nn
import torch
import os
import yaml
from torch.optim import Adam
from datetime import datetime
from PIL import Image

from loss import combined_loss


def generate_2D_gaussian_splatting(
    kernel_size,
    scale,
    rotation,
    coords,
    colours,
    image_size=(256, 256, 3),
    device="cpu",
):
    batch_size = colours.shape[0]

    # Ensure scale and rotation have the correct shape
    scale = scale.view(batch_size, 2)
    rotation = rotation.view(batch_size)

    # Compute the components of the covariance matrix
    cos_rot = torch.cos(rotation)
    sin_rot = torch.sin(rotation)

    R = torch.stack(
        [
            torch.stack([cos_rot, -sin_rot], dim=-1),
            torch.stack([sin_rot, cos_rot], dim=-1),
        ],
        dim=-2,
    )

    S = torch.diag_embed(scale)

    # Compute covariance matrix: RSS^TR^T
    covariance = R @ S @ S @ R.transpose(-1, -2)

    # Compute inverse covariance
    inv_covariance = torch.inverse(covariance)

    # Create the kernel
    x = torch.linspace(-5, 5, kernel_size, device=device)
    y = torch.linspace(-5, 5, kernel_size, device=device)
    xx, yy = torch.meshgrid(x, y, indexing="ij")
    xy = torch.stack([xx, yy], dim=-1).unsqueeze(0).expand(batch_size, -1, -1, -1)

    z = torch.einsum("bxyi,bij,bxyj->bxy", xy, -0.5 * inv_covariance, xy)
    kernel = torch.exp(z) / (
        2 * torch.tensor(np.pi, device=device) * torch.sqrt(torch.det(covariance))
    ).view(batch_size, 1, 1)

    # Normalize the kernel
    kernel_max = kernel.amax(dim=(-2, -1), keepdim=True)
    kernel_normalized = kernel / kernel_max

    # Reshape the kernel for RGB channels
    kernel_rgb = kernel_normalized.unsqueeze(1).expand(-1, 3, -1, -1)

    # Add padding to match image size
    pad_h = image_size[0] - kernel_size
    pad_w = image_size[1] - kernel_size

    if pad_h < 0 or pad_w < 0:
        raise ValueError("Kernel size should be smaller or equal to the image size.")

    padding = (pad_w // 2, pad_w // 2 + pad_w % 2, pad_h // 2, pad_h // 2 + pad_h % 2)
    kernel_rgb_padded = F.pad(kernel_rgb, padding, "constant", 0)

    # Translate the kernel
    b, c, h, w = kernel_rgb_padded.shape
    theta = torch.zeros(b, 2, 3, dtype=torch.float32, device=device)
    theta[:, 0, 0] = 1.0
    theta[:, 1, 1] = 1.0
    theta[:, :, 2] = coords

    grid = F.affine_grid(theta, size=(b, c, h, w), align_corners=True)
    kernel_rgb_padded_translated = F.grid_sample(
        kernel_rgb_padded, grid, align_corners=True
    )

    # Apply colors and sum the layers
    rgb_values_reshaped = colours.unsqueeze(-1).unsqueeze(-1)
    final_image_layers = rgb_values_reshaped * kernel_rgb_padded_translated
    final_image = final_image_layers.sum(dim=0)
    final_image = torch.clamp(final_image, 0, 1)
    final_image = final_image.permute(1, 2, 0)

    return final_image



def give_required_data(input_coords, image_size, device, image_array):

    # normalising pixel coordinates [-1,1]
    coords = torch.tensor(
        input_coords / [image_size[0], image_size[1]], device=device
    ).float()
    center_coords_normalized = torch.tensor([0.5, 0.5], device=device).float()
    coords = (center_coords_normalized - coords) * 2.0

    # Fetching the colour of the pixels in each coordinates
    colour_values = [image_array[coord[1], coord[0]] for coord in input_coords]
    colour_values_np = np.array(colour_values)
    colour_values_tensor = torch.tensor(colour_values_np, device=device).float()

    return colour_values_tensor, coords


def train_step(
    W,
    optimizer,
    persistent_mask,
    target_tensor,
    KERNEL_SIZE,
    image_size,
    device,
    lambda_param,
):
    optimizer.zero_grad()

    output = W[persistent_mask]
    scale = torch.sigmoid(output[:, 0:2])
    rotation = np.pi / 2 * torch.tanh(output[:, 2])
    colours = torch.sigmoid(output[:, 4:7])
    pixel_coords = torch.tanh(output[:, 7:9])

    g_tensor_batch = generate_2D_gaussian_splatting(
        KERNEL_SIZE, scale, rotation, pixel_coords, colours, image_size, device=device
    )

    loss = combined_loss(g_tensor_batch, target_tensor, lambda_param)
    loss.backward()

    if persistent_mask is not None:
        W.grad.data[~persistent_mask] = 0.0



    # --------------------------------
    if epoch % densification_interval == 0 and epoch > 0 :

        # Calculate the norm of gradients
        gradient_norms = torch.norm( W.grad[ persistent_mask ][ :, 7 :9 ], dim = 1, p = 2 )
        gaussian_norms = torch.norm( torch.sigmoid( W.data[ persistent_mask ][ :, 0 :2 ] ), dim = 1, p = 2 )

        sorted_grads, sorted_grads_indices = torch.sort( gradient_norms, descending = True )
        sorted_gauss, sorted_gauss_indices = torch.sort( gaussian_norms, descending = True )

        large_gradient_mask = (sorted_grads > grad_threshold)
        large_gradient_indices = sorted_grads_indices[ large_gradient_mask ]

        large_gauss_mask = (sorted_gauss > gauss_threshold)
        large_gauss_indices = sorted_gauss_indices[ large_gauss_mask ]

        common_indices_mask = torch.isin( large_gradient_indices, large_gauss_indices )
        common_indices = large_gradient_indices[ common_indices_mask ]
        distinct_indices = large_gradient_indices[ ~common_indices_mask ]

    # Split points with large coordinate gradient and large gaussian values and descale their gaussian
    if len( common_indices ) > 0 :
        print( f"Number of splitted points: {len( common_indices )}" )
        start_index = current_marker + 1
        end_index = current_marker + 1 + len( common_indices )
        persistent_mask[ start_index : end_index ] = True
        W.data[ start_index :end_index, : ] = W.data[ common_indices, : ]
        scale_reduction_factor = 1.6
        W.data[ start_index :end_index, 0 :2 ] /= scale_reduction_factor
        W.data[ common_indices, 0 :2 ] /= scale_reduction_factor
        current_marker = current_marker + len( common_indices )

    # Clone it points with large coordinate gradient and small gaussian values
    if len( distinct_indices ) > 0 :
        print( f"Number of cloned points: {len( distinct_indices )}" )
        start_index = current_marker + 1
        end_index = current_marker + 1 + len( distinct_indices )
        persistent_mask[ start_index : end_index ] = True
        W.data[ start_index :end_index, : ] = W.data[ distinct_indices, : ]

        # Calculate the movement direction based on the positional gradient
        positional_gradients = W.grad[ distinct_indices, 7 :9 ]
        gradient_magnitudes = torch.norm( positional_gradients, dim = 1, keepdim = True )
        normalized_gradients = positional_gradients / (gradient_magnitudes + 1e-8)  # Avoid division by zero

        # Define a step size for the movement
        step_size = 0.01

        # Move the cloned Gaussians
        W.data[ start_index :end_index, 7 :9 ] += step_size * normalized_gradients

        current_marker = current_marker + len( distinct_indices )
        # --------------------------------


    optimizer.step()
    return loss.item(), g_tensor_batch


def main():
    with open("config.yml", "r") as config_file:
        config = yaml.safe_load(config_file)

    KERNEL_SIZE = config["KERNEL_SIZE"]
    image_size = tuple(config["image_size"])
    primary_samples = config["primary_samples"]
    backup_samples = config["backup_samples"]
    num_epochs = config["num_epochs"]
    learning_rate = config["learning_rate"]
    image_file_name = config["image_file_name"]
    display_interval = config["display_interval"]
    lambda_param = config.get("lambda_param", 0.2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"DEVICE: {device}")

    num_samples = primary_samples + backup_samples
    image = (
        Image.open(image_file_name)
        .resize((image_size[0], image_size[0]))
        .convert("RGB")
    )
    image_array = np.array(image) / 255.0
    target_tensor = torch.tensor(image_array, dtype=torch.float32, device=device)

    coords = np.random.randint(0, image_size[:2], size=(num_samples, 2))
    colour_values, pixel_coords = give_required_data(
        coords, image_size, device, image_array
    )

    W_values = torch.cat(
        [
            torch.logit(torch.rand(num_samples, 2, device=device)),
            torch.atanh(2 * torch.rand(num_samples, 1, device=device) - 1),
            torch.logit(torch.rand(num_samples, 1, device=device)),
            torch.logit(colour_values),
            torch.atanh(pixel_coords),
        ],
        dim=1,
    )

    W = nn.Parameter(W_values)
    optimizer = Adam([W], lr=learning_rate)

    persistent_mask = torch.cat(
        [
            torch.ones(primary_samples, dtype=bool),
            torch.zeros(backup_samples, dtype=bool),
        ],
        dim=0,
    )

    now = datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
    print(f"Writing output to: {now}")
    os.makedirs(now, exist_ok=True)

    loss_history = []
    for epoch in range(num_epochs):
        loss, g_tensor_batch = train_step(
            W,
            optimizer,
            persistent_mask,
            target_tensor,
            KERNEL_SIZE,
            image_size,
            device,
            lambda_param,
        )
        loss_history.append(loss)

        if epoch % display_interval == 0:
            generated_array = g_tensor_batch.cpu().detach().numpy()
            img = Image.fromarray((generated_array * 255).astype(np.uint8))
            img.save(os.path.join(now, f"{epoch}.jpg"))
            print(
                f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss}, on {persistent_mask.sum().item()} points"
            )


if __name__ == "__main__":
    main()

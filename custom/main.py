import numpy as np
import torch
import matplotlib.pyplot as plt

from differentiable_rendering import render_image
from optimize_2d_gaussian_splatting import optimize_2d_gaussian_splatting_generator
from primitives import Splats, torch_to_np_image


def plot(name: str, image: np.ndarray, z: int) -> None:
    plt.subplot(1, 2, z)
    plt.imshow(image, cmap="hot")
    plt.title(f"{name} Image")
    plt.colorbar()


def random_init_splats(
    n_splats: int = 4,
    domain: int = 500,  # assume square domain
    radius: int = 10,
    sigma: int = 5,
) -> Splats:
    return Splats(
        positions=torch.nn.Parameter(torch.rand(n_splats, 2) * domain),
        radii=torch.nn.Parameter(torch.ones(n_splats) * radius),
        sigmas=torch.nn.Parameter(torch.ones(n_splats) * sigma),
    )


def main() -> None:
    target_splats = random_init_splats()
    torch_target_image = render_image(size=(500, 500), splats=target_splats)
    np_target_image = torch_to_np_image(torch_target_image)

    initial_optimization_splats = random_init_splats()
    generator = optimize_2d_gaussian_splatting_generator(
        target_image=torch_target_image, splats=initial_optimization_splats
    )

    for result_image in generator:
        plt.clf()
        plot("Target", np_target_image, 1)
        plot("Result", result_image, 2)
        plt.draw()
        plt.pause(0.1)

    plt.show()


if __name__ == "__main__":
    main()

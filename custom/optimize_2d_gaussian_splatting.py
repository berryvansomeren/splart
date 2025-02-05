import torch
import torch.optim as optim
from typing import Generator

from differentiable_rendering import render_image
from primitives import Splats, torch_to_np_image, NPImage


def optimize_2d_gaussian_splatting_generator(
    target_image: torch.Tensor,
    splats: Splats,
    learning_rate: float = 0.1,
    n_epochs: int = 1000,
    yield_every: int = 10,
) -> Generator[NPImage, None, None]:
    optimizer = optim.Adam(
        [splats.positions, splats.radii, splats.sigmas], lr=learning_rate
    )

    for epoch in range(n_epochs):
        optimizer.zero_grad()
        generated_image = render_image(
            size=target_image.size(),
            splats=splats,
        )
        loss = torch.mean((generated_image - target_image) ** 2)  # (Mean Squared Error)
        loss.backward(retain_graph=True)
        optimizer.step()

        if epoch % yield_every == 0:
            print(f"Epoch {epoch}/{n_epochs}, Loss: {loss.item()}")
            yield torch_to_np_image(generated_image)

    result_torch = render_image(
        size=target_image.size(),
        splats=splats,
    )
    yield torch_to_np_image(result_torch)

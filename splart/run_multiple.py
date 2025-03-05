import time

from splart.training_config import TrainingConfig, make_full_config
from splart.run_optimization import run_optimization


def make_config(
    target_image_path: str,
    texture_image_path: str,
) -> TrainingConfig:
    return make_full_config(
        target_image_path=target_image_path,
        texture_image_path=texture_image_path,
        primary_samples=15,
        n_epochs_growth_phase=300,
        growth_interval=1,
        n_growth_samples=1,
    )


def run_multiple() -> None:
    configs = [
        make_config(
            target_image_path="./input/2d_gaussian_splatting.png", texture_image_path="./brushes/canvas/canvas_1.png"
        ),
    ]
    for i, config in enumerate(configs):
        print("-" * 64)
        print(f"Running config {i}/{len(configs)-1}")
        start_time = time.perf_counter()
        run_optimization(config=config)
        end_time = time.perf_counter()
        elapsed_time = end_time - start_time
        print(f"Done with config! Took: {elapsed_time:.6f} seconds")
    print("Done with all configs!")


if __name__ == "__main__":
    run_multiple()

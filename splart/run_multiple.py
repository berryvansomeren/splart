import time

from splart.training_config import TrainingConfig, make_full_config_with_defaults
from splart.run_optimization import run_optimization


def make_config(
    target_image_path: str,
    texture_directory_path: str,
) -> TrainingConfig:
    return make_full_config_with_defaults(
        target_image_path=target_image_path,
        texture_directory_path=texture_directory_path,
        primary_samples=50,
        n_epochs_growth_phase=250,
        growth_interval=1,
        n_growth_samples=2,
    )


def run_multiple() -> None:
    configs = [
        make_config(
            target_image_path="./input/whale.jpg",
            texture_directory_path="./brushes/watercolor/",
        ),
        make_config(
            target_image_path="./input/lemur.jpg",
            texture_directory_path="./brushes/sketch/",
        ),
        make_config(
            target_image_path="./input/kingfisher.jpg",
            texture_directory_path="./brushes/oil/",
        ),
        make_config(
            target_image_path="./input/bird.jpg",
            texture_directory_path="./brushes/sketch/",
        ),
        make_config(
            target_image_path="./input/bird.jpg",
            texture_directory_path="./brushes/watercolor/",
        ),
        make_config(
            target_image_path="./input/young_deer.jpg",
            texture_directory_path="./brushes/canvas/",
        ),
        make_config(
            target_image_path="./input/zion.jpg",
            texture_directory_path="./brushes/watercolor/",
        ),
        make_config(
            target_image_path="./input/florence.jpg",
            texture_directory_path="./brushes/watercolor/",
        ),
        make_config(
            target_image_path="./input/florence.jpg",
            texture_directory_path="./brushes/oil/",
        ),
        make_config(
            target_image_path="./input/city.jpg",
            texture_directory_path="./brushes/oil/",
        ),
        make_config(
            target_image_path="./input/deer.jpg",
            texture_directory_path="./brushes/sketch/",
        ),
        make_config(
            target_image_path="./input/deer.jpg",
            texture_directory_path="./brushes/canvas/",
        ),
        make_config(
            target_image_path="./input/flamingo.jpg",
            texture_directory_path="./brushes/oil/",
        ),
        make_config(
            target_image_path="./input/flamingo.jpg",
            texture_directory_path="./brushes/canvas/",
        ),
        make_config(
            target_image_path="./input/tiger.jpg",
            texture_directory_path="./brushes/watercolor/",
        ),
        make_config(
            target_image_path="./input/frog.jpg",
            texture_directory_path="./brushes/canvas/",
        ),
    ]
    for i, config in enumerate(configs):
        print("-" * 64)
        print(f"Running config {i}/{len(configs)-1}")
        start_time = time.perf_counter()
        run_optimization(config=config)
        end_time = time.perf_counter()
        elapsed_time = end_time - start_time
        print(f"Done with config! Took: {elapsed_time:.2f} seconds")
    print("-" * 64)
    print("Done with all configs!")


if __name__ == "__main__":
    run_multiple()

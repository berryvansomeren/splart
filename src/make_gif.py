import io
import os

import subprocess
from PIL import Image


def size_bytes_to_mib(n_bytes: int) -> float:
    return n_bytes / (1024 * 1024)


def get_size_mib(data: str | bytes) -> float:
    return size_bytes_to_mib(len(data))


def load_images_from_folder(folder_path: str) -> list[str]:
    files = sorted(os.listdir(folder_path))
    image_paths = []
    for filename in files:
        image_path = os.path.join(folder_path, filename)
        if os.path.isfile(image_path) and image_path.lower().endswith(".jpg"):
            image_paths.append(image_path)
    image_paths_sorted = sorted(image_paths, key=lambda x: int("".join(c for c in x if c.isdigit())))
    return image_paths_sorted


def make_gif_for_images(image_paths: list[str]) -> bytes:
    print("Making GIF")

    FPS = 5
    FINAL_FRAME_REPEAT_SECONDS = 3
    FINAL_FRAME_REPEAT_FRAMES = FPS * FINAL_FRAME_REPEAT_SECONDS

    pil_images = []
    for i, image_path in enumerate(image_paths):
        current_image_file = Image.open(image_path)
        current_image = current_image_file.convert("RGB")  # Ensure it's in RGB mode
        pil_images.append(current_image)

    gif_buffer = io.BytesIO()
    pil_images[0].save(
        gif_buffer,
        format="GIF",
        save_all=True,
        append_images=pil_images[1:] + [pil_images[-1]] * FINAL_FRAME_REPEAT_FRAMES,
        duration=1000 / FPS,  # FPS to duration conversion (ms)
        loop=0,
    )

    original_gif = gif_buffer.getvalue()
    original_size_mib = get_size_mib(original_gif)
    print(f"GIF is {original_size_mib} MiB.")

    optimized_gif = gifsicle_optimize_in_memory(original_gif)
    size_bytes_optimized = get_size_mib(optimized_gif)
    if size_bytes_optimized != original_size_mib:
        print(f"Optimizing the GIF brought it from {original_size_mib} MiB to {size_bytes_optimized} MiB")

    return optimized_gif


def gifsicle_optimize_in_memory(gif_data: bytes, colors: int = 256, options: list[str] | None = None) -> bytes:
    if options is None:
        options = []

    if "--optimize" not in options:
        options.append("--optimize")

    TESTING_ON_WINDOWS = False
    if TESTING_ON_WINDOWS:
        exe = "C:/Users/Berry/Downloads/gifsicle-1.95-win64/gifsicle"
    else:
        exe = "gifsicle"

    command = [exe, *options, "--colors", str(colors)]
    try:
        proc = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        result_gif_data, stderr_data = proc.communicate(input=gif_data)
        if stderr_data:
            raise Exception(stderr_data.decode())

    except FileNotFoundError:
        print("The gifsicle library was not found on your system. GIF Optimization is skipped.")
        return gif_data

    return result_gif_data


def write_gif(gif_data: bytes, output_path: str) -> None:
    with open(output_path, "wb") as f:
        f.write(gif_data)


def write_gif_for_folder(output_folder: str) -> None:
    image_paths = load_images_from_folder(output_folder)
    gif_bytes = make_gif_for_images(image_paths)
    gif_path = os.path.join(output_folder, "output.gif")
    write_gif(gif_bytes, gif_path)
    print(f"Wrote GIF to {gif_path}")

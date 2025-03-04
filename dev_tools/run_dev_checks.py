import subprocess
import sys


"""Using this script instead of a Makefile to make things a bit more portable"""


def poetry_run(command) -> None:
    poetry_command = f"poetry run {command}"
    print(f'Running command "{poetry_command}"')
    try:
        subprocess.run(poetry_command, shell=True, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Command failed: {e}", file=sys.stderr)
        sys.exit(1)


def make_nice() -> None:
    poetry_run("ruff check ../")
    poetry_run("black --line-length 120 ../")
    poetry_run("mypy ../")


if __name__ == "__main__":
    make_nice()

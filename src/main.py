"""Entry point for the cosign-public-data-poc project."""

import platform


READY_MESSAGE: str = "cosign-public-data-poc ready"


def get_python_version() -> str:
    """Return the current Python runtime version string."""
    return platform.python_version()


def main() -> None:
    """Run a basic startup check for the project skeleton."""
    # Print a simple readiness message so we know the module runs correctly.
    print(READY_MESSAGE)

    # Print the active Python version for quick environment verification.
    print(f"Python version: {get_python_version()}")


if __name__ == "__main__":
    main()

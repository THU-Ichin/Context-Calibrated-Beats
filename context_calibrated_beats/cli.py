"""Installed command-line entry point for CCB."""

from CCB import main as _main


def main() -> int:
    """Run the CCB command-line interface."""
    return _main()


__all__ = ["main"]

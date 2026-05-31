from __future__ import annotations

from . import assemble


def main() -> None:
    print("Building GENIE validation design matrix (clean-room)...")
    assemble.build_and_write()
    print("Done.")


if __name__ == "__main__":
    main()

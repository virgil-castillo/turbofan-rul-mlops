"""Run the turbofan FastAPI inference service."""
from __future__ import annotations

import argparse
from collections.abc import Sequence

import uvicorn

from turbofan.inference.service import create_app


def main(argv: Sequence[str] | None = None) -> int:
    """Run a local uvicorn server for the inference API.

    Args:
        argv: Optional command-line arguments.

    Returns:
        Process exit code.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=8000, type=int)
    args = parser.parse_args(argv)
    uvicorn.run(create_app(), host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

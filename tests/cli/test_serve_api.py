"""Tests for the turbofan-serve-api command."""
from __future__ import annotations

import pytest

from turbofan.cli import serve_api


class _RunRecorder:
    """Record the arguments passed to a patched ``uvicorn.run``."""

    def __init__(self) -> None:
        """Initialize without a recorded call."""
        self.app: object | None = None
        self.kwargs: dict[str, object] = {}
        self.called: bool = False

    def __call__(self, app: object, **kwargs: object) -> None:
        """Capture the launch application and keyword arguments.

        Args:
            app: ASGI application handed to uvicorn.
            **kwargs: Server keyword arguments such as host and port.
        """
        self.app = app
        self.kwargs = kwargs
        self.called = True


class _CreateAppRecorder:
    """Record the arguments passed to a patched ``service.create_app``."""

    def __init__(self, app: object) -> None:
        """Store the sentinel app the recorder should return.

        Args:
            app: Sentinel application object to return from each call.
        """
        self._app = app
        self.kwargs: dict[str, object] = {}
        self.called: bool = False

    def __call__(self, **kwargs: object) -> object:
        """Capture the create-app keyword arguments and return the sentinel.

        Args:
            **kwargs: Keyword arguments forwarded by the CLI.

        Returns:
            The stored sentinel application object.
        """
        self.kwargs = kwargs
        self.called = True
        return self._app


def _patch_dependencies(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[_CreateAppRecorder, _RunRecorder]:
    """Patch ``create_app`` and ``uvicorn.run`` with recording fakes.

    Args:
        monkeypatch: Pytest monkeypatch fixture.

    Returns:
        The create-app and uvicorn.run recorders.
    """
    sentinel_app = object()
    create_app = _CreateAppRecorder(sentinel_app)
    run = _RunRecorder()
    monkeypatch.setattr(serve_api.service, "create_app", create_app)
    monkeypatch.setattr(serve_api.uvicorn, "run", run)
    return create_app, run


def test_serve_api_defaults_forwarded_to_create_app_and_uvicorn(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With no arguments, defaults flow into create_app and uvicorn.run."""
    create_app, run = _patch_dependencies(monkeypatch)

    code = serve_api.main([])

    assert code == 0
    assert create_app.called
    assert create_app.kwargs == {"model_name": None, "alias": None}
    assert run.called
    assert run.app is create_app(model_name=None, alias=None)
    assert run.kwargs == {"host": "127.0.0.1", "port": 8000}


def test_serve_api_parsed_arguments_forwarded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Parsed CLI arguments reach create_app and uvicorn.run unchanged."""
    create_app, run = _patch_dependencies(monkeypatch)

    code = serve_api.main(
        [
            "--host",
            "0.0.0.0",
            "--port",
            "9001",
            "--model",
            "turbofan-ridge-fd001",
            "--alias",
            "staging",
        ]
    )

    assert code == 0
    assert create_app.kwargs == {
        "model_name": "turbofan-ridge-fd001",
        "alias": "staging",
    }
    assert run.kwargs == {"host": "0.0.0.0", "port": 9001}


def test_serve_api_uses_create_app_result_as_uvicorn_app(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The application built by create_app is the one passed to uvicorn.run."""
    sentinel_app = object()

    def fake_create_app(**_kwargs: object) -> object:
        """Return the sentinel application regardless of inputs.

        Args:
            **_kwargs: Ignored create-app keyword arguments.

        Returns:
            The sentinel application object.
        """
        return sentinel_app

    run = _RunRecorder()
    monkeypatch.setattr(serve_api.service, "create_app", fake_create_app)
    monkeypatch.setattr(serve_api.uvicorn, "run", run)

    code = serve_api.main([])

    assert code == 0
    assert run.app is sentinel_app


def test_serve_api_port_must_be_integer() -> None:
    """A non-integer ``--port`` value is rejected by argparse."""
    with pytest.raises(SystemExit) as exc_info:
        serve_api.main(["--port", "not-a-number"])

    assert exc_info.value.code == 2

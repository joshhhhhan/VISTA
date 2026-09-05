"""Shared HTTP transport configuration for ARC remote environments."""

from __future__ import annotations

from typing import Any

from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


ARC_CONNECT_TIMEOUT_SECONDS = 10
ARC_READ_TIMEOUT_SECONDS = 120
ARC_CONNECT_RETRIES = 3


class ArcHTTPAdapter(HTTPAdapter):
    """Use fresh connections and never replay a submitted game action."""

    def __init__(self) -> None:
        super().__init__(
            max_retries=Retry(
                total=None,
                connect=ARC_CONNECT_RETRIES,
                read=0,
                redirect=0,
                status=0,
                other=0,
                allowed_methods=None,
                backoff_factor=0.5,
            )
        )

    def send(self, request: Any, **kwargs: Any) -> Any:
        timeout = kwargs.get("timeout")
        if timeout is None or isinstance(timeout, (int, float)):
            kwargs["timeout"] = (
                ARC_CONNECT_TIMEOUT_SECONDS,
                ARC_READ_TIMEOUT_SECONDS,
            )
        return super().send(request, **kwargs)


def configure_arc_http(owner: Any) -> bool:
    session = getattr(owner, "_session", None)
    if session is None or not callable(getattr(session, "mount", None)):
        return False
    session.headers["Connection"] = "close"
    session.mount("https://", ArcHTTPAdapter())
    session.mount("http://", ArcHTTPAdapter())
    return True

"""Proxy bypass for the launcher's literal-loopback preload call only."""
from __future__ import annotations

from contextlib import contextmanager
import ipaddress
import os
from urllib.parse import urlsplit

from robouse.teacher.admission import FrontierBlocked


@contextmanager
def direct_loopback_environment(url):
    """Restore every proxy variable exactly, including when preload fails.

    httpx eagerly initializes SOCKS proxy support even when NO_PROXY would
    exclude localhost. The original preload API has no trust_env parameter.
    Cloud CLI environments are constructed only after this synchronous scope.
    """
    try:
        parsed = urlsplit(url)
        if (
            parsed.scheme != "http"
            or parsed.username is not None
            or parsed.password is not None
            or not ipaddress.ip_address(parsed.hostname).is_loopback
            or parsed.port is None
            or not 0 < parsed.port < 65536
        ):
            raise ValueError()
    except (TypeError, ValueError):
        raise FrontierBlocked("DIRECT_PRELOAD_REQUIRES_LITERAL_LOOPBACK") from None
    saved = {
        key: value
        for key, value in os.environ.items()
        if key.lower().endswith("_proxy")
    }
    for key in saved:
        os.environ.pop(key, None)
    try:
        yield
    finally:
        for key in list(os.environ):
            if key.lower().endswith("_proxy"):
                os.environ.pop(key, None)
        os.environ.update(saved)

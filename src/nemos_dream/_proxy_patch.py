"""Force aiohttp and httpx (custom-transport form) to honor HTTP_PROXY / HTTPS_PROXY
environment variables.

Both libraries SKIP env-var proxy lookup in certain cases:
- aiohttp: always, unless `trust_env=True` is passed.
- httpx: when `transport=` is provided explicitly (as Data Designer does) or when
  the custom transport doesn't carry a `proxy` argument.

Some corporate networks require an outbound HTTP proxy to reach external
endpoints (e.g. ``integrate.api.nvidia.com``). Without this patch, Data Designer
(httpx+custom-transport) fails at runtime while plain ``httpx.get(...)`` and the
sync ``openai`` client work via default httpx behavior.

Import and call ``apply_proxy_patches()`` once before any library usage
(typically at the top of a stage runner, right after ``load_dotenv()``). It is
a no-op when no proxy env var is set.
"""

from __future__ import annotations

import os

_APPLIED = False


def _proxy_url() -> str | None:
    return (
        os.environ.get("HTTPS_PROXY")
        or os.environ.get("https_proxy")
        or os.environ.get("HTTP_PROXY")
        or os.environ.get("http_proxy")
    )


def _patch_aiohttp() -> None:
    try:
        import aiohttp
    except ImportError:
        return
    orig = aiohttp.ClientSession.__init__
    if getattr(orig, "__nemos_dream_patched__", False):
        return

    def _wrapped(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        kwargs.setdefault("trust_env", True)
        return orig(self, *args, **kwargs)

    _wrapped.__nemos_dream_patched__ = True  # type: ignore[attr-defined]
    aiohttp.ClientSession.__init__ = _wrapped  # type: ignore[assignment]


def _patch_httpx() -> None:
    try:
        import httpx
    except ImportError:
        return
    proxy = _proxy_url()
    if not proxy:
        return

    for cls_name in ("AsyncHTTPTransport", "HTTPTransport"):
        cls = getattr(httpx, cls_name, None)
        if cls is None:
            continue
        orig = cls.__init__
        if getattr(orig, "__nemos_dream_patched__", False):
            continue

        def _make_wrapper(_orig):
            def _wrapped(self, *args, **kwargs):  # type: ignore[no-untyped-def]
                if "proxy" not in kwargs and "mounts" not in kwargs:
                    kwargs["proxy"] = proxy
                return _orig(self, *args, **kwargs)
            _wrapped.__nemos_dream_patched__ = True  # type: ignore[attr-defined]
            return _wrapped

        cls.__init__ = _make_wrapper(orig)  # type: ignore[assignment]


def apply_proxy_patches() -> None:
    global _APPLIED
    if _APPLIED:
        return
    _patch_aiohttp()
    _patch_httpx()
    _APPLIED = True

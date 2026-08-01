"""Shared pytest fixtures/helpers for the splintr Python test suite."""

import pytest

from splintr import _core


def pcre2_available() -> bool:
    """Whether the installed extension was built with the optional `pcre2` feature.

    Reads the capability flag the extension module exports. Deliberately *not*
    inferred from the text of the `ValueError` that `.pcre2(True)` raises when
    the feature is absent: rewording that message would make this silently
    report "absent", and every pcre2 test would skip rather than run. A suite
    that quietly stops testing is worse than one that fails loudly.

    A missing flag means an extension built before the flag existed — a stale
    build, not a build without pcre2 — so let the `AttributeError` surface
    instead of guessing.
    """
    return bool(_core.HAS_PCRE2)


requires_pcre2 = pytest.mark.skipif(
    not pcre2_available(),
    reason="pcre2 feature not compiled into this build (compile with --features pcre2)",
)

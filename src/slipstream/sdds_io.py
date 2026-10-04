"""SDDS values retain the units and meanings of their reader; this module provides shared access to the
binding, rather than physical conversion. See docs/modules/sdds_io.md."""
from __future__ import annotations

import threading
from contextlib import contextmanager

_SDDS_LOCK = threading.Lock()


@contextmanager
def sdds_object(index: int = 0):
    """Yield ``(sdds_module, sdds_object)`` with the global lock held, releasing the object's index
    claim before the lock drops."""
    with _SDDS_LOCK:
        import sdds

        x = sdds.SDDS(index)
        try:
            yield sdds, x
        finally:
            held = getattr(x, "index", None)
            if held is not None:
                del x.index
                sdds.SDDS._used_indices.discard(held)

"""One process-wide lock around the ``sdds`` package's global index
registry.

``soliday.sdds`` (the SDDS toolkit's own Python binding) keeps an unlocked
module-level set of in-use object indices (``SDDS._used_indices``), so
constructing ``sdds.SDDS(0)`` from two threads at once raises
``ValueError: Index 0 is already in use.`` -- and the object holds its
claim on that index for as long as anything references it, so the ``del``
below has to happen *before* the lock is released, not whenever the caller
happens to drop its reference.

The lock has to be **shared across every module that touches SDDS**, not
one per module: the registry it protects is global to the ``sdds``
package, so two modules each holding their own lock still collide (found
exactly that way -- a concurrent run loading a wake table via
``wakefields.shortrange`` and a ramp table via ``lattice.ramp`` at the same
time, with a per-module lock already in place in each).

Three call sites today: ``wakefields.shortrange.ShortRangeWake``,
``lattice.ramp.RampProfile``, ``tracking.elegant.parse``. A fourth should
use this too rather than rolling its own.
"""
from __future__ import annotations

import threading
from contextlib import contextmanager

_SDDS_LOCK = threading.Lock()


@contextmanager
def sdds_object(index: int = 0):
    """Yield ``(sdds_module, sdds_object)`` with the global lock held,
    releasing the object's index claim before the lock drops::

        with sdds_object() as (sdds, x):
            x.load(str(path))
            data = np.array(x.columnData[0][0])

    Do the ``numpy`` conversions *inside* the block (the object is dead
    after it), but keep anything slow and SDDS-independent outside it --
    this serializes every SDDS-touching thread in the process.

    The index is released by hand rather than by dropping the last
    reference: ``SDDS.__del__`` is what frees it, and a ``del x`` here
    can't run it while the *caller's* own ``as (..., x)`` binding is still
    alive -- which it always is, until after this contextmanager returns
    and the lock is already gone. That's a real race, not a theoretical
    one (it reintroduced the exact "Index 0 is already in use." failure
    this module exists to prevent). Deleting ``x.index`` first is what
    stops the object's own later ``__del__`` (it guards on
    ``hasattr(self, 'index')``) from removing an index some *other* thread
    has since claimed.

    ``_used_indices`` is private to ``sdds``; there's no public release
    API. If a future version grows one, use it instead of this.
    """
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

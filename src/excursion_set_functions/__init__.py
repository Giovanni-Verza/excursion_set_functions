from __future__ import annotations

from ._core import get_max_threads, set_num_threads, omp_get_num_threads, numerical, spline, integration, analytical, __doc__, __version__

from . import python

from . import utilities

__all__ = [#"__doc__", 
           "__version__", "__doc__", 
           "get_max_threads", "set_num_threads", "omp_get_num_threads", "numerical", "spline", "integration", "analytical",
           "python", "utilities"
           ]

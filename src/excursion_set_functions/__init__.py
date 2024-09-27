from __future__ import annotations

from ._core import numerical, spline, integration, analytical, __doc__, __version__

from . import python

from . import utilities

__all__ = ["__version__", "__doc__", 
           "numerical", "spline", "integration", "analytical",
           "python", "utilities"
           ]

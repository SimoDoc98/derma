"""Dataset loaders: one module per public dataset (see CONTRIBUTING.md).

Each module has ``status()``, ``prepare()`` and ``load()``. From the command line::

    python -m derma.datasets status
    python -m derma.datasets prepare MAUS --data-dir data
"""

from . import case, maus, sad
from .common import load_prepared

DATASETS = {'CASE': case, 'MAUS': maus, 'SAD': sad}

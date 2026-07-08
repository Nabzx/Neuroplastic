"""Ensure the repository root is importable when running tests without install.

The domain packages are top-level (flat layout), so the repo root must be on
``sys.path``. An editable install (`pip install -e .`) handles this too; this
conftest makes `pytest` work straight from a fresh checkout as well.
"""

import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

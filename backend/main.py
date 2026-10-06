"""Entrypoint when service root is set to 'backend'."""
import sys
from pathlib import Path

_backend_dir = Path(__file__).resolve().parent
_root_dir = _backend_dir.parent
for p in [_backend_dir, _root_dir]:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from backend.api.main import app

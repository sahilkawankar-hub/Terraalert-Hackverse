"""Root entrypoint for Vercel and ASGI runners."""
import sys
from pathlib import Path

_root = Path(__file__).resolve().parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from backend.api.main import app

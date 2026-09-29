import sys
from pathlib import Path

# The tests import the package from the source tree.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

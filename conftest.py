# conftest.py

import os
import sys

# Ensure the project root is on sys.path for all tests
ROOT = os.path.dirname(__file__)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
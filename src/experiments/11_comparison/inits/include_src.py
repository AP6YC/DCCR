"""
Adds the folder above the one holding this file to the path.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parents[1]))

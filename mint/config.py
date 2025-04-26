from os import path
from pathlib import Path


LIB_DIR = path.dirname(path.abspath(__file__))
BASE_DIR = Path(LIB_DIR).parent
DATA_DIR = path.join(BASE_DIR, 'data')

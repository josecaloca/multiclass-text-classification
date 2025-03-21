import os
from pathlib import Path

PARENT_DIR = Path(__file__).parent.resolve().parent
DATA_DIR = PARENT_DIR / 'data'
RAW_DATA_DIR = PARENT_DIR / 'data' / 'raw'
PROCESSED_DATA_DIR = PARENT_DIR / 'data' / 'processed'

if not Path(DATA_DIR).exists():
    os.mkdir(DATA_DIR)

if not Path(RAW_DATA_DIR).exists():
    os.mkdir(RAW_DATA_DIR)

if not Path(PROCESSED_DATA_DIR).exists():
    os.mkdir(PROCESSED_DATA_DIR)

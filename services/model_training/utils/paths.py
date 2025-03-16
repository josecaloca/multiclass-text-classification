import os
from pathlib import Path

PARENT_DIR = Path(__file__).parent.resolve().parent

MODELS_DIR = PARENT_DIR / 'models'

if not Path(MODELS_DIR).exists():
    os.mkdir(MODELS_DIR)

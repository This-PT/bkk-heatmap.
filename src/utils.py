from config import FIGURES, PROCESSED, RAW


def ensure_dirs():
    for d in (RAW, PROCESSED, FIGURES):
        d.mkdir(parents=True, exist_ok=True)

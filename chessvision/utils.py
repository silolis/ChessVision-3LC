import os
from pathlib import Path

CVROOT = os.getenv("CVROOT", Path(__file__).parent.parent.as_posix())

DATA_ROOT = os.path.join(CVROOT, "data")

INPUT_SIZE = (256, 256)
BOARD_SIZE = (512, 512)
PIECE_SIZE = (64, 64)

labels = {
    "B": 0,
    "K": 1,
    "N": 2,
    "P": 3,
    "Q": 4,
    "R": 5,
    "b": 6,
    "k": 7,
    "n": 8,
    "p": 9,
    "q": 10,
    "r": 11,
    "f": 12,
}

label_names = [
    "White Bishop",
    "White King",
    "White Knight",
    "White Pawn",
    "White Queen",
    "White Rook",
    "Black Bishop",
    "Black King",
    "Black Knight",
    "Black Pawn",
    "Black Queen",
    "Black Rook",
    "Empty Square",
    "Unknown",
]

classifier_weights_dir = os.path.join(CVROOT, "weights", "classifier")
extractor_weights_dir = os.path.join(CVROOT, "weights", "extractor")

best_classifier_weights = os.path.join(CVROOT, "weights", "best_classifier.pth")
best_extractor_weights = os.path.join(CVROOT, "weights", "best_extractor.pth")


def listdir_nohidden(path):
    files = []
    for f in os.listdir(path):
        if not f.startswith("."):
            files.append(f)
    return files


def ratio(a, b):
    if a == 0 or b == 0:
        return -1
    return min(a, b) / float(max(a, b))



import argparse
import os
import random
from pathlib import Path

import cv2

CVROOT = os.getenv("CVROOT", Path(__file__).parent.parent.as_posix())

data_root = os.path.join(CVROOT, "data")

INPUT_SIZE = (256, 256)
BOARD_SIZE = (512, 512)
PIECE_SIZE = (64, 64)

test_labels = ["f", "P", "p", "R", "r", "N", "n", "B", "b", "Q", "q", "K", "k"]

labels = {"b": 6, "k": 7, "n": 8, "p": 9, "q": 10, "r": 11, "B": 0,
          "f": 12, "K": 1, "N": 2, "P": 3, "Q": 4, "R": 5}

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


class BoardExtractionError(Exception):
    def json_string(self):
        return '{"error": "true", "errorMsg": "Board extraction failed!"}'


def listdir_nohidden(path):
    files = []
    for f in os.listdir(path):
        if not f.startswith('.'):
            files.append(f)
    return files

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', metavar='indir', type=str, nargs='+',
                        help='The dir to process.')
    parser.add_argument('-f', metavar='infile', type=str, nargs='+',
                        help='The dir to process.')
    parser.add_argument('-o', metavar='outdir', type=str, nargs='+',
                        help='The dir to process.')
    args = parser.parse_args()

    indir = None
    infile = None
    outdir = None

    if args.d is not None:
        indir = args.d[0]
        infile = None
    elif args.f is not None:
        infile = args.f[0]
        indir = None

    if args.o is not None:
        outdir = args.o[0]
    else:
        raise Exception("Usage: must supply outdir (flag -o) to run this command!")

    if indir == None and infile == None:
        raise Exception("Usage: must supply either indir (-d) or infile (-f) to run this command!")

    return (infile, indir, outdir)

def writeDocumentationImage(image, name):
    cv2.imwrite("images/" + name + ".png", image)

def showImage(image, name="image"):
    print("Showing image: '%s'" % name)
    cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    cv2.imshow('image', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def drawLine(image, a, b, color, thickness=1):
    cv2.line(image, tuple(a), tuple(b), color, thickness)


def draw_contour(image, contour, color, thickness=4):
    for i in range(len(contour)):
        p1 = tuple(contour[i])
        p2 = tuple(contour[int((i+1) % len(contour))])
        drawLine(image, p1, p2, color, thickness)



def drawBoundaries (image, boundaries, color=(128, 128, 0)):
    p_min, p_max = boundaries
    cv2.rectangle(image, p_min, p_max, color)


def drawPerspective(image, perspective, thickness=4):
    (a,b,c,d) = perspective

    #    cv2.polylines(image, np.array(perspective, 'int32'), False, (0,0,255) )
    drawLine(image, a,c, (255,0,0), thickness)
    drawLine(image, b,d, (255,0,255), thickness)


def drawLines(image, lines, color=(0,0,255), thickness=2):
    for l in lines:
        l.draw(image, color, thickness)

def drawPoint(image, point, color, thickness=4):
    cv2.circle(image, tuple([int(i) for i in point]), thickness, color)


def ratio(a,b):
    if a == 0 or b == 0:
        return -1
    return min(a,b)/float(max(a,b))

def randomColor(ncol=3):
    return [(random.randint(128, 255) * (random.randint(0,100) % 2)) for x in range(ncol)]

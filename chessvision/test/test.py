import time
from pathlib import Path

import cairosvg
import chess
import chess.svg
import cv2
import numpy as np
import tlc
import torch
from PIL import Image

from chessvision.predict.classify_raw import classify_raw
from chessvision.utils import data_root, listdir_nohidden, test_labels

TEST_DATA_DIR = Path(data_root) / "test"
# LABEL_NAMES = ["B", "K", "N", "P", "Q", "R", "b", "k", "n", "p", "q", "r", "f"]
LABEL_NAMES = ["f", "P", "p", "R", "r", "N", "n", "B", "b", "Q", "q", "K", "k"]
PROJECT_NAME = "chessvision-testing"
BLACK_BOARD = Image.fromarray(np.zeros((256, 256)).astype(np.uint8))
BLACK_CROP = Image.fromarray(np.zeros((64, 64)).astype(np.uint8))


def sim(a, b):
    return sum([aa == bb for aa, bb in zip(a, b)]) / len(a)


def top_k_sim(predictions, truth, k, names):
    """
    predictions: (64, 13) probability distributions
    truth      : (64, 1)  true labels
    k          : is true label in top k predictions

    returns    : fraction of the 64 squares are k-correctly classified
    """

    sorted_predictions = np.argsort(predictions, axis=1)
    top_k = sorted_predictions[:, -k:]
    top_k_predictions = np.array([["" for _ in range(k)] for _ in range(64)])
    hits = 0

    for i in range(64):
        for j in range(k):
            top_k_predictions[i, j] = LABEL_NAMES[top_k[i, j]]

    i = 0

    for rank in range(1, 9):
        for file in ["a", "b", "c", "d", "e", "f", "g", "h"]:
            square = file + str(rank)
            square_ind = names.index(square)
            if truth[i] in top_k_predictions[square_ind]:
                hits += 1
            i += 1

    return hits / 64


def vectorize_chessboard(board):
    res = list("f" * 64)

    piecemap = board.piece_map()

    for i in range(64):
        piece = piecemap.get(i)
        if piece:
            res[i] = piece.symbol()

    return "".join(res)


def get_test_generator():
    img_filenames = listdir_nohidden(str(TEST_DATA_DIR / "raw"))
    test_imgs = np.array([cv2.imread(str(TEST_DATA_DIR / "raw/" / x)) for x in img_filenames])

    for i in range(len(test_imgs)):
        yield img_filenames[i], test_imgs[i]


def save_svg(chessboard: chess.Board, path: Path):
    svg = chess.svg.board(chessboard)
    cairosvg.svg2png(bytestring=svg.encode("utf-8"), write_to=str(path))


def run_tests(
    data_generator=None,
    run: tlc.Run | None = None,
    extractor: torch.nn.Module | None = None,
    classifier: torch.nn.Module | None = None,
    threshold=80,
) -> tlc.Run:
    if not run:
        run = tlc.init(PROJECT_NAME)

    if not data_generator:
        data_generator = get_test_generator()

    metrics_writer = tlc.MetricsTableWriter(
        run.url,
        column_schemas={
            "true_labels": tlc.CategoricalLabel("true_labels", test_labels),
            "predicted_labels": tlc.CategoricalLabel("predicted_labels", test_labels),
            "raw_imgs": tlc.PILImage("raw_imgs"),
            "rendered_board": tlc.Schema(value=tlc.ImageUrlStringValue("rendered_board")),
        },
    )

    N = 0
    test_accuracy = 0.0
    top_2_accuracy = 0.0
    top_3_accuracy = 0.0

    times = []

    with tlc.bulk_data_url_context(run.bulk_data_url):
        for filename, img in data_generator:
            print(f"***Classifying test image {filename}***")
            N += 1
            start = time.time()
            board_img, mask, predictions, chessboard, _, squares, names = classify_raw(
                img,
                filename,
                extractor,
                classifier,
                threshold=threshold,
            )

            stop = time.time()
            times.append(stop - start)

            if board_img is None:
                metrics_batch = {
                    "raw_imgs": [Image.open(str(TEST_DATA_DIR / "raw" / filename))],
                    "predicted_masks": [Image.fromarray(mask.astype(np.uint8))],
                    "extracted_board": [BLACK_BOARD],
                    "rendered_board": [""],
                    "accuracy": [0],
                    "square_crop": [BLACK_CROP],
                    "true_labels": [0],
                    "predicted_labels": [0],
                }
                metrics_writer.add_batch(metrics_batch)
                continue

            truth_file = TEST_DATA_DIR / "ground_truth/" / (filename[:-4] + ".txt")

            with open(truth_file) as truth:
                true_labels = truth.read()

            top_2_accuracy += top_k_sim(predictions, true_labels, 2, names)
            top_3_accuracy += top_k_sim(predictions, true_labels, 3, names)

            predicted_labels = vectorize_chessboard(chessboard)

            this_board_acc = sim(predicted_labels, true_labels)
            test_accuracy += this_board_acc

            labels_int = [test_labels.index(label) for label in true_labels]
            for i in range(8):
                labels_int[i * 8 : (i + 1) * 8] = list(reversed(labels_int[i * 8 : (i + 1) * 8]))
            labels_int = list(reversed(labels_int))

            predicted_labels_int = [test_labels.index(label) for label in predicted_labels]
            predicted_labels_int = list(reversed(predicted_labels_int))

            for i in range(8):
                predicted_labels_int[i * 8 : (i + 1) * 8] = list(reversed(predicted_labels_int[i * 8 : (i + 1) * 8]))

            svg_url = Path((run.bulk_data_url / "rendered_board" / (filename[:-4] + ".png")).to_str())
            svg_url.parent.mkdir(parents=True, exist_ok=True)
            save_svg(chessboard, svg_url)

            metrics_batch = {
                "raw_imgs": [Image.open(str(TEST_DATA_DIR / "raw" / filename))] * 64,
                "predicted_masks": [Image.fromarray(mask.astype(np.uint8))] * 64,
                "extracted_board": [Image.fromarray(board_img)] * 64,
                "rendered_board": [str(svg_url)] * 64,
                "accuracy": [this_board_acc] * 64,
                "square_crop": [Image.fromarray(img.squeeze()) for img in squares],
                "true_labels": labels_int,
                "predicted_labels": predicted_labels_int,
            }
            metrics_writer.add_batch(metrics_batch)

    test_accuracy /= N
    top_2_accuracy /= N
    top_3_accuracy /= N

    aggregate_data = {
        "top_1_accuracy": test_accuracy,
        "top_2_accuracy": top_2_accuracy,
        "top_3_accuracy": top_3_accuracy,
        "avg_time_per_prediction": sum(times) / N,
    }

    run.set_parameters({"test_results": aggregate_data})

    print(f"Classified {N} raw images")
    metrics_table = metrics_writer.finalize()
    run.add_metrics_table(metrics_table)
    run.set_status_completed()

    return run


if __name__ == "__main__":
    print("Running ChessVision test suite...")

    start = time.time()
    run = run_tests()
    stop = time.time()
    print(f"Tests completed in {stop-start:.1f}s")
    print("Test accuracy: {}".format(run.constants["parameters"]["top_1_accuracy"]))
    print("Top-2 accuracy: {}".format(run.constants["parameters"]["top_2_accuracy"]))
    print("Top-3 accuracy: {}".format(run.constants["parameters"]["top_3_accuracy"]))

import itertools
import time
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import tlc
from PIL import Image

from chessvision.predict.classify_raw import classify_raw
from chessvision.utils import data_root, listdir_nohidden, test_labels

test_data_dir = Path(data_root) / "test"


def plot_confusion_mtx(mtx, labels):
    plt.imshow(mtx, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels)
    plt.yticks(tick_marks, labels)

    thresh = mtx.max() / 2.0
    for i, j in itertools.product(range(mtx.shape[0]), range(mtx.shape[1])):
        plt.text(
            j, i, format(mtx[i, j], "d"), horizontalalignment="center", color="white" if mtx[i, j] > thresh else "black"
        )

    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()


def entropy(dist):
    return sum([-p * np.log(p) for p in dist if p > 0])


def avg_entropy(predictions):
    predictions = np.array(predictions)
    pred_shape = predictions.shape
    if len(pred_shape) == 3:  # (N, 64, 13)
        predictions = np.reshape(predictions, (pred_shape[0] * pred_shape[1], pred_shape[2]))
    return sum([entropy(p) for p in predictions]) / len(predictions)


def sim(a, b):
    return sum([aa == bb for aa, bb in zip(a, b)]) / len(a)


label_names = ["B", "K", "N", "P", "Q", "R", "b", "k", "n", "p", "q", "r", "f"]


def top_k_sim(predictions, truth, k, names):
    """
    predictions: (64, 13) probability distributions
    truth      : (64, 1)  true labels
    k          : is true label in top k predictions

    returns    : fraction of the 64 squares are k-correctly classified
    """

    label_names = ["B", "K", "N", "P", "Q", "R", "b", "k", "n", "p", "q", "r", "f"]
    sorted_predictions = np.argsort(predictions, axis=1)

    top_k = sorted_predictions[:, -k:]

    top_k_predictions = np.array([["" for _ in range(k)] for _ in range(64)])

    hits = 0

    for i in range(64):
        for j in range(k):
            top_k_predictions[i, j] = label_names[top_k[i, j]]

    i = 0

    for rank in range(1, 9):
        for file in ["a", "b", "c", "d", "e", "f", "g", "h"]:
            square = file + str(rank)
            square_ind = names.index(square)
            if truth[i] in top_k_predictions[square_ind]:
                hits += 1
            i += 1

    return hits / 64


def confusion_matrix(predicted, truth, N=13):
    predicted = list(predicted)
    truth = list(truth)
    if isinstance(predicted[0], str):
        for i in range(len(predicted)):
            predicted[i] = test_labels.index(predicted[i])
            truth[i] = test_labels.index(truth[i])

    mtx = np.zeros((N, N), dtype=int)

    for p, t in zip(predicted, truth):
        mtx[t, p] += 1

    return mtx


def get_hits(mtx):
    N = mtx.shape[0]
    ondiag = 0
    offdiag = 0
    for i in range(N):
        for j in range(N):
            if i == j:
                ondiag += mtx[i, j]
            else:
                offdiag += mtx[i, j]
    return ondiag, offdiag


def vectorize_chessboard1(board):
    """vectorizes a python-chess board from a1 to h8 along ranks (row-major)"""
    res = ["f"] * 64

    piecemap = board.piece_map()

    for piece in piecemap:
        res[piece] = piecemap[piece].symbol()

    return res


def vectorize_chessboard(board):
    res = list("f" * 64)

    piecemap = board.piece_map()

    for i in range(64):
        piece = piecemap.get(i)
        if piece:
            res[i] = piece.symbol()

    return "".join(res)


def get_test_generator():
    img_filenames = listdir_nohidden(str(test_data_dir / "raw"))
    test_imgs = np.array([cv2.imread(str(test_data_dir / "raw/" / x)) for x in img_filenames])

    for i in range(len(test_imgs)):
        yield img_filenames[i], test_imgs[i]


def run_tests(data_generator, extractor, classifier, threshold=80):
    run = tlc.init("chessvision-testing")
    # with tlc.bulk_data_url_context(run.bulk_data_url):

    metrics_writer = tlc.MetricsTableWriter(
        run.url,
        column_schemas={
            "true_labels": tlc.CategoricalLabel("true_labels", test_labels),
            "predicted_labels": tlc.CategoricalLabel("predicted_labels", test_labels),
            "raw_imgs": tlc.PILImage("raw_imgs"),
        },
    )

    N = 0
    test_accuracy = 0
    top_2_accuracy = 0
    top_3_accuracy = 0

    times = []

    results = {
        "raw_imgs": [],
        "board_imgs": [],
        "predictions": [],
        "chessboards": [],
        "squares": [],
        "filenames": [],
        "masks": [],
        "board_accs": [],
        "true_labels": [],
    }

    confusion_mtx = np.zeros((13, 13), dtype=int)
    errors = 0

    with tlc.bulk_data_url_context(run.bulk_data_url):
        for filename, img in data_generator:
            print(f"***Classifying test image {filename}***")
            start = time.time()
            board_img, mask, predictions, chessboard, _, squares, names = classify_raw(
                img, filename, extractor, classifier, threshold=threshold
            )

            if board_img is None:
                errors += 1
                BLACK_BOARD = Image.fromarray(np.zeros_like(mask).astype(np.uint8))
                BLACK_CROP = Image.fromarray(np.zeros((64, 64)).astype(np.uint8))
                metrics_batch = {
                    "raw_imgs": [Image.open(str(test_data_dir / "raw" / filename))],
                    "predicted_masks": [Image.fromarray(mask.astype(np.uint8))],
                    "extracted_board": [BLACK_BOARD],
                    "accuracy": [0],
                    "square_crop": [BLACK_CROP],
                    "true_labels": [0],
                    "predicted_labels": [0],
                }
                metrics_writer.add_batch(metrics_batch)
                N += 1
                continue

            stop = time.time()

            truth_file = test_data_dir / "ground_truth/" / (filename[:-4] + ".txt")
            with open(truth_file) as truth:
                true_labels = truth.read()

            top_2_accuracy += top_k_sim(predictions, true_labels, 2, names)
            top_3_accuracy += top_k_sim(predictions, true_labels, 3, names)

            times.append(stop - start)
            predicted_labels = vectorize_chessboard(chessboard)

            this_board_acc = sim(predicted_labels, true_labels)
            test_accuracy += this_board_acc

            confusion_mtx += confusion_matrix(predicted_labels, true_labels)

            results["board_accs"].append(this_board_acc)
            results["board_imgs"].append(board_img)
            results["raw_imgs"].append(img)
            results["predictions"].append(predictions)
            results["chessboards"].append(chessboard)
            results["squares"].append(squares)
            results["filenames"].append(filename)
            results["masks"].append(mask)
            results["true_labels"].append(true_labels)

            labels_int = [test_labels.index(l) for l in true_labels]
            for i in range(8):
                labels_int[i * 8 : (i + 1) * 8] = list(reversed(labels_int[i * 8 : (i + 1) * 8]))
            labels_int = list(reversed(labels_int))

            predicted_labels_int = [test_labels.index(l) for l in predicted_labels]
            predicted_labels_int = list(reversed(predicted_labels_int))
            for i in range(8):
                predicted_labels_int[i * 8 : (i + 1) * 8] = list(reversed(predicted_labels_int[i * 8 : (i + 1) * 8]))
            metrics_batch = {
                "raw_imgs": [Image.open(str(test_data_dir / "raw" / filename))] * 64,
                "predicted_masks": [Image.fromarray(mask.astype(np.uint8))] * 64,
                "extracted_board": [Image.fromarray(board_img)] * 64,
                "accuracy": [this_board_acc] * 64,
                "square_crop": [Image.fromarray(img.squeeze()) for img in squares],
                "true_labels": labels_int,
                "predicted_labels": predicted_labels_int,
            }
            metrics_writer.add_batch(metrics_batch)
            N += 1

    test_accuracy /= N
    top_2_accuracy /= N
    top_3_accuracy /= N

    aggregate_data = {
        "top_1_accuracy": test_accuracy,
        "top_2_accuracy": top_2_accuracy,
        "top_3_accuracy": top_3_accuracy,
        "avg_entropy": avg_entropy(results["predictions"]),
        "avg_time": sum(times[:-1]) / (N - 1),
    }

    run.set_parameters(aggregate_data)

    results["top_2_accuracy"] = top_2_accuracy
    results["top_3_accuracy"] = top_3_accuracy
    results["avg_entropy"] = avg_entropy(results["predictions"])
    results["avg_time"] = sum(times[:-1]) / (N - 1)
    results["acc"] = test_accuracy
    results["errors"] = errors
    results["hits"] = get_hits(confusion_mtx)
    print(f"Classified {N} raw images")

    metrics_table = metrics_writer.finalize()

    run.add_metrics_table(metrics_table)
    run.set_status_completed()

    return results


if __name__ == "__main__":
    import time

    print("Computing test accuracy...")

    test_data_gen = get_test_generator()

    start = time.time()
    results = run_tests(test_data_gen, None, None)
    stop = time.time()
    print(f"Tests completed in {stop-start:.1f}s")
    print("Test accuracy: {}".format(results["acc"]))
    # print("Top-1 accuracy: {}".format(results["top_1_accuracy"]))
    print("Top-2 accuracy: {}".format(results["top_2_accuracy"]))
    print("Top-3 accuracy: {}".format(results["top_3_accuracy"]))

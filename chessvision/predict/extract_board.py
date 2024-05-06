"""
CNN-based board extraction.
Methods to extract boards from images.
If run as main, attempts to extract boards from all images in in <indir>,
and outputs results in <outdir>

Usage:
python board_extractor.py -d ../data/images/ -o ./data/boards/
"""

import cv2
import numpy as np
import torch

from ..utils import BOARD_SIZE, ratio

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def extract_board(image, orig, model, threshold=80):
    image_batch = torch.Tensor(np.array([image])) / 255
    image_batch = image_batch.permute(0, 3, 1, 2).to(device)

    with torch.no_grad():
        logits = model(image_batch)

    probabilities = torch.sigmoid(logits)
    probabilities = probabilities[0].squeeze().cpu().numpy()
    mask = fix_mask(probabilities, threshold=threshold)

    # approximate chessboard-mask with a quadrangle
    approx = find_quadrangle(mask)
    if approx is None:
        return None, probabilities

    # scale approximation to input image size
    approx = scale_approx(approx, (orig.shape[0], orig.shape[1]))

    # extract board
    board = extract_perspective(orig, approx, BOARD_SIZE)

    board = cv2.cvtColor(board, cv2.COLOR_BGR2GRAY)
    board = cv2.flip(board, 1)  # TODO: permute approximation instead..

    return board, probabilities


def fix_mask(mask, threshold=80):
    mask *= 255
    mask = mask.astype(np.uint8)
    mask[mask > threshold] = 255
    mask[mask <= threshold] = 0
    return mask


def scale_approx(approx, orig_size):
    sf = orig_size[0] / 256.0
    scaled = np.array(approx * sf, dtype=np.uint32)
    return scaled


def rotate_quadrangle(approx):
    if approx[0, 0, 0] < approx[2, 0, 0]:
        approx = approx[[3, 0, 1, 2], :, :]
    return approx


def find_quadrangle(mask):
    try:
        contours, _ = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_TC89_KCOS)
    except:
        _, contours, _ = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_TC89_KCOS)

    if len(contours) > 1:
        # print("Found {} contour(s)".format(len(contours)))
        contours = ignore_contours(mask.shape, contours)
        # print("Filtered to {} contour(s)".format(len(contours)))

    if len(contours) == 0:
        return None

    approx = None

    # try to approximate and hope for a quad
    for i in range(len(contours)):
        cnt = contours[i]

        arclen = cv2.arcLength(cnt, True)
        candidate = cv2.approxPolyDP(cnt, 0.1 * arclen, True)

        if len(candidate) != 4:
            continue

        approx = rotate_quadrangle(candidate)
        break

    return approx


def extract_perspective(image, approx, out_size):
    w, h = out_size[0], out_size[1]

    dest = ((0, 0), (w, 0), (w, h), (0, h))

    approx = np.array(approx, np.float32)
    dest = np.array(dest, np.float32)

    coeffs = cv2.getPerspectiveTransform(approx, dest)

    return cv2.warpPerspective(image, coeffs, out_size)


def ignore_contours(img_shape, contours, min_ratio_bounding=0.6, min_area_percentage=0.35, max_area_percentage=1.0):
    ret = []
    mask_area = float(img_shape[0] * img_shape[1])

    for i in range(len(contours)):
        ca = cv2.contourArea(contours[i])
        ca /= mask_area
        if ca < min_area_percentage or ca > max_area_percentage:
            continue
        _, _, w, h = cv2.boundingRect(contours[i])
        if ratio(h, w) < min_ratio_bounding:
            continue
        ret.append(contours[i])

    return ret


if __name__ == "__main__":
    pass

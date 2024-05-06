import logging

import cv2
import timm
import torch

from chessvision.board_extraction.train_unet import load_checkpoint as load_extractor_checkpoint
from chessvision.piece_classification.train_classifier import NUM_CLASSES
from chessvision.piece_classification.train_classifier import load_checkpoint as load_classifier_checkpoint
from chessvision.predict.classify_board import classify_board
from chessvision.predict.extract_board import extract_board
from chessvision.pytorch_unet.unet.unet_model import UNet
from chessvision.utils import (
    INPUT_SIZE,
    best_classifier_weights,
    best_extractor_weights,
)

logger = logging.getLogger("chessvision")

board_model: torch.nn.Module | None = None
sq_model: torch.nn.Module | None = None


def load_classifier(checkpoint_path: str = best_classifier_weights):
    global sq_model

    if sq_model is not None:
        return sq_model

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    classifier = timm.create_model(
        "resnet18",
        num_classes=NUM_CLASSES,
        in_chans=1,
    )
    classifier, _, _, _ = load_classifier_checkpoint(classifier, None, checkpoint_path)
    classifier.eval()
    classifier.to(device)

    return classifier


def load_board_extractor(checkpoint_path: str = best_extractor_weights):
    global board_model

    if board_model is not None:
        return board_model

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    extractor = UNet(n_channels=3, n_classes=1)
    extractor = extractor.to(memory_format=torch.channels_last)
    extractor = load_extractor_checkpoint(extractor, checkpoint_path)
    extractor.eval()
    extractor.to(device)
    board_model = extractor

    return extractor


def classify_raw(
    img,
    filename="",
    board_model=None,
    sq_model=None,
    flip=False,
    threshold=80,
):
    logger.debug(f"Processing image {filename}")

    if not board_model:
        board_model = load_board_extractor()
    if not sq_model:
        sq_model = load_classifier()

    ###############################   STEP 1    #########################################

    ## Resize image
    comp_image = cv2.resize(img, INPUT_SIZE, interpolation=cv2.INTER_AREA)

    ## Extract board using CNN model and contour approximation
    logger.debug("Extracting board from image")
    try:
        board_img, mask = extract_board(comp_image, img, board_model, threshold=threshold)
    except Exception as e:
        raise e

    if board_img is None:
        return None, mask, None, None, None, None, None
    ###############################   STEP 2    #########################################

    logger.debug("Classifying squares")
    FEN, predictions, chessboard, squares, names = classify_board(board_img, sq_model, flip=flip)

    logger.debug(f"Processing image {filename}.. DONE")

    return board_img, mask, predictions, chessboard, FEN, squares, names

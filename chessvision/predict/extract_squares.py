import numpy as np


def extract_squares(board: np.ndarray, flip: bool = False) -> tuple[np.ndarray, list[str]]:
    """Extract squares from a chessboard image.

    Args:
        board: A 512x512 image of a chessboard.
        flip: Whether to flip the board.

    Returns:
        A tuple containing the cut-out squares and their names (chessboard coordinates).

    """
    ranks = ["a", "b", "c", "d", "e", "f", "g", "h"]
    files = ["1", "2", "3", "4", "5", "6", "7", "8"]

    if flip:
        ranks = list(reversed(ranks))
        files = list(reversed(files))

    squares_list = []
    names = []

    ww, hh = board.shape
    w = int(ww / 8)
    h = int(hh / 8)

    for i in range(8):
        for j in range(8):
            squares_list.append(board[i * w : (i + 1) * w, j * h : (j + 1) * h])
            names.append(ranks[j] + files[7 - i])

    squares = np.array(squares_list)
    squares = squares.reshape(squares.shape[0], 64, 64, 1)

    return squares, names

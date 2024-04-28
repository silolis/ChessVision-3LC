import os
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

# Define the folder path where the image masks are located
input_folder = "C:\\Data\\chessboard-segmentation\\masks"
output_folder = "C:\\Data\\chessboard-segmentation\\clamped"
Path(output_folder).mkdir(parents=True, exist_ok=True)

# Iterate over the files in the folder
for filename in os.listdir(input_folder):
    # Check if the file is a JPG image
    if filename.lower().endswith(".jpg"):
        print("Processing file: ", filename)
        # Read the image
        image_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, filename.replace(".JPG", ".png"))
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        # Clamp the pixel values to 0 or 255
        clamped_image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)[1]

        # Overwrite the original image with the clamped version
        cv2.imwrite(output_path, clamped_image)

        # Reload the image and check the pixel values
        reloaded = np.asarray(Image.open(output_path))

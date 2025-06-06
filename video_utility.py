import cv2
from loguru import logger
import numpy as np


def get_contourn(image):
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Create a copy of the grayscale image
    result = gray.copy()
    gray[gray > 0] = 255

    # Find contours
    contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        # Find the largest contour
        largest_contour = max(contours, key=cv2.contourArea)

        # Create a mask for the largest contour
        mask = np.zeros_like(gray)
        cv2.drawContours(mask, [largest_contour], -1, 255, -1)

        kernel = np.ones(
            (50, 50), np.uint8
        )  # Adjust kernel size for more or less dilation
        dilated_mask = cv2.dilate(
            mask, kernel, iterations=17
        )  # Adjust iterations for more or less dilation
        non_zero = cv2.findNonZero(dilated_mask)
        if non_zero is not None:
            x, y, w, h = cv2.boundingRect(non_zero)
            x = int(x + w * 0.15)
            y = int(y + h * 0.15)
            w = int(w * 0.7)
            h = int(h * 0.78)
            return y, y + h, x, x + w

    # Return the original image cropped by 8 % each direction if no contour is found
    original_height, original_width, _ = result.shape
    logger.warning("Using Fallback!")
    return (
        int(original_height * 0.1),
        int(original_height * 0.9),
        int(original_width * 0.1),
        int(original_width * 0.9),
    )


def get_roi(cap):
    frames = [cap.read() for _ in range(50)]
    for _ in range(15):
        cap.read()
    frames += [cap.read() for _ in range(60)]
    frames = [x[1] for x in frames if x[0]]
    boundaries = [get_contourn(x) for x in frames]
    y = min([bound[0] for bound in boundaries])
    y_end = max([bound[1] for bound in boundaries])
    x = min([bound[2] for bound in boundaries])
    x_end = max([bound[3] for bound in boundaries])
    return y, y_end, x, x_end

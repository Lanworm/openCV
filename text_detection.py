import cv2
import numpy as np
def text_d(image):
    gray = image
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)  # threshold
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    dilated = cv2.dilate(thresh, kernel, iterations=13)  # dilate
    contours, hierarchy = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)  # get contours

    # for each contour found, draw a rectangle around it on original image
    for contour in contours:
        # get rectangle bounding contour
        [x, y, w, h] = cv2.boundingRect(contour)

        # discard areas that are too large
        if h > 30 and w > 300:
            continue

        # # discard areas that are too small
        # if h < 5 or w < 5:
        #     continue

        # draw rectangle around contour on original image
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 255), 2)

    return image
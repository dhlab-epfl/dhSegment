import cv2
import numpy as np


def find_box_prediction(preds: np.ndarray, min_rect=True):
    contours, hierarchy = cv2.findContours(preds, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    list_boxes = list()
    if min_rect:
        for c in contours:
            rect = cv2.minAreaRect(c)
            box = cv2.boxPoints(rect)
            # Todo : test if it seems a valid box

            list_boxes.append(np.int0(box))
    else:
        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            box = np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]], dtype=int)
            # Todo : test if it seems a valid box

            list_boxes.append(box)


def cini_post_processing_fn(preds: np.ndarray, clean_predictions=False, output_basename=None):
    # class 0 -> cardboard
    # class 1 -> background
    # class 2 -> photograph

    def get_cleaned_prediction(prediction):
        # Perform Erosion and Dilation
        if not clean_predictions:
            return prediction
        opening = cv2.morphologyEx(prediction, cv2.MORPH_OPEN, np.ones((5, 5)))
        closing = cv2.medianBlur(opening, 11)
        return closing

    class_predictions = np.argmax(preds, axis=-1)

    cardboard_prediction = get_cleaned_prediction((class_predictions == 0).astype(np.uint8))
    _, contours, hierarchy = cv2.findContours(cardboard_prediction, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cardboard_contour = np.concatenate(contours)  # contours[np.argmax([cv2.contourArea(c) for c in contours])]
    cardboard_rectangle = cv2.minAreaRect(cardboard_contour)

    image_prediction = (class_predictions == 2).astype(np.uint8)
    # Force the image prediction to be inside the extracted cardboard
    mask = np.zeros_like(image_prediction)
    cv2.fillConvexPoly(mask, cv2.boxPoints(cardboard_rectangle).astype(np.int32), 1)
    image_prediction = mask * image_prediction
    image_prediction = get_cleaned_prediction(image_prediction)
    _, contours, hierarchy = cv2.findContours(image_prediction, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    # Take the biggest contour or two biggest if similar size (two images in the page)
    image_contour = contours[0] if len(contours) == 1 or (cv2.contourArea(contours[0]) > 0.5*cv2.contourArea(contours[1])) \
        else np.concatenate(contours[0:2])
    image_rectangle = cv2.minAreaRect(image_contour)

    if output_basename is not None:
        #TODO
        pass
    return cardboard_rectangle, image_rectangle

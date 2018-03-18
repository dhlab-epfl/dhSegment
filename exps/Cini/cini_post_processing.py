import cv2
import numpy as np
from doc_seg.utils import dump_pickle


def cini_post_processing_fn(preds: np.ndarray,
                            clean_predictions=False,
                            advanced=False,
                            output_basename=None):
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

    # get cardboard rectangle
    cardboard_prediction = get_cleaned_prediction((class_predictions == 0).astype(np.uint8))
    _, contours, hierarchy = cv2.findContours(cardboard_prediction, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cardboard_contour = np.concatenate(contours)  # contours[np.argmax([cv2.contourArea(c) for c in contours])]
    cardboard_rectangle = cv2.minAreaRect(cardboard_contour)
    # If extracted cardboard too small compared to scan size, get cardboard+image prediction
    if cv2.contourArea(cv2.boxPoints(cardboard_rectangle)) < 0.20*cardboard_prediction.size:
        cardboard_prediction = get_cleaned_prediction(((class_predictions == 0) | (class_predictions == 2)).astype(np.uint8))
        _, contours, hierarchy = cv2.findContours(cardboard_prediction, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cardboard_contour = np.concatenate(contours)  # contours[np.argmax([cv2.contourArea(c) for c in contours])]
        cardboard_rectangle = cv2.minAreaRect(cardboard_contour)

    image_prediction = (class_predictions == 2).astype(np.uint8)
    if advanced:
        # Force the image prediction to be inside the extracted cardboard
        mask = np.zeros_like(image_prediction)
        cv2.fillConvexPoly(mask, cv2.boxPoints(cardboard_rectangle).astype(np.int32), 1)
        image_prediction = mask * image_prediction
        eroded_mask = cv2.erode(mask, np.ones((20, 20)))
        image_prediction = image_prediction | (~cardboard_prediction & eroded_mask)

    image_prediction = get_cleaned_prediction(image_prediction)
    _, contours, hierarchy = cv2.findContours(image_prediction, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    # Take the biggest contour or two biggest if similar size (two images in the page)
    image_contour = contours[0] if len(contours) == 1 or (cv2.contourArea(contours[0]) > 0.5*cv2.contourArea(contours[1])) \
        else np.concatenate(contours[0:2])
    image_rectangle = cv2.minAreaRect(image_contour)

    if output_basename is not None:
        dump_pickle(output_basename+'.pkl', {
            'shape': preds.shape[:2],
            'cardboard_rectangle': cardboard_rectangle,
            'image_rectangle': image_rectangle
        })
    return cardboard_rectangle, image_rectangle
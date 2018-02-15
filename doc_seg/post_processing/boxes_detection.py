import cv2
import numpy as np
import math


def find_box(predictions: np.array, mode: str='min_rectangle', min_area: float=0.2, p_arc_length: float=0.01):
    """

    :param predictions: Uint8 binary 2D array
    :param mode: 'min_rectangle', 'quadrilateral', rectangle
    :param min_area:
    :return:
    """
    _, contours, _ = cv2.findContours(predictions, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    found_box = None
    biggest_area = 0

    def validate_box(box):
        # Aproximate area computation (TODO eventually make a proper area computation)
        approx_area = (np.max(box[:, 0]) - np.min(box[:, 0])) * (np.max(box[:, 1] - np.min(box[:, 1])))
        if approx_area > min_area * predictions.size and approx_area > biggest_area:  # If box is bigger than 20% of the image size

            # Correct out of range corners
            box = np.maximum(box, 0)
            box = np.stack((np.minimum(box[:, 0], predictions.shape[1]),
                            np.minimum(box[:, 1], predictions.shape[0])), axis=1)

            return box

    if mode not in ['quadrilateral', 'min_rectangle', 'rectangle']:
        raise NotImplementedError
    if mode == 'quadrilateral':
        for c in contours:
            epsilon = p_arc_length * cv2.arcLength(c, True)
            cnt = cv2.approxPolyDP(c, epsilon, True)
            box = np.vstack(simplify_douglas_peucker(cnt[:, 0, :], 4))
            if len(box) != 4:
                mode = 'min_rectangle'
                print('Quadrilateral has {} points. Switching to minimal rectangle mode'.format(len(box)))
            else:
                found_box = validate_box(box)
    if mode == 'min_rectangle':
        for c in contours:
            rect = cv2.minAreaRect(c)
            box = np.int0(cv2.boxPoints(rect))
            found_box = validate_box(box)
    elif mode == 'rectangle':
        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            box = np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]], dtype=int)
            found_box = validate_box(box)

    return found_box


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


# Ramer-Douglas-Peucker from :
# https://stackoverflow.com/questions/37946754/python-ramer-douglas-peucker-rdp-algorithm-with-number-of-points-instead-of


def dist_squared(p1, p2):
    return pow((p1[0] - p2[0]), 2) + pow((p1[1] - p2[1]), 2)


class Line(object):
    def __init__(self, p1, p2):
        self.p1 = p1
        self.p2 = p2
        self.length_squared = dist_squared(self.p1, self.p2)

    def get_ratio(self, point):
        segment_length = self.length_squared
        if segment_length == 0:
            return dist_squared(point, self.p1)
        return ((point[0] - self.p1[0]) * (self.p2[0] - self.p1[0]) +
                (point[1] - self.p1[1]) * (self.p2[1] - self.p1[1])) / segment_length

    def distance_to_squared(self, point):
        t = self.get_ratio(point)

        if t < 0:
            return dist_squared(point, self.p1)
        if t > 1:
            return dist_squared(point, self.p2)

        return dist_squared(point, [
            self.p1[0] + t * (self.p2[0] - self.p1[0]),
            self.p1[1] + t * (self.p2[1] - self.p1[1])
        ])

    def distance_to(self, point):
        return math.sqrt(self.distance_to_squared(point))


def simplify_douglas_peucker(points, points_to_keep):
    weights = list()
    length = len(points)

    def douglas_peucker(start, end):
        if end > start + 1:
            line = Line(points[start], points[end])
            max_dist = -1
            max_dist_index = 0

            for i in range(start + 1, end):
                dist = line.distance_to_squared(points[i])
                if dist > max_dist:
                    max_dist = dist
                    max_dist_index = i

            weights.insert(max_dist_index, max_dist)

            douglas_peucker(start, max_dist_index)
            douglas_peucker(max_dist_index, end)

    douglas_peucker(0, length - 1)
    weights.insert(0, float("inf"))
    weights.append(float("inf"))

    weights_descending = weights
    weights_descending = sorted(weights_descending, reverse=True)

    max_tolerance = weights_descending[points_to_keep - 1]
    result = [
        point for i, point in enumerate(points) if weights[i] >= max_tolerance
    ]

    return result

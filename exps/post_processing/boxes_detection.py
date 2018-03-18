import cv2
import numpy as np
from scipy.spatial import KDTree


def find_box(predictions: np.array, mode: str='min_rectangle', min_area: float=0.2,
             p_arc_length: float=0.01, n_max_boxes=1):
    """

    :param predictions: Uint8 binary 2D array
    :param mode: 'min_rectangle', 'quadrilateral', rectangle
    :param min_area:
    :param p_arc_length: when 'qualidrateral' mode is chosen
    :param n_max_boxes:
    :return: n_max_boxes of 4 corners [[x1,y2], [x2,y2], ... ]
    """
    _, contours, _ = cv2.findContours(predictions, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours is None:
        print('No contour found')
        return None
    # found_box = None
    found_boxes = list()

    h_img, w_img = predictions.shape[:2]

    def validate_box(box: np.array) -> (np.array, float):
        # Aproximate area computation (TODO eventually make a proper area computation)
        approx_area = (np.max(box[:, 0]) - np.min(box[:, 0])) * (np.max(box[:, 1] - np.min(box[:, 1])))
        # if approx_area > min_area * predictions.size and approx_area > biggest_area:
        if approx_area > min_area * predictions.size:

            # Correct out of range corners
            box = np.maximum(box, 0)
            box = np.stack((np.minimum(box[:, 0], predictions.shape[1]),
                            np.minimum(box[:, 1], predictions.shape[0])), axis=1)

            # return box
            return box, approx_area

    if mode not in ['quadrilateral', 'min_rectangle', 'rectangle']:
        raise NotImplementedError
    if mode == 'quadrilateral':
        for c in contours:
            epsilon = p_arc_length * cv2.arcLength(c, True)
            cnt = cv2.approxPolyDP(c, epsilon, True)
            # box = np.vstack(simplify_douglas_peucker(cnt[:, 0, :], 4))

            # Find extreme points in Convex Hull
            hull_points = cv2.convexHull(cnt, returnPoints=True)
            # points = cnt
            points = hull_points
            if len(points) > 4:
                # Find closes points to corner using nearest neighbors
                tree = KDTree(points[:, 0, :])
                _, ul = tree.query((0, 0))
                _, ur = tree.query((w_img, 0))
                _, dl = tree.query((0, h_img))
                _, dr = tree.query((w_img, h_img))
                box = np.vstack([points[ul, 0, :], points[ur, 0, :],
                                 points[dr, 0, :], points[dl, 0, :]])
            elif len(hull_points) == 4:
                box = hull_points[:, 0, :]
            else:
                    continue
            # Todo : test if it looks like a rectangle (2 sides must be more or less parallel)
            # todo : (otherwise we may end with strange quadrilaterals)
            if len(box) != 4:
                mode = 'min_rectangle'
                print('Quadrilateral has {} points. Switching to minimal rectangle mode'.format(len(box)))
            else:
                # found_box = validate_box(box)
                found_boxes.append(validate_box(box))
    if mode == 'min_rectangle':
        for c in contours:
            rect = cv2.minAreaRect(c)
            box = np.int0(cv2.boxPoints(rect))
            # found_box = validate_box(box)
            found_boxes.append(validate_box(box))
    elif mode == 'rectangle':
        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            box = np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]], dtype=int)
            # found_box = validate_box(box)
            found_boxes.append(validate_box(box))
    # sort by area
    found_boxes = [fb for fb in found_boxes if fb is not None]
    found_boxes = sorted(found_boxes, key=lambda x: x[1], reverse=True)
    if n_max_boxes == 1:
        if found_boxes != []:
            return found_boxes[0][0]
        else:
            return None
    else:
        return [fb[0] for i, fb in enumerate(found_boxes) if i <= n_max_boxes]


# Ramer-Douglas-Peucker from :
# https://stackoverflow.com/questions/37946754/python-ramer-douglas-peucker-rdp-algorithm-with-number-of-points-instead-of


# def dist_squared(p1, p2):
#     return pow((p1[0] - p2[0]), 2) + pow((p1[1] - p2[1]), 2)


# class Line(object):
#     def __init__(self, p1, p2):
#         self.p1 = p1
#         self.p2 = p2
#         self.length_squared = dist_squared(self.p1, self.p2)
#
#     def get_ratio(self, point):
#         segment_length = self.length_squared
#         if segment_length == 0:
#             return dist_squared(point, self.p1)
#         return ((point[0] - self.p1[0]) * (self.p2[0] - self.p1[0]) +
#                 (point[1] - self.p1[1]) * (self.p2[1] - self.p1[1])) / segment_length
#
#     def distance_to_squared(self, point):
#         t = self.get_ratio(point)
#
#         if t < 0:
#             return dist_squared(point, self.p1)
#         if t > 1:
#             return dist_squared(point, self.p2)
#
#         return dist_squared(point, [
#             self.p1[0] + t * (self.p2[0] - self.p1[0]),
#             self.p1[1] + t * (self.p2[1] - self.p1[1])
#         ])
#
#     def distance_to(self, point):
#         return math.sqrt(self.distance_to_squared(point))
#
#
# def simplify_douglas_peucker(points, points_to_keep):
#     weights = list()
#     length = len(points)
#
#     def douglas_peucker(start, end):
#         if end > start + 1:
#             line = Line(points[start], points[end])
#             max_dist = -1
#             max_dist_index = 0
#
#             for i in range(start + 1, end):
#                 dist = line.distance_to_squared(points[i])
#                 if dist > max_dist:
#                     max_dist = dist
#                     max_dist_index = i
#
#             weights.insert(max_dist_index, max_dist)
#
#             douglas_peucker(start, max_dist_index)
#             douglas_peucker(max_dist_index, end)
#
#     douglas_peucker(0, length - 1)
#     weights.insert(0, float("inf"))
#     weights.append(float("inf"))
#
#     weights_descending = weights
#     weights_descending = sorted(weights_descending, reverse=True)
#
#     max_tolerance = weights_descending[points_to_keep - 1]
#     result = [
#         point for i, point in enumerate(points) if weights[i] >= max_tolerance
#     ]
#
#     return result

import cv2
import numpy as np
import math
from shapely import geometry
from scipy.spatial import KDTree


def find_boxes(boxes_mask: np.ndarray,
               mode: str= 'min_rectangle',
               min_area: float=0.,
               p_arc_length: float=0.01,
               n_max_boxes=math.inf) -> list:
    """
    Finds the coordinates of the box in the binary image `boxes_mask`.

    :param boxes_mask: Binary image: the mask of the box to find. uint8, 2D array
    :param mode: 'min_rectangle' : minimum enclosing rectangle, can be rotated
                 'rectangle' : minimum enclosing rectangle, not rotated
                 'quadrilateral' : minimum polygon approximated by a quadrilateral
    :param min_area: minimum area of the box to be found. A value in percentage of the total area of the image.
    :param p_arc_length: used to compute the epsilon value to approximate the polygon with a quadrilateral.
                         Only used when 'quadrilateral' mode is chosen.
    :param n_max_boxes: maximum number of boxes that can be found (default inf).
                        This will select n_max_boxes with largest area.
    :return: list of length n_max_boxes containing boxes with 4 corners [[x1,y1], ..., [x4,y4]]
    """

    assert len(boxes_mask.shape) == 2, \
        'Input mask must be a 2D array ! Mask is now of shape {}'.format(boxes_mask.shape)

    contours, _ = cv2.findContours(boxes_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours is None:
        print('No contour found')
        return None
    found_boxes = list()

    h_img, w_img = boxes_mask.shape[:2]

    def validate_box(box: np.array) -> (np.array, float):
        """

        :param box: array of 4 coordinates with format [[x1,y1], ..., [x4,y4]]
        :return: (box, area)
        """
        polygon = geometry.Polygon([point for point in box])
        if polygon.area > min_area * boxes_mask.size:

            # Correct out of range corners
            box = np.maximum(box, 0)
            box = np.stack((np.minimum(box[:, 0], boxes_mask.shape[1]),
                            np.minimum(box[:, 1], boxes_mask.shape[0])), axis=1)

            # return box
            return box, polygon.area

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
            found_boxes.append(validate_box(box))
    elif mode == 'rectangle':
        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            box = np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]], dtype=int)
            found_boxes.append(validate_box(box))
    # sort by area
    found_boxes = [fb for fb in found_boxes if fb is not None]
    found_boxes = sorted(found_boxes, key=lambda x: x[1], reverse=True)
    if n_max_boxes == 1:
        if found_boxes:
            return found_boxes[0][0]
        else:
            return None
    else:
        return [fb[0] for i, fb in enumerate(found_boxes) if i < n_max_boxes]

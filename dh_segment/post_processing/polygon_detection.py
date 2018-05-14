#!/usr/bin/env python

import cv2
import numpy as np
import math
from shapely import geometry


def find_polygonal_regions(image_mask: np.ndarray, min_area: float=0.1, n_max_polygons: int=math.inf):

    _, contours, _ = cv2.findContours(image_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours is None:
        print('No contour found')
        return None
    found_polygons = list()

    for c in contours:
        polygon = geometry.Polygon([point[0] for point in c])
        # Check that polygon has area greater than minimal area
        if polygon.area >= min_area*np.prod(image_mask.shape[:2]):
            found_polygons.append(
                (np.array([point for point in polygon.exterior.coords], dtype=np.uint), polygon.area)
            )

    # sort by area
    found_polygons = [fp for fp in found_polygons if fp is not None]
    found_polygons = sorted(found_polygons, key=lambda x: x[1], reverse=True)

    if found_polygons:
        return [fp[0] for i, fp in enumerate(found_polygons) if i <= n_max_polygons]
    else:
        return None

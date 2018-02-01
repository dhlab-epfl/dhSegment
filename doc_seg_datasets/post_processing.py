import numpy as np
from scipy.ndimage import label
from skimage.measure import label as skimage_label
from typing import Tuple, List
from scipy.signal import convolve2d
from skimage.graph import MCP_Connect
from skimage.morphology import skeletonize
from sklearn.metrics.pairwise import euclidean_distances
from collections import defaultdict
import cv2


def vertical_local_maxima(probs):
    local_maxima = np.zeros_like(probs, dtype=bool)
    local_maxima[1:-1] = (probs[1:-1] > probs[:-2]) & (probs[2:] < probs[1:-1])
    return local_maxima


def hysteresis_thresholding(probs: np.array, candidates: np.array, low_threshold: float, high_threshold: float):
    low_mask = candidates & (probs > low_threshold)
    # Connected components extraction
    label_components, count = label(low_mask, np.ones((3, 3)))
    # Keep components with high threshold elements
    good_labels = np.unique(label_components[low_mask & (probs > high_threshold)])
    label_masks = np.zeros((count + 1,), bool)
    label_masks[good_labels] = 1
    return label_masks[label_components]


def remove_borders(mask, margin=5):
    tmp = mask.copy()
    tmp[:margin] = 1
    tmp[-margin:] = 1
    tmp[:, :margin] = 1
    tmp[:, -margin:] = 1
    label_components, count = label(tmp, np.ones((3, 3)))
    result = mask.copy()
    result[label_components == label_components[0, 0]] = 0
    return result


def extract_line_polygons(lines_mask):
    # Make sure one-pixel wide 8-connected mask
    lines_mask = skeletonize(lines_mask)

    class MakeLineMCP(MCP_Connect):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.connections = dict()
            self.scores = defaultdict(lambda: np.inf)

        def create_connection(self, id1, id2, pos1, pos2, cost1, cost2):
            k = (min(id1, id2), max(id1, id2))
            s = cost1 + cost2
            if self.scores[k] > s:
                self.connections[k] = (pos1, pos2, s)
                self.scores[k] = s

        def get_connections(self, subsample=5):
            results = dict()
            for k, (pos1, pos2, s) in self.connections.items():
                path = np.concatenate([self.traceback(pos1), self.traceback(pos2)[::-1]])
                results[k] = path[::subsample]
            return results

        def goal_reached(self, int_index, float_cumcost):
            if float_cumcost > 0:
                return 2
            else:
                return 0

    # Find extremities points
    end_points_candidates = np.stack(np.where((convolve2d(lines_mask, np.ones((3, 3)), mode='same') == 2) & lines_mask)).T
    connected_components = skimage_label(lines_mask, connectivity=2)
    # Group endpoint by connected components and keep only the two points furthest away
    d = defaultdict(list)
    for pt in end_points_candidates:
        d[connected_components[pt[0], pt[1]]].append(pt)
    end_points = []
    for pts in d.values():
        d = euclidean_distances(np.stack(pts), np.stack(pts))
        i, j = np.unravel_index(d.argmax(), d.shape)
        end_points.append(pts[i])
        end_points.append(pts[j])
    end_points = np.stack(end_points)

    mcp = MakeLineMCP(~lines_mask)
    mcp.find_costs(end_points)
    connections = mcp.get_connections()
    if not np.all(np.array(sorted([i for k in connections.keys() for i in k])) == np.arange(len(end_points))):
        print('Warning : extract_line_polygons seems weird')
    return [c[:, None, :] for c in connections.values()]


def line_extraction_v0(probs, sigma, low_threshold, high_threshold):
    probs_line = probs[:, :, 1]
    # smooth
    probs2 = cv2.GaussianBlur(probs_line, (int(3*sigma)*2+1,int(3*sigma)*2+1), sigma)
    local_maxima = vertical_local_maxima(probs2)
    lines_mask = hysteresis_thresholding(probs2, local_maxima, low_threshold, high_threshold)
    # Remove lines touching border
    lines_mask = remove_borders(lines_mask)
    # Extract polygons from line mask
    contours = extract_line_polygons(lines_mask)
    return contours, lines_mask


def upscale_coordinates(list_points: List[np.array], ratio: Tuple[float, float]):  # list of (N,1,2) cv2 points
    return np.array(
        [(round(p[0, 0]*ratio[1]), round(p[0, 1]*ratio[0])) for p in list_points]
    )[:, None, :].astype(int)


def dibco_binarization(probabilities_mask):
    # # Otsu's thresholding
    # blur = cv2.GaussianBlur(probabilities_mask, (5, 5), 0)
    # thresh_val, bin_img = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # return bin_img

    return probabilities_mask > 0.5
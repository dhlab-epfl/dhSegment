from skimage.graph import MCP_Connect
from skimage.morphology import skeletonize
from skimage.measure import label as skimage_label
from sklearn.metrics.pairwise import euclidean_distances
from scipy.signal import convolve2d
from collections import defaultdict
import numpy as np


def find_lines(lines_mask: np.ndarray):
    """
    Returns the longest central line for each connected component in the given binary mask.

    :param lines_mask: Binary mask of the detected line-areas
    :return: a list of Opencv-style polygonal lines (each contour encoded as [N,1,2] elements where each tuple is (x,y) )
    """
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

    if np.sum(lines_mask) == 0:
        return []
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
        print('Warning : find_lines seems weird')
    return [c[:, None, ::-1] for c in connections.values()]

# Copyright © 2021 Warren Weckesser
# modified by Nukelawe to use a Chebyshev metric

"""
`voronoi_chebyshev` creates Voronoi cells (polygons) for a set of points in the plane
using the chebyshev metric. The function requires the Shapely library
(https://pypi.org/project/Shapely/).
"""

import math
import numpy as np

def _chebyshev_closest(p0, p1, xmin, xmax, ymin, ymax):
    """
    Return the vertices of a polygon bounding the set of points where the chebyshev
    distance to p0 is less than the chebyshev distance to p1. The region is clipped to the
    bounding box defined by (xmin, xmax, ymin, ymax).
    """
    x0, y0 = p0
    x1, y1 = p1
    width = abs(x1 - x0)
    height = abs(y1 - y0)
    if width == height == 0:
        raise ValueError('points must not be equal')

    flip = False
    if width > height:
        xmin, xmax, ymin, ymax = ymin, ymax, xmin, xmax
        x0, y0, x1, y1 = y0, x0, y1, x1
        width, height = height, width
        flip = True

    mid = 0.5*(y0 + y1)
    if width == 0:
        if y0 < y1:
            p = np.array([[xmin, mid],
                          [xmax, mid],
                          [xmax, ymin],
                          [xmin, ymin],
                          [xmin, mid]])
        else:
            p = np.array([[xmin, mid],
                          [xmin, ymax],
                          [xmax, ymax],
                          [xmax, mid],
                          [xmin, mid]])
    else:
        s = math.copysign(1, y1 - y0)
        pa = np.array([x1 - s*0.5*height, mid])
        pb = np.array([x0 + s*0.5*height, mid])
        if x0 < x1 and y0 < y1:
            pa = [x1 - .5*height, mid]
            pb = [x0 + .5*height, mid]
            xmax = pb[0] + (mid - ymin)
            xmin = pa[0] - (ymax - mid)
            p = np.array([pa, pb,
                          [xmax, ymin],
                          [xmin, ymin],
                          [xmin, ymax],
                          pa])
        elif x0 < x1 and y0 > y1:
            pa = [x1 - .5*height, mid]
            pb = [x0 + .5*height, mid]
            xmax = pb[0] + (ymax - mid)
            xmin = pa[0] - (mid - ymin)
            p = np.array([pa, pb,
                          [xmax, ymax],
                          [xmin, ymax],
                          [xmin, ymin],
                          pa])
        elif x0 > x1 and y0 < y1:
            pa = [x0 - .5*height, mid]
            pb = [x1 + .5*height, mid]
            xmax = pb[0] + (ymax - mid)
            xmin = pa[0] - (mid - ymin)
            p = np.array([pa, pb,
                          [xmax, ymax],
                          [xmax, ymin],
                          [xmin, ymin],
                          pa])
        else: # x0 > x1 and y0 > y1
            pa = [x0 - .5*height, mid]
            pb = [x1 + .5*height, mid]
            xmax = pb[0] + (mid - ymin)
            xmin = pa[0] - (ymax - mid)
            p = np.array([pa, pb,
                          [xmax, ymin],
                          [xmax, ymax],
                          [xmin, ymax],
                          pa])
    if flip:
        p = p[:, ::-1]
    return p


def voronoi_chebyshev(points, districts, xmin, xmax, ymin, ymax):
    """
    Compute Voronoi cells using the Chebyshev metric.

    The cells (polygons represented as arrays of 2-d points) are clipped
    to the bounding box defined by `xmin`, `xmax`, `ymin`, `ymax`.

    The return value is a list of numpy arrays.  The i-th list
    has shape (n[i], 2), where n[i] is the number of vertices
    in the Voronoi cell around `points[i]`.

    There must be no duplicate points.
    """
    from shapely.geometry import Polygon
    from shapely.ops import unary_union

    regions = []
    for i0 in range(len(points)):
        p0 = points[i0]
        poly = []
        for i1 in range(len(points)):
            if i1 == i0:
                continue
            p1 = points[i1]
            p = _chebyshev_closest(p0, p1, xmin, xmax, ymin, ymax)
            poly.append(Polygon(p))

        region = Polygon(np.array([
            [xmin,ymin],
            [xmin,ymax],
            [xmax,ymax],
            [xmax,ymin],
            [xmin,ymin]
        ]))
        for r in poly:
            region = region.intersection(r)
        regions.append(region)

    # merge cells in same district
    cells = {}
    for district in districts.keys():
        stations = [regions[i] for i in districts[district]]
        boundary = unary_union(stations).boundary
        cells[district] = np.column_stack(boundary.xy)
    return cells

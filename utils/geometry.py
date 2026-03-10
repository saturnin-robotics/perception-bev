"""
utils/geometry.py
======================================================================
Geometric utilities for perception-bev.

Phase 1 — Foundation:
    hex_to_rgb()        convert hex color to normalized RGB tuple
    bbox_corners_2d()   4 corners of an oriented 2D bounding box
    bbox_corners_3d()   8 corners of an oriented 3D bounding box
    bbox_edges_3d()     12 edges connecting the 8 corners (wireframe)
    iou_2d()            Intersection over Union between two 2D boxes

Future phases will add:
    Phase 2 — project_lidar_to_image(), rotation_matrix()
    Phase 3 — fit_lane_polynomial(), point_to_lane_distance()
    Phase 4 — iou_3d(), mahalanobis_distance()
    Phase 5 — ego_to_world(), transform_point_cloud()
"""

import numpy as np

def hex_to_rgb(hex_color: str) -> tuple:

    hex_color = hex_color.lstrip("#")
    r = int(hex_color[0:2], 16) / 255.0
    g = int(hex_color[2:4], 16) / 255.0
    b = int(hex_color[4:6], 16) / 255.0
    return (r, g, b)

def bbox_corners_2d(cx: float, cy: float,
                    l: float, w: float,
                    yaw: float= 0.0) -> np.ndarray:
    
    corners = np.array([
        [l/2, w/2], 
        [l/2, -w/2],
        [-l/2, -w/2],
        [-l/2, w/2]
    ])

    # 2d rotation matrix around Z

    cos_yaw = np.cos(yaw)
    sin_yaw = np.sin(yaw)

    R = np.array([
        [cos_yaw, -sin_yaw],
        [sin_yaw, cos_yaw]
    ])

    # Rotate then translate the center
    rotated = (R @ corners.T).T
    rotated[:, 0] +=cx
    rotated[:, 1] +=cy

    return rotated



def bbox_corners_3d(center: np.ndarray,
                    size: np.ndarray,
                    yaw: float = 0.0) -> np.ndarray:
    
    cx, cy, cz = center
    l, w, h = size
   
   
   
    corners = np.array([
    [-l/2, -w/2, -h/2],   # p0 bottom rear-right
    [-l/2,  w/2, -h/2],   # p1 bottom rear-left
    [ l/2,  w/2, -h/2],   # p2 bottom front-left
    [ l/2, -w/2, -h/2],   # p3 bottom front-right
    [-l/2, -w/2,  h/2],   # p4 top rear-right
    [-l/2,  w/2,  h/2],   # p5 top rear-left
    [ l/2,  w/2,  h/2],   # p6 top front-left
    [ l/2, -w/2,  h/2],   # p7 top front-right
    ])

    cos_yaw = np.cos(yaw)
    sin_yaw = np.sin(yaw)

    R = np.array([
        [cos_yaw, -sin_yaw, 0],
        [sin_yaw, cos_yaw,  0],
        [0,       0,        1]
    ])
     
    rotated = (R @ corners.T).T
    rotated[:, 0] += cx
    rotated[:, 1] += cy
    rotated[:, 2] += cz



    return rotated

def bbox_edges_3d() -> list:
    """
    Return the 12 edges of a 3D bounding box as index pairs.

    Designed to be used with bbox_corners_3d() output:
        corners = bbox_corners_3d(center, size, yaw)
        for i, j in bbox_edges_3d():
            draw_line(corners[i], corners[j])

    Returns:
        list of [int, int]: 12 pairs of corner indices

    Edge groups:
        [0-3]  bottom face  (4 edges)
        [4-7]  top face     (4 edges)
        [8-11] verticals    (4 edges)
    """
    return [
        # Bottom face
        [0, 1], [1, 2], [2, 3], [3, 0],
        # Top face
        [4, 5], [5, 6], [6, 7], [7, 4],
        # Vertical edges
        [0, 4], [1, 5], [2, 6], [3, 7],
    ]

def iou_2d(box1: list, box2: list) -> float:
    """
    Compute Intersection over Union (IoU) between two 2D bounding boxes.

    Args:
        box1 (list): [cx, cy, l, w] — center + dimensions
        box2 (list): [cx, cy, l, w] — center + dimensions

    Returns:
        float: IoU score in range [0.0, 1.0]
               0.0 = no overlap
               1.0 = perfect overlap

    Example:
        >>> iou_2d([0, 0, 4, 2], [1, 0, 4, 2])
        0.428...
    """

    def to_minmax(box):
        cx, cy, l, w = box

        return cx - l/2, cy -w/2, cx + l/2, cy + w/2
    
    ax1, ay1, ax2, ay2 = to_minmax(box1)
    bx1, by1, bx2, by2 = to_minmax(box2)

    # intersection rectangle
    ix1 = max(ax1, bx1)
    ix2 = min(ax2, bx2)
    iy1 = max(ay1, by1)
    iy2 = min(ay2, by2)

    inter_area = max(0.0, ix2 -ix1) * max(0.0, iy2 - iy1)

    if inter_area == 0.0:
        return 0.0
    
    area1 = (ax2 - ax1) * (ay2 - ay1)
    area2 = (bx2 - bx1) * (by2 - by1)

    union_area = area1 + area2 - inter_area

    return inter_area / union_area if union_area > 0 else 0.0
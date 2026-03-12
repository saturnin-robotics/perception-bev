"""
Here will be defined the complete pipeline necessary for the perception task. These are not using DL that we'll see later
    
Pipeline stages:
    1. filter_roi()            Region of Interest + range filter
    2. voxel_downsample()      Density reduction via voxelization (Open3D)
    3. remove_ground_ransac()  Ground / obstacle separation (Open3D RANSAC)
    4. cluster_dbscan()        Object clustering + bbox extraction (sklearn)
    5. extract_drivable_zone() Drivable area polygon (scipy Convex Hull)
"""
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List
from sklearn.cluster import DBSCAN



try:
    import open3d as o3d
    HAS_OPEN3D = True
except ImportError:
    HAS_OPEN3D = False

##########################################
# Detection , Single detect object
###########################################

@dataclass
class Detection:
    center:     np.ndarray
    size:       np.ndarray
    yaw:        float    = 0.0
    cls:        str      = "unknown"   
    n_points:   int      = 0
    score:      float    = 1.0
    cluster_id: int      = -1

    def to_bbox4(self) ->list:
        """ Return [cx, cy, l, w] for IoU compute"""
        return [
            float(self.center[0]),
            float(self.center[1]),
            float(self.size[0]),
            float(self.size[1])
        ]
    def __repr__(self) ->str:
        return (f"Detection ("
                f"cls='{self.cls}', "
                f"center=[{self.center[0]:.1f}, {self.center[1]:.1f}, {self.center[2]:.1f}], "
                f"size = [{self.size[0]:.1f}, {self.size[1]:.1f}, {self.size[2]:.1f}], "
                f"pts={self.n_points})")
    
######################################################
# Frame Processed , pipeline output for one frame
#####################################################

@dataclass
class ProcessedFrame:
    xyz_obstacles:      np.ndarray
    xyz_ground:         np.ndarray
    intensity:          np.ndarray
    detections:         List[Detection] = field(default_factory=list)
    drivable_poly:      Optional[np.ndarray] = None
    n_clusters:         int = 0

###############################
# Filter Regioon Of Interest
###############################

def filter_roi(xyz: np.ndarray, intensity: np.ndarray, cfg: dict):
    """
        This function return a region of interest depending on thresholding mask
    """

    roi = cfg["roi"]
    li = cfg["lidar"]
    dist = np.linalg.norm(xyz, axis=1)

    mask = np.array(
        (dist < li["max_range"]) & (dist > li ["min_range"]) &
        (xyz[:, 0] < roi["x_max"]) & (xyz[:, 0] > roi["x_min"]) &
        (xyz[:, 1] < roi["y_max"]) & (xyz[:, 1] > roi["y_min"]) &
        (xyz[:, 2] < roi["z_max"]) & (xyz[:, 2] > roi["z_min"])
            )
    return xyz[mask], intensity[mask]

###########################################
# Voxel Downsampling
###########################################

def voxel_downsample(xyz: np.ndarray, intensity: np.ndarray, voxel_size: float = 0.15):

    if not HAS_OPEN3D or voxel_size <=0 or len(xyz):
        return xyz, intensity
    
    pcd = o3d.geometry.PoinCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz.astype(np.float64))

    colors = np.zeros((len(xyz), 3), dtype=np.float64)
    colors[:, 0] = np.clip(intensity, 0.0, 1.0)

    pcd.colors = o3d.utility.Vector3dVector(colors)

    downpcd = pcd.voxel_down_sample(voxel_size)

    xyz_d = np.asarray(downpcd.points, dtype=np.float64)
    int_d = np.asarray(downpcd.colors)[:,0].astype(np.float64)

    return xyz_d, int_d

################################################
# Remove Ground RANSAc
##############################################

def remove_ground_ransac(xyz: np.ndarray, intensity: np.ndarray, cfg: dict):

    g = cfg["ground_removal"]

    if not HAS_OPEN3D:
        threshold = cfg["lidar"]["ground_z_threshold"]
        mask_obs = xyz[:, 2] > threshold
        return xyz[mask_obs], intensity[mask_obs], xyz[~mask_obs]
    

    pcd = o3d.geometry.PointCloud()

    pcd.points = o3d.utility.Vector3dVector(xyz.astype(np.float64))

    # RANSAC : find the plane ax + by + cz + d = 0that maximise inliers
     
    _, inliers =  pcd.segment_plane(distance_threshold = g["distance_threshold"],
                                              ransac_n = g["ransac_n"],
                                              num_iterations = g["num_iterations"])
    # separate ground and obstacles

    inliers_set = set(inliers)
    mask_ground = np.array([i in inliers_set for i in range(len(xyz))])
    mask_obs = ~mask_ground

    # Extra refinement: obstacles below Z threshold go back to ground
    mask_obs &= xyz[:, 2] > cfg['lidar']['ground_z_threshold']

    return xyz[mask_obs], intensity[mask_obs], xyz[mask_ground]

##############################################################
# DBSCAN Clustering and 3d BBOX
##############################################################


def _classify_by_size(l: float, w: float, h: float, cfg: dict) -> str:

    """ Function that classiffy by size and return the object class if 
        rules(conditions) are respected and return "unknown" otherwise
    """
    
    rules = cfg.get("classification", {})

    for class_name, bounds in rules.items():
        if not bounds:
            continue

        def in_range(value, key):
            if key not in bounds:
                return True
            lo, hi = bounds[key]

            return lo <= value <= hi
        if in_range(l ,'l') and in_range(w, 'w') and in_range(h, 'h'):
            return class_name

    return "unknown"

def cluster_dbscan(xyz: np.ndarray, intensity: np.ndarray, cfg: dict) -> List[Detection]:
    """
    Cluster obstacle points with DBSCAN and extract bounding boxes.
 
    For each valid cluster:
        1. Compute axis-aligned bounding box (center + size)
        2. Estimate heading angle via 2D PCA on XY plane
        3. Classify by bounding box dimensions
        4. Filter clusters outside size limits
 
    Args:
        xyz       (np.ndarray): obstacle points (N, 3)
        intensity (np.ndarray): obstacle intensities (N,)
        cfg       (dict):       full configuration dictionary
 
    Returns:
        List[Detection]: list of detected objects
    """

    d = cfg["dbscan"]
    b = cfg["bbox"]

    if len(xyz)  <= d["min_samples"]:
        return []
    
    # Now Dbscan return  a label per point

    labels = DBSCAN(
        eps= d['eps'],
        min_samples= d['min_samples'],
        n_jobs= -1
    ).fit_predict(xyz)

    detections =[]

    # remove the noise
    for label in set(labels):
        if label < 0:
            continue
        cluster_pts = xyz[labels==label]
        n_pts = len(cluster_pts)
        # Filter by point count
        if n_pts < d["min_points"] or n_pts > d["max_points"]:
            continue
        # Bounding box from min/max extents
        mn     = cluster_pts.min(axis=0)
        mx     = cluster_pts.max(axis=0)
        center = ((mn + mx) / 2.0).astype(np.float32)
        size   = (mx - mn).astype(np.float32)       # [l, w, h]

        # Filter by physical size limits
        size_min = np.array(b["min_size"])
        size_max = np.array(b["max_size"])
        if np.any(size < size_min) or np.any(size > size_max):
            continue

        # Estimate heading angle via PCA on XY plane
        pts_2d  = cluster_pts[:, :2] - cluster_pts[:, :2].mean(axis=0)
        yaw     = 0.0
        if len(pts_2d) >= 2:
            try:
                _, _, Vt = np.linalg.svd(pts_2d, full_matrices=False)
                yaw = float(np.arctan2(Vt[0, 1], Vt[0, 0]))
            except np.linalg.LinAlgError:
                yaw = 0.0

        # Dimension-based classification
        cls = _classify_by_size(
            float(size[0]), float(size[1]), float(size[2]), cfg
        )

        detections.append(Detection(
            center=center,
            size=size,
            yaw=yaw,
            cls=cls,
            n_points=n_pts,
            cluster_id=int(label)
        ))

    return detections
 
 
# ══════════════════════════════════════════════════════
#  Stage 5 — Drivable zone extraction
# ══════════════════════════════════════════════════════
 
def extract_drivable_zone(xyz_ground: np.ndarray,
                          roi_cfg: dict) -> Optional[np.ndarray]:
    """
    Estimate the drivable area using a Convex Hull on ground points.
 
    Only ground points in front of the ego vehicle are considered.
    The result is a 2D polygon (N, 2) displayed in green in the BEV.
 
    Limitation: Convex Hull is always convex — curved roads will be
    approximated. Phase 3 will improve this with lane detection.
 
    Args:
        xyz_ground (np.ndarray): ground points from RANSAC (M, 3)
        roi_cfg    (dict):       roi section of the configuration
 
    Returns:
        np.ndarray: polygon vertices (K, 2) in XY, or None if failed
    """
    if len(xyz_ground) < 10:
        return None
 
    try:
        from scipy.spatial import ConvexHull
 
        # Keep only ground points ahead of the ego vehicle
        mask  = (xyz_ground[:, 0] > 0) & \
                (xyz_ground[:, 0] < roi_cfg["x_max"])
        pts2d = xyz_ground[mask, :2]
 
        if len(pts2d) < 4:
            return None
 
        # Subsample for speed (ConvexHull is O(n log n))
        if len(pts2d) > 2000:
            idx   = np.random.choice(len(pts2d), 2000, replace=False)
            pts2d = pts2d[idx]
 
        hull = ConvexHull(pts2d)
        return pts2d[hull.vertices].astype(np.float32)
 
    except Exception:
        return None
 
 
# ══════════════════════════════════════════════════════
#  PerceptionPipeline — orchestrates all stages
# ══════════════════════════════════════════════════════
 
class PerceptionPipeline:
    """
    Full perception pipeline — runs all 5 stages in sequence.
 
    Usage:
        pipeline = PerceptionPipeline(cfg)
        result   = pipeline.process(frame)
 
        print(result.detections)       # list of Detection objects
        print(result.drivable_poly)    # (K, 2) polygon or None
    """
 
    def __init__(self, cfg: dict):
        self.cfg = cfg
 
    def process(self, frame) -> ProcessedFrame:
        """
        Process a single LidarFrame through the full pipeline.
 
        Args:
            frame (LidarFrame): raw input frame from any loader
 
        Returns:
            ProcessedFrame: pipeline output with detections and drivable area
        """
        xyz       = frame.xyz.copy()
        intensity = frame.intensity.copy()
 
        # Stage 1 — ROI filter
        xyz, intensity = filter_roi(xyz, intensity, self.cfg)
 
        # Stage 2 — Voxel downsampling
        xyz, intensity = voxel_downsample(
            xyz, intensity, self.cfg["lidar"]["voxel_size"]
        )
 
        # Stage 3 — Ground removal
        xyz_obs, int_obs, xyz_gnd = remove_ground_ransac(
            xyz, intensity, self.cfg
        )
 
        # Stage 4 — DBSCAN clustering
        detections = cluster_dbscan(xyz_obs, int_obs, self.cfg)
 
        # Stage 5 — Drivable zone
        drivable = extract_drivable_zone(xyz_gnd, self.cfg["roi"])
 
        return ProcessedFrame(
            xyz_obstacles=xyz_obs,
            xyz_ground=xyz_gnd,
            intensity=int_obs,
            detections=detections,
            drivable_poly=drivable,
            n_clusters=len(detections)
        )
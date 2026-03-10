"""
sensors/loader.py
======================================================
Universal LiDAR loader — multi-source abstraction

All sources return a standardized LidarFrame object containing:
    - xyz:       (N, 3) float32  — 3D point coordinates
    - intensity: (N,)   float32  — return intensity per point

Supported sources:
    KITTILoader      .bin files  — Nx4 float32 (x, y, z, intensity)
    NuScenesLoader   .bin files  — Nx5 float32 (x, y, z, intensity, ring)
    PCDLoader        .pcd files  — ASCII or binary via Open3D
    SyntheticLoader  generated   — realistic simulated scene
    LidarLoader      facade      — single entry point for all sources
"""
import os
import glob
import numpy as np
import copy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Optional

try:
    import open3d as o3d
    HAS_OPEN3D = True
except ImportError:
    HAS_OPEN3D = False


"""
    Standardized LiDAR frame — common structure for all sources.

    Attributes:
        xyz       (np.ndarray): point coordinates, shape (N, 3), float32
        intensity (np.ndarray): return intensity per point, shape (N,), float32
        timestamp (float):      acquisition time in seconds
        frame_id  (int):        sequential frame index
        source    (str):        data source identifier
        metadata  (dict):       source-specific extra data (ring index, file path...)

"""
    

@dataclass
class LidarFrame:
    xyz: np.ndarray
    intensity: np.ndarray
    timestamp: float = 0.0
    frame_id: int    = 0
    source: str      = "unknown"
    metadata: dict = field(default_factory=dict)

    @property
    def n_points(self) -> int:
        "Number of points in the current frame"
        return len(self.xyz)
    
    def __repr__(self):
        return (f"LidarFrame("
                f"id={self.frame_id}), "
                f"pts={self.n_points:,}, "
                f"src='{self.source}')")
    
# ===============================
# KITTILoader
# ===============================

class KITTILoader:
    """
    Loads Velodyne LiDAR scans from the KITTI dataset.

    File Format : binary float32, Nx4(x) (x, y, z, intensity)
    """

    def __init__(self, velodyne_dir: str):
        self.files = sorted(glob.glob(os.path.join(velodyne_dir, "*.bin")))
        
        if not self.files:
            raise FileNotFoundError(
                f"No .bin files found in: {velodyne_dir}"
            )
        self._idx=0

    def __len__(self) ->int:
        return len(self.files)
    
    def __iter__(self) ->Iterator[LidarFrame]:
        """ Iterate over all frames"""
        for i, filepath in enumerate(self.files):
            yield self._load(filepath, i)

    def _load(self, filepath: str, frame_id: int) -> LidarFrame:
        """Load a single .bin file and retun a LidarFrame"""
        raw = np.fromfile(filepath, dtype=np.float32).reshape(-1, 4)
        return LidarFrame(
            xyz= raw[:, :3].copy(),
            intensity= raw[:, 3].copy(),
            frame_id=frame_id,
            source="kitti",
            metadata={"file": filepath}
        )
    
    def read_next(self) -> Optional[LidarFrame]:
        """Read the next frame and returns None when all frames ae consumed"""

        if self._idx >= len(self.files):
            return None
        frame = self._load(self.files[self._idx], self._idx)
        self._idx +=1
        return frame
    

# ############################################################
# Nuscenes Loader
# ############################################################

class NuScenesLoader:
    """
    Loads LiDAR sweeps from the nuScenes dataset.

    File format: binary float32, Nx5 columns (x, y, z, intensity, ring_index)
    Expected layout: sweeps/LIDAR_TOP/*.bin

    Args:
        sweep_dir (str): path to the folder containing .bin sweep files
    """
    def __init__(self, sweep_dir: str):
        self.files = sorted(glob.glob(os.path.join(sweep_dir, "*.bin")))

        if not self.files:
            raise FileNotFoundError( f"No .bin file found in: {sweep_dir}")
        
        self._idx = 0
    
    def __len__(self) -> int:
        return len(self.files
                   )
    
    def __iter__(self) -> Iterator[LidarFrame]:
        """ Iterate over all sweeps"""
        for i, filepath in enumerate(self.files):
            yield self._load(filepath, i)

    def _load(self, filepath: str, frame_id: int) -> LidarFrame:
        raw = np.fromfile(filepath, dtype=np.float32).reshape(-1, 5)
        return LidarFrame(
            xyz= raw[:, :3].copy(),
            intensity= raw[:,3].copy(),
            frame_id= frame_id,
            source="nuscenes",
            metadata={
                "ring": raw[:, 4].copy(),
                "file": filepath
            }
        ) 
    def read_next(self) -> Optional[LidarFrame]:
        if self._idx >= len(self.files):
            return None
        frame = self._load(self.files[self._idx], self._idx)
        self._idx += 1
        return frame
    
##########################################################
# PCDLoader
#########################################################

class PCDLoader:
    """
    Load .pcd files (ASCII or binary) using open3d 
    Accepts either a single .pcd file or a directory of .pcd files.

    Args:
        path (str): path to a .pcd file or a folder containing .pcd files
    """

    def __init__(self, path: str):
        if not HAS_OPEN3D:
            raise ImportError(
                "open3d is required for PCDLoader.\n"
                "Install with: conda install -c open3d-admin open3d"
            )

        p = Path(path)
        if p.is_dir():
            self.files = sorted(p.glob("*.pcd"))
        elif p.suffix == ".pcd":
            self.files = [p]
        else:
            raise ValueError(f"Expected a .pcd file or directory, got: {path}")

        if not self.files:
            raise FileNotFoundError(f"No .pcd files found in: {path}")

        self._idx = 0

    def __len__(self) -> int:
        return len(self.files)

    def __iter__(self) -> Iterator[LidarFrame]:
        """Iterate over all .pcd files in order."""
        for i, filepath in enumerate(self.files):
            yield self._load(str(filepath), i)

    def _load(self, filepath: str, frame_id: int) -> LidarFrame:
        """Load a single .pcd file and return a LidarFrame."""
        pcd = o3d.io.read_point_cloud(filepath)
        xyz = np.asarray(pcd.points, dtype=np.float32)

        # Use color channel 0 as intensity if colors are available
        if pcd.has_colors():
            intensity = np.asarray(pcd.colors)[:, 0].astype(np.float32)
        else:
            intensity = np.ones(len(xyz), dtype=np.float32)

        return LidarFrame(
            xyz=xyz,
            intensity=intensity,
            frame_id=frame_id,
            source="pcd",
            metadata={"file": filepath}
        )

    def read_next(self) -> Optional[LidarFrame]:
        """Read the next .pcd file. Returns None when all files are consumed."""
        if self._idx >= len(self.files):
            return None
        frame = self._load(str(self.files[self._idx]), self._idx)
        self._idx += 1
        return frame
    
# =====================================================
#  SyntheticLoader
# ======================================================

class SyntheticLoader:
    """
    Generates realistic synthetic LiDAR frames for development and testing.
    No hardware or dataset required.

    Each frame contains:
        - Flat ground plane (noisy, ~N/2 points)
        - Object point clouds (sampled from bounding box volumes)
        - Ambient noise points

    Args:
        n_frames  (int): total number of frames to generate
        n_points  (int): approximate number of points per frame
    """

    # Static scene definition — objects with initial position and velocity
    _SCENE_TEMPLATE = [
        {"type": "car",        "pos": [ 15.0,  0.0,  0.0], "v": [0.50,  0.00, 0.0]},
        {"type": "car",        "pos": [ 25.0, -3.5,  0.0], "v": [0.40,  0.00, 0.0]},
        {"type": "car",        "pos": [ 35.0,  0.0,  0.0], "v": [0.60,  0.00, 0.0]},
        {"type": "pedestrian", "pos": [  8.0,  4.0,  0.0], "v": [0.00,  0.08, 0.0]},
        {"type": "pedestrian", "pos": [  6.0, -4.0,  0.0], "v": [0.00, -0.06, 0.0]},
        {"type": "cyclist",    "pos": [ 12.0,  2.5,  0.0], "v": [0.25,  0.00, 0.0]},
    ]

    # Approximate bounding box dimensions per class [l, w, h] in meters
    _DIMENSIONS = {
        "car":        [4.5, 1.9, 1.5],
        "truck":      [8.0, 2.5, 3.0],
        "pedestrian": [0.5, 0.5, 1.8],
        "cyclist":    [1.8, 0.7, 1.7],
    }

    def __init__(self, n_frames: int = 10000, n_points: int = 25000):
        self.n_frames  = n_frames
        self.n_points  = n_points
        self._frame_id = 0
        self._rng      = np.random.default_rng(seed=42)

        # Deep copy so each instance has its own mutable scene state
        self._objects = copy.deepcopy(self._SCENE_TEMPLATE)
        for obj in self._objects:
            obj["pos"] = np.array(obj["pos"], dtype=float)
            obj["v"]   = np.array(obj["v"],   dtype=float)

    def _sample_object_points(self, obj: dict, n: int) -> np.ndarray:
        """Sample n points uniformly inside the object's bounding box."""
        dims = self._DIMENSIONS.get(obj["type"], [1.0, 1.0, 1.0])
        pts  = self._rng.uniform(-0.5, 0.5, (n, 3)) * np.array(dims)
        pts += obj["pos"]
        return pts.astype(np.float32)

    def read_next(self) -> Optional[LidarFrame]:
        """
        Generate the next synthetic frame.
        Returns None when n_frames have been generated.
        """
        if self._frame_id >= self.n_frames:
            return None

        # Move all objects forward by one time step
        for obj in self._objects:
            obj["pos"] += obj["v"] + self._rng.normal(0, 0.015, 3)

        # Ground plane 
        n_ground = self.n_points // 2
        gx = self._rng.uniform(-5,  40, n_ground)
        gy = self._rng.uniform(-8,   8, n_ground)
        gz = self._rng.normal(-1.8, 0.04, n_ground)
        gi = self._rng.uniform(0.1, 0.4,  n_ground)
        ground_xyz = np.stack([gx, gy, gz], axis=1).astype(np.float32)

        # Object points
        obj_xyz_list = []
        obj_int_list = []
        n_per_obj = max(80, (self.n_points // 3) // len(self._objects))

        for obj in self._objects:
            pts = self._sample_object_points(obj, n_per_obj)
            obj_xyz_list.append(pts)
            obj_int_list.append(
                self._rng.uniform(0.5, 1.0, n_per_obj).astype(np.float32)
            )

        # Ambient noise 
        n_noise   = 300
        noise_xyz = self._rng.uniform(
            [-5, -8, -2], [40, 8, 3], (n_noise, 3)
        ).astype(np.float32)
        noise_int = self._rng.uniform(0.0, 0.2, n_noise).astype(np.float32)

        # Combine all points
        xyz_all = np.vstack(
            [ground_xyz] + obj_xyz_list + [noise_xyz]
        )
        int_all = np.concatenate(
            [gi] + obj_int_list + [noise_int]
        ).astype(np.float32)

        frame = LidarFrame(
            xyz=xyz_all,
            intensity=int_all,
            frame_id=self._frame_id,
            source="synthetic",
            metadata={"objects": copy.deepcopy(self._objects)}
        )

        self._frame_id += 1
        return frame

    def __iter__(self) -> Iterator[LidarFrame]:
        """Iterate — resets frame counter at the start."""
        self._frame_id = 0
        while True:
            frame = self.read_next()
            if frame is None:
                break
            yield frame


# ======================================================
#  LidarLoader — facade (single entry point)
# ===================================================

class LidarLoader:
    """
    Facade — single entry point for all LiDAR data sources.

    Hides the complexity of individual loaders behind a clean interface.
    The rest of the project only needs to know about LidarLoader.

    Usage:
        loader = LidarLoader.synthetic()
        loader = LidarLoader.kitti("data/kitti/velodyne/")
        loader = LidarLoader.nuscenes("data/nuscenes/sweeps/LIDAR_TOP/")
        loader = LidarLoader.pcd("data/scans/")
        loader = LidarLoader.from_path("data/unknown_dataset/")  # auto-detect
    """

    @staticmethod
    def kitti(velodyne_dir: str) -> KITTILoader:
        """Load KITTI Velodyne .bin files from a directory."""
        return KITTILoader(velodyne_dir)

    @staticmethod
    def nuscenes(sweep_dir: str) -> NuScenesLoader:
        """Load nuScenes LiDAR sweeps from a directory."""
        return NuScenesLoader(sweep_dir)

    @staticmethod
    def pcd(path: str) -> PCDLoader:
        """Load .pcd files from a file or directory."""
        return PCDLoader(path)

    @staticmethod
    def synthetic(n_frames: int = 10000,
                  n_points: int = 25000) -> SyntheticLoader:
        """Generate synthetic LiDAR frames (no hardware required)."""
        return SyntheticLoader(n_frames=n_frames, n_points=n_points)

    @staticmethod
    def from_path(path: str):
        """
        Auto-detect the data format from file extensions and return
        the appropriate loader.

        Detection logic:
            *.pcd files found  → PCDLoader
            *.bin + 4 cols     → KITTILoader
            *.bin + 5 cols     → NuScenesLoader
        """
        p = Path(path)

        if not p.exists():
            raise FileNotFoundError(f"Path does not exist: {path}")

        if p.is_dir():
            pcd_files = list(p.glob("*.pcd"))
            bin_files = list(p.glob("*.bin"))

            if pcd_files:
                return PCDLoader(path)

            if bin_files:
                # Detect KITTI (4 cols) vs nuScenes (5 cols)
                sample = np.fromfile(str(bin_files[0]), dtype=np.float32)
                n_cols = 5 if len(sample) % 5 == 0 else 4
                if n_cols == 5:
                    return NuScenesLoader(path)
                else:
                    return KITTILoader(path)

        elif p.suffix == ".pcd":
            return PCDLoader(path)

        elif p.suffix == ".bin":
            return KITTILoader(str(p.parent))

        raise ValueError(
            f"Could not detect data format for: {path}\n"
            f"Supported: .pcd, .bin (KITTI or nuScenes)"
        )

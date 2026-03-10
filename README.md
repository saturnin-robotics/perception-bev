# 🚗 perception-bev

> **Bird's Eye View perception system for autonomous driving**  
> LiDAR point cloud processing · Multi-Object Tracking · BEV visualization

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)
![Vedo](https://img.shields.io/badge/Vedo-2026+-green)
![Open3D](https://img.shields.io/badge/Open3D-0.19-orange)
![License](https://img.shields.io/badge/License-MIT-lightgrey)
![Status](https://img.shields.io/badge/Status-In%20Progress-yellow)

---

## 📌 Overview

Real-time 3D perception pipeline that processes LiDAR point clouds and renders a
Bird's Eye View with detected and tracked objects, drivable area, lanes,
and velocity vectors.

```
LiDAR Input → ROI Filter → Ground Removal (RANSAC) → Clustering (DBSCAN)
    → Classification → Kalman Tracking → BEV Visualization (Vedo)
```

---

## 🗺️ Roadmap

### 🔲 Phase 1 — Foundation
- [x] Project structure & centralized config (`params.yaml`)
- [x] Multi-source LiDAR loader (KITTI, nuScenes, PCD, Synthetic)
- [ ] ROI filtering + voxel downsampling (Open3D)
- [ ] Ground removal via RANSAC (Open3D `segment_plane`)
- [ ] Object clustering via DBSCAN (sklearn)
- [ ] Dimension-based classification (car, truck, pedestrian, cyclist)
- [ ] Drivable area extraction (Convex Hull on ground points)
- [ ] Kalman 6D filter per track `[x, y, z, vx, vy, vz]`
- [ ] Hungarian algorithm for detection-to-track association
- [ ] Split-screen BEV + Ego View visualizer (Vedo)
- [ ] Real-time HUD (FPS, object count, class legend)
- [ ] Velocity arrows + trajectory trails
- [ ] Multi-threaded pipeline (perception thread + Vedo timer)

### 🔲 Phase 2 — Sensor Fusion
- [ ] Camera integration (OpenCV video stream)
- [ ] LiDAR → image projection (extrinsic calibration)
- [ ] Point cloud colorization from camera image
- [ ] YOLOv8 2D detection on camera frames
- [ ] Label enrichment: YOLO class → DBSCAN cluster

### 🔲 Phase 3 — Lane & Map
- [ ] Lane marking detection from ground points
- [ ] Lane polynomial fitting (numpy)
- [ ] Road boundary estimation
- [ ] Local occupancy grid map (BEV top-down)
- [ ] HD map integration (OpenDRIVE / Lanelet2)

### 🔲 Phase 4 — Deep Learning Detection
- [ ] PointPillars 3D object detection
- [ ] CenterPoint inference
- [ ] Replace DBSCAN pipeline with DL detector
- [ ] Confidence score integration into tracker

### 🔲 Phase 5 — World Model
- [ ] Ego-vehicle localization (kiss-icp SLAM)
- [ ] Scene graph construction (objects + relations)
- [ ] Trajectory prediction (constant velocity / learned)
- [ ] Risk assessment per tracked object
- [ ] Toward a World Model representation

---

## 🏗️ Architecture

```
perception-bev/
├── main.py                   # Entry point (CLI + threading)
├── requirements.txt          # Python dependencies
├── .gitignore
├── README.md
├── config/
│   └── params.yaml           # All parameters (ROI, DBSCAN, Kalman, colors...)
├── sensors/
│   └── loader.py             # KITTI / nuScenes / PCD / Synthetic loader
├── processing/
│   └── pipeline.py           # ROI → Voxel → RANSAC → DBSCAN → Drivable zone
├── tracking/
│   └── mot.py                # Kalman 6D + Hungarian algorithm (MOT)
├── visualization/
│   └── bev.py                # Split-screen BEV + Ego View (Vedo)
└── utils/
    ├── config.py             # YAML config loader
    └── geometry.py           # Geometric utilities (bbox, IoU, rotations)
```

---

## ⚙️ Installation

### Requirements
- Python 3.11
- Conda (recommended)
- NVIDIA GPU (optional — CPU fallback available)

### Setup

```bash
# 1. Create conda environment
conda create -n perception_env python=3.11 -y
conda activate perception_env

# 2. Install Open3D via conda (required before pip)
conda install -c open3d-admin -c conda-forge open3d -y

# 3. Install remaining dependencies
pip install -r requirements.txt
```

### Verify installation

```bash
python -c "
import numpy, scipy, sklearn, yaml
import open3d as o3d
import vedo, filterpy, cv2
print('open3d  ', o3d.__version__)
print('vedo    ', vedo.__version__)
print('=== Ready ===')
"
```

---

## 🚀 Usage

```bash
# Synthetic data (no hardware required)
python main.py

# KITTI dataset
python main.py --source kitti --path data/kitti/velodyne/ --loop

# PCD files (folder or single file)
python main.py --source pcd --path data/my_scans/

# nuScenes sweeps
python main.py --source nuscenes --path data/nuscenes/sweeps/LIDAR_TOP/

# Auto-detect format
python main.py --source auto --path data/my_dataset/

# Custom options
python main.py --fps 30 --loop --config my_config.yaml
```

---

## 🖥️ BEV Visualization Layers

| Layer | Content | Type |
|-------|---------|------|
| 0 | Dark background + distance rings (10/20/.../50m) | Static |
| 1 | Drivable area (RANSAC ground + Convex Hull) | Dynamic |
| 2 | Ground point cloud (gray) | Dynamic |
| 3 | Obstacle point cloud (heatmap by intensity) | Dynamic |
| 4 | 3D bounding boxes wireframe (color by class) | Dynamic |
| 5 | Velocity arrows | Dynamic |
| 6 | Trajectory trails (last 30 positions) | Dynamic |
| 7 | Ego-vehicle (white box) | Dynamic |
| 8 | HUD (FPS, counts, legend) | Dynamic |

### Keyboard shortcuts

| Key | Action |
|-----|--------|
| `SPACE` | Pause / Resume |
| `Q` | Quit |
| `R` | Reset cameras |
| `S` | Screenshot |

---

## 🎨 Object Classes & Colors

| Class | Color | Hex |
|-------|-------|-----|
| Car | 🟡 Yellow | `#F5C518` |
| Truck | 🟠 Orange | `#FF6B35` |
| Pedestrian | 🔵 Cyan | `#00D4FF` |
| Cyclist | 🟣 Magenta | `#FF3CAC` |
| Unknown | ⚪ Gray | `#AAAAAA` |

---

## 📦 Dependencies

| Package | Version | Role |
|---------|---------|------|
| vedo | ≥ 2023.5 | 3D BEV visualization |
| open3d | ≥ 0.17 | RANSAC, voxel downsampling, PCD I/O |
| scikit-learn | ≥ 1.3 | DBSCAN clustering |
| filterpy | ≥ 1.4 | Kalman filter |
| scipy | ≥ 1.10 | Hungarian algorithm, Convex Hull |
| numpy | ≥ 1.24 | Point cloud operations |
| opencv-python | ≥ 4.8 | Camera integration (optional) |

---

## 📄 License

MIT License — see [LICENSE](LICENSE) file.

---

## 🙏 Acknowledgements

- [Vedo](https://vedo.embl.es) — Scientific 3D visualization library
- [Open3D](http://www.open3d.org) — 3D data processing
- [KITTI Dataset](https://www.cvlibs.net/datasets/kitti/) — Benchmark dataset
- [nuScenes](https://www.nuscenes.org) — Autonomous driving dataset
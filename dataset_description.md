# KITTI Object Detection Dataset & Conversion Taxonomy

The **KITTI Vision Benchmark Suite** is a standardized high-resolution dataset recorded from moving vehicles around Karlsruhe, Germany. It provides stereo camera pairs, LiDAR point clouds, and high-fidelity 2D/3D annotation metrics for autonomous vehicle evaluation.

---

## 1. Ground Truth Format Comparison

### Option A: Raw KITTI Text Format (Source Annotations)
Each image is accompanied by an equivalent `.txt` file specifying up to 15 parameters per row:

```txt
# Column Index Definitions:
# 0      : Object Type (e.g., 'Car', 'Pedestrian', 'Cyclist', 'Truck', 'Tram', 'DontCare')
# 1      : Truncation level [0.0 to 1.0] (float)
# 2      : Occlusion level integer (0 = fully visible, 1 = partly occluded, 2 = heavily, 3 = unknown)
# 3      : Observation angle alpha [-pi to pi] (float)
# 4..7   : 2D Bounding Box pixel rectangle [left, top, right, bottom] (floats)
# 8..10  : 3D Dimension heights, widths, lengths (floats)
# 11..13 : 3D Camera coordinates x, y, z center (floats)
# 14     : Heading rotation angle ry [-pi to pi] (float)
```

**Example Row:**
```txt
Car 0.00 0 -1.57 586.23 173.40 701.12 254.12 1.48 1.62 3.51 1.25 1.65 11.20 -1.59
```

---

### Option B: Normalized YOLOv8 Text Format (Target Input)
YOLOv8 requires simple space-separated values scaled from `0.0` to `1.0` reflecting the image boundaries:

```txt
[class_id] [x_center] [y_center] [width] [height]
```

**Normalization Formulas:**
Given image resolution bounds ($W_{img}$, $H_{img}$) and 2D pixel coordinates ($x_{min}, y_{min}, x_{max}, y_{max}$):

$$x_{center} = \frac{x_{min} + x_{max}}{2.0 \cdot W_{img}}$$

$$y_{center} = \frac{y_{min} + y_{max}}{2.0 \cdot H_{img}}$$

$$width = \frac{x_{max} - x_{min}}{W_{img}}$$

$$height = \frac{y_{max} - y_{min}}{H_{img}}$$

---

## 2. Taxonomic Mapping Grid

To configure fine-tuning for Autopilot HUD capabilities, the dataset filters out distant or secondary target categories and groups others into 5 critical focus categories:

| KITTI Source Label | mapped Target ID | YOLO Name | Detection Type |
| :--- | :---: | :--- | :--- |
| `Car`, `Van` | **0** | `car` | Vehicle |
| `Pedestrian`, `Person_sitting` | **1** | `pedestrian` | Vulnerable Obstacle |
| `Cyclist` | **2** | `cyclist` | Vulnerable Obstacle |
| `Truck` | **3** | `truck` | Heavy Vehicle |
| `Tram` | **4** | `tram` | Transit Train |

*Background labels (like `DontCare`, `Misc`, or `SittingOnGround`) are systematically ignored during loading to avoid penalizing the neural engine.*

---

## 3. Directory Topology
Preprocessing automatically partitions raw resources into the folder hierarchy formatted for easy parsing inside the `data.yaml` layout:

```yaml
dataset/
├── data.yaml              <-- Dataset specification
├── images/
│   ├── train/             <-- 80% Training Frame files (.png)
│   └── val/               <-- 20% Evaluation Frame files (.png)
└── labels/
    ├── train/             <-- Normalized label descriptions (.txt)
    └── val/               <-- Normalized label descriptions (.txt)
```

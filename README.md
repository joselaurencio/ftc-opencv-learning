# ftc-opencv-learning repository

Progress toward real-time yellow ball detection for FIRST Tech Challenge drivetrain autonomous decision making.

## overview

This repository contains Python + OpenCV experiments focused on color-based object detection. The long-term goal is to develop a robust vision pipeline that can be ported to EasyOpenCV / VisionPortal on the robot for identifying yellow game elements and providing positional data (x-coordinate) to the drivetrain.

## current scripts

- `test_opencv.py`  
  Basic webcam test and OpenCV installation verification.

- `hsv_tuner.py`  
  Interactive trackbar tool for tuning HSV color ranges in real time. Critical for adapting to field lighting conditions.

- `ball_detection.py`  
  Main detection pipeline. Performs HSV thresholding, morphological noise reduction, contour detection, and outputs center coordinates with simple left/center/right decisions.

## pipeline details

1. BGR → HSV conversion
2. `inRange()` thresholding using tuned bounds `[16, 85, 97]` – `[35, 255, 255]`
3. Morphological opening + closing (kernel 7x7)
4. Gaussian blur on mask
5. `findContours()` with area filtering (>800 px)
6. Bounding box + centroid calculation
7. Largest contour selection for primary target

## usage

```bash
# Activate environment
source opencv-env/bin/activate

# Run tuner
python hsv_tuner.py

# Run detection
python ball_detection.py
```

Press `q` to exit any script.

## future work

- Frame-to-frame position smoothing
- Multi-object tracking
- Region of Interest (ROI) implementation
- Performance benchmarking
- Java port for FTC SDK + VisionPortal

## setup requirements

- Python 3
- OpenCV with contrib modules
- Working webcam

---

**Last Updated:** July 2026  
**Status:** Local prototyping phase
# **Defend The Republic: Autonomous Aerial Computer Vision System**

## 📖 Project Overview

This repository contains the computer vision subsystem for the **Defend The Republic (DTR)** autonomous aerial competition team at Indiana University Bloomington.

As a Junior Intelligent Systems Engineering student, my role focused on developing a **low-latency object detection and range-finding pipeline**. The system enables an autonomous blimp/drone to identify game pieces (balloons and goals), calculate their precise 3D position relative to the vehicle, and feed navigation data to the flight controller in real time.

---

## 🎯 Key Capabilities

- **Object Detection:** Identifies 8 distinct classes (Orange/Yellow Circles, Squares, Triangles; Purple/Green Balls) using a custom-trained YOLOv8 model.  
- **Edge Optimization:** Optimized for Raspberry Pi 4, achieving **180ms inference time** (down from 1s) via quantization and input scaling.  
- **3D Localization:** Algorithms compute distance (meters) and angles (θₓ, θᵧ) from 2D bounding boxes.

---

## 🛠️ Tech Stack

**Hardware:** Raspberry Pi 4 Model B, Pi Camera Module  
**ML Frameworks:** PyTorch, Ultralytics YOLOv8  
**Data Processing:** Roboflow, OpenCV  
**Techniques:** INT8 Quantization, Transfer Learning, Trigonometric Range Finding

---

## ⚙️ System Architecture

The pipeline processes video input to output navigation vectors for the flight controller.

```mermaid
graph TD
    A[Pi Camera Input] -->|320x320 Stream| B(YOLOv8 Int8 Model)
    B -->|Bounding Boxes| C{Confidence > 0.5}
    C -->|Yes| D[Object ID Extraction]
    C -->|No| A
    D --> E[Distance Calculation]
    D --> F[Angle Calculation]
    E & F --> G[Output Vector]
    G -->|UDP/Serial| H[Flight Controller]

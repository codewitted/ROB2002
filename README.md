This repository is a companion resource for the Evaluation for Robotics module, ROB2002, at the University of Lincoln. You will find here details about the development environment enabling work with the robots, the description of practical tasks for each week, and practical instructions for the assignment.

## Workshops
For a list of all the workshops offered as part of ROB2002 and additional resources, please refer to the [Wiki](https://github.com/LCAS/ROB2002/wiki).
The rest are instructions to you the files and code provided for the evaluation for robotics assessment ROB2002-2425
# Robotic Counting Demo (Baseline vs. Enhanced)

This repository contains ROS2 Python scripts to demonstrate a robotic counting experiment under three different environments:

- Environment 1 (Open warehouse)
- Environment 2 (Moderate obstacles)
- Environment 3 (Heavily cluttered)

9 boxes were placed (3 red, 3 green, 3 blue) in each environment.

## Nodes Overview

1. combined_detector.py 
   - Subscribes to a color + depth camera feed (topics: `/limo/depth_camera_link/image_raw`, `/limo/depth_camera_link/depth/image_raw`, etc.).  
   - Detects colored boxes in 2D, converts them to 3D in the global frame (`odom`), and publishes each detection as a `PoseStamped` on `/object_location`.

2. counter_3d.py 
   - Subscribes to `/object_location`.  
   - Maintains a list of unique objects, skipping any new pose within 0.3–0.6 m of an existing one.  
   - Prints “`Total Unique Objects: X`” each time a new object is added.

3. baseline_sweeper.py 
   - A simple corner-based approach with minimal coverage.  
   - Visits corners, returns home.  
   - Typically faster but misses occluded objects.

4. enhanced_sweeper.py 
   - A multi-waypoint or zigzag approach covering the entire warehouse area.  
   - Slower but more comprehensive coverage.

## Installation

1. Clone this repo into your ROS2 workspace:
   ```bash
   cd ~/your_ros2_ws/src
   git clone https://github.com/codewitted.git
   cd ..
   colcon build
   source install/setup.bash

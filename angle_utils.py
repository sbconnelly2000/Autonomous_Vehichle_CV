# angle_utils.py
"""
Angle calculation utilities.
Given pixel coordinates of an object's bounding-box center,
return the horizontal and vertical angles between the camera center
and the object in degrees.
"""

import math

# === CAMERA PARAMETERS ===
FOCAL_LENGTH_PX = 700    # Adjust if your camera calibration changes
IMG_WIDTH = 640          # your Pi cam resized frame width
IMG_HEIGHT = 640         # your Pi cam resized frame height


def get_object_angles_px(x_center, y_center,
                         img_width=IMG_WIDTH,
                         img_height=IMG_HEIGHT,
                         focal_px=FOCAL_LENGTH_PX):
    """
    Convert bounding-box pixel center into (theta_x, theta_y) in degrees.

    +theta_x  → object is to the right of center
    -theta_x  → object is to the left

    +theta_y  → object is ABOVE the center
    -theta_y  → object is BELOW the center
    """
    # camera's optical center
    cx = img_width / 2.0
    cy = img_height / 2.0

    # pixel offset from image center
    dx = x_center - cx
    dy = y_center - cy

    # angles using arctan model
    theta_x = math.degrees(math.atan(dx / focal_px))
    theta_y = math.degrees(math.atan(dy / focal_px))

    # flip Y so positive = up
    return theta_x, -theta_y


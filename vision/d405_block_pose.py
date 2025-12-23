import pyrealsense2 as rs
import numpy as np
import cv2
import json

# -----------------------------
# Load camera calibration
# -----------------------------
with open("camera_calib.json", "r") as f:
    calib = json.load(f)

R = np.array(calib["R"])
t = np.array(calib["t"])

T_cam_world = np.eye(4)
T_cam_world[:3, :3] = R
T_cam_world[:3, 3] = t


# -----------------------------
# Start RealSense
# -----------------------------
pipeline = rs.pipeline()
config = rs.config()

config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

pipeline.start(config)

profile = pipeline.get_active_profile()
intr = profile.get_stream(rs.stream.color)\
              .as_video_stream_profile().get_intrinsics()

fx, fy = intr.fx, intr.fy
cx, cy = intr.ppx, intr.ppy


def pixel_to_camera(u, v, depth):
    z = depth[v, u] * 0.001  # mm → meters
    if z <= 0:
        return None
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    return np.array([x, y, z])


def cam_to_world(p_cam):
    p_h = np.append(p_cam, 1.0)
    return (T_cam_world @ p_h)[:3]


# -----------------------------
# Capture one frame
# -----------------------------
frames = pipeline.wait_for_frames()
color_frame = frames.get_color_frame()
depth_frame = frames.get_depth_frame()

color = np.asanyarray(color_frame.get_data())
depth = np.asanyarray(depth_frame.get_data())


# -----------------------------
# OpenCV block detection
# (example: red blocks)
# -----------------------------
hsv = cv2.cvtColor(color, cv2.COLOR_BGR2HSV)

lower_red = np.array([0, 120, 70])
upper_red = np.array([10, 255, 255])

mask = cv2.inRange(hsv, lower_red, upper_red)
mask = cv2.medianBlur(mask, 5)

contours, _ = cv2.findContours(
    mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
)

block_poses = []

for cnt in contours:
    area = cv2.contourArea(cnt)
    if area < 800:
        continue

    M = cv2.moments(cnt)
    if M["m00"] == 0:
        continue

    u = int(M["m10"] / M["m00"])
    v = int(M["m01"] / M["m00"])

    p_cam = pixel_to_camera(u, v, depth)
    if p_cam is None:
        continue

    p_world = cam_to_world(p_cam)

    T_obj = np.eye(4)
    T_obj[:3, 3] = p_world

    block_poses.append(T_obj)

    # Debug draw
    cv2.circle(color, (u, v), 6, (0, 255, 0), -1)


pipeline.stop()

print("Detected blocks:", len(block_poses))
for i, T in enumerate(block_poses):
    print(f"\nBlock {i} pose:\n", T)

cv2.imshow("detections", color)
cv2.waitKey(0)
cv2.destroyAllWindows()

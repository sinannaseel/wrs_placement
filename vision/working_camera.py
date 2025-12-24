import cv2
import json
import numpy as np
import pyrealsense2 as rs
import wrs.basis.robot_math as rm


GREEN_LOW  = (35, 80, 70)
GREEN_HIGH = (85, 255, 255)

with open("cameras/manual_calibration.json", "r") as f:
    extrinsic = np.array(json.load(f)["affine_mat"])

ctx = rs.context()
devices = ctx.query_devices()
if len(devices) < 2:
    raise RuntimeError("Less than 2 RealSense cameras connected")

serial = devices[1].get_info(rs.camera_info.serial_number)
print("[Camera] Using serial:", serial)

def main():
    ctx = rs.context()
    devices = ctx.query_devices()
    if len(devices) < 2:
        raise RuntimeError("Less than 2 RealSense cameras connected")

    serial = devices[1].get_info(rs.camera_info.serial_number)
    print("[Camera] Using serial:", serial)
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device(serial)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    profile = pipeline.start(config)
    
    try:
        align = rs.align(rs.stream.color)
        intr = profile.get_stream(rs.stream.color)\
                      .as_video_stream_profile().get_intrinsics()

        print("[Camera] Ready (ID 1)")

        ROBOT_POS = np.array([0, 0, 0])
        ROBOT_ROT = np.eye(3)

        while True:
            frames = pipeline.wait_for_frames()
            frames = align.process(frames)

            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()

            color = np.asanyarray(color_frame.get_data())
            hsv = cv2.cvtColor(color, cv2.COLOR_BGR2HSV)

            mask = cv2.inRange(hsv, GREEN_LOW, GREEN_HIGH)
            cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for c in cnts:
                if cv2.contourArea(c) < 300:
                    continue

                rect = cv2.minAreaRect(c)
                (cx, cy), (_, _), angle = rect

                u, v = int(cx), int(cy)
                depth = depth_frame.get_distance(u, v)
                if depth == 0:
                    continue

                cam_xyz = np.array(
                    rs.rs2_deproject_pixel_to_point(intr, [u, v], depth)
                )

                cam_xyz_h = np.append(cam_xyz, 1.0)
                tcp = rm.homomat_from_posrot(ROBOT_POS, ROBOT_ROT)
                world_xyz = (tcp @ extrinsic @ cam_xyz_h)[:3]

                print("\n[Cube Pose]")
                print("Position (world):", world_xyz)
                print("Yaw (deg):", angle)

                box = cv2.boxPoints(rect).astype(int)
                cv2.drawContours(color, [box], 0, (0,255,0), 2)

            cv2.imshow("Cube View (Cam 1)", color)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        print("\n[INFO] Stopped by user")

    except Exception as e:
        print("\n[ERROR]", e)

    finally:
        if pipeline is not None:
            pipeline.stop()
        cv2.destroyAllWindows()
        print("[INFO] Resources released")



if __name__ == "__main__":
    main()

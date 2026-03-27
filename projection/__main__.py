# -*- coding: utf-8 -*-
"""
球场标定命令行工具。

使用方式：
    # 交互式标定（推荐）
    python -m projection.calibration_cli interactive 1.mp4 --frame 30 --output my_calibration.json

    # 自动线条标定
    python -m projection.calibration_cli auto 1.mp4 --frame 30 --output my_calibration.json --debug

    # 验证标定
    python -m projection.calibration_cli verify 1.mp4 --calibration my_calibration.json --frame 100
"""

import argparse
import sys

import cv2
import numpy as np

from projection.interactive_calibrator import (
    InteractiveCalibrator,
    LineBasedCalibrator,
    PITCH_CORNERS_METER_CENTER,
)


def interactive_calibration(args: argparse.Namespace) -> None:
    """交互式标定：用户点击 4 个角点。"""
    calibrator = InteractiveCalibrator()

    # 加载帧
    if not calibrator.load_frame_from_video(args.video, args.frame):
        print(f"Error: Cannot open video: {args.video}", file=sys.stderr)
        return

    print(f"Loaded frame {args.frame} from {args.video}")
    print(f"Frame size: {calibrator.frame_w}x{calibrator.frame_h}")
    print()
    print("Instructions:")
    print("  Click on the 4 field corners in order:")
    print("    1. TOP-LEFT corner of the visible pitch")
    print("    2. TOP-RIGHT corner of the visible pitch")
    print("    3. BOTTOM-RIGHT corner of the visible pitch")
    print("    4. BOTTOM-LEFT corner of the visible pitch")
    print()
    print("  Controls:")
    print("    'r' - Reset/clear all points")
    print("    'u' - Undo last point")
    print("    'c' - Compute homography and save")
    print("    'q' - Quit without saving")
    print()
    print("NOTE: Click on the CORNERS of the FIELD LINES (line intersections), not on players!")

    clicked_pts: list = []
    labels = calibrator.get_click_labels()

    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            clicked_pts.append((x, y))
            calibrator.click_point(x, y, label=labels[len(clicked_pts) - 1] if len(clicked_pts) <= 4 else None)
            # 绘制
            display = calibrator.frame.copy()
            display = calibrator.draw_clicks(display)
            # 标注下一个要点的位置
            if len(clicked_pts) <= 4:
                cv2.putText(
                    display,
                    f"Click corner {len(clicked_pts)}/4: {labels[len(clicked_pts) - 1] if len(clicked_pts) <= 4 else ''}",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (0, 255, 0),
                    2,
                )
            cv2.imshow("Interactive Calibration", display)

    display = calibrator.frame.copy()
    cv2.namedWindow("Interactive Calibration")
    cv2.setMouseCallback("Interactive Calibration", mouse_callback)

    cv2.putText(
        display,
        "Click 4 corners: 1=TL, 2=TR, 3=BR, 4=BL (press 'c' to compute, 'q' to quit)",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2,
    )
    cv2.imshow("Interactive Calibration", display)

    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            print("Cancelled.")
            cv2.destroyAllWindows()
            return
        elif key == ord("r"):
            # Reset
            clicked_pts.clear()
            calibrator.clicked_points.clear()
            calibrator.correspondence_labels.clear()
            display = calibrator.frame.copy()
            cv2.putText(
                display,
                "Reset. Click 4 corners: 1=TL, 2=TR, 3=BR, 4=BL (press 'c' to compute, 'q' to quit)",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )
            cv2.imshow("Interactive Calibration", display)
            print("Reset.")
        elif key == ord("u"):
            # Undo
            if clicked_pts:
                clicked_pts.pop()
                calibrator.clicked_points.pop()
                calibrator.correspondence_labels.pop()
                display = calibrator.frame.copy()
                display = calibrator.draw_clicks(display)
                if len(clicked_pts) < 4:
                    cv2.putText(
                        display,
                        f"Click corner {len(clicked_pts) + 1}/4: {labels[len(clicked_pts)]}",
                        (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 0),
                        2,
                    )
                cv2.imshow("Interactive Calibration", display)
                print(f"Undid. Points remaining: {len(clicked_pts)}")
        elif key == ord("c") or len(clicked_pts) >= 4:
            break

    cv2.destroyAllWindows()

    if len(clicked_pts) < 4:
        print("Need at least 4 points. Cancelled.")
        return

    # 计算单应矩阵
    H, quality = calibrator.compute_homography()

    if H is not None:
        print(f"\nHomography computed. Quality: {quality}")
        calibrator.save_calibration(args.output, H)
        print(f"Saved to {args.output}")

        # 验证：投影一些点
        print("\nVerification — projecting corner points back:")
        H_inv = np.linalg.inv(H)
        for name, (mx, my) in PITCH_CORNERS_METER_CENTER.items():
            pt = np.array([mx, my, 1.0], dtype=np.float32)
            proj = H_inv @ pt
            proj = proj / proj[2]
            print(f"  {name}: world({mx:.1f}, {my:.1f}) -> image({proj[0]:.0f}, {proj[1]:.0f})")
    else:
        print("Failed to compute homography. Try again with better corner selections.")


def auto_calibration(args: argparse.Namespace) -> None:
    """自动线条标定。"""
    calibrator = LineBasedCalibrator()

    # 加载帧
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"Error: Cannot open video: {args.video}", file=sys.stderr)
        return

    cap.set(cv2.CAP_PROP_POS_FRAMES, args.frame)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        print(f"Error: Cannot read frame {args.frame}", file=sys.stderr)
        return

    print(f"Frame {args.frame}: {frame.shape[1]}x{frame.shape[0]}")
    print("Running line-based calibration...")

    # 检测线条
    lines = calibrator.detect_lines(frame)
    if lines is not None:
        print(f"Detected {len(lines)} line segments")
    else:
        print("No lines detected.")

    # 找交点
    intersections = calibrator.find_intersections(lines)
    print(f"Found {len(intersections)} intersection points")

    # 完整标定
    H, quality = calibrator.calibrate(frame)

    if H is not None:
        print(f"\nCalibration succeeded! Quality: {quality}")
        import json

        data = {
            "H": H.tolist(),
            "coord_system": "meter_center",
            "pitch_length_m": 105.0,
            "pitch_width_m": 68.0,
        }
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        print(f"Saved to {args.output}")

        # 验证投影
        H_inv = np.linalg.inv(H)
        print("\nVerification — projecting corner points back:")
        for name, (mx, my) in PITCH_CORNERS_METER_CENTER.items():
            pt = np.array([mx, my, 1.0], dtype=np.float32)
            proj = H_inv @ pt
            proj = proj / proj[2]
            print(f"  {name}: world({mx:.1f}, {my:.1f}) -> image({proj[0]:.0f}, {proj[1]:.0f})")

        # 调试可视化
        if args.debug:
            corners = calibrator.filter_quadrilateral(intersections)
            vis = calibrator.visualize(frame, lines, corners, intersections)
            cv2.namedWindow("Calibration Debug")
            cv2.imshow("Calibration Debug", vis)
            print("\nDebug window open. Press any key to close.")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
    else:
        print("\nCalibration FAILED. Try:")
        print("  1. Use a different frame (with clearer field lines)")
        print("  2. Use interactive calibration instead:")
        print(f"     python -m projection.calibration_cli interactive {args.video}")


def verify_calibration(args: argparse.Namespace) -> None:
    """验证标定结果。"""
    from projection.interactive_calibrator import InteractiveCalibrator

    # 加载标定
    H, info = InteractiveCalibrator.load_calibration(args.calibration)
    print(f"Loaded calibration from {args.calibration}")
    print(f"Coordinate system: {info['coord_system']}")

    # 加载帧
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"Error: Cannot open video: {args.video}", file=sys.stderr)
        return

    cap.set(cv2.CAP_PROP_POS_FRAMES, args.frame)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        print(f"Error: Cannot read frame {args.frame}", file=sys.stderr)
        return

    print(f"Frame {args.frame}: {frame.shape[1]}x{frame.shape[0]}")

    # 投影球场角点并绘制
    H_inv = np.linalg.inv(H)
    display = frame.copy()

    for name, (mx, my) in PITCH_CORNERS_METER_CENTER.items():
        pt = np.array([mx, my, 1.0], dtype=np.float32)
        proj = H_inv @ pt
        proj = proj / proj[2]
        x, y = int(proj[0]), int(proj[1])
        if 0 <= x < frame.shape[1] and 0 <= y < frame.shape[0]:
            color = (0, 255, 0)
            cv2.circle(display, (x, y), 15, color, -1)
            cv2.putText(display, name, (x + 15, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            print(f"  {name}: ({mx:.1f}, {my:.1f}) -> ({x}, {y})")

    print("\nGreen circles show where pitch corners project to. Verify they match actual field corners!")
    print("Press any key to close.")
    cv2.namedWindow("Calibration Verify")
    cv2.imshow("Calibration Verify", display)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def main() -> None:
    parser = argparse.ArgumentParser(description="球场标定工具")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # interactive subcommand
    interact_parser = subparsers.add_parser(
        "interactive", help="交互式标定：点击 4 个角点"
    )
    interact_parser.add_argument("video", help="视频文件路径或摄像头索引")
    interact_parser.add_argument(
        "--frame", type=int, default=30, help="用于标定的帧序号 (default: 30)"
    )
    interact_parser.add_argument(
        "--output", default="calibration.json", help="标定输出文件路径"
    )
    interact_parser.set_defaults(handler=interactive_calibration)

    # auto subcommand
    auto_parser = subparsers.add_parser(
        "auto", help="自动线条标定：Hough 线条检测"
    )
    auto_parser.add_argument("video", help="视频文件路径")
    auto_parser.add_argument(
        "--frame", type=int, default=30, help="用于标定的帧序号 (default: 30)"
    )
    auto_parser.add_argument(
        "--output", default="calibration.json", help="标定输出文件路径"
    )
    auto_parser.add_argument(
        "--debug", action="store_true", help="显示调试可视化窗口"
    )
    auto_parser.set_defaults(handler=auto_calibration)

    # verify subcommand
    verify_parser = subparsers.add_parser(
        "verify", help="验证标定：投影角点并显示"
    )
    verify_parser.add_argument("video", help="视频文件路径")
    verify_parser.add_argument(
        "--calibration", required=True, help="标定 JSON 文件路径"
    )
    verify_parser.add_argument(
        "--frame", type=int, default=30, help="验证用的帧序号"
    )
    verify_parser.set_defaults(handler=verify_calibration)

    args = parser.parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()

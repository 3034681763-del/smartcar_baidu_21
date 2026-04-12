import argparse
import time

import cv2


def build_parser():
    parser = argparse.ArgumentParser(
        description="Realtime camera preview with FPS overlay for Jetson/Linux."
    )
    parser.add_argument(
        "--device",
        default="0",
        help="Camera device index or path, e.g. 0 or /dev/video0",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=640,
        help="Requested capture width",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=480,
        help="Requested capture height",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=60,
        help="Requested camera FPS",
    )
    parser.add_argument(
        "--codec",
        default="MJPG",
        help="Requested FOURCC codec, e.g. MJPG or YUYV",
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=0,
        help="How long to run in seconds. Use 0 for no limit",
    )
    parser.add_argument(
        "--warmup",
        type=float,
        default=2.0,
        help="Warmup time before measuring in seconds",
    )
    parser.add_argument(
        "--window-name",
        default="camera_fps_test",
        help="Preview window name",
    )
    return parser


def parse_device(device_arg):
    if device_arg.isdigit():
        return int(device_arg)
    return device_arg


def draw_overlay(frame, instant_fps, average_fps):
    lines = [
        f"Instant FPS: {instant_fps:5.2f}",
        f"Average FPS: {average_fps:5.2f}",
    ]
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.7
    thickness = 2
    margin = 12
    line_gap = 10

    sizes = [cv2.getTextSize(text, font, scale, thickness) for text in lines]
    max_width = max(size[0][0] for size in sizes)
    total_height = sum(size[0][1] + size[1] for size in sizes) + line_gap * (len(lines) - 1)

    x2 = frame.shape[1] - margin
    x1 = max(x2 - max_width - 24, 0)
    y1 = margin
    y2 = min(y1 + total_height + 24, frame.shape[0])

    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), -1)

    cursor_y = y1 + 22
    for index, text in enumerate(lines):
        size, baseline = sizes[index]
        text_x = x2 - size[0] - 12
        text_y = cursor_y + size[1]
        cv2.putText(
            frame,
            text,
            (text_x, text_y),
            font,
            scale,
            (0, 255, 0),
            thickness,
            cv2.LINE_AA,
        )
        cursor_y = text_y + baseline + line_gap


def main():
    args = build_parser().parse_args()
    device = parse_device(args.device)

    cap = cv2.VideoCapture(device, cv2.CAP_V4L2)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open camera: {args.device}")

    if len(args.codec) != 4:
        raise ValueError("FOURCC codec must be exactly 4 characters, e.g. MJPG")

    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*args.codec))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv2.CAP_PROP_FPS, args.fps)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    actual_fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
    actual_codec = "".join(chr((actual_fourcc >> 8 * i) & 0xFF) for i in range(4))

    print("=" * 60)
    print("Camera FPS Test")
    print(f"Device:    {args.device}")
    print(f"Requested: {args.width}x{args.height} @ {args.fps} FPS, codec={args.codec}")
    print(f"Actual:    {actual_width}x{actual_height} @ {actual_fps:.2f} FPS, codec={actual_codec}")
    print(f"Warmup:    {args.warmup:.1f}s")
    print(f"Duration:  {'unlimited' if args.duration == 0 else str(args.duration) + 's'}")
    print("Preview:   on")
    print("Quit:      press q in the preview window")
    print("=" * 60)

    cv2.namedWindow(args.window_name, cv2.WINDOW_NORMAL)

    warmup_end = time.time() + args.warmup
    while time.time() < warmup_end:
        ok, _ = cap.read()
        if not ok:
            raise RuntimeError("Camera read failed during warmup")

    frame_count = 0
    start_time = time.time()
    last_fps_update_time = start_time
    last_fps_update_count = 0
    instant_fps = 0.0
    average_fps = 0.0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Camera read failed during measurement")
                break

            frame_count += 1
            now = time.time()
            elapsed = now - start_time
            window_elapsed = now - last_fps_update_time

            if window_elapsed >= 0.25:
                instant_fps = (frame_count - last_fps_update_count) / window_elapsed
                average_fps = frame_count / elapsed if elapsed > 0 else 0.0
                last_fps_update_time = now
                last_fps_update_count = frame_count

            if int(elapsed) != int(max(elapsed - 0.01, 0)) and elapsed >= 1.0:
                print(
                    f"[{elapsed:5.1f}s] instant_fps={instant_fps:6.2f} avg_fps={average_fps:6.2f}"
                )

            draw_overlay(frame, instant_fps, average_fps)
            cv2.imshow(args.window_name, frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

            if args.duration > 0 and elapsed >= args.duration:
                break
    finally:
        total_elapsed = max(time.time() - start_time, 1e-6)
        final_avg_fps = frame_count / total_elapsed
        print("=" * 60)
        print(f"Captured frames: {frame_count}")
        print(f"Measured time:   {total_elapsed:.2f}s")
        print(f"Average FPS:     {final_avg_fps:.2f}")
        print("=" * 60)
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

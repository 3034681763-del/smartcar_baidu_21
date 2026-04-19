import time

import cv2
import pyudev


def _sort_video_nodes(nodes):
    def sort_key(node):
        try:
            return int(str(node).rsplit("video", 1)[1])
        except (IndexError, ValueError):
            return 10**9

    return sorted(set(nodes), key=sort_key)


def list_camera_nodes_by_path(device_arg):
    if isinstance(device_arg, str):
        if device_arg.startswith("/dev/video"):
            return [device_arg]
        if device_arg.isdigit():
            return [int(device_arg)]
    elif isinstance(device_arg, int):
        return [device_arg]

    context = pyudev.Context()
    candidates = []
    for device in context.list_devices(subsystem="video4linux"):
        device_node = getattr(device, "device_node", None)
        if not device_node:
            continue

        end_index = device.device_path.rfind("/video4linux")
        cleaned_string = device.device_path[:end_index] if end_index != -1 else device.device_path
        if cleaned_string.endswith(str(device_arg)):
            candidates.append(device_node)

    return _sort_video_nodes(candidates)


def open_camera_from_device_arg(
    device_arg,
    width=640,
    height=480,
    fps=60,
    codec="MJPG",
    role_name="Camera",
):
    candidates = list_camera_nodes_by_path(device_arg)
    if not candidates:
        raise RuntimeError(f"{role_name} physical path not found: {device_arg}")

    last_error = None
    for candidate in candidates:
        cap = cv2.VideoCapture(candidate, cv2.CAP_V4L2)
        if not cap.isOpened():
            cap = cv2.VideoCapture(candidate)
        if not cap.isOpened():
            last_error = f"{role_name} failed to open: {candidate}"
            continue

        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*codec))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        cap.set(cv2.CAP_PROP_FPS, fps)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        ok = False
        for _ in range(5):
            ok, _ = cap.read()
            if ok:
                break
            time.sleep(0.02)

        if not ok:
            last_error = f"{role_name} camera node is not streaming frames: {candidate}"
            cap.release()
            continue

        return cap, candidate, candidates

    if last_error:
        raise RuntimeError(last_error)
    raise RuntimeError(f"{role_name} failed to open: {device_arg}")

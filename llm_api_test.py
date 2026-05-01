#!/usr/bin/env python3
import argparse
import json
import multiprocessing
import os
import time

import cv2
import numpy as np

from camera_device_utils import open_camera_from_device_arg
from env_loader import load_local_env
from get_json import load_params
from infer_server_client import InferClient1, OCRPipeline, model_configs, serve_model_process
from order_ai_client import OrderAIClient
from pest_vlm_client import PestVlmClient
from Process_manage import ProcessManager
from shared_memory_manager import SharedMemoryManager
from tool_func import (
    DEFAULT_ANIMAL_LABEL_ALIASES,
    crop_detection_with_padding,
    get_animal_box_in_roi,
)


TASK_SHM_KEY = "shm_task"
CROP_SHM_KEY = "shm_crop"
FRAME_SHAPE = (640, 640, 3)


def build_parser():
    parser = argparse.ArgumentParser(
        description="Full-chain test for all large-model API call scenes in this project."
    )
    parser.add_argument(
        "--task-device",
        default="1-2.1:1.0",
        help="Task camera device path, /dev/videoX, or physical path like 1-2.1:1.0.",
    )
    parser.add_argument("--width", type=int, default=640, help="Task camera capture width.")
    parser.add_argument("--height", type=int, default=480, help="Task camera capture height.")
    parser.add_argument("--fps", type=int, default=60, help="Task camera capture FPS.")
    parser.add_argument("--codec", default="MJPG", help="Task camera FOURCC codec.")
    parser.add_argument("--skip-order", action="store_true", help="Skip order OCR + LLM test.")
    parser.add_argument("--skip-pest", action="store_true", help="Skip pest detection + VLM test.")
    parser.add_argument(
        "--use-pest-detection-crop",
        action="store_true",
        help="Enable animal detection + crop before VLM. Default is to send the full captured frame.",
    )
    parser.add_argument(
        "--use-existing-services",
        action="store_true",
        help="Do not start model processes; connect to already running task/word services.",
    )
    parser.add_argument("--timeout", type=float, default=12.0, help="Large-model API timeout.")
    parser.add_argument("--retries", type=int, default=2, help="Large-model API retry count.")
    parser.add_argument("--ocr-retries", type=int, default=8, help="OCR pipeline retry count.")
    parser.add_argument(
        "--startup-wait",
        type=float,
        default=0.5,
        help="Seconds to wait after starting model processes before creating clients.",
    )
    return parser


def capture_task_frame(args, phase_name):
    if len(args.codec) != 4:
        raise ValueError("FOURCC codec must be exactly 4 characters")

    cap, actual_device, candidates = open_camera_from_device_arg(
        args.task_device,
        width=args.width,
        height=args.height,
        fps=args.fps,
        codec=args.codec,
        role_name="Task",
    )

    window_name = f"LLM API Test - {phase_name}"
    print("=" * 60)
    print(f"[Full LLM Test] Capture task camera for {phase_name}")
    print(f"Task path:  {args.task_device}")
    print(f"Task nodes: {candidates}")
    print(f"Using:      {actual_device}")
    print("Press q to capture current frame.")
    print("=" * 60)

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    captured = None
    frame_count = 0
    start_time = time.time()

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError("task camera frame read failed")

            frame_count += 1
            elapsed = time.time() - start_time
            fps_text = frame_count / elapsed if elapsed > 0 else 0.0
            preview = frame.copy()
            cv2.putText(
                preview,
                f"{phase_name} | press q to capture | FPS {fps_text:.1f}",
                (12, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.imshow(window_name, preview)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                captured = frame.copy()
                print(f"[Full LLM Test] Captured frame for {phase_name}.")
                break
    finally:
        cap.release()
        cv2.destroyWindow(window_name)

    if captured is None:
        raise RuntimeError("no task camera frame captured")
    if captured.shape[:2] != FRAME_SHAPE[:2]:
        captured = cv2.resize(captured, (FRAME_SHAPE[1], FRAME_SHAPE[0]))
    return captured


def setup_runtime(use_existing_services):
    shm_manager = SharedMemoryManager()
    shm_manager.create_block(TASK_SHM_KEY, size=np.prod(FRAME_SHAPE))
    shm_manager.create_block(CROP_SHM_KEY, size=np.prod(FRAME_SHAPE))

    manager = None
    if not use_existing_services:
        manager = ProcessManager()
        for cfg in model_configs:
            manager.add_process(target=serve_model_process, args=(cfg, shm_manager))
        manager.start_all()

    return shm_manager, manager


def build_clients(shm_manager):
    clients = {}
    for cfg in model_configs:
        clients[cfg["name"]] = InferClient1(cfg["name"], shm_manager, cfg["port"])
    ocr_reader = OCRPipeline(
        shm_manager,
        task_client=clients["task"],
        word_client=clients["word"],
        crop_key=CROP_SHM_KEY,
    )
    return clients, ocr_reader


def load_order_cfg():
    task_cfg = load_params("motion.json").get("TASK_CONFIG", {})
    return task_cfg.get("ORDER_DELIVERY", {})


def load_pest_cfg():
    task_cfg = load_params("motion.json").get("TASK_CONFIG", {})
    return task_cfg.get("PEST_CONFIRM", {})


def test_order_chain(frame, args, shm_manager, ocr_reader):
    cfg = load_order_cfg()
    shm_manager.write_image(TASK_SHM_KEY, frame)

    result_amount = int(cfg.get("ocr_result_amount", cfg.get("item_count", 2)))
    target_label = cfg.get("text_label_aliases", ["text"])[0]
    sort_by = cfg.get("ocr_sort_by", "y")

    print("=" * 60)
    print("[Full LLM Test] Order chain")
    print(f"OCR target label: {target_label}, result_amount={result_amount}, sort_by={sort_by}")

    texts = ocr_reader.read_texts(
        shm_key=TASK_SHM_KEY,
        shape=FRAME_SHAPE,
        dtype=np.uint8,
        target_label=target_label,
        result_amount=result_amount,
        sort_by=sort_by,
        retries=args.ocr_retries,
    )
    print(f"[Full LLM Test] OCR texts: {json.dumps(texts, ensure_ascii=False)}")
    if not texts:
        raise RuntimeError("OCR returned no text; check text detection label, image, and OCR model")

    client = OrderAIClient(timeout_s=args.timeout, max_retries=args.retries)
    result = client.parse_order(
        texts,
        valid_goods=cfg.get("valid_goods", []),
        valid_buildings=cfg.get("valid_buildings", []),
        item_count=int(cfg.get("item_count", 2)),
    )
    print("[Full LLM Test] Order AI result:")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def test_pest_chain(frame, args, shm_manager, task_client):
    cfg = load_pest_cfg()
    shm_manager.write_image(TASK_SHM_KEY, frame)

    print("=" * 60)
    print("[Full LLM Test] Pest chain")

    if args.use_pest_detection_crop:
        print("[Full LLM Test] Pest mode: detection crop enabled")
        detections = task_client(TASK_SHM_KEY, FRAME_SHAPE, np.uint8)
        box = get_animal_box_in_roi(
            detections,
            roi=cfg.get("confirm_roi"),
            target_pose=tuple(cfg.get("align_target_pose", [320, 260])),
            label_aliases=cfg.get("animal_label_aliases", DEFAULT_ANIMAL_LABEL_ALIASES),
            min_score=float(cfg.get("min_score", 0.5)),
        )
        if box is None:
            labels = [getattr(det, "label", None) for det in (detections or [])]
            raise RuntimeError(f"no animal box detected; labels={labels}")

        print(
            "[Full LLM Test] Animal detection: "
            f"label={getattr(box, 'label', None)} score={float(getattr(box, 'score', 0.0)):.3f} "
            f"bbox={tuple(getattr(box, 'bbox', ())) }"
        )
        vlm_input = crop_detection_with_padding(
            frame,
            box,
            pad=int(cfg.get("crop_pad", 32)),
        )
        if vlm_input is None:
            raise RuntimeError("failed to crop animal detection")
    else:
        print("[Full LLM Test] Pest mode: full-frame VLM input (default)")
        # Keep the original detection-and-crop path available behind
        # --use-pest-detection-crop, but default to the full captured frame
        # so animal VLM testing does not depend on detector labels.
        vlm_input = frame

    client = PestVlmClient(timeout_s=args.timeout, max_retries=args.retries)
    result = client.classify(vlm_input)
    print("[Full LLM Test] Pest VLM result:")
    print(json.dumps({"result": result}, ensure_ascii=False, indent=2))
    return result


def validate_args(args):
    load_local_env()
    provider = os.environ.get("LLM_PROVIDER", "aistudio").strip().lower().replace("-", "_")
    if provider in ("ai_studio", "aistudio"):
        if not os.environ.get("AI_STUDIO_API_KEY") and not os.environ.get("AISTUDIO_API_KEY"):
            raise RuntimeError('AI_STUDIO_API_KEY is not set. Example: export AI_STUDIO_API_KEY="your_access_token"')
    elif not os.environ.get("QIANFAN_API_KEY"):
        raise RuntimeError('QIANFAN_API_KEY is not set. Example: export QIANFAN_API_KEY="your_api_key"')
    if args.skip_order and args.skip_pest:
        raise RuntimeError("both tests are skipped")


def main():
    args = build_parser().parse_args()
    try:
        validate_args(args)
    except Exception as exc:
        print(f"[Full LLM Test] Argument error: {exc}")
        return 2

    multiprocessing.set_start_method("spawn", force=True)
    shm_manager = None
    manager = None
    failures = []

    try:
        shm_manager, manager = setup_runtime(args.use_existing_services)
        if manager is not None:
            time.sleep(max(0.0, args.startup_wait))
        clients, ocr_reader = build_clients(shm_manager)

        if not args.skip_order:
            try:
                order_frame = capture_task_frame(args, "order OCR + LLM")
                test_order_chain(order_frame, args, shm_manager, ocr_reader)
            except Exception as exc:
                failures.append(("order", exc))
                print(f"[Full LLM Test] Order chain failed: {exc}")

        if not args.skip_pest:
            try:
                pest_frame = capture_task_frame(args, "pest detection + VLM")
                test_pest_chain(pest_frame, args, shm_manager, clients["task"])
            except Exception as exc:
                failures.append(("pest", exc))
                print(f"[Full LLM Test] Pest chain failed: {exc}")
    finally:
        if manager is not None:
            manager.terminate_all()
        if shm_manager is not None:
            shm_manager.release_all()

    print("=" * 60)
    if failures:
        print("[Full LLM Test] Failed items:")
        for name, exc in failures:
            print(f"  - {name}: {exc}")
        return 1

    print("[Full LLM Test] All enabled full-chain large-model tests passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

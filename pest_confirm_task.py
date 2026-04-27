import time

import numpy as np

from get_json import load_params
from pest_vlm_client import PestVlmClient
from tool_func import (
    DEFAULT_ANIMAL_LABEL_ALIASES,
    crop_detection_with_padding,
    get_animal_box_in_roi,
)


class PestConfirmTaskExecutor:
    """Confirm four animal cards after seeding and before irrigation."""

    def __init__(
        self,
        base,
        task_client=None,
        task_shm=None,
        task_shm_key="shm_task",
        tracking_callback=None,
        motion_file="motion.json",
        vlm_client=None,
    ):
        self.base = base
        self.task_client = task_client
        self.task_shm = task_shm
        self.task_shm_key = task_shm_key
        self.tracking_callback = tracking_callback
        self.motion_file = motion_file
        self.vlm_client = vlm_client or PestVlmClient()

    def run(self):
        cfg = self._load_cfg()
        if self.task_client is None:
            print("[PestConfirm] Task model client is not configured.")
            return None
        if self.task_shm is None:
            print("[PestConfirm] Task shared memory is not configured.")
            return None

        confirm_count = int(cfg.get("confirm_count", 4))
        results = []
        print(f"[PestConfirm] Start animal confirmation, count={confirm_count}.")

        for index in range(confirm_count):
            print(f"[PestConfirm] Confirm card {index + 1}/{confirm_count}.")
            if not self._align_leftmost_animal(cfg):
                return self._fail(f"Failed to align animal card {index + 1}.")

            crop = self._crop_leftmost_animal(cfg)
            if crop is None:
                return self._fail(f"Failed to crop animal card {index + 1}.")

            try:
                result = self.vlm_client.classify(crop)
            except Exception as exc:
                return self._fail(f"VLM classify failed for card {index + 1}: {exc}")

            results.append(int(result))
            print(f"[PestConfirm] Card {index + 1} result={result}, results={results}")

            if index < confirm_count - 1:
                action = cfg.get("advance_action", "Front")
                countdown = float(cfg.get("advance_countdown", 0.2))
                if not self.base.execute_base_motion(action, countdown=countdown):
                    return self._fail(f"Advance action failed: {action}")
                time.sleep(float(cfg.get("settle_after_advance_s", 0.15)))

        self.base.MOD_STOP()
        print(f"[PestConfirm] Finished: {results}")
        return results

    def has_animal(self, cfg=None):
        cfg = cfg or self._load_cfg()
        detections = self.task_client(self.task_shm_key, (640, 640, 3), np.uint8)
        box = get_animal_box_in_roi(
            detections,
            roi=cfg.get("confirm_roi"),
            target_pose=tuple(cfg.get("align_target_pose", [320, 260])),
            label_aliases=cfg.get("animal_label_aliases", DEFAULT_ANIMAL_LABEL_ALIASES),
            min_score=float(cfg.get("min_score", 0.5)),
        )
        return box is not None

    def _align_leftmost_animal(self, cfg):
        if self.tracking_callback is None:
            print("[PestConfirm] Tracking callback is unavailable, skip visual alignment.")
            return True
        return self.tracking_callback(
            target_pose=tuple(cfg.get("align_target_pose", [320, 260])),
            cam_pose=cfg.get("cam_pose", "L"),
            timeout_s=float(cfg.get("track_timeout_s", 4.0)),
            max_missed_frames=int(cfg.get("max_missed_frames", 6)),
            recover_pause_s=float(cfg.get("recover_pause_s", 0.06)),
            recover_timeout_s=float(cfg.get("recover_timeout_s", 0.5)),
            selector=get_animal_box_in_roi,
            selector_kwargs={
                "roi": cfg.get("confirm_roi"),
                "target_pose": tuple(cfg.get("align_target_pose", [320, 260])),
                "label_aliases": cfg.get("animal_label_aliases", DEFAULT_ANIMAL_LABEL_ALIASES),
                "min_score": float(cfg.get("min_score", 0.5)),
            },
        )

    def _crop_leftmost_animal(self, cfg):
        detections = self.task_client(self.task_shm_key, (640, 640, 3), np.uint8)
        box = get_animal_box_in_roi(
            detections,
            roi=cfg.get("confirm_roi"),
            target_pose=tuple(cfg.get("align_target_pose", [320, 260])),
            label_aliases=cfg.get("animal_label_aliases", DEFAULT_ANIMAL_LABEL_ALIASES),
            min_score=float(cfg.get("min_score", 0.5)),
        )
        if box is None:
            return None

        frame = self.task_shm.read_image(self.task_shm_key, (640, 640, 3), np.uint8)
        return crop_detection_with_padding(frame, box, pad=int(cfg.get("crop_pad", 32)))

    def _load_cfg(self):
        task_cfg = load_params(self.motion_file).get("TASK_CONFIG", {})
        cfg = task_cfg.get("PEST_CONFIRM", {})
        cfg.setdefault("confirm_count", 4)
        cfg.setdefault("animal_label_aliases", DEFAULT_ANIMAL_LABEL_ALIASES)
        cfg.setdefault("advance_action", "Front")
        cfg.setdefault("align_target_pose", [320, 260])
        cfg.setdefault("confirm_roi", [80, 120, 420, 560])
        return cfg

    def _fail(self, reason):
        self.base.MOD_STOP()
        print(f"[PestConfirm] Task failed: {reason}")
        return None

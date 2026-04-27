import time

import numpy as np

from get_json import load_params
from tool_func import (
    DEFAULT_ANIMAL_LABEL_ALIASES,
    get_animal_box_by_rank,
    same_detection_area_visible,
)


class ShootingTaskExecutor:
    """Shoot harmful animal cards according to the previous pest confirmation result."""

    def __init__(
        self,
        base,
        task_client=None,
        task_shm_key="shm_task",
        tracking_callback=None,
        motion_file="motion.json",
    ):
        self.base = base
        self.task_client = task_client
        self.task_shm_key = task_shm_key
        self.tracking_callback = tracking_callback
        self.motion_file = motion_file

    def run(self, pest_results):
        cfg = self._load_cfg()
        if self.task_client is None:
            print("[Shooting] Task model client is not configured.")
            return False
        if not pest_results:
            print("[Shooting] No pest confirmation result, skip shooting.")
            return False

        harmful_indexes = [index for index, result in enumerate(pest_results) if int(result) == 1]
        if not harmful_indexes:
            print("[Shooting] No harmful animal target.")
            return True

        print(f"[Shooting] Start shooting harmful targets: {harmful_indexes}")
        cleared_indexes = []
        max_retry = int(cfg.get("max_retry_per_target", 5))

        for target_index in harmful_indexes:
            visible_rank = int(target_index) - len([idx for idx in cleared_indexes if idx < target_index])
            if visible_rank < 0:
                visible_rank = 0

            target_cleared = False
            for attempt in range(1, max_retry + 1):
                print(
                    f"[Shooting] Target index={target_index}, rank={visible_rank}, "
                    f"attempt {attempt}/{max_retry}."
                )

                target_box = self._get_target_box(visible_rank, cfg)
                if target_box is None:
                    print(f"[Shooting] Target index={target_index} is already invisible.")
                    target_cleared = True
                    break

                if not self._align_target(visible_rank, cfg):
                    print(f"[Shooting] Align failed for target index={target_index}.")
                    continue

                target_box = self._get_target_box(visible_rank, cfg) or target_box
                if not self._shoot_once(cfg):
                    print(f"[Shooting] Shoot command failed for target index={target_index}.")
                    continue

                time.sleep(float(cfg.get("settle_after_shoot_s", 0.25)))
                if self._confirm_target_disappeared(target_box, cfg):
                    print(f"[Shooting] Target index={target_index} cleared.")
                    target_cleared = True
                    break

                print(f"[Shooting] Target index={target_index} still visible, retry.")

            if target_cleared:
                cleared_indexes.append(target_index)
            else:
                print(f"[Shooting] Give up target index={target_index} after {max_retry} attempts.")

        self.base.MOD_STOP()
        print("[Shooting] Shooting task finished.")
        return True

    def _get_target_box(self, rank, cfg):
        detections = self.task_client(self.task_shm_key, (640, 640, 3), np.uint8)
        return get_animal_box_by_rank(
            detections,
            rank=rank,
            roi=cfg.get("shoot_roi"),
            label_aliases=cfg.get("animal_label_aliases", DEFAULT_ANIMAL_LABEL_ALIASES),
            min_score=float(cfg.get("min_score", 0.5)),
        )

    def _align_target(self, rank, cfg):
        if self.tracking_callback is None:
            print("[Shooting] Tracking callback is unavailable, skip visual alignment.")
            return True
        return self.tracking_callback(
            target_pose=tuple(cfg.get("shoot_target_pose", [320, 260])),
            cam_pose=cfg.get("cam_pose", "shoot"),
            timeout_s=float(cfg.get("track_timeout_s", 4.0)),
            max_missed_frames=int(cfg.get("max_missed_frames", 6)),
            recover_pause_s=float(cfg.get("recover_pause_s", 0.06)),
            recover_timeout_s=float(cfg.get("recover_timeout_s", 0.5)),
            selector=get_animal_box_by_rank,
            selector_kwargs={
                "rank": rank,
                "roi": cfg.get("shoot_roi"),
                "label_aliases": cfg.get("animal_label_aliases", DEFAULT_ANIMAL_LABEL_ALIASES),
                "min_score": float(cfg.get("min_score", 0.5)),
            },
        )

    def _shoot_once(self, cfg):
        return self.base.execute_shoot_instruction(
            cfg.get("shoot_command", {"cmd": "SysMode", "flag": 9}),
            wait_timeout_s=float(cfg.get("shoot_ack_timeout_s", 3.0)),
        )

    def _confirm_target_disappeared(self, target_box, cfg):
        confirm_frames = int(cfg.get("disappear_confirm_frames", 3))
        interval_s = float(cfg.get("disappear_check_interval_s", 0.08))
        invisible_count = 0

        for _ in range(max(confirm_frames * 3, confirm_frames)):
            detections = self.task_client(self.task_shm_key, (640, 640, 3), np.uint8)
            visible = same_detection_area_visible(
                detections,
                target_box,
                label_aliases=cfg.get("animal_label_aliases", DEFAULT_ANIMAL_LABEL_ALIASES),
                min_score=float(cfg.get("min_score", 0.5)),
                iou_threshold=float(cfg.get("same_target_iou", 0.25)),
            )
            if visible:
                invisible_count = 0
            else:
                invisible_count += 1
                if invisible_count >= confirm_frames:
                    return True
            time.sleep(interval_s)
        return False

    def _load_cfg(self):
        task_cfg = load_params(self.motion_file).get("TASK_CONFIG", {})
        cfg = task_cfg.get("PEST_SHOOT", {})
        cfg.setdefault("animal_label_aliases", DEFAULT_ANIMAL_LABEL_ALIASES)
        cfg.setdefault("shoot_target_pose", [320, 260])
        cfg.setdefault("shoot_roi", [40, 80, 600, 600])
        cfg.setdefault("shoot_command", {"cmd": "SysMode", "flag": 9})
        return cfg

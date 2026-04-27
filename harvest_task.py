import time

import numpy as np

from get_json import load_params
from tool_func import get_leftmost_box


class HarvestTaskExecutor:
    """Task 4: harvest blue/yellow fruits from left to right."""

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
        self.results = []

    def run(self):
        cfg = self._load_cfg()
        target_count = int(cfg.get("target_count", 8))
        self.results = []
        print("[Harvest] Start harvest task.")

        for index in range(1, target_count + 1):
            fruit_box = self._get_leftmost_fruit(cfg)
            if fruit_box is None:
                return self._fail(f"No fruit detected for index {index}.")

            fruit_type = self._fruit_type_from_label(getattr(fruit_box, "label", ""), cfg)
            if not fruit_type:
                return self._fail(f"Unknown fruit label: {getattr(fruit_box, 'label', None)}")

            if not self._align_fruit(cfg):
                return self._fail(f"Failed to align fruit {index}.")

            action = cfg.get("arm_actions", {}).get(fruit_type)
            if not action:
                return self._fail(f"Missing harvest arm action for fruit_type={fruit_type}.")

            print(
                "[Harvest] Pick fruit "
                f"{index}/{target_count}: type={fruit_type}, label={getattr(fruit_box, 'label', '')}, action={action}"
            )
            if not self.base.execute_arm_instruction(
                action,
                countdown=float(cfg.get("arm_action_countdown", 0.2)),
            ):
                return self._fail(f"Arm harvest action failed: {action}.")

            self.results.append(
                {
                    "index": index,
                    "fruit_type": fruit_type,
                    "label": str(getattr(fruit_box, "label", "")),
                    "action": action,
                }
            )

            if index < target_count and not self._advance_to_next(cfg):
                return self._fail(f"Failed to advance after fruit {index}.")

        safe_action = cfg.get("arm_actions", {}).get("safe", "ArmHarvestSafe")
        if safe_action:
            self.base.execute_arm_instruction(
                safe_action,
                countdown=float(cfg.get("arm_action_countdown", 0.2)),
            )
        self.base.MOD_STOP()
        print(f"[Harvest] Task finished. results={self.results}")
        return list(self.results)

    def has_fruit(self):
        cfg = self._load_cfg()
        return self._get_leftmost_fruit(cfg) is not None

    def _get_leftmost_fruit(self, cfg):
        if self.task_client is None:
            return None
        detections = self.task_client(self.task_shm_key, (640, 640, 3), np.uint8)
        return get_leftmost_box(
            detections,
            label_aliases=cfg.get("fruit_label_aliases", ["blue", "yellow"]),
            min_score=float(cfg.get("min_score", 0.5)),
            roi=cfg.get("fruit_roi", None),
        )

    def _align_fruit(self, cfg):
        if self.tracking_callback is None or self.task_client is None:
            print("[Harvest] Tracking unavailable, skip fruit alignment.")
            return True
        print("[Harvest] Align leftmost fruit.")
        return self.tracking_callback(
            target_pose=tuple(cfg.get("fruit_target_pose", [320, 260])),
            cam_pose=cfg.get("cam_pose", "L"),
            timeout_s=float(cfg.get("track_timeout_s", 4.0)),
            max_missed_frames=int(cfg.get("max_missed_frames", 6)),
            recover_pause_s=float(cfg.get("recover_pause_s", 0.06)),
            recover_timeout_s=float(cfg.get("recover_timeout_s", 0.5)),
            selector=get_leftmost_box,
            selector_kwargs={
                "label_aliases": cfg.get("fruit_label_aliases", ["blue", "yellow"]),
                "min_score": float(cfg.get("min_score", 0.5)),
                "roi": cfg.get("fruit_roi", None),
            },
        )

    def _advance_to_next(self, cfg):
        action = cfg.get("advance_action", "HarvestAdvanceStep")
        print(f"[Harvest] Advance to next fruit -> {action}")
        return self.base.execute_chassis_instruction(
            action,
            countdown=float(cfg.get("advance_countdown", cfg.get("action_countdown", 0.2))),
        )

    @staticmethod
    def _fruit_type_from_label(label, cfg):
        label = str(label or "").lower()
        aliases = cfg.get("fruit_type_aliases", {})
        for fruit_type, labels in aliases.items():
            if isinstance(labels, str):
                labels = [labels]
            if label in {str(item).lower() for item in labels}:
                return str(fruit_type)
        if label in ("blue", "yellow"):
            return label
        return None

    def _fail(self, reason):
        self.base.MOD_STOP()
        print(f"[Harvest] Task failed: {reason}")
        return []

    def _load_cfg(self):
        task_cfg = load_params(self.motion_file).get("TASK_CONFIG", {})
        cfg = task_cfg.get("HARVEST", {})
        cfg.setdefault("fruit_label_aliases", ["blue", "yellow"])
        cfg.setdefault("fruit_type_aliases", {"blue": ["blue"], "yellow": ["yellow"]})
        cfg.setdefault(
            "arm_actions",
            {
                "blue": "ArmHarvestBlueToBin",
                "yellow": "ArmHarvestYellowToBin",
                "safe": "ArmHarvestSafe",
            },
        )
        return cfg

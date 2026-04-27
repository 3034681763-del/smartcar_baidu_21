import numpy as np

from get_json import load_params
from tool_func import get_biggest_box


class SortTaskExecutor:
    """Task 5: sort harvested fruits into color-specific and small warehouses."""

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

    def run(self, harvest_results=None):
        cfg = self._load_cfg()
        harvest_results = list(harvest_results or [])
        total_count = int(cfg.get("total_count", 8))
        first_count = int(cfg.get("first_stage_count", 4))
        second_count = total_count - first_count
        if len(harvest_results) < total_count:
            return self._fail(f"Need {total_count} harvest results, got {len(harvest_results)}.")

        print(f"[Sort] Start sort task with harvest_results={harvest_results}")

        warehouse_box = self._get_warehouse_box(cfg, stage="color")
        if warehouse_box is None:
            return self._fail("No color warehouse detected.")
        warehouse_type = self._warehouse_type_from_label(getattr(warehouse_box, "label", ""), cfg)
        if not warehouse_type:
            return self._fail(f"Unknown warehouse label: {getattr(warehouse_box, 'label', None)}")

        if not self._align_warehouse(cfg, stage="color"):
            return self._fail("Failed to align color warehouse.")

        for index in range(first_count):
            item = harvest_results[index]
            if not self._place_item(item, cfg, stage="color", warehouse_type=warehouse_type):
                return self._fail(f"Failed to place first-stage item {index + 1}.")

        if second_count > 0:
            if not self._move_to_small_warehouse(cfg):
                return self._fail("Failed to move to small warehouse.")
            if not self._align_warehouse(cfg, stage="small"):
                return self._fail("Failed to align small warehouse.")
            for index in range(first_count, total_count):
                item = harvest_results[index]
                if not self._place_item(item, cfg, stage="small", warehouse_type="small"):
                    return self._fail(f"Failed to place second-stage item {index + 1}.")

        safe_action = cfg.get("arm_actions", {}).get("safe", "ArmSortSafe")
        if safe_action:
            self.base.execute_arm_instruction(
                safe_action,
                countdown=float(cfg.get("arm_action_countdown", 0.2)),
            )
        self.base.MOD_STOP()
        print("[Sort] Task finished.")
        return True

    def has_warehouse(self):
        cfg = self._load_cfg()
        return self._get_warehouse_box(cfg, stage="color") is not None

    def _place_item(self, item, cfg, stage, warehouse_type):
        fruit_type = str(item.get("fruit_type") or "").strip()
        action = self._place_action_for(fruit_type, cfg, stage, warehouse_type)
        if not action:
            print(f"[Sort] Missing place action: stage={stage}, fruit_type={fruit_type}")
            return False
        print(
            "[Sort] Place item "
            f"index={item.get('index')} fruit_type={fruit_type} stage={stage} action={action}"
        )
        return self.base.execute_arm_instruction(
            action,
            countdown=float(cfg.get("arm_action_countdown", 0.2)),
        )

    def _place_action_for(self, fruit_type, cfg, stage, warehouse_type):
        arm_actions = cfg.get("arm_actions", {})
        if stage == "small":
            return arm_actions.get("small") or arm_actions.get(f"{fruit_type}_small")
        return (
            arm_actions.get(fruit_type)
            or arm_actions.get(warehouse_type)
            or arm_actions.get(f"{fruit_type}_to_{warehouse_type}")
        )

    def _align_warehouse(self, cfg, stage):
        if self.tracking_callback is None or self.task_client is None:
            print(f"[Sort] Tracking unavailable for {stage} warehouse, skip.")
            return True
        print(f"[Sort] Align {stage} warehouse.")
        target_pose_key = "small_warehouse_target_pose" if stage == "small" else "warehouse_target_pose"
        return self.tracking_callback(
            target_pose=tuple(cfg.get(target_pose_key, [320, 260])),
            cam_pose=cfg.get("cam_pose", "L"),
            timeout_s=float(cfg.get("track_timeout_s", 4.0)),
            max_missed_frames=int(cfg.get("max_missed_frames", 6)),
            recover_pause_s=float(cfg.get("recover_pause_s", 0.06)),
            recover_timeout_s=float(cfg.get("recover_timeout_s", 0.5)),
            selector=get_biggest_box,
            selector_kwargs={
                "target_labels": self._warehouse_labels(cfg, stage),
                "min_score": float(cfg.get("min_score", 0.5)),
            },
        )

    def _move_to_small_warehouse(self, cfg):
        action = cfg.get("move_to_small_warehouse_action", "SortMoveToSmallWarehouse")
        print(f"[Sort] Move to small warehouse -> {action}")
        return self.base.execute_chassis_instruction(
            action,
            countdown=float(cfg.get("action_countdown", 0.2)),
        )

    def _get_warehouse_box(self, cfg, stage):
        if self.task_client is None:
            return None
        detections = self.task_client(self.task_shm_key, (640, 640, 3), np.uint8)
        return get_biggest_box(
            detections,
            target_labels=self._warehouse_labels(cfg, stage),
            min_score=float(cfg.get("min_score", 0.5)),
        )

    def _warehouse_labels(self, cfg, stage):
        if stage == "small":
            return cfg.get("small_warehouse_label_aliases", ["small_warehouse"])
        return cfg.get("warehouse_label_aliases", ["blue", "yellow", "warehouse"])

    @staticmethod
    def _warehouse_type_from_label(label, cfg):
        label = str(label or "").lower()
        aliases = cfg.get("warehouse_type_aliases", {})
        for warehouse_type, labels in aliases.items():
            if isinstance(labels, str):
                labels = [labels]
            if label in {str(item).lower() for item in labels}:
                return str(warehouse_type)
        if label in ("blue", "yellow"):
            return label
        return None

    def _fail(self, reason):
        self.base.MOD_STOP()
        print(f"[Sort] Task failed: {reason}")
        return False

    def _load_cfg(self):
        task_cfg = load_params(self.motion_file).get("TASK_CONFIG", {})
        cfg = task_cfg.get("SORT", {})
        cfg.setdefault("warehouse_label_aliases", ["blue", "yellow", "warehouse"])
        cfg.setdefault("small_warehouse_label_aliases", ["small_warehouse"])
        cfg.setdefault("warehouse_type_aliases", {"blue": ["blue"], "yellow": ["yellow"]})
        cfg.setdefault(
            "arm_actions",
            {
                "blue": "ArmSortBluePlace",
                "yellow": "ArmSortYellowPlace",
                "small": "ArmSortSmallPlace",
                "safe": "ArmSortSafe",
            },
        )
        return cfg

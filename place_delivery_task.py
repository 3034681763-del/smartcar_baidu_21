import time

import numpy as np

from get_json import load_params
from tool_func import get_biggest_box


class PlaceDeliveryTaskExecutor:
    """Task 7: place picked goods by unit and receiver-name grid."""

    def __init__(
        self,
        base,
        ocr_reader=None,
        task_client=None,
        task_shm_key="shm_task",
        tracking_callback=None,
        motion_file="motion.json",
    ):
        self.base = base
        self.ocr_reader = ocr_reader
        self.task_client = task_client
        self.task_shm_key = task_shm_key
        self.tracking_callback = tracking_callback
        self.motion_file = motion_file

    def run(self, order):
        cfg = self._load_cfg()
        print(f"[PlaceDelivery] Start placing task with order: {order}")

        items_by_building = self._group_items_by_building(order, cfg)
        if not items_by_building:
            return self._fail("No valid delivery items.")

        if "1" in items_by_building:
            if not self._process_unit("1", items_by_building["1"], cfg):
                return self._fail("Failed to place goods for unit 1.")

        if "2" in items_by_building:
            if not self._move_unit1_to_unit2(cfg):
                return self._fail("Failed to move from unit 1 to unit 2.")
            if not self._process_unit("2", items_by_building["2"], cfg):
                return self._fail("Failed to place goods for unit 2.")

        return_action = cfg.get("return_lane_action", "")
        if return_action:
            print(f"[PlaceDelivery] Return lane -> {return_action}")
            if not self.base.execute_chassis_instruction(
                return_action,
                countdown=float(cfg.get("action_countdown", 0.2)),
            ):
                return self._fail("Failed to return lane.")

        safe_action = cfg.get("safe_arm_action", "ArmSafe")
        if safe_action:
            self.base.execute_arm_instruction(
                safe_action,
                countdown=float(cfg.get("arm_action_countdown", 0.2)),
            )
        self.base.MOD_STOP()
        print("[PlaceDelivery] Task finished.")
        return True

    def has_unit(self, building="1"):
        cfg = self._load_cfg()
        if self.task_client is None:
            return False
        detections = self.task_client(self.task_shm_key, (640, 640, 3), np.uint8)
        labels = self._unit_labels(building, cfg)
        box = get_biggest_box(
            detections,
            target_labels=labels,
            min_score=float(cfg.get("min_score", 0.5)),
        )
        return box is not None

    def _process_unit(self, building, items, cfg):
        print(f"[PlaceDelivery] Process unit {building}, items={items}")
        if not self._align_unit(building, cfg):
            return False
        if not self._back_for_name_view(cfg):
            return False

        name_grid = self._read_name_grid(cfg)
        if not name_grid:
            return False

        for item in items:
            name = str(item.get("name") or "").strip()
            goods = str(item.get("goods") or "").strip()
            pos = self._find_name_position(name, name_grid)
            if pos is None:
                print(f"[PlaceDelivery] Name not found: {name}, grid={name_grid}")
                return False
            action = self._place_action_for_position(pos, cfg)
            if not action:
                print(f"[PlaceDelivery] Missing place action for pos={pos}")
                return False
            print(
                "[PlaceDelivery] Place item "
                f"goods={goods}, building={building}, name={name}, pos={pos}, action={action}"
            )
            if not self.base.execute_arm_instruction(
                action,
                countdown=float(cfg.get("arm_action_countdown", 0.2)),
            ):
                return False
        return True

    def _align_unit(self, building, cfg):
        if self.tracking_callback is None or self.task_client is None:
            print(f"[PlaceDelivery] Tracking unavailable for unit {building}, skip.")
            return True

        labels = self._unit_labels(building, cfg)
        target_pose = tuple(cfg.get("unit_target_pose", [320, 260]))
        print(f"[PlaceDelivery] Align unit {building}, labels={labels}")
        return self.tracking_callback(
            target_pose=target_pose,
            cam_pose=cfg.get("cam_pose", "L"),
            timeout_s=float(cfg.get("track_timeout_s", 4.0)),
            max_missed_frames=int(cfg.get("max_missed_frames", 6)),
            recover_pause_s=float(cfg.get("recover_pause_s", 0.06)),
            recover_timeout_s=float(cfg.get("recover_timeout_s", 0.5)),
            selector=get_biggest_box,
            selector_kwargs={
                "target_labels": labels,
                "min_score": float(cfg.get("min_score", 0.5)),
            },
        )

    def _back_for_name_view(self, cfg):
        action = cfg.get("back_for_name_view_action", "DeliveryBack30")
        print(f"[PlaceDelivery] Back for name view -> {action}")
        return self.base.execute_chassis_instruction(
            action,
            countdown=float(cfg.get("action_countdown", 0.2)),
        )

    def _move_unit1_to_unit2(self, cfg):
        action = cfg.get("unit1_to_unit2_action", "DeliveryUnit1To2")
        print(f"[PlaceDelivery] Move unit 1 to unit 2 -> {action}")
        return self.base.execute_chassis_instruction(
            action,
            countdown=float(cfg.get("action_countdown", 0.2)),
        )

    def _read_name_grid(self, cfg):
        if self.ocr_reader is None:
            print("[PlaceDelivery] OCR reader is not configured.")
            return {}

        expected_count = int(cfg.get("text_expected_count", 6))
        stable_times = int(cfg.get("ocr_stable_times", 2))
        retry_count = int(cfg.get("ocr_retry_count", 8))
        last_signature = None
        last_grid = {}
        stable_hits = 0

        for attempt in range(1, retry_count + 1):
            items = self.ocr_reader.read_text_items(
                shm_key=self.task_shm_key,
                shape=(640, 640, 3),
                dtype=np.uint8,
                target_label=cfg.get("text_label_aliases", ["text"])[0],
                result_amount=expected_count,
                sort_by="y",
                pad=int(cfg.get("text_pad", 12)),
                retries=int(cfg.get("ocr_inner_retries", 2)),
                require_count=expected_count,
            )
            grid = self._sort_text_items_to_grid(items, cfg)
            signature = self._grid_signature(grid)
            print(f"[PlaceDelivery] Name grid {attempt}/{retry_count}: {signature}")
            if not grid:
                last_signature = None
                stable_hits = 0
                time.sleep(float(cfg.get("ocr_retry_interval_s", 0.1)))
                continue
            if signature == last_signature:
                stable_hits += 1
            else:
                last_signature = signature
                last_grid = grid
                stable_hits = 1
            if stable_hits >= stable_times:
                return last_grid
            time.sleep(float(cfg.get("ocr_retry_interval_s", 0.1)))

        return last_grid

    def _sort_text_items_to_grid(self, items, cfg):
        rows = int(cfg.get("text_grid_rows", 2))
        cols = int(cfg.get("text_grid_cols", 3))
        expected_count = rows * cols
        if not items or len(items) != expected_count:
            detected_count = 0 if not items else len(items)
            print(f"[PlaceDelivery] Need {expected_count} text boxes, detected {detected_count}.")
            return {}

        sorted_by_y = sorted(items, key=lambda item: float(item["center"][1]))
        grid = {}
        for row_index in range(rows):
            row_items = sorted_by_y[row_index * cols:(row_index + 1) * cols]
            row_items = sorted(row_items, key=lambda item: float(item["center"][0]))
            for col_index, item in enumerate(row_items):
                text = str(item.get("text", "")).strip()
                if not text:
                    return {}
                grid[(row_index + 1, col_index + 1)] = text
        return grid

    @staticmethod
    def _grid_signature(grid):
        if not grid:
            return ""
        return "|".join(f"{row},{col}:{grid[(row, col)]}" for row, col in sorted(grid))

    @staticmethod
    def _find_name_position(name, grid):
        if not name:
            return None
        for pos, text in grid.items():
            if text == name:
                return pos
        for pos, text in grid.items():
            if name in text or text in name:
                return pos
        return None

    @staticmethod
    def _place_action_for_position(pos, cfg):
        row, col = pos
        action_grid = cfg.get("place_action_grid", [])
        try:
            return action_grid[row - 1][col - 1]
        except (IndexError, TypeError):
            return f"ArmPlaceR{row}C{col}"

    def _group_items_by_building(self, order, cfg):
        valid_buildings = {str(item) for item in cfg.get("valid_buildings", ["1", "2"])}
        grouped = {}
        for item in (order or {}).get("items", []):
            building = str(item.get("building") or "").strip()
            name = str(item.get("name") or "").strip()
            goods = str(item.get("goods") or "").strip()
            if building not in valid_buildings or not name or not goods:
                print(f"[PlaceDelivery] Skip invalid item: {item}")
                continue
            grouped.setdefault(building, []).append(item)
        return grouped

    def _unit_labels(self, building, cfg):
        aliases = cfg.get("unit_label_aliases", {})
        if isinstance(aliases, dict):
            labels = aliases.get(str(building), [])
        else:
            labels = aliases
        if labels:
            return labels
        return [f"unit_{building}", f"building_{building}"]

    def _fail(self, reason):
        self.base.MOD_STOP()
        print(f"[PlaceDelivery] Task failed: {reason}")
        return False

    def _load_cfg(self):
        task_cfg = load_params(self.motion_file).get("TASK_CONFIG", {})
        cfg = task_cfg.get("PLACE_DELIVERY", {})
        cfg.setdefault("valid_buildings", ["1", "2"])
        cfg.setdefault("text_label_aliases", ["text"])
        cfg.setdefault("unit_label_aliases", {})
        cfg.setdefault("place_action_grid", [
            ["ArmPlaceR1C1", "ArmPlaceR1C2", "ArmPlaceR1C3"],
            ["ArmPlaceR2C1", "ArmPlaceR2C2", "ArmPlaceR2C3"],
        ])
        return cfg

import time

import numpy as np

from get_json import load_params
from order_ai_client import OrderAIClient
from tool_func import get_biggest_box


class OrderDeliveryTaskExecutor:
    """Task 6/7: read order, parse it, move to shelf, and pick two goods."""

    def __init__(
        self,
        base,
        ocr_reader=None,
        task_client=None,
        task_shm_key="shm_task",
        tracking_callback=None,
        motion_file="motion.json",
        ai_client=None,
    ):
        self.base = base
        self.ocr_reader = ocr_reader
        self.task_client = task_client
        self.task_shm_key = task_shm_key
        self.tracking_callback = tracking_callback
        self.motion_file = motion_file
        self.ai_client = ai_client or OrderAIClient()
        self.last_order = None

    def run(self):
        cfg = self._load_cfg()
        print("[OrderDelivery] Start order and pickup task.")

        if not self._align_text(cfg):
            return self._fail("Failed to align order text.")
        if not self._push_order_rod(cfg):
            return self._fail("Failed to push order rod.")

        time.sleep(float(cfg.get("image_settle_s", 0.5)))
        first_texts = self._read_stable_texts(cfg, phase_name="first order")
        if not first_texts:
            return self._fail("Failed to read first order text.")
        first_order = self._parse_order(first_texts, cfg)
        if first_order is None:
            return self._fail("Failed to parse first order.")

        if not self._prepare_common_order_view(cfg):
            return self._fail("Failed to prepare common order camera view.")
        time.sleep(float(cfg.get("image_settle_s", 0.5)))
        common_texts = self._read_stable_texts(cfg, phase_name="common order")
        if not common_texts:
            return self._fail("Failed to read common order text.")
        common_order = self._parse_order(common_texts, cfg)
        if common_order is None:
            return self._fail("Failed to parse common order.")

        final_order = self._merge_orders(first_order, common_order)
        self.last_order = final_order
        print(f"[OrderDelivery] Final order: {final_order}")

        if not self._move_to_shelf(cfg):
            return self._fail("Failed to move to shelf.")
        if not self._align_shelf(cfg):
            return self._fail("Failed to align shelf.")
        pick_action_map = self._build_goods_pick_action_map(cfg)
        if not pick_action_map:
            return self._fail("Failed to build goods pick action map.")
        if not self._pick_two_goods(final_order, cfg, pick_action_map):
            return self._fail("Failed to pick goods.")

        self.base.MOD_STOP()
        print("[OrderDelivery] Task finished.")
        return True

    def has_order_machine(self):
        cfg = self._load_cfg()
        if self.task_client is None:
            return False
        detections = self.task_client(self.task_shm_key, (640, 640, 3), np.uint8)
        box = get_biggest_box(
            detections,
            target_labels=cfg.get("order_machine_label_aliases", ["order_machine"]),
            min_score=float(cfg.get("min_score", 0.5)),
        )
        return box is not None

    def _align_text(self, cfg):
        return self._track_target(
            cfg,
            label_aliases=cfg.get("text_label_aliases", ["text"]),
            target_pose=tuple(cfg.get("order_text_target_pose", [320, 240])),
            phase_name="order text",
        )

    def _push_order_rod(self, cfg):
        return self.base.execute_wait_instruction(
            cfg.get("push_order_command", {"cmd": "SysMode", "flag": 10}),
            wait_timeout_s=float(cfg.get("push_ack_timeout_s", 3.0)),
            action_name="push order rod",
        )

    def _prepare_common_order_view(self, cfg):
        action = cfg.get("common_order_camera_action", "ArmOrderCameraView")
        print(f"[OrderDelivery] Prepare common order view -> {action}")
        return self.base.execute_arm_instruction(
            action,
            countdown=float(cfg.get("arm_action_countdown", 0.2)),
        )

    def _move_to_shelf(self, cfg):
        action = cfg.get("move_to_shelf_action", "movelong")
        print(f"[OrderDelivery] Move to shelf -> {action}")
        return self.base.execute_chassis_instruction(
            action,
            countdown=float(cfg.get("action_countdown", 0.2)),
        )

    def _align_shelf(self, cfg):
        return self._track_target(
            cfg,
            label_aliases=cfg.get("shelf_label_aliases", ["shelf"]),
            target_pose=tuple(cfg.get("shelf_target_pose", [320, 260])),
            phase_name="shelf",
        )

    def _prepare_shelf_overview(self, cfg):
        action = cfg.get("shelf_overview_camera_action", "ArmShelfOverview")
        print(f"[OrderDelivery] Prepare shelf overview -> {action}")
        return self.base.execute_arm_instruction(
            action,
            countdown=float(cfg.get("arm_action_countdown", 0.2)),
        )

    def _build_goods_pick_action_map(self, cfg):
        if self.task_client is None:
            return {}
        if not self._prepare_shelf_overview(cfg):
            return {}
        time.sleep(float(cfg.get("shelf_overview_settle_s", 0.4)))

        stable_times = int(cfg.get("goods_layout_confirm_frames", 2))
        retry_count = int(cfg.get("goods_layout_retry_count", 8))
        last_signature = None
        stable_hits = 0
        last_map = {}

        for attempt in range(1, retry_count + 1):
            detections = self.task_client(self.task_shm_key, (640, 640, 3), np.uint8)
            layout = self._detect_goods_layout(detections, cfg)
            signature = self._goods_layout_signature(layout)
            print(f"[OrderDelivery] Goods layout {attempt}/{retry_count}: {signature}")
            if not layout:
                stable_hits = 0
                last_signature = None
                time.sleep(float(cfg.get("goods_layout_retry_interval_s", 0.08)))
                continue
            if signature == last_signature:
                stable_hits += 1
            else:
                stable_hits = 1
                last_signature = signature
                last_map = layout
            if stable_hits >= stable_times:
                print(f"[OrderDelivery] Stable goods action map: {last_map}")
                return last_map
            time.sleep(float(cfg.get("goods_layout_retry_interval_s", 0.08)))

        return last_map

    def _detect_goods_layout(self, detections, cfg):
        rows = int(cfg.get("shelf_grid_rows", 4))
        cols = int(cfg.get("shelf_grid_cols", 2))
        slot_action_grid = cfg.get("slot_action_grid") or []
        aliases = cfg.get("goods_label_aliases", {})
        alias_to_goods = self._build_alias_to_goods(aliases, cfg.get("valid_goods", []))
        target_labels = set(alias_to_goods.keys())
        min_score = float(cfg.get("goods_min_score", cfg.get("min_score", 0.5)))

        candidates = []
        for det in detections or []:
            label = str(getattr(det, "label", "")).lower()
            if label not in target_labels:
                continue
            if float(getattr(det, "score", 0.0)) < min_score:
                continue
            if getattr(det, "center", None) is None:
                continue
            candidates.append(det)

        expected_count = rows * cols
        if len(candidates) < int(cfg.get("goods_min_detect_count", expected_count)):
            print(f"[OrderDelivery] Goods layout needs {expected_count}, detected {len(candidates)}.")
            return {}

        candidates = sorted(candidates, key=lambda det: (float(det.center[1]), float(det.center[0])))[:expected_count]
        action_map = {}
        for row_index in range(rows):
            row_items = candidates[row_index * cols:(row_index + 1) * cols]
            row_items = sorted(row_items, key=lambda det: float(det.center[0]))
            for col_index, det in enumerate(row_items):
                goods = alias_to_goods.get(str(getattr(det, "label", "")).lower())
                action = self._slot_action_for(row_index, col_index, cols, slot_action_grid)
                if not goods or not action:
                    continue
                action_map[goods] = action
                print(
                    "[OrderDelivery] Goods slot "
                    f"row={row_index + 1} col={col_index + 1} goods={goods} action={action}"
                )
        return action_map

    @staticmethod
    def _build_alias_to_goods(goods_label_aliases, valid_goods):
        alias_to_goods = {}
        for goods in valid_goods:
            goods_key = str(goods).lower()
            alias_to_goods[goods_key] = goods_key
        for goods, aliases in goods_label_aliases.items():
            goods_key = str(goods).lower()
            alias_to_goods[goods_key] = goods_key
            if isinstance(aliases, str):
                aliases = [aliases]
            for alias in aliases:
                alias_to_goods[str(alias).lower()] = goods_key
        return alias_to_goods

    @staticmethod
    def _slot_action_for(row_index, col_index, cols, slot_action_grid):
        try:
            return slot_action_grid[row_index][col_index]
        except (IndexError, TypeError):
            slot_index = row_index * int(cols) + col_index + 1
            return f"ArmPickSlot{slot_index}"

    @staticmethod
    def _goods_layout_signature(layout):
        if not layout:
            return ""
        return "|".join(f"{goods}:{layout[goods]}" for goods in sorted(layout))

    def _pick_two_goods(self, order, cfg, dynamic_action_map):
        items = order.get("items", [])[: int(cfg.get("item_count", 2))]
        static_action_map = cfg.get("pick_action_map", {})
        allow_static_fallback = bool(cfg.get("allow_static_pick_fallback", False))
        for index, item in enumerate(items, start=1):
            goods = item.get("goods")
            action = dynamic_action_map.get(goods)
            if not action and allow_static_fallback:
                action = static_action_map.get(goods)
            if not action:
                print(f"[OrderDelivery] Missing pick action for goods={goods}")
                return False
            print(f"[OrderDelivery] Pick item {index}: goods={goods}, action={action}")
            if not self.base.execute_arm_instruction(
                action,
                countdown=float(cfg.get("arm_action_countdown", 0.2)),
            ):
                return False
        return True

    def _track_target(self, cfg, label_aliases, target_pose, phase_name):
        if self.tracking_callback is None or self.task_client is None:
            print(f"[OrderDelivery] Tracking unavailable for {phase_name}, skip.")
            return True
        print(f"[OrderDelivery] Align {phase_name}.")
        return self.tracking_callback(
            target_pose=target_pose,
            cam_pose=cfg.get("cam_pose", "L"),
            timeout_s=float(cfg.get("track_timeout_s", 4.0)),
            max_missed_frames=int(cfg.get("max_missed_frames", 6)),
            recover_pause_s=float(cfg.get("recover_pause_s", 0.06)),
            recover_timeout_s=float(cfg.get("recover_timeout_s", 0.5)),
            selector=get_biggest_box,
            selector_kwargs={
                "target_labels": label_aliases,
                "min_score": float(cfg.get("min_score", 0.5)),
            },
        )

    def _read_stable_texts(self, cfg, phase_name):
        if self.ocr_reader is None:
            return []
        stable_times = int(cfg.get("ocr_stable_times", 2))
        retry_count = int(cfg.get("ocr_retry_count", 6))
        result_amount = int(cfg.get("ocr_result_amount", 2))
        sort_by = cfg.get("ocr_sort_by", "y")
        target_label = cfg.get("text_label_aliases", ["text"])[0]
        last_texts = None
        stable_hits = 0

        for attempt in range(1, retry_count + 1):
            texts = self.ocr_reader.read_texts(
                shm_key=self.task_shm_key,
                shape=(640, 640, 3),
                dtype=np.uint8,
                target_label=target_label,
                result_amount=result_amount,
                sort_by=sort_by,
                retries=int(cfg.get("ocr_inner_retries", 4)),
            )
            normalized = self._normalize_texts(texts)
            print(f"[OrderDelivery] OCR {phase_name} {attempt}/{retry_count}: {normalized}")
            if not normalized:
                stable_hits = 0
                last_texts = None
                continue
            if normalized == last_texts:
                stable_hits += 1
            else:
                last_texts = normalized
                stable_hits = 1
            if stable_hits >= stable_times:
                return normalized
            time.sleep(float(cfg.get("ocr_retry_interval_s", 0.1)))
        return last_texts or []

    def _parse_order(self, texts, cfg):
        try:
            return self.ai_client.parse_order(
                texts,
                valid_goods=cfg.get("valid_goods", []),
                valid_buildings=cfg.get("valid_buildings", []),
                item_count=int(cfg.get("item_count", 2)),
            )
        except Exception as exc:
            print(f"[OrderDelivery] AI parse failed: {exc}")
            return None

    def _merge_orders(self, first_order, common_order):
        first_items = first_order.get("items", [])
        common_items = common_order.get("items", [])
        merged = []
        max_len = max(len(first_items), len(common_items))
        for index in range(max_len):
            first = first_items[index] if index < len(first_items) else {}
            common = common_items[index] if index < len(common_items) else {}
            merged.append(
                {
                    "goods": common.get("goods") or first.get("goods"),
                    "building": common.get("building") or first.get("building"),
                    "name": common.get("name") or first.get("name"),
                }
            )
        return {"items": merged}

    @staticmethod
    def _normalize_texts(texts):
        if not texts:
            return []
        normalized = []
        for text in texts:
            value = str(text).strip()
            if value:
                normalized.append(value)
        return normalized

    def _fail(self, reason):
        self.base.MOD_STOP()
        print(f"[OrderDelivery] Task failed: {reason}")
        return False

    def _load_cfg(self):
        task_cfg = load_params(self.motion_file).get("TASK_CONFIG", {})
        cfg = task_cfg.get("ORDER_DELIVERY", {})
        cfg.setdefault("order_machine_label_aliases", ["order_machine", "order_kiosk"])
        cfg.setdefault("text_label_aliases", ["text"])
        cfg.setdefault("shelf_label_aliases", ["shelf", "goods_shelf"])
        cfg.setdefault("goods_label_aliases", {})
        cfg.setdefault("valid_goods", [])
        cfg.setdefault("valid_buildings", [])
        cfg.setdefault("pick_action_map", {})
        cfg.setdefault("slot_action_grid", [
            ["ArmPickSlot1", "ArmPickSlot2"],
            ["ArmPickSlot3", "ArmPickSlot4"],
            ["ArmPickSlot5", "ArmPickSlot6"],
            ["ArmPickSlot7", "ArmPickSlot8"],
        ])
        return cfg

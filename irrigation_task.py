import time
from dataclasses import dataclass, field

from get_json import load_params
from tool_func import (
    DEFAULT_IRRIGATION_BOARD_LABEL_ALIASES,
    DEFAULT_IRRIGATION_PLACE_LABEL_ALIASES,
    DEFAULT_IRRIGATION_SUPPLY_LABEL_ALIASES,
    get_irrigation_board_box,
    get_irrigation_place_box,
    get_irrigation_supply_box,
    parse_irrigation_need_text,
)


@dataclass
class IrrigationRuntimeState:
    tower_need: dict = field(default_factory=lambda: {1: None, 2: None})
    tower_done: dict = field(default_factory=lambda: {1: 0, 2: 0})
    tower_finished: dict = field(default_factory=lambda: {1: False, 2: False})
    supply_counts: dict = field(default_factory=dict)
    supply_order: list = field(default_factory=list)
    current_supply_zone: str = "a"
    carry_count: int = 0
    completed: bool = False


class IrrigationTaskExecutor:
    """Config-driven irrigation task flow modeled after the latest Task1 style."""

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
        self.runtime = None

    def run(self, supply_counts=None):
        cfg = self._load_cfg()
        self.runtime = IrrigationRuntimeState(
            supply_counts=self._build_supply_counts(cfg, supply_counts),
            supply_order=list(cfg.get("supply_order", ["a", "b", "c"])),
            current_supply_zone=str(cfg.get("supply_order", ["a"])[0]).lower(),
        )

        print("[Irrigation] Start irrigation task flow.")
        self._execute_chassis(cfg["actions"].get("enter"))

        for tower_id in cfg.get("tower_order", [1, 2]):
            if not self._run_single_tower(int(tower_id), cfg):
                return False

        self.runtime.completed = True
        self._execute_chassis(cfg["actions"].get("return"), countdown=cfg.get("action_countdown", 0.2))
        self.base.MOD_STOP()
        print("[Irrigation] Irrigation task flow finished.")
        return True

    def _run_single_tower(self, tower_id, cfg):
        print(f"[Irrigation] Process tower {tower_id}.")
        need_count = self._read_tower_need(tower_id, cfg)
        if need_count is None:
            return self._fail(f"Failed to read demand for tower {tower_id}.", cfg)

        self.runtime.tower_need[tower_id] = need_count
        print(f"[Irrigation] Tower {tower_id} need count: {need_count}")

        while self.runtime.tower_done[tower_id] < need_count:
            if not self._pick_one_block(cfg):
                return self._fail(f"Failed to pick block for tower {tower_id}.", cfg)
            if not self._place_one_block(tower_id, cfg):
                return self._fail(f"Failed to place block for tower {tower_id}.", cfg)
            self.runtime.tower_done[tower_id] += 1
            print(
                f"[Irrigation] Tower {tower_id} progress: "
                f"{self.runtime.tower_done[tower_id]}/{need_count}"
            )

        self.runtime.tower_finished[tower_id] = True
        print(f"[Irrigation] Tower {tower_id} completed.")
        return True

    def _read_tower_need(self, tower_id, cfg):
        action_key = cfg["actions"].get(f"tower{tower_id}_read")
        self._execute_chassis(action_key, countdown=cfg.get("action_countdown", 0.2))

        retries = int(cfg.get("read_retry_count", 1))
        default_need = cfg.get("default_need_count", 1)
        max_need = int(cfg.get("max_need_count", 3))
        result_amount = int(cfg.get("ocr_result_amount", 3))
        sort_by = cfg.get("ocr_sort_by", "y")

        for attempt in range(retries + 1):
            if cfg.get("use_visual_alignment", True):
                aligned = self._track_target(
                    selector=get_irrigation_board_box,
                    selector_kwargs={"label_aliases": cfg.get("board_label_aliases")},
                    target_pose=tuple(cfg.get("board_target_pose", [320, 230])),
                    cfg=cfg,
                )
                if not aligned:
                    print(f"[Irrigation] Tower {tower_id} board align failed on attempt {attempt + 1}.")
                    continue

            texts = self._read_texts(result_amount=result_amount, sort_by=sort_by)
            need_count = parse_irrigation_need_text(texts, default=default_need, max_need=max_need)
            if need_count is not None:
                return need_count

        return None

    def _pick_one_block(self, cfg):
        zone = self._ensure_supply_zone(cfg)
        if zone is None:
            print("[Irrigation] All supply zones are exhausted.")
            return False

        action_key = cfg["actions"].get(f"supply_{zone}")
        self._execute_chassis(action_key, countdown=cfg.get("action_countdown", 0.2))

        if cfg.get("use_visual_alignment", True):
            aligned = self._track_target(
                selector=get_irrigation_supply_box,
                selector_kwargs={"label_aliases": cfg.get("supply_label_aliases")},
                target_pose=tuple(cfg.get("supply_target_pose", [320, 250])),
                cfg=cfg,
            )
            if not aligned:
                return False

        self._execute_arm(cfg["arm_actions"].get("pick_prepare"), cfg)
        self._execute_arm(cfg["arm_actions"].get("pick"), cfg)
        self._execute_arm(cfg["arm_actions"].get("lift"), cfg)
        self.runtime.carry_count = 1

        if self.runtime.supply_counts.get(zone, 0) > 0:
            self.runtime.supply_counts[zone] -= 1
        print(
            f"[Irrigation] Picked one block from zone {zone}, "
            f"remaining={self.runtime.supply_counts.get(zone, 0)}"
        )
        return True

    def _place_one_block(self, tower_id, cfg):
        action_key = cfg["actions"].get(f"tower{tower_id}_place")
        self._execute_chassis(action_key, countdown=cfg.get("action_countdown", 0.2))

        if cfg.get("use_visual_alignment", True):
            aligned = self._track_target(
                selector=get_irrigation_place_box,
                selector_kwargs={"label_aliases": cfg.get("place_label_aliases")},
                target_pose=tuple(cfg.get("place_target_pose", [320, 250])),
                cfg=cfg,
            )
            if not aligned:
                return False

        self._execute_arm(cfg["arm_actions"].get("place_prepare"), cfg)
        self._execute_arm(cfg["arm_actions"].get("place"), cfg)
        self._execute_arm(cfg["arm_actions"].get("release"), cfg)
        self._execute_arm(cfg["arm_actions"].get("safe"), cfg)
        self.runtime.carry_count = 0
        return True

    def _track_target(self, selector, selector_kwargs, target_pose, cfg):
        if self.tracking_callback is None or self.task_client is None:
            print("[Irrigation] Tracking callback or task client is unavailable, skip visual align.")
            return True
        return self.tracking_callback(
            target_pose=target_pose,
            cam_pose=cfg.get("cam_pose", "L"),
            timeout_s=float(cfg.get("track_timeout_s", 4.0)),
            max_missed_frames=int(cfg.get("max_missed_frames", 6)),
            recover_pause_s=float(cfg.get("recover_pause_s", 0.06)),
            recover_timeout_s=float(cfg.get("recover_timeout_s", 0.5)),
            selector=selector,
            selector_kwargs=selector_kwargs,
        )

    def _read_texts(self, result_amount=3, sort_by="y"):
        if self.ocr_reader is None:
            return []
        return self.ocr_reader.read_texts(
            shm_key=self.task_shm_key,
            shape=(640, 640, 3),
            result_amount=result_amount,
            sort_by=sort_by,
        )

    def _ensure_supply_zone(self, cfg):
        if not self.runtime.supply_order:
            return None

        current = self.runtime.current_supply_zone
        while self.runtime.supply_counts.get(current, 0) <= 0:
            next_zone = self._advance_supply_zone()
            if next_zone is None:
                return None
            current = next_zone

        return current

    def _advance_supply_zone(self):
        order = self.runtime.supply_order
        current = self.runtime.current_supply_zone
        try:
            index = order.index(current)
        except ValueError:
            self.runtime.current_supply_zone = order[0]
            return order[0]

        if index + 1 >= len(order):
            return None

        self.runtime.current_supply_zone = order[index + 1]
        print(f"[Irrigation] Advance supply zone -> {self.runtime.current_supply_zone}")
        return self.runtime.current_supply_zone

    def _execute_chassis(self, action_key, countdown=0.2):
        if not action_key:
            return True
        return self.base.execute_chassis_instruction(action_key, countdown=countdown)

    def _execute_arm(self, action_key, cfg):
        if not action_key:
            return True
        return self.base.execute_arm_instruction(
            action_key,
            countdown=float(cfg.get("arm_action_countdown", 0.15)),
        )

    def _fail(self, reason, cfg):
        self.base.MOD_STOP()
        self._execute_chassis(cfg["actions"].get("abort"), countdown=cfg.get("abort_countdown", 0.2))
        print(f"[Irrigation] Task failed: {reason}")
        return False

    def _build_supply_counts(self, cfg, override_counts=None):
        counts = {str(k).lower(): int(v) for k, v in cfg.get("supply_counts", {}).items()}
        if override_counts:
            for key, value in override_counts.items():
                counts[str(key).lower()] = int(value)
        return counts

    def _load_cfg(self):
        task_cfg = load_params(self.motion_file).get("TASK_CONFIG", {})
        irrigation_cfg = task_cfg.get("IRRIGATION", {})
        irrigation_cfg.setdefault("tower_order", [1, 2])
        irrigation_cfg.setdefault("supply_order", ["a", "b", "c"])
        irrigation_cfg.setdefault("board_label_aliases", DEFAULT_IRRIGATION_BOARD_LABEL_ALIASES)
        irrigation_cfg.setdefault("supply_label_aliases", DEFAULT_IRRIGATION_SUPPLY_LABEL_ALIASES)
        irrigation_cfg.setdefault("place_label_aliases", DEFAULT_IRRIGATION_PLACE_LABEL_ALIASES)
        irrigation_cfg.setdefault("actions", {})
        irrigation_cfg.setdefault("arm_actions", {})
        irrigation_cfg.setdefault("supply_counts", {"a": 2, "b": 2, "c": 2})
        return irrigation_cfg

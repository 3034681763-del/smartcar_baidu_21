import math

import numpy as np


DEFAULT_SEED_LABEL_ALIASES = {
    "large": ["largeround", "seed_large", "largecylinder", "seeding_large"],
    "medium": ["mediumround", "seed_medium", "mediumcylinder", "seeding_medium"],
    "small": ["smallround", "seed_small", "smallcylinder", "seeding_small"],
}

DEFAULT_SEED_DROP_SLOT_MAP = {
    "large": "a",
    "medium": "b",
    "small": "c",
}

DEFAULT_SEED_COLOR_RULES = {
    "medium": "blue_bias",
}

DEFAULT_IRRIGATION_BOARD_LABEL_ALIASES = [
    "waterneed",
    "tower_board",
    "irrigation_board",
    "water_demand_board",
    "droplet_board",
]

DEFAULT_IRRIGATION_SUPPLY_LABEL_ALIASES = [
    "waterblock",
    "irrigation_block",
    "water_cube",
    "cube",
    "bluecube",
]

DEFAULT_IRRIGATION_PLACE_LABEL_ALIASES = [
    "watertower",
    "towerplatform",
    "irrigation_place",
    "place_slot",
    "tower_tray",
]


def _normalize_targets(target_labels):
    if target_labels is None:
        return None
    if isinstance(target_labels, str):
        return {target_labels}
    return {label for label in target_labels if label}


def _iter_valid_detections(detections, min_score=0.5):
    if not detections:
        return
    for detection in detections:
        if getattr(detection, "score", 0.0) >= min_score:
            yield detection


def _filter_detections(detections, target_labels=None, min_score=0.5):
    targets = _normalize_targets(target_labels)
    filtered = []
    for detection in _iter_valid_detections(detections, min_score=min_score):
        label = getattr(detection, "label", None)
        if targets is None or label in targets:
            filtered.append(detection)
    return filtered


def _box_area(detection):
    x1, y1, x2, y2 = getattr(detection, "bbox", (0, 0, 0, 0))
    return max(0.0, float(x2) - float(x1)) * max(0.0, float(y2) - float(y1))


def _bbox_size(detection):
    x1, y1, x2, y2 = getattr(detection, "bbox", (0, 0, 0, 0))
    return max(0.0, float(x2) - float(x1)), max(0.0, float(y2) - float(y1))


def _bbox_center_margin_ok(detection, edge_margin=5, frame_width=640, frame_height=640):
    x1, y1, x2, y2 = getattr(detection, "bbox", (0, 0, 0, 0))
    if float(x1) <= edge_margin or float(x2) >= frame_width - edge_margin:
        return False
    if float(y1) <= edge_margin or float(y2) >= frame_height - edge_margin:
        return False
    return True


def _collect_seed_candidates(detections, label_aliases=None, min_score=0.5):
    aliases = label_aliases or DEFAULT_SEED_LABEL_ALIASES
    seed_labels = set()
    for labels in aliases.values():
        seed_labels.update(labels)

    candidates = _filter_detections(
        detections,
        target_labels=seed_labels,
        min_score=min_score,
    )
    unique = []
    seen = set()
    for detection in candidates:
        key = (
            getattr(detection, "label", None),
            tuple(round(float(v), 2) for v in getattr(detection, "bbox", (0, 0, 0, 0))),
        )
        if key in seen:
            continue
        seen.add(key)
        unique.append(detection)
    return unique


def _extract_box_patch(image, detection, patch_radius=12):
    if image is None:
        return None

    height, width = image.shape[:2]
    cx = int(round(float(detection.center[0])))
    cy = int(round(float(detection.center[1])))
    x1 = max(0, cx - patch_radius)
    y1 = max(0, cy - patch_radius)
    x2 = min(width, cx + patch_radius + 1)
    y2 = min(height, cy + patch_radius + 1)
    if x1 >= x2 or y1 >= y2:
        return None
    patch = image[y1:y2, x1:x2]
    if patch.size == 0:
        return None
    return patch


def get_box_color_stats(image, detection, patch_radius=12):
    patch = _extract_box_patch(image, detection, patch_radius=patch_radius)
    if patch is None:
        return None

    mean_bgr = patch.reshape(-1, patch.shape[-1]).mean(axis=0)
    blue, green, red = (float(mean_bgr[0]), float(mean_bgr[1]), float(mean_bgr[2]))
    return {
        "blue": blue,
        "green": green,
        "red": red,
        "blue_bias": blue - red,
        "green_bias": green - red,
    }


def _classify_seed_boxes_with_color(
    detections,
    image,
    label_aliases=None,
    min_score=0.5,
    patch_radius=12,
):
    candidates = _collect_seed_candidates(
        detections,
        label_aliases=label_aliases,
        min_score=min_score,
    )
    if image is None or len(candidates) < 3:
        return None

    enriched = []
    for detection in candidates:
        stats = get_box_color_stats(image, detection, patch_radius=patch_radius)
        if stats is None:
            continue
        enriched.append(
            {
                "detection": detection,
                "stats": stats,
                "area": _box_area(detection),
            }
        )
    if len(enriched) < 3:
        return None

    medium_item = max(enriched, key=lambda item: item["stats"]["blue_bias"])
    remaining = [item for item in enriched if item["detection"] is not medium_item["detection"]]
    if len(remaining) < 2:
        return None

    large_item = max(remaining, key=lambda item: item["area"])
    small_item = min(remaining, key=lambda item: item["area"])

    return {
        "large": large_item["detection"],
        "medium": medium_item["detection"],
        "small": small_item["detection"],
        "color_stats": {
            "large": large_item["stats"],
            "medium": medium_item["stats"],
            "small": small_item["stats"],
        },
        "source": "color_assist",
    }


def get_only_box(detections, min_score=0.5):
    filtered = _filter_detections(detections, min_score=min_score)
    if not filtered:
        return None
    return filtered[0]


def get_biggest_box(detections, target_labels=None, min_score=0.5):
    filtered = _filter_detections(detections, target_labels=target_labels, min_score=min_score)
    if not filtered:
        return None
    return max(filtered, key=_box_area)


def get_closest_box(detections, target_pose=(320, 240), target_labels=None, min_score=0.5):
    filtered = _filter_detections(detections, target_labels=target_labels, min_score=min_score)
    if not filtered:
        return None

    tx, ty = target_pose
    return min(
        filtered,
        key=lambda det: math.hypot(float(det.center[0]) - float(tx), float(det.center[1]) - float(ty)),
    )


def get_seed_box(detections, size_key, label_aliases=None, min_score=0.5):
    aliases = label_aliases or DEFAULT_SEED_LABEL_ALIASES
    return get_biggest_box(
        detections,
        target_labels=aliases.get(size_key, []),
        min_score=min_score,
    )


def get_irrigation_board_box(detections, label_aliases=None, min_score=0.5):
    aliases = label_aliases or DEFAULT_IRRIGATION_BOARD_LABEL_ALIASES
    return get_biggest_box(detections, target_labels=aliases, min_score=min_score)


def get_irrigation_supply_box(detections, label_aliases=None, min_score=0.5):
    aliases = label_aliases or DEFAULT_IRRIGATION_SUPPLY_LABEL_ALIASES
    return get_biggest_box(detections, target_labels=aliases, min_score=min_score)


def get_irrigation_place_box(detections, label_aliases=None, min_score=0.5):
    aliases = label_aliases or DEFAULT_IRRIGATION_PLACE_LABEL_ALIASES
    return get_biggest_box(detections, target_labels=aliases, min_score=min_score)


def parse_irrigation_need_text(texts, default=None, max_need=3):
    if texts is None:
        return default

    if not isinstance(texts, (list, tuple)):
        texts = [texts]

    normalized = []
    for item in texts:
        if item is None:
            continue
        if isinstance(item, (list, tuple)):
            normalized.append(" ".join(str(part) for part in item if part is not None))
        else:
            normalized.append(str(item))

    digit_map = {
        "一": 1,
        "壹": 1,
        "1": 1,
        "一滴": 1,
        "二": 2,
        "贰": 2,
        "两": 2,
        "2": 2,
        "二滴": 2,
        "三": 3,
        "叁": 3,
        "3": 3,
        "三滴": 3,
    }

    for text in normalized:
        for token, value in digit_map.items():
            if token in text and value <= max_need:
                return value
        digits = [int(ch) for ch in text if ch.isdigit()]
        for digit in digits:
            if 0 < digit <= max_need:
                return digit

    return default


def has_all_seed_targets(
    detections,
    label_aliases=None,
    min_score=0.5,
    edge_margin=5,
    frame_width=640,
    frame_height=640,
    min_area=0.0,
    return_details=False,
):
    aliases = label_aliases or DEFAULT_SEED_LABEL_ALIASES
    details = {}
    for size_key in ("large", "medium", "small"):
        box = get_seed_box(detections, size_key, label_aliases=aliases, min_score=min_score)
        if box is None:
            details[size_key] = {"ready": False, "reason": "missing", "box": None}
            return details if return_details else False
        area = _box_area(box)
        if area < float(min_area):
            details[size_key] = {"ready": False, "reason": "too_small", "box": box, "area": area}
            return details if return_details else False
        if not _bbox_center_margin_ok(
            box,
            edge_margin=edge_margin,
            frame_width=frame_width,
            frame_height=frame_height,
        ):
            details[size_key] = {"ready": False, "reason": "edge", "box": box, "area": area}
            return details if return_details else False
        details[size_key] = {"ready": True, "reason": "ok", "box": box, "area": area}
    details["ready"] = True
    return details if return_details else True


def judge_seed_layout(
    detections,
    image=None,
    label_aliases=None,
    drop_slot_map=None,
    sort_reverse=False,
    min_score=0.5,
    use_color_assist=True,
    color_patch_radius=12,
):
    """
    Determine where large/medium/small cylinders currently sit.

    Returns:
        {
            "ordered_boxes": {"large": det, "medium": det, "small": det},
            "slot_map": {"large": "2", "medium": "1", "small": "3"},
            "transport_map": {
                "large": {
                    "pick_slot": "2",
                    "drop_slot": "a",
                    "go_action": "Seed2ToA",
                },
                ...
            },
            "sorted_keys": ["medium", "large", "small"],
            "signature": "large:2|medium:1|small:3"
        }
    """
    aliases = label_aliases or DEFAULT_SEED_LABEL_ALIASES
    drop_map = drop_slot_map or DEFAULT_SEED_DROP_SLOT_MAP
    selected = {}
    color_info = {}
    source = "label_only"

    if use_color_assist:
        color_layout = _classify_seed_boxes_with_color(
            detections,
            image=image,
            label_aliases=aliases,
            min_score=min_score,
            patch_radius=color_patch_radius,
        )
        if color_layout is not None:
            selected = {
                "large": color_layout["large"],
                "medium": color_layout["medium"],
                "small": color_layout["small"],
            }
            color_info = color_layout.get("color_stats", {})
            source = color_layout.get("source", source)

    if not selected:
        for size_key in ("large", "medium", "small"):
            box = get_seed_box(detections, size_key, label_aliases=aliases, min_score=min_score)
            if box is None:
                return None
            selected[size_key] = box

    for size_key in ("large", "medium", "small"):
        color_info.setdefault(
            size_key,
            get_box_color_stats(image, selected[size_key], patch_radius=color_patch_radius),
        )

    sorted_items = sorted(
        selected.items(),
        key=lambda item: float(item[1].center[0]),
        reverse=bool(sort_reverse),
    )
    slot_names = ["1", "2", "3"]
    slot_map = {
        sorted_items[index][0]: slot_names[index]
        for index in range(len(sorted_items))
    }

    transport_map = {}
    for size_key, pick_slot in slot_map.items():
        drop_slot = drop_map[size_key]
        transport_map[size_key] = {
            "pick_slot": pick_slot,
            "drop_slot": drop_slot,
            "go_action": f"Seed{pick_slot}To{drop_slot.upper()}",
        }

    return {
        "ordered_boxes": selected,
        "slot_map": slot_map,
        "transport_map": transport_map,
        "sorted_keys": [item[0] for item in sorted_items],
        "signature": "|".join(f"{key}:{slot_map[key]}" for key in ("large", "medium", "small")),
        "color_info": color_info,
        "source": source,
    }

import math


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


def has_all_seed_targets(
    detections,
    label_aliases=None,
    min_score=0.5,
    edge_margin=5,
    frame_width=640,
):
    aliases = label_aliases or DEFAULT_SEED_LABEL_ALIASES
    for size_key in ("large", "medium", "small"):
        box = get_seed_box(detections, size_key, label_aliases=aliases, min_score=min_score)
        if box is None:
            return False
        x1, _, x2, _ = getattr(box, "bbox", (0, 0, 0, 0))
        if float(x1) <= edge_margin or float(x2) >= frame_width - edge_margin:
            return False
    return True


def judge_seed_layout(
    detections,
    label_aliases=None,
    drop_slot_map=None,
    sort_reverse=False,
    min_score=0.5,
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

    for size_key in ("large", "medium", "small"):
        box = get_seed_box(detections, size_key, label_aliases=aliases, min_score=min_score)
        if box is None:
            return None
        selected[size_key] = box

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
    }

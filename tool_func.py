def get_only_box(detections):
    """Return the current detection box when the scene contains a single target."""
    if not detections:
        return None
    return detections[0]

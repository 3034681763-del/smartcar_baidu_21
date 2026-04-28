import os

from env_loader import load_local_env


LEGACY_LANE_MODEL_PATH = "/home/jetson/workspace_plus/vehicle_wbt_21th_lane/src/cnn_auto.nb"
LEGACY_TASK_MODEL_PATH = "Global_V2"

SRC_LANE_MODEL_PATH = "/home/jetson/smartcar_baidu_21/src/cnn_lane.nb"
SRC_TASK_MODEL_PATH = "/home/jetson/smartcar_baidu_21/src/target_det"


def get_model_profile():
    load_local_env()
    return os.environ.get("SMARTCAR_MODEL_PROFILE", "legacy").strip().lower()


def use_src_models():
    return get_model_profile() in ("src", "new", "jetson", "local")


def get_lane_model_path():
    load_local_env()
    explicit_path = os.environ.get("LANE_MODEL_PATH")
    if explicit_path:
        return explicit_path
    return SRC_LANE_MODEL_PATH if use_src_models() else LEGACY_LANE_MODEL_PATH


def get_task_model_path():
    load_local_env()
    explicit_path = os.environ.get("TASK_MODEL_PATH")
    if explicit_path:
        return explicit_path
    return SRC_TASK_MODEL_PATH if use_src_models() else LEGACY_TASK_MODEL_PATH

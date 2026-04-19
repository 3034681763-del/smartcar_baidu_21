import time

from get_json import load_params


class SegmentedPID:
    def __init__(
        self,
        kp_small,
        kp_medium,
        kp_large,
        ke_small,
        ke_medium,
        ke_max,
        ki,
        kd,
        int_sep_thr,
        int_max,
        output_max,
    ):
        self.kp_small = kp_small
        self.kp_medium = kp_medium
        self.kp_large = kp_large
        self.ke_small = ke_small
        self.ke_medium = ke_medium
        self.ke_max = ke_max
        self.ki = ki
        self.kd = kd
        self.int_sep_thr = int_sep_thr
        self.int_max = int_max
        self.output_max = output_max
        self._int_sum = 0.0
        self._last_error = 0.0
        self._last_time = None

    def reset(self):
        self._int_sum = 0.0
        self._last_error = 0.0
        self._last_time = None

    def _select_kp(self, error):
        abs_error = abs(error)
        if abs_error <= self.ke_small:
            return self.kp_small
        if abs_error <= self.ke_medium:
            return self.kp_medium
        return self.kp_large

    def __call__(self, error, current_time):
        if self._last_time is None:
            dt = 0.0
        else:
            dt = current_time - self._last_time

        kp = self._select_kp(error)
        p_term = kp * error

        if abs(error) <= self.int_sep_thr and dt > 0:
            self._int_sum += error * dt
        self._int_sum = max(-self.int_max, min(self._int_sum, self.int_max))
        i_term = self.ki * self._int_sum

        if dt > 0:
            d_term = self.kd * (error - self._last_error) / dt
        else:
            d_term = 0.0

        self._last_error = error
        self._last_time = current_time

        output = p_term + i_term + d_term
        return max(-self.output_max, min(output, self.output_max))


class BoxPidAligner:
    """Champion-style box alignment controller for a single tracked target."""

    def __init__(self, params="Fast", config_file="param.json"):
        config = load_params(config_file).get("TRACKING_PID", {})
        mode_cfg = config.get(params, config.get("Fast", {}))
        if not mode_cfg:
            raise ValueError(f"Missing TRACKING_PID config for mode: {params}")

        self.params = params
        self.x_threshold = float(mode_cfg.get("x_threshold", 15.0))
        self.y_threshold = float(mode_cfg.get("y_threshold", 15.0))
        self.judge_count = int(mode_cfg.get("judge_count", 3))
        self.countx = self.judge_count
        self.county = self.judge_count

        self.pid_x = SegmentedPID(**mode_cfg["x_pid"])
        self.pid_y = SegmentedPID(**mode_cfg["y_pid"])

    def reset(self):
        self.pid_x.reset()
        self.pid_y.reset()
        self.countx = self.judge_count
        self.county = self.judge_count

    def update(self, center, target_pose=(320, 240), cam_pose="L"):
        now = time.time()
        target_x = target_pose[0] if len(target_pose) > 0 else False
        target_y = target_pose[1] if len(target_pose) > 1 else False

        if cam_pose != "shoot":
            if target_x is not False and target_x is not None:
                x_err = -(float(center[0]) - float(target_x))
                out_x = self.pid_x(x_err, now)
                flag_x = False
                if abs(x_err) < self.x_threshold:
                    self.countx -= 1
                else:
                    self.countx = self.judge_count
                if self.countx <= 0:
                    flag_x = True
            else:
                x_err = None
                out_x = 0.0
                flag_x = True

            if target_y is not False and target_y is not None:
                y_err = -(float(center[1]) - float(target_y))
                out_y = self.pid_y(y_err, now)
                flag_y = False
                if abs(y_err) < self.y_threshold:
                    self.county -= 1
                else:
                    self.county = self.judge_count
                if self.county <= 0:
                    flag_y = True
            else:
                y_err = None
                out_y = 0.0
                flag_y = True

            aligned = flag_x and flag_y
        else:
            x_err = -(float(center[0]) - float(target_x))
            y_err = -(float(center[1]) - float(target_y))
            out_x = self.pid_x(x_err, now)
            out_y = self.pid_y(y_err, now)

            if abs(x_err) < self.x_threshold:
                self.countx -= 1
            else:
                self.countx = self.judge_count

            if abs(y_err) < self.y_threshold:
                self.county -= 1
            else:
                self.county = self.judge_count

            aligned = self.countx <= 0 and self.county <= 0

        if cam_pose == "R":
            motion = {
                "cmd": "Motion",
                "mode": 1,
                "pos_x": -float(out_y),
                "pos_y": float(out_x),
                "z_angle": 0.0,
            }
        elif cam_pose == "shoot":
            motion = {
                "cmd": "Motion",
                "mode": 1,
                "pos_x": 0.0,
                "pos_y": 0.0,
                "z_angle": float(out_x) / 20.0,
            }
        else:
            motion = {
                "cmd": "Motion",
                "mode": 1,
                "pos_x": float(out_y),
                "pos_y": -float(out_x),
                "z_angle": 0.0,
            }

        return aligned, motion, {"x_err": x_err, "y_err": y_err}

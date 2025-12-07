import numpy as np
import mujoco

from .create_path import create_path

# userdata layout (indices into data.userdata)
USERDATA_INIT_FLAG = 0   # 0.0 = not initialized, 1.0 = initialized
USERDATA_WP_INDEX  = 1   # current waypoint index (stored as float, cast to int)

# Control gains and parameters
_K_V = 1.0            # gain on forward error
_K_W = 2.0            # gain on heading error
_WAYPOINT_TOL = 0.10  # [m] distance at which a waypoint is reached
_MAX_CTRL = 1.0       # clip commands to [-1, 1]

# Internal (Python-side) cache for the path only
_PATH_CACHE = None


# Helper: get (x, y, yaw) of the car body in world frame
def _get_car_pose(model, data, body_name="car"):
    """
    Returns (x, y, yaw) of the given body in world coordinates.
    yaw is extracted from the body's rotation matrix assuming z-up.
    """
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)

    # World position of body (3,)
    pos = data.xpos[body_id].copy()
    x, y = pos[0], pos[1]

    # World rotation matrix of body (3x3), stored flat in row-major order
    R_flat = data.xmat[body_id].copy()
    R = R_flat.reshape(3, 3)

    # Heading is body x-axis in world frame
    heading = R[:, 0]
    yaw = np.arctan2(heading[1], heading[0])

    return x, y, yaw


# Helper: initialize userdata and path cache
def _init_if_needed(model, data):
    """
    Initialize path and userdata slots on first call.
    Uses:
      userdata[USERDATA_INIT_FLAG] as 0/1 flag
      userdata[USERDATA_WP_INDEX]  as current waypoint index
    """
    global _PATH_CACHE

    # Ensure userdata is large enough
    if model.nuserdata < 2:
        raise RuntimeError(
            f"traj_control needs at least 2 userdata slots, "
            f"but model.nuserdata = {model.nuserdata}. "
            f"Add <size nuserdata='2'/> (or larger) to your XML."
        )

    init_flag = data.userdata[USERDATA_INIT_FLAG]

    if init_flag == 0.0:
        # First time called: build the path
        path = create_path(model, data)
        path = np.asarray(path, dtype=float)

        if path.shape[1] != 2:
            raise ValueError(
                f"create_path must return waypoints with shape (N, 2); "
                f"got shape {path.shape}"
            )

        _PATH_CACHE = path

        # Initialize userdata state
        data.userdata[USERDATA_INIT_FLAG] = 1.0  # mark as initialized
        data.userdata[USERDATA_WP_INDEX] = 0.0   # waypoint index = 0


# Main controller: follow trajectory in _PATH_CACHE using userdata state
def traj_control(model, data, time=None):
    """
    Trajectory-following controller.

    Arguments:
      model : mujoco.MjModel
      data  : mujoco.MjData
      time  : (optional) current simulation time, unused here

    Returns:
      drive_ctrl  : np.ndarray of shape (2,), actuator commands:
                [0] drive_forward
                [1] drive_turn
    """
    global _PATH_CACHE

    _init_if_needed(model, data)

    # If path was not created for some reason, just stop
    if _PATH_CACHE is None or _PATH_CACHE.shape[0] == 0:
        return np.zeros(model.nu, dtype=float)

    # Read current waypoint index from userdata
    wp_idx = int(data.userdata[USERDATA_WP_INDEX])

    # If finished all waypoints, stop
    if wp_idx >= _PATH_CACHE.shape[0]:
        return np.zeros(model.nu, dtype=float)

    # Current robot pose
    x, y, yaw = _get_car_pose(model, data, body_name="car")

    # Current target waypoint
    tx, ty = _PATH_CACHE[wp_idx]
    dx = tx - x
    dy = ty - y
    dist = np.hypot(dx, dy)

    # Check if we reached this waypoint
    if dist < _WAYPOINT_TOL:
        wp_idx += 1
        data.userdata[USERDATA_WP_INDEX] = float(wp_idx)

        if wp_idx >= _PATH_CACHE.shape[0]:
            # Done with all waypoints
            return np.zeros(model.nu, dtype=float)

        # Update target to new waypoint
        tx, ty = _PATH_CACHE[wp_idx]
        dx = tx - x
        dy = ty - y
        dist = np.hypot(dx, dy)

    # Transform position error into body frame
    cos_yaw = np.cos(yaw)
    sin_yaw = np.sin(yaw)

    ex_body = cos_yaw * dx + sin_yaw * dy      # forward error
    ey_body = -sin_yaw * dx + cos_yaw * dy     # lateral error (not used)

    # Desired heading to waypoint
    desired_heading = np.arctan2(dy, dx)

    # Heading error (wrapped to [-pi, pi])
    heading_error = np.arctan2(
        np.sin(desired_heading - yaw),
        np.cos(desired_heading - yaw),
    )

    # Simple proportional controller for unicycle-like behavior
    v_cmd = _K_V * ex_body          # forward velocity command
    w_cmd = _K_W * heading_error    # turn rate command

    # Map to actuators and clip
    forward_ctrl = np.clip(v_cmd, -_MAX_CTRL, _MAX_CTRL)
    turn_ctrl = np.clip(w_cmd, -_MAX_CTRL, _MAX_CTRL)

    # Build full control vector
    drive_ctrl = np.array([forward_ctrl, turn_ctrl])

    return drive_ctrl

import numpy as np
import mujoco
from .traj_controllers.traj_pid import traj_pid as pid_control
from .traj_controllers.traj_mpc import traj_mpc as mpc_control

from .create_path import create_path

# userdata layout (indices into data.userdata)
USERDATA_INIT_FLAG = 0   # 0.0 = not initialized, 1.0 = initialized
USERDATA_WP_INDEX  = 1   # current waypoint index (stored as float, cast to int)

# Parameters
_WAYPOINT_TOL = 0.40  # [m] distance at which a waypoint is reached
_MAX_DRV = 0.75       # clip commands
_MAX_TRN = 1.0

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

def _interpolate_path(path) :
    '''
    Interpolates a given path with n extra points between each waypoint
    
    :param path: list of tuples of (x,y) points
    :param n: number of evenly spaced points to add between each waypoint
    '''
    waypoints = np.array(path)
    if waypoints.shape[0] < 2:
        raise ValueError("Need at least two waypoints to interpolate a path.")
    
    segments = []
    for (x0, y0, z0), (x1, y1, z1) in zip(waypoints[:-1], waypoints[1:]):
        #n = round(np.sqrt((x1-x0)**2 + (y1-y0)**2) / 2) + 2            # interpolate based on distance b/t points
        n = 2                                                           # for now just return same points
        xs = np.linspace(x0, x1, n)
        ys = np.linspace(y0, y1, n)
        zs = np.linspace(z0, z1, n)
        seg = np.column_stack((xs, ys, zs))
        segments.append(seg)
    
    # Stack all segments into one array and delete repeated points
    path_interp = np.vstack([seg[:-1] for seg in segments] + [segments[-1][-1:]])
    
    return path_interp


# Helper: initialize userdata and path cache
def _init_if_needed(model, data, scene) :
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
        path = create_path(model, data, scene)
        path = np.asarray(path, dtype=float)

        if path.shape[1] != 3:
            raise ValueError(
                f"create_path must return waypoints with shape (N, 3); "
                f"got shape {path.shape}"
            )

        path_interp = _interpolate_path(path)
        _PATH_CACHE = path_interp

        # Initialize userdata state
        data.userdata[USERDATA_INIT_FLAG] = 1.0  # mark as initialized
        data.userdata[USERDATA_WP_INDEX] = 0.0   # waypoint index = 0


# Main controller: follow trajectory in _PATH_CACHE using userdata state
def traj_control(model, data, scene, time=None):
    """
    Trajectory-following controller. Modularity with either PID or
    MPC controller

    Arguments:
      model : mujoco.MjModel
      data  : mujoco.MjData
      scene : int for the scenario
      time  : (optional) current simulation time, unused here

    Returns:
      drive_ctrl  : np.ndarray of shape (2,), actuator commands:
                [0] drive_forward
                [1] drive_turn
    """
    global _PATH_CACHE

    _init_if_needed(model, data, scene)

    nu = 2  # number of actuators to control here (just drive and turn)

    # If path was not created for some reason, just stop
    if _PATH_CACHE is None or _PATH_CACHE.shape[0] == 0:
        return np.zeros(nu, dtype=float)

    # Read current waypoint index from userdata
    wp_idx = int(data.userdata[USERDATA_WP_INDEX])

    # If finished all waypoints, stop
    if wp_idx >= _PATH_CACHE.shape[0]:
        return np.zeros(nu, dtype=float)

    # Current robot pose
    x, y, yaw = _get_car_pose(model, data, body_name="car")

    # Current target waypoint
    tx, ty, _ = _PATH_CACHE[wp_idx]
    dx = tx - x
    dy = ty - y
    dist = np.hypot(dx, dy)

    # print(f"Way-point Target = ({tx},{ty})")
    # print(f"Current Position = ({x},{y})")

    # Check if we reached this waypoint
    if dist < _WAYPOINT_TOL:
        print(f"Waypoint {wp_idx} reached")
        wp_idx += 1
        data.userdata[USERDATA_WP_INDEX] = float(wp_idx)

        if wp_idx >= _PATH_CACHE.shape[0]:
            # Done with all waypoints
            return np.zeros(nu, dtype=float)

        # Update target to new waypoint
        tx, ty, _ = _PATH_CACHE[wp_idx]
        dx = tx - x
        dy = ty - y
        dist = np.hypot(dx, dy)

    v_cmd, w_cmd = pid_control(dx, dy, yaw)
    # v_cmd, w_cmd = mpc_control(model, data)

    # Map to actuators and clip
    forward_ctrl = np.clip(v_cmd, -_MAX_DRV, _MAX_DRV)
    turn_ctrl = np.clip(w_cmd, -_MAX_TRN, _MAX_TRN)

    # Build full control vector
    drive_ctrl = np.array([forward_ctrl, turn_ctrl])

    return drive_ctrl

import numpy as np

# Control gains and parameters
_K_V = 1.0            # gain on forward error
_K_W = 5.0            # gain on heading error

def traj_pid(dx, dy, yaw) -> tuple[float, float]: 
    '''
    This is a simple proportional controller that amplifies the error from target positions

    :param dx: distance to desired position in x
    :param dy: distance to desired position in y
    :param yaw: current yaw of model
    :return: control inputs for drive and turn commands
    :rtype: tuple[float, float]
    '''

    # Transform position error into body frame
    cos_yaw = np.cos(yaw)
    sin_yaw = np.sin(yaw)

    ex_body = cos_yaw * dx + sin_yaw * dy      # forward error
    #ey_body = -sin_yaw * dx + cos_yaw * dy     # lateral error (not used)

    # Desired heading to waypoint
    desired_heading = np.arctan2(dy, dx)

    # Heading error (wrapped to [-pi, pi])
    heading_error = np.arctan2(
        np.sin(desired_heading - yaw),
        np.cos(desired_heading - yaw),
    )
    
    # print(f"Desired Heading = {desired_heading}")
    # print(f"Current Heading = {yaw}")
    # print(f"heading error = {heading_error}")

    # Simple proportional controller for unicycle-like behavior
    v_cmd = max(0, _K_V * ex_body - _K_W * abs(heading_error))       # forward velocity command
    w_cmd = _K_W * heading_error                                # turn rate command
    # print(f"drive error = {ex_body}")
    # print(f"headi error = {heading_error}")
    return (v_cmd, w_cmd)
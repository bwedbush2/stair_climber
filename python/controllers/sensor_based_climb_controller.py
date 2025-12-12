import numpy as np
import mujoco

if not hasattr(mujoco, "climb_filter_state"):
    mujoco.climb_filter_state = {
        "filtered_target": 0.0,
        "last_valid_raw": 0.0,
        "first_run": True
    }


def get_sensor_value(model, data, sensor_name):
    id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, sensor_name)
    if id == -1: 
        return -1.0
    adr = model.sensor_adr[id]
    return data.sensordata[adr]

def climb_control(model, data):
    # PD Gains
    kp = 1 
    kd = 0.2
    
    # Filter and Slew settings
    ALPHA = 0.05
    DECAY_RATE = 0.90
    MAX_SLEW_RATE = 0.005

    SENSOR_Z_DIFF = 0.1 
    COS_45 = 0.7071
    SIN_45 = 0.7071


    d_low = get_sensor_value(model, data, "floor_sensL")
    d_high = get_sensor_value(model, data, "floor_sensU")
    state = mujoco.climb_filter_state

    raw_target = 0.0

    if d_low > 0 and d_high > 0:
        # Calculate slope angle from sensor diff
        delta_d = d_high - d_low
        rise = SENSOR_Z_DIFF - (delta_d * SIN_45)
        run = delta_d * COS_45
        angle = np.arctan2(rise, run)
        
        raw_target = -np.clip(angle, -1.0, 1.0)
        state["last_valid_raw"] = raw_target
    else:
        # Decay last known value if sensors miss
        state["last_valid_raw"] *= DECAY_RATE
        raw_target = state["last_valid_raw"]

    if state["first_run"]:
        state["filtered_target"] = raw_target
        state["first_run"] = False
    else:
        # Low-pass filter
        new_smoothed = (ALPHA * raw_target) + ((1 - ALPHA) * state["filtered_target"])
        
        # Slew rate limit to prevent vibration
        delta = new_smoothed - state["filtered_target"]
        clamped_delta = np.clip(delta, -MAX_SLEW_RATE, MAX_SLEW_RATE)
        state["filtered_target"] += clamped_delta
        
    final_target = state["filtered_target"]


    # Joint control
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "climb_L")
    if joint_id == -1: 
        return 0.0

    qpos_adr = model.jnt_qposadr[joint_id]
    qvel_adr = model.jnt_dofadr[joint_id]

    current_pos = data.qpos[qpos_adr]
    current_vel = data.qvel[qvel_adr]


    
    error = final_target - current_pos
    derivative = 0 - current_vel 
    control_signal = (kp * error) + (kd * derivative)

    return np.clip(control_signal, -1.0, 1.0)
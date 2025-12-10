import numpy as np
import mujoco

# --- STATE STORAGE ---
if not hasattr(mujoco, "climb_filter_state"):
    mujoco.climb_filter_state = {
        "filtered_target": 0.0,
        "last_valid_raw": 0.0, # Remembers the last good sensor reading
        "first_run": True
    }

def get_sensor_value(model, data, sensor_name):
    id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, sensor_name)
    if id == -1: return -1.0
    adr = model.sensor_adr[id]
    return data.sensordata[adr]

def climb_control(model, data):
    """
    Adjusts the rocker-bogie angle with Decay Logic and Slew Rate Limiting 
    to completely eliminate sensor jitter and microbouncing.
    """
    # --- 1. CONFIGURATION ---
    kp = 5 
    kd = 0.2
    
    # FILTER SETTINGS
    ALPHA = 0.05            # Low-pass smoothing factor
    DECAY_RATE = 0.90       # How fast to return to 0 if sensors miss (0.98 = slow decay)
    MAX_SLEW_RATE = 0.005   # Max radians the joint target can change per simulation step

    s_low_name = "floor_sensL"
    s_high_name = "floor_sensU"
    
    SENSOR_Z_DIFF = 0.1 
    COS_45 = 0.7071
    SIN_45 = 0.7071

    # --- 2. GET SENSOR DATA ---
    d_low = get_sensor_value(model, data, s_low_name)
    d_high = get_sensor_value(model, data, s_high_name)
    
    state = mujoco.climb_filter_state

    # --- 3. CALCULATE RAW TARGET (With Dropout Protection) ---
    raw_target = 0.0

    if d_low > 0 and d_high > 0:
        # Valid Reading: Calculate angle normally
        delta_d = d_high - d_low
        rise = SENSOR_Z_DIFF - (delta_d * SIN_45)
        run = delta_d * COS_45
        angle = np.arctan2(rise, run)
        
        # Invert for correct joint direction
        raw_target = -np.clip(angle, -1.0, 1.0)
        
        # Save this as a "known good" value
        state["last_valid_raw"] = raw_target
    else:
        # Invalid Reading (Sensor miss/infinity):
        # Instead of snapping to 0.0 (which causes violent shaking),
        # we slowly DECAY the last known good value towards 0.
        state["last_valid_raw"] *= DECAY_RATE
        raw_target = state["last_valid_raw"]

    # --- 4. LOW-PASS FILTER ---
    if state["first_run"]:
        state["filtered_target"] = raw_target
        state["first_run"] = False
    else:
        # Apply standard smoothing
        new_smoothed = (ALPHA * raw_target) + ((1 - ALPHA) * state["filtered_target"])
        
        # --- 5. SLEW RATE LIMITER (The Anti-Vibration Fix) ---
        # Calculate how much the target WANTS to change
        delta = new_smoothed - state["filtered_target"]
        
        # Clamp that change to the maximum allowed speed
        clamped_delta = np.clip(delta, -MAX_SLEW_RATE, MAX_SLEW_RATE)
        
        # Apply the clamped change
        state["filtered_target"] += clamped_delta
        
    final_target = state["filtered_target"]

    # --- 6. COMPUTE PD CONTROL ---
    joint_name = "climb_L"
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    if joint_id == -1: return 0.0

    qpos_adr = model.jnt_qposadr[joint_id]
    qvel_adr = model.jnt_dofadr[joint_id]

    current_pos = data.qpos[qpos_adr]
    current_vel = data.qvel[qvel_adr]

    error = final_target - current_pos
    derivative = 0 - current_vel 
    
    control_signal = (kp * error) + (kd * derivative)

    return np.clip(control_signal, -1.0, 1.0)
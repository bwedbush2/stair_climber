import numpy as np
import mujoco

# --- STATE STORAGE ---
# We attach a storage dictionary to the function itself.
# This allows us to remember the "previous" value without needing a class.
if not hasattr(mujoco, "climb_filter_state"):
    mujoco.climb_filter_state = {
        "filtered_target": 0.0,
        "first_run": True
    }

def get_sensor_value(model, data, sensor_name):
    id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, sensor_name)
    if id == -1: return -1.0
    adr = model.sensor_adr[id]
    return data.sensordata[adr]

def climb_control(model, data):
    """
    Adjusts the rocker-bogie angle with signal smoothing to prevent jitter.
    """
    # --- 1. CONFIGURATION ---
    kp = 3.5       # Slightly reduced stiffness
    kd = 0.15      # Increased damping
    
    # Smoothing Factor (0.0 to 1.0)
    # 0.05 = Very smooth, slow reaction
    # 0.50 = Less smooth, fast reaction
    # 1.00 = No smoothing (jittery)
    ALPHA = 0.1 

    s_low_name = "floor_sensL"
    s_high_name = "floor_sensU"
    
    SENSOR_Z_DIFF = 0.1 
    COS_45 = 0.7071
    SIN_45 = 0.7071

    # --- 2. GET SENSOR DATA ---
    d_low = get_sensor_value(model, data, s_low_name)
    d_high = get_sensor_value(model, data, s_high_name)
    
    # --- 3. CALCULATE RAW TARGET ---
    raw_target = 0.0

    if d_low > 0 and d_high > 0:
        delta_d = d_high - d_low
        
        # Geometric calculation
        rise = SENSOR_Z_DIFF - (delta_d * SIN_45)
        run = delta_d * COS_45
        
        # Use arctan2 for continuous angle calculation (handles vertical walls naturally)
        angle = np.arctan2(rise, run)
        
        # Invert and clamp (Positive Slope = Negative Joint Angle)
        raw_target = -np.clip(angle, -1.0, 1.0)

    # --- 4. APPLY SMOOTHING (Low-Pass Filter) ---
    # Retrieve previous smoothed value
    state = mujoco.climb_filter_state
    
    if state["first_run"]:
        state["filtered_target"] = raw_target
        state["first_run"] = False
    else:
        # Filter Equation: Output = (α * New) + ((1 - α) * Old)
        state["filtered_target"] = (ALPHA * raw_target) + ((1 - ALPHA) * state["filtered_target"])
        
    final_target = state["filtered_target"]

    # --- 5. COMPUTE PD CONTROL ---
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
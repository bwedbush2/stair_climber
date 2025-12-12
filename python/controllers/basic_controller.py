import numpy as np
import mujoco

def climb_control(model, data):
    # Params
    target_pos = 1.0
    kp = 0.5
    kd = 0.005
    joint_name = "climb_L" 
    
    # Get joint ID and addy
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    if joint_id == -1:
        return 0.0

    qpos_adr = model.jnt_qposadr[joint_id]
    qvel_adr = model.jnt_dofadr[joint_id]

    current_pos = data.qpos[qpos_adr]
    current_vel = data.qvel[qvel_adr]

    # PD Calculation
    error = target_pos - current_pos
    derivative = 0 - current_vel
    
    control_signal = (kp * error) + (kd * derivative)

    return np.clip(control_signal, -1.0, 1.0)
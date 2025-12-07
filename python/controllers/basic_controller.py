import numpy as np
import mujoco

def climb_control(model, data):
    """
    Computes a control signal using PD logic.
    
    Args:
        model, data: MuJoCo objects
        target_pos (float): The desired joint angle or position.
        kp (float): Proportional gain (stiffness).
        kd (float): Derivative gain (damping).
        
    Returns:
        float: Control signal clipped between -1 and 1.
    """
    target_pos=1.0
    kp=0.5
    kd=0.005
    
    # --- 1. CONFIGURATION ---
    joint_name = "climb_L" 
    
    # --- 2. GET JOINT STATE ---
    # Get the ID of the joint
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    
    if joint_id == -1:
        print(f"Error: Joint '{joint_name}' not found.")
        return 0.0

    # Get the address in the qpos/qvel arrays
    # (qpos_adr is necessary because some joints have multiple degrees of freedom)
    qpos_adr = model.jnt_qposadr[joint_id]
    qvel_adr = model.jnt_dofadr[joint_id]

    # Read current position and velocity
    current_pos = data.qpos[qpos_adr]
    current_vel = data.qvel[qvel_adr]

    # --- 3. COMPUTE PD CONTROL ---
    # Equation: u = Kp * (target - current) + Kd * (0 - velocity)
    # Note: Target velocity is usually 0 for holding a position
    error = target_pos - current_pos
    derivative = 0 - current_vel # We want 0 velocity at the target
    
    control_signal = (kp * error) + (kd * derivative)

    # --- 4. CLAMP OUTPUT ---
    # Restrict value to hardware limits [-1, 1]
    climb_val = np.clip(control_signal, -1.0, 1.0)
    
    return climb_val
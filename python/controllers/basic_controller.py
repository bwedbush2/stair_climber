import numpy as np
import mujoco
import os 

def climb_control(model, data):
    # """
    # Sets the control signal for a specific actuator to 0.5.
    # """
    # # 1. Find the ID of the actuator by its name in the XML
    # # If you don't use names, you can hardcode the index (e.g., data.ctrl[0])
    # actuator_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_name)
    
    # # 2. Check if the actuator was found (returns -1 if not found)
    # if actuator_id != -1:
    #     # 3. Apply the control value
    #     data.ctrl[actuator_id] = 0.5
    # else:
    #     print(f"Warning: Actuator '{actuator_name}' not found.")
    climb_val = 0.5
    return climb_val 
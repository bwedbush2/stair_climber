import time
import mujoco
import mujoco.viewer
import numpy as np
import os 
import sys 
# Bin Control Offset
BIN_OFFSET = 0.0 

def get_chassis_pitch(data, model):
    """Reads IMU quaternion and converts to Pitch angle (radians)"""
    try:
        sens_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, "sensor_chassis_quat")
        adr = model.sensor_adr[sens_id]
        q = data.sensordata[adr:adr+4] 
        sinp = 2 * (q[0] * q[2] - q[3] * q[1])
        if abs(sinp) >= 1: pitch = np.sign(sinp) * (np.pi / 2)
        else: pitch = np.arcsin(sinp)
        return pitch
    except:
        return 0.0

def main():
    print("Loading simulation...")
    # --- CROSS-PLATFORM PATH SETUP ---
# 1. Get the absolute path to the folder containing THIS script
#    (Assumes this script is inside: python/controllers/)
script_dir = os.path.dirname(os.path.abspath(__file__))

# 2. Construct the path to the XML file
#    Structure:
#      root/
#      ├── mujoco/
#      │    └── ray_simulation.xml
#      └── python/
#           └── controllers/
#                └── your_script.py  <-- We are here
#
#    Logic: Go UP two levels (".." -> python, ".." -> root), then DOWN into "mujoco"
xml_folder = os.path.join(script_dir, "..", "..", "mujoco")
XML_PATH = os.path.join(xml_folder, "ray_simulation.xml")

# 3. Clean the path (resolves '..' and fixes slashes for Windows/Mac)
XML_PATH = os.path.normpath(XML_PATH)

# --- Verification Print (Optional) ---
if not os.path.exists(XML_PATH):
    print(f"Error: XML not found at {XML_PATH}")
else:
    print(f"XML found: {XML_PATH}")
    model = mujoco.MjModel.from_xml_path(XML_PATH)
    data = mujoco.MjData(model)

    print("\n=======================================")
    print("🎛️  UI SLIDER CONTROL ACTIVE")
    print("---------------------------------------")
    print("1. Expand the 'Controls' menu on the right.")
    print("2. Drag the sliders to drive/climb.")
    print("3. The BIN will still balance AUTOMATICALLY.")
    print("=======================================")

    # Launch viewer (No keyboard callbacks needed)
    with mujoco.viewer.launch_passive(model, data) as viewer:
        
        # Expand the control panel by default
        viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONVEXHULL] = 0 # Optional: Clean up visual
        
        while viewer.is_running():
            step_start = time.time()

            # --- 1. AUTOMATIC BIN CONTROL ---
            # We ONLY write to Actuator 3 (Bin). 
            # We leave Actuators 0, 1, and 2 alone so the UI can control them.
            chassis_pitch = get_chassis_pitch(data, model)
            bin_target = np.clip(-chassis_pitch + BIN_OFFSET, -1.0, 1.0)
            data.ctrl[3] = bin_target

            # --- 2. PHYSICS STEP ---
            mujoco.mj_step(model, data)
            viewer.sync()

            # Keep real-time
            time_until_next_step = model.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

if __name__ == "__main__":
    main()
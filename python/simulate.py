import mujoco
import mujoco.viewer
import numpy as np
import time
import os
import sys
import traj_planning.traj_control
from traj_planning.traj_control import traj_control

# --- CROSS-PLATFORM PATH SETUP ---
# 1. Get the absolute path to the folder containing this script
script_dir = os.path.dirname(os.path.abspath(__file__))

# 2. Construct the path to the XML file safely
#    Windows will automatically use '\', Mac/Linux will use '/'
#    Structure:  root/
#                  ├── mujoco/
#                  │     └── ray_simulation.xml
#                  └── python/
#                        └── simulate.py (this file)
xml_folder = os.path.join(script_dir, "..", "mujoco")
XML_PATH = os.path.join(xml_folder, "ray_simulation.xml")

# 3. Clean the path (resolves '..' and fixes slashes for specific OS)
XML_PATH = os.path.normpath(XML_PATH)

def get_body_pitch(model, data, body_name):
    """Helper to get the pitch angle (y-axis rotation)."""
    # Safety check: ensure body exists
    try:
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    except Exception:
        return 0.0 # Return 0 if body not found to prevent crash

    w, x, y, z = data.xquat[body_id]
    # Quaternion to Pitch conversion
    pitch = np.arcsin(2 * (w * y - z * x))
    return pitch

def controller(model, data):
    """
    Controller for RAY_Scenario_3
    """
    # Lookup IDs (Doing this inside the loop is slower but safer for dev)
    # For high performance, move these lookups to 'run_simulation' init
    id_drive = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "drive_forward")
    id_turn = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "drive_turn")
    id_climb = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "actuator_climb")
    id_level = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "level_bin")

    # 1. Drive Forward constant
    drive, turn = traj_control(model,data)
    data.ctrl[id_drive] = drive
    data.ctrl[id_turn] = turn
    # 2. Steering (Slight sine wave)
    #data.ctrl[id_turn] = 0.1 * np.sin(data.time)

    # 3. Lift Bogies
    data.ctrl[id_climb] = 0

    # 4. Active Leveling
    chassis_pitch = get_body_pitch(model, data, "car")
    data.ctrl[id_level] = -chassis_pitch 

def run_simulation():
    # --- OS DIAGNOSTICS ---
    print(f"Operating System: {sys.platform}")
    print(f"Script Location:  {script_dir}")
    print(f"Looking for XML:  {XML_PATH}")

    # Verify file exists before asking MuJoCo to load it
    if not os.path.exists(XML_PATH):
        print("\n" + "="*40)
        print("CRITICAL ERROR: XML FILE NOT FOUND")
        print("="*40)
        print(f"The script expected the file here:\n{XML_PATH}")
        print("\nPlease ensure your folder structure is:")
        print("  /ParentFolder")
        print("     /mujoco")
        print("         ray_simulation.xml")
        print("     /python")
        print("         simulate.py")
        return

    try:
        model = mujoco.MjModel.from_xml_path(XML_PATH)
    except ValueError as e:
        print(f"MuJoCo Load Error: {e}")
        return
    data = mujoco.MjData(model)

    # Waypoints Setup
    id_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'car')
    start_pos = model.body_pos[id_body]
    WAYPOINTS = [
            start_pos,
            (5.0, 0.0, 1),      # Ramp bottom
            (5.0, -3.5, 1),     # Curb approach
            (4.0, -3.5, 1),     # Curb top
            (4.0, -7, 1),       # Path
            (4.0, -10, 1),      # Stairs
            (4.0, -12, 1),      # Porch
    ]

    with mujoco.viewer.launch_passive(model, data) as viewer:
        print("\nSimulation Started.")
        
        # Initial Camera
        viewer.cam.distance = 6.0
        viewer.cam.lookat[:] = [3.0, -5.0, 1.0]
        
        # TIMING VARIABLES
        dt = model.opt.timestep       # Physics timestep (usually 0.002)
        target_framerate = 30         # Hz (Render speed)
        render_interval = 1.0 / target_framerate
        last_render_time = 0.0

        while viewer.is_running():
            step_start = time.time()

            # 1. CONTROLLER & PHYSICS
            # Run physics!
            controller(model, data)
            mujoco.mj_step(model, data)

            # 2. RENDER (Only run this if enough time has passed)
            if data.time - last_render_time >= render_interval:
                
                # Update markers only when we render
                with viewer.lock(): 
                    # Draw Waypoint Lines
                    for i in range(len(WAYPOINTS) - 1):
                        if viewer.user_scn.ngeom >= viewer.user_scn.maxgeom: break
                        mujoco.mjv_connector(
                            viewer.user_scn.geoms[viewer.user_scn.ngeom],
                            type=mujoco.mjtGeom.mjGEOM_CAPSULE,
                            width=0.05,
                            from_=np.array(WAYPOINTS[i]),
                            to=np.array(WAYPOINTS[i+1])
                        )
                        viewer.user_scn.geoms[viewer.user_scn.ngeom].rgba = np.array([1, 1, 0, 1])
                        viewer.user_scn.ngeom += 1
                
                # Sync Viewer
                viewer.sync()
                last_render_time = data.time
            # 4. Time keeping
            time_until_next_step = model.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

if __name__ == "__main__":
    run_simulation()
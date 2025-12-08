import mujoco
import mujoco.viewer
import numpy as np
import time
import os
import sys

# ==========================================
# 🔌 CONTROL IMPORTS
# ==========================================

# 1. Trajectory Control Import
#    Located in: python/traj_planning/traj_control.py
try:
    from traj_planning.traj_control import traj_control as current_traj_control
    from traj_planning.create_path import create_path
except ImportError:
    print("⚠️ Warning: Could not import trajectory controller file")

# 2. Climb Control Import
try:
    from controllers.basic_controller import climb_control as current_climb_control
except ImportError:
    print("⚠️ Warning: Could not import controllers file.")

# ==========================================
# ⚙️ GLOBAL SETTINGS
# ==========================================
# These flags will be set by the user input at runtime
USE_TRAJECTORY_CONTROL = True
USE_CLIMB_CONTROL = True

# ==========================================
# 📂 PATH SETUP
# ==========================================
# 1. Get the absolute path to the folder containing this script
script_dir = os.path.dirname(os.path.abspath(__file__))

# 2. Construct the path to the XML file safely
xml_folder = os.path.join(script_dir, "..", "mujoco")
XML_PATH = os.path.join(xml_folder, "ray_simulation.xml")

# 3. Clean the path (resolves '..' and fixes slashes for specific OS)
XML_PATH = os.path.normpath(XML_PATH)

# ==========================================
# 🤖 ROBOT DEFINITION & SCENARIOS
# ==========================================
ROBOT_XML = """
    <body name="car" pos="{START_POS}"> 
      <freejoint/>
      <inertial pos="0.1 0 0" mass="10" diaginertia="0.5 0.5 0.8"/>

      <geom name="chassis_visual" type="mesh" mesh="chassis" pos="-0.405 0.275 -0.05" euler="90 0 0" 
            density="0" contype="0" conaffinity="0" group="1" rgba="1 1 1 0.5"/>
      <geom name="belly" type="box" size=".3 .15 .04" pos="0 0 0.05" rgba=".8 .2 .2 1"/>

      <site name="sens_chassis" pos="0 0 0.2" size=".02" rgba="1 0 0 1"/>

      <body name="leveling_base" pos="-0.1 0 0.15">
        <joint name="bin_pitch" axis="0 1 0" damping="10.0" range="-60 60"/>
        <geom type="cylinder" size=".03 .12" euler="90 0 0" rgba=".2 .2 .2 1"/>
        <geom name="platform_flat" type="box" size=".14 .14 .01" pos="0 0 0.04" rgba=".3 .3 .3 1"/>
        
        <body name="the_bin" pos="0 0 0.05">
            <inertial pos="0 0 0.1" mass="8" diaginertia="0.1 0.1 0.1"/>
            <geom name="bin_visual" type="box" size=".13 .13 .12" pos="0 0 0.12" rgba="1 .6 0 1"/>
        </body>
      </body>

    <body name="left_bogie" pos="0.25 .25 0">
        <joint name="climb_L" axis="0 1 0" damping="50.0"/>
        
        <site name="sens_bogie" pos="0.1 0 0" size=".02" rgba="0 1 0 1"/>

        <geom type="box" size=".15 .04 .01" pos="0 .03 0" rgba="0.5 0.5 0.5 1"/>
        <geom class="skid_plate" fromto=".13 0 0 -.13 0 0"/>
        <body name="L_Front" pos=".13 0 0"> <joint name="L1" class="wheel"/> <geom class="wheel"/> </body>
        <body name="L_Mid" pos="-.13 0 0"> <joint name="L2" class="wheel"/> <geom class="wheel"/> </body>
      </body>
      <body name="left_wheel_rear" pos="-0.3 0.25 0"> <joint name="L3" class="wheel"/> <geom class="wheel"/> </body>
    <body name="right_bogie" pos="0.25 -.25 0">
        <joint name="climb_R" axis="0 1 0" damping="50.0"/>
        <geom type="box" size=".15 .04 .01" pos="0 -.03 0" rgba="0.5 0.5 0.5 1"/>
        <geom class="skid_plate" fromto=".13 0 0 -.13 0 0"/>
        <body name="R_Front" pos=".13 0 0"> <joint name="R1" class="wheel"/> <geom class="wheel"/> </body>
        <body name="R_Mid" pos="-.13 0 0"> <joint name="R2" class="wheel"/> <geom class="wheel"/> </body>
      </body>
      <body name="right_wheel_rear" pos="-0.3 -.25 0"> <joint name="R3" class="wheel"/> <geom class="wheel"/> </body>
    </body>
"""

def get_scenario_1():
    print("\n--- Generating Scenario 1: Simple Curb ---")
    return """
    <geom name="floor" type="plane" size="5 5 .1" material="grid"/>
    <body name="curb" pos="2 0 0.075">
        <geom type="box" size="1 2 0.075" material="concrete" rgba="0.5 0.5 0.5 1"/>
    </body>
    """, "0 0 0.2"

def get_scenario_2():
    print("\n--- Generating Scenario 2: Navigation Target ---")
    return """
    <geom name="floor" type="plane" size="10 10 .1" material="grid"/>
    <geom name="wall" type="box" size="0.2 2 0.5" pos="3 0 0.5" rgba="0.8 0.2 0.2 1"/>
    <body name="target1" pos="3 3 0">
        <geom type="sphere" size="0.2" rgba="0 1 0 0.5" contype="0" conaffinity="0"/>
        <light pos="0 0 2" dir="0 0 -1" diffuse="0 1 0"/>
    </body>
    <body name="target2" pos="6 0 0">
        <geom type="sphere" size="0.2" rgba="0 1 0 0.5" contype="0" conaffinity="0"/>
        <light pos="0 0 2" dir="0 0 -1" diffuse="0 1 0"/>
    </body>
    """, "0 0 0.2"

def get_scenario_3():
    print("\n--- Generating Scenario 3: Stair Pyramid ---")
    return """
    <geom name="floor" type="plane" size="10 10 .1" material="grid"/>
    <body name="stair_pyramid" pos="2 0 0">
        <geom name="u1" type="box" size="1 1 .05" pos="0 0 0.05" material="concrete"/>
        <geom name="u2" type="box" size="1 1 .05" pos="0.3 0 0.15" material="concrete"/>
        <geom name="u3" type="box" size="1 1 .05" pos="0.6 0 0.25" material="concrete"/>
        <geom name="u4" type="box" size="1 1 .05" pos="0.9 0 0.35" material="concrete"/>
        <geom name="plat" type="box" size="1 1 .05" pos="1.5 0 0.45" material="concrete"/>
        <geom name="d1" type="box" size="1 1 .05" pos="2.1 0 0.35" material="concrete"/>
        <geom name="d2" type="box" size="1 1 .05" pos="2.4 0 0.25" material="concrete"/>
        <geom name="d3" type="box" size="1 1 .05" pos="2.7 0 0.15" material="concrete"/>
        <geom name="d4" type="box" size="1 1 .05" pos="3.0 0 0.05" material="concrete"/>
    </body>
    """, "0 0 0.2"

def get_scenario_4():
    print("\n--- Generating Scenario 4: Full Mission ---")
    stairs_xml = ""
    for i in range(10):
        stairs_xml += f'<geom name="s{i}" type="box" size="1 .2 .1" pos="4 {-7.7 - (i*0.4)} {0.1 + (i*0.1)}" material="concrete"/>\n'

    return """
    <geom name="street" type="plane" size="15 15 .1" material="asphalt"/>
    <body name="van" pos="0 0 0.6">
        <geom type="box" size="1.5 1 0.05" material="metal"/>
        <geom type="box" size="1.5 .05 0.5" pos="0 1.05 0.5" material="metal" rgba=".3 .4 .5 0.5"/>
        <geom type="box" size="1.5 .05 0.5" pos="0 -1.05 0.5" material="metal" rgba=".3 .4 .5 0.5"/>
        <geom type="box" size=".05 1 0.5" pos="-1.55 0 0.5" material="metal" rgba=".3 .4 .5 0.5"/>
    </body>
    <body name="ramp" pos="3.0 0 0.32">
        <geom type="box" size="1.6 1 0.05" euler="0 12 0" material="ramp_surface" friction="2.0 0.005 0.0001"/>
    </body>
    <body name="sidewalk" pos="4 -3.5 0.075"> 
        <geom type="box" size="5 1.5 0.075" material="concrete"/>
    </body>
    <body name="path" pos="4 -6 0.075">
        <geom type="box" size="1 1.5 0.075" material="concrete"/>
    </body>
    """ + stairs_xml + """
    <body name="porch" pos="4 -12 1.1">
        <geom type="box" size="2 1.2 0.1" material="wood"/>
        <geom type="box" size="1 .1 1.5" pos="0 -1.2 1.5" rgba=".4 .2 .1 1"/>
    </body>
    """, "0 0 0.8"

# ==========================================
# 🛠️ XML BUILDER
# ==========================================

def build_xml(scenario_id):
    world_xml = ""
    start_pos = "0 0 0.2"
    
    if scenario_id == 1:
        world_xml, start_pos = get_scenario_1()
    elif scenario_id == 2:
        world_xml, start_pos = get_scenario_2()
    elif scenario_id == 3:
        world_xml, start_pos = get_scenario_3()
    elif scenario_id == 4:
        world_xml, start_pos = get_scenario_4()
    else:
        print("Invalid ID. Exiting.")
        return False

    full_xml = f"""
<mujoco model="RAY_Scenario_{scenario_id}">
  <compiler autolimits="true"/>
  <asset>
    <texture name="grid" type="2d" builtin="checker" width="512" height="512" rgb1=".1 .2 .3" rgb2=".2 .3 .4"/>
    <material name="grid" texture="grid" texrepeat="1 1" texuniform="true" reflectance=".2"/>
    <texture name="concrete" type="2d" builtin="flat" rgb1=".7 .7 .7" width="512" height="512"/>
    <material name="concrete" texture="concrete" reflectance=".1"/>
    <texture name="metal" type="2d" builtin="flat" rgb1=".4 .5 .6" width="512" height="512"/>
    <material name="metal" texture="metal" reflectance=".5"/>
    <texture name="asphalt" type="2d" builtin="flat" rgb1=".2 .2 .2" width="512" height="512"/>
    <material name="asphalt" texture="asphalt" reflectance=".1"/>
    <texture name="grip_tape" type="2d" builtin="checker" width="512" height="512" rgb1=".1 .1 .1" rgb2=".2 .2 .2"/>
    <material name="ramp_surface" texture="grip_tape" texrepeat="2 2" reflectance="0"/>
    <texture name="wood" type="2d" builtin="flat" rgb1=".6 .4 .3" width="512" height="512"/>
    <material name="wood" texture="wood" reflectance=".1"/>
    <mesh name="chassis" file="assets/chassis.stl" scale="0.001 0.001 0.001"/>
  </asset>

  <default>
    <joint damping="0.5"/>
    <default class="wheel">
      <geom type="cylinder" size=".06 .05" rgba="0.2 0.2 0.2 1" euler="90 0 0" condim="3" friction="2.5 0.005 0.0001"/>
      <joint type="hinge" axis="0 1 0"/>
    </default>
    <default class="skid_plate">
        <geom type="capsule" size=".055" rgba="0.3 0.3 0.3 1" condim="3" friction="0.8 0.005 0.0001"/>
    </default>
  </default>

  <worldbody>
    <light pos="0 0 10" dir="0 0 -1" directional="true"/>
    {world_xml}
    {ROBOT_XML.format(START_POS=start_pos)}
  </worldbody>

  <tendon>
    <fixed name="forward_combined">
        <joint joint="L1" coef="0.5"/> <joint joint="L2" coef="0.5"/> <joint joint="L3" coef="0.5"/>
        <joint joint="R1" coef="0.5"/> <joint joint="R2" coef="0.5"/> <joint joint="R3" coef="0.5"/>
    </fixed>
    <fixed name="turn_combined">
        <joint joint="L1" coef="-0.5"/> <joint joint="L2" coef="-0.5"/> <joint joint="L3" coef="-0.5"/>
        <joint joint="R1" coef="0.5"/> <joint joint="R2" coef="0.5"/> <joint joint="R3" coef="0.5"/>
    </fixed>
    
    <fixed name="climb_axle">
      <joint joint="climb_L" coef="1"/> <joint joint="climb_R" coef="1"/>
    </fixed>
  </tendon>
  <equality>
      <joint joint1="climb_L" joint2="climb_R" polycoef="0 1 0 0 0" solimp="0.99 0.99 0.01" solref="0.005 1"/>
  </equality>
  <actuator>
    <motor name="drive_forward" tendon="forward_combined" ctrlrange="-1 1" gear="40"/>
    <motor name="drive_turn" tendon="turn_combined" ctrlrange="-1 1" gear="40"/>
    
    <position name="actuator_climb" joint="climb_L" ctrlrange="-1.5 1.5" kp="10000"/> 
    
    <position name="level_bin" joint="bin_pitch" ctrlrange="-1 1" kp="500"/>
  </actuator>
  
  <sensor>
    <jointpos name="sensor_bin_angle" joint="bin_pitch"/>
    
    <actuatorpos name="sensor_bogie_angle" actuator="actuator_climb"/>

    <framequat name="sensor_chassis_quat" objtype="site" objname="sens_chassis"/>
    <gyro name="sensor_chassis_gyro" site="sens_chassis"/>
  </sensor>
<size nuserdata='2'/>

</mujoco>
    """

    os.makedirs(xml_folder, exist_ok=True)
    try:
        with open(XML_PATH, "w") as f:
            f.write(full_xml)
        print(f"✅ Generated XML at: {XML_PATH}")
        return True
    except Exception as e:
        print(f"❌ Error writing XML file: {e}")
        return False

# ==========================================
# 🎮 SIMULATION CONTROL
# ==========================================

def get_body_pitch(model, data, body_name):
    """Helper to get the pitch angle (y-axis rotation)."""
    try:
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    except Exception:
        return 0.0 

    w, x, y, z = data.xquat[body_id]
    pitch = np.arcsin(2 * (w * y - z * x))
    return pitch

def controller(model, data):
    """
    Controller logic
    """
    id_drive = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "drive_forward")
    id_turn = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "drive_turn")
    id_climb = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "actuator_climb")
    id_level = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "level_bin")

    # 1. Drive & Turn (Only if enabled)
    if USE_TRAJECTORY_CONTROL:
        drive, turn = current_traj_control(model, data)
        data.ctrl[id_drive] = drive
        data.ctrl[id_turn] = turn
    # else:
    #     # Stop motors if control disabled
    #     data.ctrl[id_drive] = 0.0
    #     data.ctrl[id_turn] = 0.0

    # 2. Climb Control (Only if enabled)
    if USE_CLIMB_CONTROL:
        climb_val = current_climb_control(model, data)
        data.ctrl[id_climb] = climb_val
    # else:
    #     # Default behavior (e.g., hold 0)
    #     data.ctrl[id_climb] = 0.0

    # 3. Active Leveling (Always on)
    chassis_pitch = get_body_pitch(model, data, "car")
    data.ctrl[id_level] = -chassis_pitch 

def run_simulation(scene: int):
    print(f"\n🚀 Loading Simulation from: {XML_PATH}")

    if not os.path.exists(XML_PATH):
        print("CRITICAL ERROR: XML FILE NOT FOUND")
        return

    try:
        model = mujoco.MjModel.from_xml_path(XML_PATH)
    except ValueError as e:
        print(f"MuJoCo Load Error: {e}")
        return

    data = mujoco.MjData(model)

    # Waypoints Setup (for visualization)    
    WAYPOINTS = create_path(model, data, scene)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        print("\nSimulation Started. Close the window to stop.")
        
        # Initial Camera Setup
        viewer.cam.distance = 6.0
        viewer.cam.lookat[:] = [3.0, -5.0, 1.0]
        
        # TIMING VARIABLES
        dt = model.opt.timestep
        target_framerate = 30
        render_interval = 1.0 / target_framerate
        last_render_time = 0.0
        
        while viewer.is_running():
            step_start = time.time()
            
            # 1. Controller & Physics
            controller(model, data)
            mujoco.mj_step(model, data)
            
            # 2. Render Loop
            if data.time - last_render_time >= render_interval:
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
                
                viewer.sync()
                last_render_time = data.time

            # 3. Time keeping
            time_until_next_step = model.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

# ==========================================
# 🚀 MAIN EXECUTION
# ==========================================

if __name__ == "__main__":
    print("=========================================")
    print("      RAY ROBOT SIMULATION RUNNER        ")
    print("=========================================")
    print("1. Simple Curb (Sanity Check)")
    print("2. Navigation Target (Steering)")
    print("3. Stair Pyramid (Climb & Descend)")
    print("4. Full Mission (Truck -> Porch)")
    print("=========================================")
    
    # 1. Select Scenario
    valid_choice = False
    choice = 0
    while not valid_choice:
        try:
            selection = input("\nSelect Scenario (1-4): ")
            choice = int(selection)
            if 1 <= choice <= 4:
                build_success = build_xml(choice)
                if build_success:
                    valid_choice = True
            else:
                print("❌ Please enter a number between 1 and 4.")
        except ValueError:
            print("❌ Invalid input. Please enter a number.")

    # 2. Select Control Modes
    print("\n-----------------------------------------")
    print("CONTROL SETTINGS")
    print("-----------------------------------------")
    
    # Trajectory Control
    valid_traj = False
    while not valid_traj:
        user_in = input("Enable Trajectory Control? (y/n): ").strip().lower()
        if user_in == 'y':
            USE_TRAJECTORY_CONTROL = True
            valid_traj = True
        elif user_in == 'n':
            USE_TRAJECTORY_CONTROL = False
            valid_traj = True
        else:
            print("Please enter 'y' or 'n'.")

    # Climb Control
    valid_climb = False
    while not valid_climb:
        user_in = input("Enable Climb Control? (y/n): ").strip().lower()
        if user_in == 'y':
            USE_CLIMB_CONTROL = True
            valid_climb = True
        elif user_in == 'n':
            USE_CLIMB_CONTROL = False
            valid_climb = True
        else:
            print("Please enter 'y' or 'n'.")

    # 3. Launch
    run_simulation(choice)
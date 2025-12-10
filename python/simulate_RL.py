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
try:
    from traj_planning.traj_control import traj_control as current_traj_control
    from traj_planning.create_path import create_path
except ImportError:
    print("⚠️ Warning: Could not import trajectory controller file. Using dummy.")
    current_traj_control = lambda m, d, s: (0.0, 0.0)
    create_path = lambda m, d, s: []

# 2. Climb Control Import (Standard)
try:
    from controllers.climbing_controller import climb_control as current_climb_control
except ImportError:
    pass

# ### --- RL INTEGRATION START --- ###
# 3. RL Model Import (Stable Baselines3)
try:
    from stable_baselines3 import PPO
    RL_AVAILABLE = True
except ImportError:
    print("⚠️ Warning: Stable-Baselines3 not found. RL control disabled.")
    RL_AVAILABLE = False
# ### --- RL INTEGRATION END --- ###

# ==========================================
# ⚙️ GLOBAL SETTINGS
# ==========================================
USE_TRAJECTORY_CONTROL = True
USE_CLIMB_CONTROL = True
USE_RL_FOR_CLIMB = False # Will toggle this in main()

# ==========================================
# 📂 PATH SETUP
# ==========================================
script_dir = os.path.dirname(os.path.abspath(__file__))
xml_folder = os.path.join(script_dir, "..", "mujoco")
XML_PATH = os.path.join(xml_folder, "ray_simulation.xml")
XML_PATH = os.path.normpath(XML_PATH)

# Path to your trained model
MODEL_PATH = os.path.join(script_dir, "Trained Climbing Models", "Version 1 (Stable)") # Assumes the .zip file is in the same folder

# ==========================================
# 🤖 ROBOT DEFINITION & SCENARIOS
# ==========================================
# (Keeping your exact XML definitions)
ROBOT_XML = """
    <body name="car" pos="{START_POS}"> 
      <freejoint/>
      <inertial pos="0.1 0 0" mass="10" diaginertia="0.5 0.5 0.8"/>

      <geom name="chassis_visual" type="mesh" mesh="chassis" pos="-0.405 0.275 -0.05" euler="90 0 0" 
            density="0" contype="0" conaffinity="0" group="1" rgba="1 1 1 0.5"/>
      <geom name="belly" type="box" size=".3 .15 .04" pos="0 0 0.05" rgba=".8 .2 .2 1"/>

      <site name="sens_chassis" pos="0 0 0.2" size=".02" rgba="1 0 0 1"/>

      <body name="laser_array" pos="0.25 0 0">
        <site name="laser_1_site" pos="0.2 0 0.05" euler="0 135 0" size=".01" rgba="1 0 0 1"/>
        <site name="laser_2_site" pos="0.2 0 0.15" euler="0 135 0" size=".01" rgba="1 0 0 1"/>
        <site name="laser_3_site" pos="0 0 0.35" euler="0 90 0" size=".01" rgba="1 0 0 1"/>
      </body>
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
    <position name="actuator_climb" joint="climb_L" ctrlrange="-1.5 1.5" kp="3000" forcerange="-400 400"/>
    
    <position name="level_bin" joint="bin_pitch" ctrlrange="-1 1" kp="500"/>
  </actuator>
  
  <sensor>
    <jointpos name="sensor_bin_angle" joint="bin_pitch"/>
    
    <actuatorpos name="sensor_bogie_angle" actuator="actuator_climb"/>

    <framequat name="sensor_chassis_quat" objtype="site" objname="sens_chassis"/>
    <gyro name="sensor_chassis_gyro" site="sens_chassis"/>

    <rangefinder name="floor_sensL" site="laser_1_site"/>
    <rangefinder name="floor_sensU" site="laser_2_site"/>
    <rangefinder name="wall_sens" site="laser_3_site"/>
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
    

def draw_laser_beams(viewer, model, data):
    """
    Draws visible lines for the rangefinders.
    """
    for i in range(1, 4):
        sensor_name = f"laser_{i}" if i <= 3 else "wall_sens" # Handle naming variation
        # Actually easier to use the explicit names in our XML:
        if i == 1: s_name = "floor_sensL"; site_name="laser_1_site"
        elif i == 2: s_name = "floor_sensU"; site_name="laser_2_site"
        elif i == 3: s_name = "wall_sens"; site_name="laser_3_site"

        try:
            sens_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, s_name)
            site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
            
            start_pos = data.site_xpos[site_id]
            mat = data.site_xmat[site_id].reshape(3, 3)
            
            adr = model.sensor_adr[sens_id]
            dist = data.sensordata[adr]
            
            if dist < 0: dist = 2.0 
            
            direction = mat[:, 2] 
            end_pos = start_pos + (direction * dist)
            
            if viewer.user_scn.ngeom < viewer.user_scn.maxgeom:
                mujoco.mjv_connector(
                    viewer.user_scn.geoms[viewer.user_scn.ngeom],
                    type=mujoco.mjtGeom.mjGEOM_CAPSULE,
                    width=0.01,
                    from_=start_pos,
                    to=end_pos
                )
                if dist < 0.2:
                    viewer.user_scn.geoms[viewer.user_scn.ngeom].rgba = np.array([1, 0, 0, 1])
                else:
                    viewer.user_scn.geoms[viewer.user_scn.ngeom].rgba = np.array([0, 1, 0, 0.5])
                    
                viewer.user_scn.ngeom += 1
                
                if viewer.user_scn.ngeom < viewer.user_scn.maxgeom:
                    mujoco.mjv_initGeom(
                        viewer.user_scn.geoms[viewer.user_scn.ngeom],
                        type=mujoco.mjtGeom.mjGEOM_SPHERE,
                        size=[0.02, 0, 0],
                        pos=end_pos,
                        mat=np.eye(3).flatten(),
                        rgba=np.array([1, 0, 0, 1])
                    )
                    viewer.user_scn.ngeom += 1
        except Exception as e:
            pass 
        
        
# ==========================================
# 🎮 CONTROLLER & HELPERS
# ==========================================

def get_sensor_value(model, data, sensor_name):
    try:
        sens_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, sensor_name)
        adr = model.sensor_adr[sens_id]
        return data.sensordata[adr]
    except Exception:
        return -1.0
    
def get_body_pitch(model, data, body_name):
    try:
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    except Exception:
        return 0.0 

    w, x, y, z = data.xquat[body_id]
    pitch = np.arcsin(2 * (w * y - z * x))
    return pitch

# ### --- RL OBSERVATION BUILDER --- ###
def get_rl_observation(model, data):
    """Reconstructs the 7-value array expected by the trained model"""
    # 1. Pitch & Roll
    q = data.qpos[3:7] 
    pitch = np.arcsin(np.clip(2 * (q[0]*q[2] - q[3]*q[1]), -1, 1))
    roll  = np.arctan2(2*(q[0]*q[1]+q[2]*q[3]), 1-2*(q[1]**2+q[2]**2))
    
    # 2. Bogie Actuator State
    climb_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "actuator_climb")
    bogie = data.ctrl[climb_id]
    
    # 3. Sensors
    l1 = get_sensor_value(model, data, "floor_sensL")
    l2 = get_sensor_value(model, data, "floor_sensU")
    l3 = get_sensor_value(model, data, "wall_sens")
    l1 = 2.0 if l1 < 0 else l1
    l2 = 2.0 if l2 < 0 else l2
    l3 = 2.0 if l3 < 0 else l3

    # 4. Velocity
    vel = np.linalg.norm(data.qvel[:2])

    id_drive = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "drive_forward")
    drive = data.ctrl[id_drive]
    
    return np.array([pitch, roll, bogie, l1, l2, l3, vel, drive], dtype=np.float32)

# Global memory for RL smoothing
rl_filtered_action = 0.0

def controller(model, data, scene, rl_agent=None):
    global rl_filtered_action

    # IDs
    id_drive = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "drive_forward")
    id_turn = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "drive_turn")
    id_climb = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "actuator_climb")
    id_level = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "level_bin")

    # 1. DEFAULT: Trajectory Planner
    drive_cmd, turn_cmd = 0.0, 0.0
    if USE_TRAJECTORY_CONTROL:
        drive_cmd, turn_cmd = current_traj_control(model, data, scene)
        data.ctrl[id_drive] = drive_cmd
        data.ctrl[id_turn] = turn_cmd  # Either Trajectory OR RL

    # 2. RL CLIMB & TERMINAL GUIDANCE
    climb_cmd = 0.0

    if USE_CLIMB_CONTROL and rl_agent is not None:
        # Get Obs
        obs = get_rl_observation(model, data)
        wall_dist = obs[5]  # Index 5 is 'wall_sens'

        # Predict [Climb, Turn]
        action, _ = rl_agent.predict(obs, deterministic=True)

        # Smoothing (Vectorized for both)
        # Note: rl_filtered_action should now be initialized as np.array([0.0, 0.0])
        alpha = 0.2
        rl_filtered_action = (alpha * action) + ((1 - alpha) * rl_filtered_action)
        climb_cmd = rl_filtered_action[0]
        data.ctrl[id_climb] = climb_cmd

    # 4. LEVELING
    data.ctrl[id_level] = -get_body_pitch(model, data, "car")

def run_simulation(scene: int, rl_agent=None):
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

    # Waypoints Setup  
    WAYPOINTS = create_path(model, data, scene)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        print("\nSimulation Started. Close the window to stop.")
        
        viewer.cam.distance = 6.0
        viewer.cam.lookat[:] = [3.0, -5.0, 1.0]
        
        dt = model.opt.timestep
        target_framerate = 30
        render_interval = 1.0 / target_framerate
        last_render_time = 0.0
        
        while viewer.is_running():
            step_start = time.time()
            
            # 1. Controller & Physics
            controller(model, data, scene, rl_agent)
            
            # Physics Step (Replicating the 20-step ratio used in training is optional 
            # but recommended for RL accuracy. For visual smoothness we do 1 step here)
            mujoco.mj_step(model, data)
            
            # 2. Render Loop
            if data.time - last_render_time >= render_interval:
                with viewer.lock(): 
                    viewer.user_scn.ngeom = 0 
                    draw_laser_beams(viewer, model, data)
                    
                    # Draw Waypoints
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
                
                # Debug Prints
                if int(data.time * 100) % 30 == 0:
                        d1 = get_sensor_value(model, data, "floor_sensU")
                        d2 = get_sensor_value(model, data, "floor_sensL")
                        d3 = get_sensor_value(model, data, "wall_sens")
                        # print(f"L1: {d1:.2f} | L2: {d2:.2f} | L3: {d3:.2f}")

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
    if input("Enable Trajectory Control? (y/n): ").lower() == 'n':
        USE_TRAJECTORY_CONTROL = False
    else:
        USE_TRAJECTORY_CONTROL = True

    # Climb Control
    if input("Enable Climb Control? (y/n): ").lower() == 'n':
        USE_CLIMB_CONTROL = False
    else:
        USE_CLIMB_CONTROL = True
        
        # ASK FOR RL MODE
        if RL_AVAILABLE:
            rl_in = input("   ↳ Use Trained RL Brain for Climbing? (y/n): ").lower()
            if rl_in == 'y':
                USE_RL_FOR_CLIMB = True
            else:
                USE_RL_FOR_CLIMB = False
                print("   ↳ Using Standard Logic Controller.")
        else:
            print("   ↳ RL Library missing. Using Standard Logic.")
            USE_RL_FOR_CLIMB = False

    # 3. Load Model (If needed)
    agent = None
    if USE_RL_FOR_CLIMB:
        print(f"\n🧠 Loading RL Model: {MODEL_PATH}...")
        try:
            agent = PPO.load(MODEL_PATH, device="cpu")
            print("✅ Model Loaded Successfully!")
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            print("   Falling back to Standard Logic.")
            USE_RL_FOR_CLIMB = False

    # 4. Launch
    run_simulation(choice, rl_agent=agent)
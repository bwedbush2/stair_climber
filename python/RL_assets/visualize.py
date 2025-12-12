import time
import numpy as np
import mujoco
import mujoco.viewer
from stable_baselines3 import PPO
import os

NUM_ROBOTS = 10
SPACING = 3.0
MODEL_PATH = "ray_stairs_policy"

ROBOT_TEMPLATE = """
    <body name="car_{ID}" pos="{START_X} {OFFSET_Y} {START_Z}"> 
      <freejoint/>
      <inertial pos="0.1 0 0" mass="10" diaginertia="0.5 0.5 0.8"/>
      <geom name="chassis_{ID}" type="mesh" mesh="chassis" pos="-0.405 0.275 -0.05" euler="90 0 0" 
            density="0" contype="0" conaffinity="0" group="1" rgba="{RGBA}"/>
      <geom name="belly_{ID}" type="box" size=".3 .15 .04" pos="0 0 0.05" rgba=".8 .2 .2 1"/>

      <site name="sens_chassis_{ID}" pos="0 0 0.2" size=".02" rgba="1 0 0 0"/>

      <body name="laser_array_{ID}" pos="0.25 0 0">
        <site name="laser_1_site_{ID}" pos="0.2 0 0.05" euler="0 135 0" size=".01" rgba="1 0 0 0"/>
        <site name="laser_2_site_{ID}" pos="0.2 0 0.15" euler="0 135 0" size=".01" rgba="1 0 0 0"/>
        <site name="laser_3_site_{ID}" pos="0 0 0.35" euler="0 90 0" size=".01" rgba="1 0 0 0"/>
      </body>

      <body name="leveling_base_{ID}" pos="-0.1 0 0.15">
        <joint name="bin_pitch_{ID}" axis="0 1 0" damping="10.0" range="-60 60"/>
        <geom type="cylinder" size=".03 .12" euler="90 0 0" rgba=".2 .2 .2 1"/>
        <geom name="platform_flat_{ID}" type="box" size=".14 .14 .01" pos="0 0 0.04" rgba=".3 .3 .3 1"/>
        <body name="the_bin_{ID}" pos="0 0 0.05">
            <inertial pos="0 0 0.1" mass="8" diaginertia="0.1 0.1 0.1"/>
            <geom name="bin_visual_{ID}" type="box" size=".13 .13 .12" pos="0 0 0.12" rgba="1 .6 0 1"/>
        </body>
      </body>

      <body name="left_bogie_{ID}" pos="0.25 .25 0">
        <joint name="climb_L_{ID}" axis="0 1 0" damping="50.0"/> 
        <site name="sens_bogie_{ID}" pos="0.1 0 0" size=".02" rgba="0 1 0 0"/>
        <geom type="box" size=".15 .04 .01" pos="0 .03 0" rgba="0.5 0.5 0.5 1"/>
        <geom class="skid_plate" fromto=".13 0 0 -.13 0 0"/>
        <body name="L_Front_{ID}" pos=".13 0 0"> <joint name="L1_{ID}" class="wheel"/> <geom class="wheel"/> </body>
        <body name="L_Mid_{ID}" pos="-.13 0 0"> <joint name="L2_{ID}" class="wheel"/> <geom class="wheel"/> </body>
      </body>
      <body name="left_wheel_rear_{ID}" pos="-0.3 0.25 0"> <joint name="L3_{ID}" class="wheel"/> <geom class="wheel"/> </body>

      <body name="right_bogie_{ID}" pos="0.25 -.25 0">
        <joint name="climb_R_{ID}" axis="0 1 0" damping="50.0"/>
        <geom type="box" size=".15 .04 .01" pos="0 -.03 0" rgba="0.5 0.5 0.5 1"/>
        <geom class="skid_plate" fromto=".13 0 0 -.13 0 0"/>
        <body name="R_Front_{ID}" pos=".13 0 0"> <joint name="R1_{ID}" class="wheel"/> <geom class="wheel"/> </body>
        <body name="R_Mid_{ID}" pos="-.13 0 0"> <joint name="R2_{ID}" class="wheel"/> <geom class="wheel"/> </body>
      </body>
      <body name="right_wheel_rear_{ID}" pos="-0.3 -.25 0"> <joint name="R3_{ID}" class="wheel"/> <geom class="wheel"/> </body>
    </body>
"""


def generate_swarm_xml():
    script_dir = os.path.dirname(os.path.abspath(__file__))

    world_content = ""
    actuator_content = ""
    sensor_content = ""
    tendon_content = ""
    equality_content = ""

    for i in range(NUM_ROBOTS):
        step_height = np.random.uniform(0.08, 0.22)
        step_depth = np.random.uniform(0.25, 0.50)
        num_steps = np.random.randint(3, 10)
        offset_y = i * SPACING

        color_r = 1.0 - (i / NUM_ROBOTS)
        color_g = (i / NUM_ROBOTS)
        rgba = f"{color_r:.2f} {color_g:.2f} 0.2 1"

        current_x = 2.0
        current_z = 0.0

        for s in range(num_steps):
            z_center = current_z + (step_height / 2)
            world_content += f"""
            <geom name="s_u_{i}_{s}" type="box" size="{step_depth / 2} 1 {step_height / 2}" 
                  pos="{current_x + step_depth / 2} {offset_y} {z_center}" material="concrete"/>
            """
            current_x += step_depth
            current_z += step_height

        plat_len = 1.5
        world_content += f"""
        <geom name="plat_{i}" type="box" size="{plat_len / 2} 1 {step_height / 2}" 
              pos="{current_x + plat_len / 2} {offset_y} {current_z - step_height / 2}" material="concrete"/>
        """
        current_x += plat_len

        for s in range(num_steps):
            current_z -= step_height
            z_center = current_z + (step_height / 2)
            world_content += f"""
            <geom name="s_d_{i}_{s}" type="box" size="{step_depth / 2} 1 {step_height / 2}" 
                  pos="{current_x + step_depth / 2} {offset_y} {z_center}" material="concrete"/>
            """
            current_x += step_depth

        world_content += ROBOT_TEMPLATE.format(ID=i, START_X=0.5, START_Z=0.5, OFFSET_Y=offset_y, RGBA=rgba)

        tendon_content += f"""
        <fixed name="fwd_{i}"><joint joint="L1_{i}" coef="0.5"/><joint joint="L2_{i}" coef="0.5"/><joint joint="L3_{i}" coef="0.5"/>
                              <joint joint="R1_{i}" coef="0.5"/><joint joint="R2_{i}" coef="0.5"/><joint joint="R3_{i}" coef="0.5"/></fixed>
        <fixed name="turn_{i}"><joint joint="L1_{i}" coef="-0.5"/><joint joint="L2_{i}" coef="-0.5"/><joint joint="L3_{i}" coef="-0.5"/>
                               <joint joint="R1_{i}" coef="0.5"/><joint joint="R2_{i}" coef="0.5"/><joint joint="R3_{i}" coef="0.5"/></fixed>
        """

        equality_content += f"""
        <joint joint1="climb_L_{i}" joint2="climb_R_{i}" polycoef="0 1 0 0 0" solimp="0.99 0.99 0.01" solref="0.005 1"/>
        """

        actuator_content += f"""
        <motor name="drive_{i}" tendon="fwd_{i}" ctrlrange="-1 1" gear="40"/>
        <motor name="steer_{i}" tendon="turn_{i}" ctrlrange="-1 1" gear="40"/>
        <position name="climb_{i}" joint="climb_L_{i}" ctrlrange="-1.5 1.5" kp="3000" forcerange="-400 400"/>
        <position name="level_{i}" joint="bin_pitch_{i}" ctrlrange="-1 1" kp="500"/>
        """

        sensor_content += f"""
        <rangefinder name="sL_{i}" site="laser_1_site_{i}"/>
        <rangefinder name="sU_{i}" site="laser_2_site_{i}"/>
        <rangefinder name="sW_{i}" site="laser_3_site_{i}"/>
        """

    full_xml = f"""
    <mujoco model="RAY_Swarm">
      <compiler autolimits="true"/>
      <asset>
        <mesh name="chassis" file="{script_dir}/../../mujoco/assets/chassis.stl" scale="0.001 0.001 0.001"/>
        <texture name="concrete" type="2d" builtin="flat" rgb1=".7 .7 .7" width="512" height="512"/>
        <material name="concrete" texture="concrete" reflectance=".1"/>
        <texture name="grid" type="2d" builtin="checker" width="512" height="512" rgb1=".1 .2 .3" rgb2=".2 .3 .4"/>
        <material name="grid" texture="grid" texrepeat="1 1" texuniform="true" reflectance=".2"/>
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
        <geom name="floor" type="plane" size="50 100 .1" material="grid"/>
        {world_content}
      </worldbody>
      <tendon>{tendon_content}</tendon>
      <equality>{equality_content}</equality>
      <actuator>{actuator_content}</actuator>
      <sensor>{sensor_content}</sensor>
    </mujoco>
    """
    return full_xml


def get_sensor(model, data, name):
    id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, name)
    val = data.sensordata[model.sensor_adr[id]]
    return 2.0 if val < 0 else val


def get_pitch(model, data, idx):
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"car_{idx}")
    q = data.xquat[body_id]
    return np.arcsin(np.clip(2 * (q[0] * q[2] - q[3] * q[1]), -1, 1))


def get_roll(model, data, idx):
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"car_{idx}")
    q = data.xquat[body_id]
    return np.arctan2(2 * (q[0] * q[1] + q[2] * q[3]), 1 - 2 * (q[1] ** 2 + q[2] ** 2))


if __name__ == "__main__":
    print(f"Starting XML generation for {NUM_ROBOTS} robots...")
    xml_string = generate_swarm_xml()

    model = mujoco.MjModel.from_xml_string(xml_string)
    data = mujoco.MjData(model)

    print("Attempting to load the trained policy...")
    try:
        agent = PPO.load(MODEL_PATH, device="cpu")
        print("Policy loaded successfully.")
    except:
        print("Model file not found. Running robots without AI control.")
        agent = None

    filtered_actions = np.zeros(NUM_ROBOTS)
    target_speeds = np.random.uniform(0.3, 0.7, NUM_ROBOTS)

    print("Launching MuJoCo visualization window...")
    with mujoco.viewer.launch_passive(model, data) as viewer:

        viewer.cam.lookat[:] = [5.0, (NUM_ROBOTS * SPACING) / 2, 0.0]
        viewer.cam.distance = NUM_ROBOTS * 4.0
        viewer.cam.elevation = -45
        viewer.cam.azimuth = 90

        while viewer.is_running():
            start_time = time.time()

            for i in range(NUM_ROBOTS):
                pitch = get_pitch(model, data, i)
                roll = get_roll(model, data, i)
                body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"car_{i}")
                z_pos = data.xpos[body_id][2]

                climb_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"climb_{i}")
                drive_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"drive_{i}")
                level_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"level_{i}")

                if abs(pitch) > 1.0 or abs(roll) > 1.0 or z_pos < 0.0:
                    data.ctrl[drive_id] = 0.0
                    continue

                bogie_val = data.ctrl[climb_id]
                vel_sensor = 0.5
                l1 = get_sensor(model, data, f"sL_{i}")
                l2 = get_sensor(model, data, f"sU_{i}")
                l3 = get_sensor(model, data, f"sW_{i}")

                obs = np.array([pitch, roll, bogie_val, l1, l2, l3, vel_sensor, target_speeds[i]], dtype=np.float32)

                if agent:
                    action, _ = agent.predict(obs, deterministic=True)
                    raw = action[0]
                else:
                    raw = 0.0

                filtered_actions[i] = (0.2 * raw) + (0.8 * filtered_actions[i])

                data.ctrl[climb_id] = filtered_actions[i]
                data.ctrl[drive_id] = target_speeds[i]
                data.ctrl[level_id] = -pitch

            for _ in range(20):
                mujoco.mj_step(model, data)

            viewer.sync()

            time.sleep(0.02)
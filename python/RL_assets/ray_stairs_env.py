import gymnasium as gym
from gymnasium import spaces
import numpy as np
import mujoco
import os

# Define the base robot XML
ROBOT_XML_TEMPLATE = """
    <body name="car" pos="{START_X} 0 {START_Z}"> 
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


class RayProceduralEnv(gym.Env):
    def __init__(self, render_mode=None):
        super(RayProceduralEnv, self).__init__()

        self.render_mode = render_mode
        self.viewer = None
        self.script_dir = os.path.dirname(os.path.abspath(__file__))

        # 1. ACTION SPACE: Climb Only [1D]
        self.action_space = spaces.Box(low=np.array([-1.5], dtype=np.float32),
                                       high=np.array([1.5], dtype=np.float32), dtype=np.float32)

        # 2. OBS SPACE: 8 values
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(8,), dtype=np.float32)

        self.model = None
        self.data = None
        self.target_speed = 0.4
        self.current_step_height = 0.1  # Tracked for reward scaling
        self.filtered_action = 0.0
        self.last_raw_action = 0.0

        self._generate_and_load_xml()

    def _generate_and_load_xml(self):
        # 1. RANDOMIZE PARAMETERS
        self.current_step_height = np.random.uniform(0.08, 0.22)
        step_depth = np.random.uniform(0.25, 0.50)
        num_steps = np.random.randint(3, 12) # Reduced max from 20 to 12 to keep map size reasonable

        self.mode = "CLIMB" if np.random.random() < 0.7 else "DESCEND"

        # --- BUILD GEOMETRY ---
        stair_xml = ""
        current_x = 2.0
        current_z = 0.0

        # A. BUILD UP STAIRS
        for i in range(num_steps):
            z_center = current_z + (self.current_step_height / 2)
            stair_xml += f"""
            <geom name="step_u_{i}" type="box" size="{step_depth / 2} 1 {self.current_step_height / 2}" 
                  pos="{current_x + step_depth / 2} 0 {z_center}" material="concrete"/>
            """
            current_x += step_depth
            current_z += self.current_step_height

        # B. BUILD PLATFORM
        plat_len = 1.5
        # Platform center is at the same Z height as the last step's center
        plat_z_center = current_z - (self.current_step_height / 2)
        
        stair_xml += f"""
        <geom name="platform" type="box" size="{plat_len / 2} 1 {self.current_step_height / 2}" 
              pos="{current_x + plat_len / 2} 0 {plat_z_center}" material="concrete"/>
        """
        current_x += plat_len
        
        # C. BUILD DOWN STAIRS (The missing piece!)
        # We start from current_z (top surface) and step down
        for i in range(num_steps):
            current_z -= self.current_step_height # Drop surface level
            z_center = current_z + (self.current_step_height / 2) # Center of the block
            
            stair_xml += f"""
            <geom name="step_d_{i}" type="box" size="{step_depth / 2} 1 {self.current_step_height / 2}" 
                  pos="{current_x + step_depth / 2} 0 {z_center}" material="concrete"/>
            """
            current_x += step_depth

        # --- SPAWN LOGIC ---
        if self.mode == "CLIMB":
            start_x = 0.5
            start_z = 0.2
        else:
            # DESCEND MODE: Spawn in the middle of the platform
            # We subtract half the platform length to get to its center (current_x is at the end edge of plat + steps)
            # Actually, current_x is now way past the platform (at the bottom of down stairs).
            # We need to calculate platform center explicitly.
            
            # Recalculate platform center X:
            # Start of stairs (2.0) + (Up Steps * Depth) + (Plat Len / 2)
            plat_center_x = 2.0 + (num_steps * step_depth) + (plat_len / 2)
            
            start_x = plat_center_x
            start_z = (num_steps * self.current_step_height) + 0.3 # Safe drop height above platform

        # --- FULL XML ---
        full_xml = f"""
        <mujoco model="RAY_Procedural">
          <compiler autolimits="true"/>
          <asset>
            <mesh name="chassis" file="{self.script_dir}/../../mujoco/assets/chassis.stl" scale="0.001 0.001 0.001"/>
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
            <geom name="floor" type="plane" size="50 10 .1" material="grid"/>
            {stair_xml}
            {ROBOT_XML_TEMPLATE.format(START_X=start_x, START_Z=start_z)}
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
            <rangefinder name="floor_sensL" site="laser_1_site"/>
            <rangefinder name="floor_sensU" site="laser_2_site"/>
            <rangefinder name="wall_sens" site="laser_3_site"/>
          </sensor>
        </mujoco>
        """
        self.model = mujoco.MjModel.from_xml_string(full_xml)
        self.data = mujoco.MjData(self.model)

        self.drive_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "drive_forward")
        self.turn_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "drive_turn")
        self.climb_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "actuator_climb")
        self.bin_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "level_bin")
        
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._generate_and_load_xml()
        
        self.target_speed = np.random.uniform(0.2, 0.7)
        
        # 1. Orientation
        self.data.qpos[3:7] = [1, 0, 0, 0] 
        self.data.qpos[1] = np.random.uniform(-0.1, 0.1)

        # 2. SPAWN HEIGHT FIX
        # Was 0.2. Increase to 0.3 to drop it safely onto the floor.
        # This prevents "floor clipping explosions"
        if self.mode == "CLIMB":
             # Force Z to be safe
             self.data.qpos[2] = 0.5
        
        self.filtered_action = 0.0
        self.last_raw_action = 0.0
        
        # Close old viewer
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
            
        mujoco.mj_step(self.model, self.data)
        
        # 3. FORCE RENDER (Open the window NOW)
        if self.render_mode == "human":
            self._render_frame()
            
        return self._get_obs(), {}


    def step(self, action):
        raw_action = action[0]  # Single action (Climb)

        alpha = 0.2
        self.filtered_action = (alpha * raw_action) + ((1 - alpha) * self.filtered_action)

        # Controls (Turn is locked)
        self.data.ctrl[self.drive_id] = self.target_speed
        self.data.ctrl[self.climb_id] = self.filtered_action
        self.data.ctrl[self.turn_id] = 0.0
        self.data.ctrl[self.bin_id] = -self._get_pitch()

        for _ in range(20):
            mujoco.mj_step(self.model, self.data)

        # --- REWARDS ---# ... inside step() ...

        # --- REWARD LOGIC ---
        vel_x = self.data.qvel[0]
        
        # 1. PROGRESS (Velocity Only)
        # We want it to move forward. 
        # Using 'vel_x' directly is robust. 
        # If it goes backwards, this becomes negative (Good!).
        reward_progress = vel_x * 50.0
        
        # 2. STABILITY
        reward_stability = -abs(self._get_pitch()) * 2.0
        
        # 3. ENERGY (Hinge)
        reward_energy_base = -np.square(raw_action) * 0.15
        excess = max(0, abs(raw_action) - 1.2)
        reward_energy_excess = -np.square(excess) * 10.0
        
        # 4. CONTEXTUAL SAFETY
        wall_dist = self._read_sensor("wall_sens")
        pitch = self._get_pitch()
                
        # LOGIC:
        # 1. Wall Check: Is the wall far away? (> 1.2m)
        wall_is_clear = np.clip((wall_dist - 1.2) * 2.0, 0.0, 1.0)

        robot_is_flat = 1.0 - np.clip(abs(pitch) * 5.0, 0.0, 1.0)

        safety_factor = wall_is_clear * robot_is_flat
        reward_context = (reward_energy_base + reward_energy_excess) * 5.0 * safety_factor
        total_energy = reward_energy_base + reward_energy_excess + reward_context

        reward_smooth = -np.square(raw_action - self.last_raw_action) * 2.0
        reward_heading = -abs(self._get_yaw()) * 2.0 
        
        # 5. TIME PENALTY (The "Hurry Up" Fix)
        # Flat penalty every step. Forces it to finish ASAP.
        reward_time = -1.0 

        total_reward = (reward_progress + reward_stability + total_energy + 
                        reward_smooth + reward_heading + reward_time)

        self.last_raw_action = raw_action

        # TERMINATION
        terminated = False
        if abs(self._get_pitch()) > 1.0 or abs(self._get_roll()) > 1.0: terminated = True; total_reward -= 50.0

        # Success (X > 12.0)
        if self.data.qpos[0] > 12.0:
            terminated = True

            # --- DIFFICULTY BONUS ---
            # Base Reward: 1000
            # Height Bonus: Scale by step height (0.08 to 0.22)
            # A 22cm step gives roughly DOUBLE the reward of an 8cm step.
            difficulty_mult = 1.0 + (self.current_step_height * 10.0)
            total_reward += 1000.0 * difficulty_mult

        return self._get_obs(), total_reward, terminated, False, {}

    # --- HELPERS ---
    def _get_obs(self):
        pitch = self._get_pitch()
        roll = self._get_roll()
        bogie = self.data.ctrl[self.climb_id]
        vel = np.linalg.norm(self.data.qvel[:2])
        l1 = self._read_sensor("floor_sensL")
        l2 = self._read_sensor("floor_sensU")
        l3 = self._read_sensor("wall_sens")
        return np.array([pitch, roll, bogie, l1, l2, l3, vel, self.target_speed], dtype=np.float32)

    def _get_pitch(self):
        q = self.data.qpos[3:7]
        return np.arcsin(np.clip(2 * (q[0] * q[2] - q[3] * q[1]), -1, 1))

    def _get_roll(self):
        q = self.data.qpos[3:7]
        return np.arctan2(2 * (q[0] * q[1] + q[2] * q[3]), 1 - 2 * (q[1] ** 2 + q[2] ** 2))

    def _get_yaw(self):
        q = self.data.qpos[3:7]
        return np.arctan2(2 * (q[0] * q[3] + q[1] * q[2]), 1 - 2 * (q[2] ** 2 + q[3] ** 2))

    def _read_sensor(self, name):
        try:
            id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, name)
            adr = self.model.sensor_adr[id]
            val = self.data.sensordata[adr]
            return 2.0 if val < 0 else val
        except:
            return 0.0

    def _render_frame(self):
        # 1. Launch Viewer if it doesn't exist
        if self.viewer is None:
            import mujoco.viewer
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
            
            # --- CAMERA AUTO-ALIGN ---
            # This locks the camera to the robot ('car')
            self.viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
            self.viewer.cam.trackbodyid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "car")
            
            # Set a nice view angle (Behind and above)
            self.viewer.cam.distance = 4.0
            self.viewer.cam.elevation = -25
            self.viewer.cam.azimuth = 90

        # 2. Update the window
        self.viewer.sync()

    def close(self):
        if self.viewer: self.viewer.close()
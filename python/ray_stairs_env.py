import gymnasium as gym
from gymnasium import spaces
import numpy as np
import mujoco
import os

class RayFocusedStairs(gym.Env):
    def __init__(self, render_mode=None):
        super(RayFocusedStairs, self).__init__()
        
        # 1. SETUP (Load Scenario 3)
        # We assume you ran 'build_ray.py' -> Option 3 beforehand
        script_dir = os.path.dirname(os.path.abspath(__file__))
        xml_path = os.path.join(script_dir, "..", "mujoco", "ray_simulation.xml")
        xml_path = os.path.normpath(xml_path)
        
        if not os.path.exists(xml_path):
            raise FileNotFoundError("XML not found! Run 'build_ray.py' (Scenario 3) first.")

        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        
        self.drive_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "drive_forward")
        self.climb_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "actuator_climb")
        self.bin_id   = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "level_bin")
        
        # ACTION: Climb Only
        self.action_space = spaces.Box(low=np.array([-1.5]), high=np.array([1.5]), dtype=np.float32)

        # OBSERVATION: 7 values
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(7,), dtype=np.float32)

        self.render_mode = render_mode
        self.viewer = None
        self.last_action = np.array([0.0])

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        
        # SPAWN: Facing the Stairs (Stairs are at X=2.0)
        # Start at X=0.5 to get to the action faster
        self.data.qpos[0] = 0.5 
        self.data.qpos[1] = 0.0 + np.random.uniform(-0.1, 0.1)
        self.data.qpos[2] = 0.2 
        
        # Heading: +X direction (Quat [1, 0, 0, 0])
        self.data.qpos[3:7] = [1, 0, 0, 0]
        
        self.last_action = np.array([0.0])
        mujoco.mj_step(self.model, self.data)
        return self._get_obs(), {}

    def step(self, action):
        # 1. Action
        drive_cmd = 0.4 # Slow, steady speed for learning
        climb_cmd = action[0]
        
        self.data.ctrl[self.drive_id] = drive_cmd
        self.data.ctrl[self.climb_id] = climb_cmd
        self.data.ctrl[self.bin_id] = -self._get_pitch()

        # 2. Physics (20 steps)
        for _ in range(20):
            mujoco.mj_step(self.model, self.data)

        # 3. REWARDS (SCALED DOWN)
        
        # A. Forward Progress (Max Velocity is approx 0.5)
        # Scale: 0.0 to ~2.0
        vel_x = self.data.qvel[0]
        reward_progress = vel_x * 4.0 
        
        # B. Stability (Penalty)
        # Scale: 0.0 to -0.5
        reward_stability = -abs(self._get_pitch()) * 1.0
        
        # C. Efficiency (Penalty)
        # Scale: 0.0 to -0.1
        reward_energy = -np.square(climb_cmd) * 0.1
        reward_smooth = -np.square(climb_cmd - self.last_action[0]) * 0.5
        
        # D. Stall Penalty (CRITICAL)
        # If we are not moving (vel < 0.05) but trying to drive, lose points.
        # This fixes the "holding legs up forever" bug.
        reward_stall = 0.0
        if vel_x < 0.05:
            reward_stall = -0.5

        # TOTAL
        reward = reward_progress + reward_stability + reward_energy + reward_smooth + reward_stall
        self.last_action = action

        # 4. TERMINATION
        terminated = False
        
        # Fail: Flipped
        if abs(self._get_pitch()) > 1.0 or abs(self._get_roll()) > 1.0:
            terminated = True
            reward -= 10.0 # Small penalty (not -100)
            
        # Success: Crossed Pyramid (X > 3.5)
        if self.data.qpos[0] > 3.5:
            terminated = True
            reward += 20.0 # Moderate bonus (not +1000)

        if self.render_mode == "human":
            self._render_frame()

        return self._get_obs(), reward, terminated, False, {}

    # --- HELPERS ---
    def _get_obs(self):
        pitch = self._get_pitch()
        roll = self._get_roll()
        bogie = self.data.ctrl[self.climb_id]
        vel = np.linalg.norm(self.data.qvel[:2])
        l1 = self._read_sensor("floor_sensL")
        l2 = self._read_sensor("floor_sensU")
        l3 = self._read_sensor("wall_sens")
        return np.array([pitch, roll, bogie, l1, l2, l3, vel], dtype=np.float32)

    def _get_pitch(self):
        q = self.data.qpos[3:7] 
        return np.arcsin(np.clip(2 * (q[0]*q[2] - q[3]*q[1]), -1, 1))

    def _get_roll(self):
        q = self.data.qpos[3:7]
        return np.arctan2(2*(q[0]*q[1]+q[2]*q[3]), 1-2*(q[1]**2+q[2]**2))

    def _read_sensor(self, name):
        try:
            id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, name)
            adr = self.model.sensor_adr[id]
            val = self.data.sensordata[adr]
            return 2.0 if val < 0 else val
        except: return 0.0

    def _render_frame(self):
        if self.viewer is None:
            import mujoco.viewer
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
        self.viewer.sync()
    
    def close(self):
        if self.viewer: self.viewer.close()
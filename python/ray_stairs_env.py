import gymnasium as gym
from gymnasium import spaces
import numpy as np
import mujoco
import os


class RayFocusedStairs(gym.Env):
    def __init__(self, render_mode=None):
        super(RayFocusedStairs, self).__init__()

        # 1. SETUP
        script_dir = os.path.dirname(os.path.abspath(__file__))
        xml_path = os.path.join(script_dir, "..", "mujoco", "ray_simulation.xml")
        if not os.path.exists(xml_path):
            raise FileNotFoundError("XML not found! Run 'build_ray.py' (Scenario 3) first.")

        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)

        self.drive_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "drive_forward")
        self.climb_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "actuator_climb")
        self.bin_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "level_bin")

        self.action_space = spaces.Box(low=np.array([-1.5], dtype=np.float32),
                                       high=np.array([1.5], dtype=np.float32), dtype=np.float32)

        # OBSERVATION: Added +1 size for "Current Target Speed"
        # It's helpful (but not strictly required) for the robot to KNOW how fast it's trying to go.
        # Let's keep it simple at 7 for now, assuming velocity sensor covers it.
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(7,), dtype=np.float32)

        self.render_mode = render_mode
        self.viewer = None

        # State
        self.target_speed = 0.4  # Default
        self.filtered_action = 0.0
        self.last_raw_action = 0.0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)

        # --- FIX 1: VARIABLE SPEED TRAINING ---
        # Pick a random speed for this episode.
        # Range: 0.2 (Crawl) to 1.0 (Full Sprints)
        self.target_speed = np.random.uniform(0.2, 0.6)

        # --- FIX 2: SPAWN DISTRIBUTION (80/20) ---
        rand = np.random.random()
        if rand < 0.8:
            # Climb Start
            self.data.qpos[0] = 0.5 + np.random.uniform(0, 1.0)
            self.data.qpos[2] = 0.2
        else:
            # Descend Start
            self.data.qpos[0] = 3.5
            self.data.qpos[2] = 0.6

        self.data.qpos[1] = 0.0 + np.random.uniform(-0.1, 0.1)
        self.data.qpos[3:7] = [1, 0, 0, 0]

        self.filtered_action = 0.0
        self.last_raw_action = 0.0

        mujoco.mj_step(self.model, self.data)
        return self._get_obs(), {}

    def step(self, action):
        raw_action = action[0]

        # Smoothing
        alpha = 0.2
        self.filtered_action = (alpha * raw_action) + ((1 - alpha) * self.filtered_action)

        # --- APPLY VARIABLE SPEED ---
        self.data.ctrl[self.drive_id] = self.target_speed  # Use the randomized speed
        self.data.ctrl[self.climb_id] = self.filtered_action
        self.data.ctrl[self.bin_id] = -self._get_pitch()

        for _ in range(20):
            mujoco.mj_step(self.model, self.data)

        # --- REWARD LOGIC ---

        # A. Progress (Relative to Target Speed)
        # We reward it for matching the target speed, not just "going fast".
        vel_x = self.data.qvel[0]

        # Error: How far are we from target speed?
        speed_error = abs(vel_x - self.target_speed)

        # Reward: 2.0 minus error (If perfect match, +2.0. If stalled, negative).
        reward_progress = (self.target_speed - speed_error) * 5.0

        # B. Stability
        reward_stability = -abs(self._get_pitch()) * 2.0

        # C. Energy & Lazy Penalty
        reward_energy = -np.square(raw_action) * 3.0

        wall_dist = self._read_sensor("wall_sens")
        reward_lazy = 0.0
        if wall_dist > 1.5 and abs(raw_action) > 0.1:
            reward_lazy = -5.0 * abs(raw_action)

            # D. Smoothness
        reward_smooth = -np.square(raw_action - self.last_raw_action) * 5.0

        total_reward = reward_progress + reward_stability + reward_energy + reward_lazy + reward_smooth
        self.last_raw_action = raw_action

        # TERMINATION
        terminated = False
        if abs(self._get_pitch()) > 1.0 or abs(self._get_roll()) > 1.0:
            terminated = True
            total_reward -= 50.0

        if self.data.qpos[0] > 6.0:
            terminated = True
            total_reward += 1000.0

        if self.render_mode == "human":
            self._render_frame()

        return self._get_obs(), total_reward, terminated, False, {}

    # ... (Keep helpers: _get_obs, _get_pitch, _read_sensor, _render_frame) ...
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
        return np.arcsin(np.clip(2 * (q[0] * q[2] - q[3] * q[1]), -1, 1))

    def _get_roll(self):
        q = self.data.qpos[3:7]
        return np.arctan2(2 * (q[0] * q[1] + q[2] * q[3]), 1 - 2 * (q[1] ** 2 + q[2] ** 2))

    def _read_sensor(self, name):
        try:
            id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, name)
            adr = self.model.sensor_adr[id]
            val = self.data.sensordata[adr]
            return 2.0 if val < 0 else val
        except:
            return 0.0

    def _render_frame(self):
        if self.viewer is None:
            import mujoco.viewer
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
        self.viewer.sync()

    def close(self):
        if self.viewer: self.viewer.close()
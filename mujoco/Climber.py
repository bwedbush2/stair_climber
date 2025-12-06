import time
import mujoco
import mujoco.viewer
import numpy as np

# ================= CONFIGURATION =================
CLIMB_ANGLE_ATTACK = -0.9  # Lift high to clear curb
CLIMB_ANGLE_HOLD = -0.6    # Maintain angle while climbing
# =================================================

class AutoClimber:
    def __init__(self):
        self.state = "CRUISE"
        self.bogie_target = 0.0
        self.timer = 0.0

    def update(self, pitch, bumper_hit):
        """State Machine based on Impact and Pitch"""
        
        # STATE 1: CRUISE (Flat driving)
        if self.state == "CRUISE":
            self.bogie_target = 0.0
            
            # TRIGGER: If we hit a wall/curb OR we are already tilted up
            if bumper_hit:
                print("💥 BUMP! Engaging Attack Mode")
                self.state = "ATTACK"
                self.timer = time.time()
            elif pitch > 0.2:
                self.state = "CLIMBING"

        # STATE 2: ATTACK (Lift legs immediately after impact)
        elif self.state == "ATTACK":
            self.bogie_target = CLIMB_ANGLE_ATTACK
            
            # If we have been attacking for 1 second, switch to steady climbing
            # This gives time for the front wheels to get ON TOP of the step
            if time.time() - self.timer > 1.0:
                self.state = "CLIMBING"

        # STATE 3: CLIMBING (Steady ascent)
        elif self.state == "CLIMBING":
            self.bogie_target = CLIMB_ANGLE_HOLD
            
            # EXIT: If robot levels out (reached top)
            if pitch < 0.1:
                print("✅ Crested. Returning to Cruise.")
                self.state = "CRUISE"
                
        return self.bogie_target, self.state

# GLOBAL INPUTS
manual_throttle = 0.0
manual_turn = 0.0

def key_callback(keycode):
    global manual_throttle, manual_turn
    if keycode == 265: manual_throttle += 0.2 # Up
    elif keycode == 264: manual_throttle -= 0.2 # Down
    elif keycode == 263: manual_turn += 0.2 # Left
    elif keycode == 262: manual_turn -= 0.2 # Right
    elif keycode == 32:  manual_throttle = 0.0 # Space
    
    manual_throttle = np.clip(manual_throttle, -1.0, 1.0)
    manual_turn = np.clip(manual_turn, -1.0, 1.0)

def main():
    print("Loading simulation...")
    model = mujoco.MjModel.from_xml_path("ray_simulation.xml")
    data = mujoco.MjData(model)
    climber = AutoClimber()

    # Get Sensor IDs
    l_bump_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, "touch_L")
    r_bump_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, "touch_R")
    
    with mujoco.viewer.launch_passive(model, data, key_callback=key_callback) as viewer:
        while viewer.is_running():
            step_start = time.time()

            # 1. READ SENSORS
            # Pitch
            sens_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, "sensor_chassis_quat")
            adr = model.sensor_adr[sens_id]
            q = data.sensordata[adr:adr+4]
            sinp = 2 * (q[0] * q[2] - q[3] * q[1])
            pitch = np.arcsin(np.clip(sinp, -1, 1))

            # Bumpers (Check if force > 0.1)
            bump_L = data.sensordata[model.sensor_adr[l_bump_id]]
            bump_R = data.sensordata[model.sensor_adr[r_bump_id]]
            is_hitting = (bump_L > 0.1) or (bump_R > 0.1)

            # 2. RUN LOGIC
            bogie_cmd, mode = climber.update(pitch, is_hitting)
            
            # 3. APPLY CONTROLS
            # If in ATTACK or CLIMB mode, boost throttle to ensure we have power
            final_throttle = manual_throttle
            if mode != "CRUISE" and manual_throttle > 0:
                final_throttle = 1.0 # Full power for climbing

            data.ctrl[0] = final_throttle
            data.ctrl[1] = manual_turn
            data.ctrl[2] = bogie_cmd
            data.ctrl[3] = np.clip(-pitch, -1.0, 1.0) # Auto-Bin

            mujoco.mj_step(model, data)
            viewer.sync()
            
            time.sleep(model.opt.timestep)

if __name__ == "__main__":
    main()
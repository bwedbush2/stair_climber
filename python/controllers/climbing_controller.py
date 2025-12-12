import numpy as np
import mujoco

# MATH HELPERS
def get_body_pitch(model, data, body_name="car"):
    try:
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    except Exception: return 0.0
    q = data.xquat[body_id]
    sinp = 2 * (q[0] * q[2] - q[3] * q[1])
    if abs(sinp) >= 1: pitch = np.sign(sinp) * (np.pi / 2) 
    else: pitch = np.arcsin(sinp)
    return pitch

# CLIMBER LOGIC CLASS
def get_sensor_value(model, data, sensor_name):
    """
    Returns the scalar value of a sensor by name.
    Returns -1.0 if the sensor name is invalid.
    """
    try:
        # 1. Get the ID of the sensor string
        sens_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, sensor_name)
        
        # 2. Get the array address (index) for this sensor
        # (This is needed because some sensors like quaternions use 4 slots)
        adr = model.sensor_adr[sens_id]
        
        # 3. Read the value from the global sensor array
        # Rangefinders return a single scalar distance (meters)
        return data.sensordata[adr]
        
    except Exception:
        # Warning: Sensor not found
        return -1.0
    
class AdaptiveClimber:
    def __init__(self):
        self.bogie_angle = 0.0
        self.stall_timer = 0.0
        
        # --- TUNING PARAMETERS ---
        self.LIFT_SPEED = 0.01      # Faster lift to react quickly
        self.LOWER_SPEED = 0.01    # Slower lower to "pull" heavy loads
        
        self.MAX_LIFT = -1.2        # Max possible lift
        self.ATTACK_LIMIT = -0.6    # Max lift allowed while chassis is still flat
        
        self.STALL_VEL_THRESH = 0.1 
        self.TILT_THRESH = 0.25     # Rad (~15 deg). Above this = "On Stairs"

    def update(self, data, throttle_cmd, pitch, dt):
        
        # 1. Get Velocity
        speed = np.sqrt(data.qvel[0]**2 + data.qvel[1]**2)
        
        # 2. Check Stall
        is_stalled = (abs(throttle_cmd) > 0.3) and (speed < self.STALL_VEL_THRESH)
        
        # 3. LOGIC TREE
        
        # === CASE A: ROBOT IS TILTED UP (Mid-Climb) ===
        if pitch > self.TILT_THRESH:
            if is_stalled:
                # Stuck on slope -> Flatten to gain traction
                self.bogie_angle += self.LOWER_SPEED
            else:
                # Moving up slope -> Lift to clear next step
                self.bogie_angle -= self.LIFT_SPEED

        # === CASE B: ROBOT IS FLAT (Curb / First Step) ===
        else:
            if is_stalled:
                self.stall_timer += dt
                
                # CRITICAL FIX: THE "HOOK AND PULL"
                if self.bogie_angle < self.ATTACK_LIMIT:
                    # FORCE LOWER: Press front wheel into the step to pull middle wheel up
                    self.bogie_angle += self.LOWER_SPEED
                else:
                    # Standard Attack: Lift nose to find the step
                    self.bogie_angle -= self.LIFT_SPEED
            
            else:
                self.stall_timer = 0.0
                # Moving on flat ground -> Relax to 0
                self.bogie_angle += self.LOWER_SPEED
            
        # 4. Safety Clamps
        self.bogie_angle = np.clip(self.bogie_angle, self.MAX_LIFT, 0.0)
            
        return self.bogie_angle

# Global Instance
climber_logic = AdaptiveClimber()

def climb_control(model, data):
    drive_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "drive_forward")
    current_throttle = data.ctrl[drive_id]
    current_pitch = get_body_pitch(model, data, "car")

    floor_dist_lower = get_sensor_value(model, data, "floor_sensL")
    floor_dist_upper = get_sensor_value(model, data, "floor_sensU")
    wall_dist = get_sensor_value(model, data, "wall_sens")

    # if wall_dist > 0 and wall_dist < 0.75:
    #     print("Wall detected")
    #     print(wall_dist)
    
    # if floor_dist_upper - floor_dist_lower < 0.1:
    #     print("No floor detected! (step)")
    # else:
    #     print("floor detected! (step)")

    return climber_logic.update(data, current_throttle, current_pitch, model.opt.timestep)
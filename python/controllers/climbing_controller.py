import numpy as np
import mujoco

# ==========================================
# 📐 MATH HELPERS
# ==========================================

def get_body_pitch(model, data, body_name="car"):
    """
    Returns the global pitch angle (radians) of a body.
    Positive Pitch = Nose Tipping Up
    """
    try:
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    except Exception:
        return 0.0

    q = data.xquat[body_id]
    w, x, y, z = q[0], q[1], q[2], q[3]
    
    # Quaternion to Pitch (Rotation around Y-axis)
    sinp = 2 * (w * y - z * x)
    if abs(sinp) >= 1:
        pitch = np.sign(sinp) * (np.pi / 2) 
    else:
        pitch = np.arcsin(sinp)

    return pitch

# ==========================================
# 🧠 CLIMBER LOGIC CLASS
# ==========================================

class AdaptiveClimber:
    def __init__(self):
        self.bogie_angle = 0.0
        self.stall_timer = 0.0
        
        # --- TUNING PARAMETERS ---
        self.LIFT_SPEED = 0.02      # Speed to raise legs
        self.LOWER_SPEED = 0.02     # Speed to lower legs
        self.MAX_LIFT = -1.2        # Max lift angle
        self.STALL_VEL_THRESH = 0.1 # m/s (Below this = Stalled)
        self.TILT_THRESH = 0.25     # Rad (~15 deg). Above this = "On Stairs"

    def update(self, data, throttle_cmd, pitch, dt):
        """
        New Logic:
        1. Flat Ground: Lift if stalled (to climb curb).
        2. Tilted Up:
           - If Stalled: LOWER legs (Straighten to grab traction).
           - If Moving: LIFT legs (Prepare for next step).
        """
        
        # 1. Get True Velocity
        speed = np.sqrt(data.qvel[0]**2 + data.qvel[1]**2)
        
        # 2. Check Stall Condition
        is_stalled = (abs(throttle_cmd) > 0.2) and (speed < self.STALL_VEL_THRESH)
        
        # 3. SPLIT LOGIC: ARE WE TILTED?
        
        if pitch > self.TILT_THRESH:
            # === CASE: ON STAIRS (Tilted Up) ===
            
            if is_stalled:
                # STRATEGY: "Connect better with ground"
                # If we are stuck on the slope, flatten out to maximize wheelbase/traction
                self.bogie_angle += self.LOWER_SPEED
            else:
                # STRATEGY: "Moving up the step"
                # If we have momentum, keep the nose up to clear the NEXT step
                self.bogie_angle -= self.LIFT_SPEED
                
        else:
            # === CASE: FLAT GROUND / CURB APPROACH ===
            
            if is_stalled:
                self.stall_timer += dt
                # Only lift if stuck for a moment (avoids jitter)
                if self.stall_timer > 0.1:
                    # STRATEGY: Climb the Curb
                    self.bogie_angle -= self.LIFT_SPEED
            else:
                self.stall_timer = 0.0
                # STRATEGY: Cruise
                # Slowly return to flat
                self.bogie_angle += self.LOWER_SPEED
            
        # 4. Safety Clamps
        self.bogie_angle = np.clip(self.bogie_angle, self.MAX_LIFT, 0.0)
            
        return self.bogie_angle

# Global Instance
climber_logic = AdaptiveClimber()

# ==========================================
# 🎮 MAIN CONTROLLER FUNCTION
# ==========================================

def climb_control(model, data):
    # 1. Get Inputs
    drive_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "drive_forward")
    current_throttle = data.ctrl[drive_id]
    
    # 2. Get Pitch (New Input)
    current_pitch = get_body_pitch(model, data, "car")
    
    # 3. Run Logic
    new_climb_angle = climber_logic.update(data, current_throttle, current_pitch, model.opt.timestep)
    
    return new_climb_angle
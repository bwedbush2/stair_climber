import numpy as np
import mujoco

def get_body_pitch(model, data, body_name="car"):
    try:
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    except Exception: 
        return 0.0
    q = data.xquat[body_id]
    sinp = 2 * (q[0] * q[2] - q[3] * q[1])
    if abs(sinp) >= 1: 
        pitch = np.sign(sinp) * (np.pi / 2) 
    else: 
        pitch = np.arcsin(sinp)
    return pitch

def get_sensor_value(model, data, sensor_name):
    try:
        sens_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, sensor_name)
        adr = model.sensor_adr[sens_id]
        return data.sensordata[adr]
    except Exception:
        return -1.0
    
class AdaptiveClimber:
    def __init__(self):
        self.bogie_angle = 0.0
        self.stall_timer = 0.0
        
        # Tuning parameters
        self.LIFT_SPEED = 0.01
        self.LOWER_SPEED = 0.01
        self.MAX_LIFT = -1.2
        self.ATTACK_LIMIT = -0.6
        self.STALL_VEL_THRESH = 0.1 
        self.TILT_THRESH = 0.25

    def update(self, data, throttle_cmd, pitch, dt):
        speed = np.sqrt(data.qvel[0]**2 + data.qvel[1]**2)
        is_stalled = (abs(throttle_cmd) > 0.3) and (speed < self.STALL_VEL_THRESH)
        
        if pitch > self.TILT_THRESH:
            if is_stalled:
                self.bogie_angle += self.LOWER_SPEED
            else:
                self.bogie_angle -= self.LIFT_SPEED
        else:
            if is_stalled:
                self.stall_timer += dt
                # Traction/Hook adjustment
                if self.bogie_angle < self.ATTACK_LIMIT:
                    self.bogie_angle += self.LOWER_SPEED
                else:
                    self.bogie_angle -= self.LIFT_SPEED
            else:
                self.stall_timer = 0.0
                self.bogie_angle += self.LOWER_SPEED
            
        self.bogie_angle = np.clip(self.bogie_angle, self.MAX_LIFT, 0.0)
        return self.bogie_angle

climber_logic = AdaptiveClimber()

def climb_control(model, data):
    drive_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "drive_forward")
    current_throttle = data.ctrl[drive_id]
    current_pitch = get_body_pitch(model, data, "car")

    # Sensor readings for external logic if needed
    floor_dist_lower = get_sensor_value(model, data, "floor_sensL")
    floor_dist_upper = get_sensor_value(model, data, "floor_sensU")
    wall_dist = get_sensor_value(model, data, "wall_sens")

    return climber_logic.update(data, current_throttle, current_pitch, model.opt.timestep)
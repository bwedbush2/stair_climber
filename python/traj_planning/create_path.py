import mujoco
# import re

'''
This file looks at the scenario we are in and outputs a path
to follow. This should output (x,y) and maybe heading angle??

For now I will hard code the way points into each scenario
'''

def _get_scenario(model, scene: int):
    '''
    what scenario are we in?
    Expects 'RAY_Scenario_{int}'
    '''
    # name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_MODEL, 0)
    # match = re.search(r"Scenario_(\d+)", name)
    # if not match:
    #         raise ValueError(f"Failed extracting scenorio number from: '{name}'")
    # scenario = int(match.group(1))
    return scene

def _get_start_xy(model, body_name: str = "car"):
    """
    Returns the starting (x, y) position of the robot as defined in the model
    """
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)

    # model.body_pos is the initial position from the XML: [x, y, z]
    x, y, _ = model.body_pos[body_id]

    return float(x), float(y)


def create_path(model, data, scene: int) -> list[tuple[float, float]]:
    """
    Creates a list of (x, y) waypoints based on the scenario encoded in model.name.
    The first waypoint is the robot's current (x, y) position, and the last waypoint
    is the goal position. THESE POINTS ARE CURRENTLY HARD CODED.
    """
    scenario = _get_scenario(model, scene)
    x0, y0 = _get_start_xy(model, body_name="car")

    # Each list should start at some nominal start position and end at a goal position.
    scenario_paths = {
        1: [
            (x0, y0),       # start pos
            (2.0, 0.0),     # goal (middle of curb)
        ],
        2: [
            (x0, y0),       # start pos
            (3.0, 3.0),     # target1
            (6.0, 0.0),      # target2
        ],
        3: [
            (x0, y0),       # start pos
            (1.5, 0),       # top of stairs
            (3.5, 0),       # goal (downstairs and clearance)
        ],
        4: [
            (x0, y0),       # start pos
            (5.0, 0.0),     # down the ramp
            (5.0, -3.5),     # up the curb
            (4.0, -3.5),     # center of the curb
            (4.0, -7),       # before the stairs
            (4.0, -10),      # up the stairs
            (4.0, -12),     # goal (porch)
        ],
    }

    if scenario not in scenario_paths:
        raise ValueError(f"No path defined for scenario {scenario}")

    waypoints = list(scenario_paths[scenario])  # shallow copy so we can modify

    return waypoints
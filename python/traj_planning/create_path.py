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
    is the goal position.
    """
    scenario = _get_scenario(model, scene)
    x0, y0 = _get_start_xy(model, body_name="car")

    scenario_paths = {
        1: [
            (x0, y0, 2),       # start pos
            (2.0, 0.0, 2),     # goal (middle of curb)
        ],
        2: [
            (x0, y0, 2),       # start pos
            (3.0, 3.0, 2),     # target1
            (6.0, 0.0, 2),     # target2
            (6.0, -2.0, 2),      # target 3
            (3.0, -5.0, 2),     # target 4
            (-2.0, -5.0, 2),     # target 5
            (x0, y0, 2),         # back to start
        ],
        3: [
            (x0, y0, 2),       # start pos
            (4.0, 0, 2),       # top of stairs
            (8.5, 0, 2),       # goal (downstairs and clearance)
        ],
        4: [
            (x0, y0, 2),       # start pos
            (5.3, 0.0, 2),
            (5.5, 0.0, 2),     # down the ramp
            (5.5, -1.0, 2),
            (5.5, -3.3, 2),
            (5.5, -3.5, 2),    # up the curb
            
            # --- HOUSE 1 (x=4) ---
            (4.0, -3.5, 2),    # align with House 1
            (4.0, -7, 2),      # approach stairs
            (4.0, -10, 2),     # climb stairs
            (4.0, -12.5, 2),   # HOUSE 1 PORCH
            (4.0, -10, 2),     # reverse down
            (4.0, -7, 2),      # bottom of stairs
            (4.0, -3.5, 2),    # back to sidewalk

            # --- HOUSE 2 (x=-2) ---
            (-2.0, -3.5, 2),   # travel along sidewalk
            (-2.0, -7, 2),     # approach stairs
            (-2.0, -10, 2),    # climb stairs
            (-2.0, -12.5, 2),  # HOUSE 2 PORCH
            (-2.0, -10, 2),    # reverse down
            (-2.0, -7, 2),     # bottom of stairs
            (-2.0, -3.5, 2),   # back to sidewalk

            # --- HOUSE 3 (x=-8) V-SHAPED CLIMB ---
            (-8.0, -3.5, 2),   # travel along sidewalk
            (-8.0, -7.7, 2),   # start of stairs
            (-10.0, -9.5, 2),  # apex of V-staircase (bowing out)
            (-8.0, -11.3, 2),  # return to center at top of stairs
            (-8.0, -12.5, 2),  # HOUSE 3 PORCH (Final Goal)
        ],
    }

    if scenario not in scenario_paths:
        raise ValueError(f"No path defined for scenario {scenario}")

    waypoints = list(scenario_paths[scenario])

    return waypoints
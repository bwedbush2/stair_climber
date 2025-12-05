"""
Controllers package initializer.

This file exposes all controller classes so they can be imported as:

    from controllers import BasicController, MPCController, FSMController

It also provides a dictionary `AVAILABLE_CONTROLLERS` for selecting controllers
by string name inside simulate.py.
"""

from .basic_controller import BasicController
from .mpc_controller import MPCController
from .fsm_controller import FSMController  


# Dictionary for string-based controller selection
AVAILABLE_CONTROLLERS = {
    "basic": BasicController,
    "mpc": MPCController,
    "fsm": FSMController,
}

__all__ = [
    "BasicController",
    "MPCController",
    "FSMController",
    "AVAILABLE_CONTROLLERS",
]


### Import examples ###

## from controllers import FSMController
# OR 
## from controllers import AVAILABLE_CONTROLLERS
## controller = AVAILABLE_CONTROLLERS["fsm"](model, data, params)


### Switch between controllers ###

# controller_name = "fsm"     # or "basic", "mpc"
# ControllerClass = AVAILABLE_CONTROLLERS[controller_name]
# controller = ControllerClass(model, data, params)

import IsaacGymSimulator_shapenet as isaac_sim


from isaacgym import gymapi
from isaacgym import gymutil
from isaacgym import gymtorch
from isaacgym.torch_utils import *

if __name__ == "__main__":
    # Add custom arguments
    custom_parameters = [
        {"name": "--controller", "type": str, "default": "ik",
        "help": "Controller to use for Franka. Options are {ik, osc}"},
        {"name": "--num_envs", "type": int, "default": 256, "help": "Number of environments to create"},
        {"name": "--object", "type": str, "default": "025_mug", "help": "Object name as in YCB dataset"},
        {"name": "--quality_type", "type": str, "default": "None", "help": "Choose from ['top1', 'top2', 'bottom1', 'bottom2']"},
        {"name": "--headless", "type": str, "default": "Off", "help": "Headless=On has no graphics but faster simulations"},
    ]
    args = gymutil.parse_arguments(
        description="test",
        custom_parameters=custom_parameters,
    )
    isaac = isaac_sim.IsaacGymSim(args)


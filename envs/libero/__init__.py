import os
import sys
from pathlib import Path

# Add external/LIBERO to Python path so we can import from libero directly
libero_dir = Path(__file__).resolve().parent.parent.parent / "external" / "LIBERO"
if str(libero_dir) not in sys.path:
    sys.path.insert(0, str(libero_dir))

from libero.libero.envs import OffScreenRenderEnv
from libero.libero import benchmark

def make_libero_env(task_id: str, camera_setup: str):
    """
    Creates a LIBERO environment configured for the pi0.5 standard camera setup.
    
    Args:
        task_id: The ID or name of the task to instantiate.
        camera_setup: The camera configuration to use (e.g., 'pi0.5').
    """
    # Look up the task suite. For this script we assume it's part of libero_spatial
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict["libero_spatial"]()
    
    # Retrieve the bddl file for the requested task ID
    task = task_suite.get_task(task_id)
    task_bddl_file = os.path.join(task_suite.get_task_bddl_file_path(task_id))

    # Standard camera setup for pi0.5 (typically front and wrist cameras)
    if camera_setup == "pi0.5":
        camera_args = {
            "camera_heights": 256,
            "camera_widths": 256,
            "camera_depths": False,
            "camera_names": ["agentview", "robot0_eye_in_hand"],
        }
    else:
        # Default fallback
        camera_args = {
            "camera_heights": 128,
            "camera_widths": 128,
        }

    env_args = {
        "bddl_file_name": task_bddl_file,
        **camera_args
    }
    
    env = OffScreenRenderEnv(**env_args)
    return env

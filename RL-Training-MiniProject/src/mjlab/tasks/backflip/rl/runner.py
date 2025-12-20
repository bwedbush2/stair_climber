import os

import wandb
from rsl_rl.runners import OnPolicyRunner

from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.velocity.rl.exporter import (
  attach_onnx_metadata,
  export_velocity_policy_as_onnx,
)

import os
import wandb
from rsl_rl.runners import OnPolicyRunner
from mjlab.rl import RslRlVecEnvWrapper

class BackflipOnPolicyRunner(OnPolicyRunner):
    """
    Runner for the Backflip task. 
    Inherits directly from OnPolicyRunner to handle standard training and saving.
    """
    env: RslRlVecEnvWrapper

    def save(self, path: str, infos=None):
        """Save the model and training information."""
        # This calls the standard rsl_rl save function which saves 'model_*.pt'
        super().save(path, infos)
        
        # Optional: Save to WandB if you are using it
        if getattr(self, "logger_type", None) in ["wandb"]:
            # Check if the file exists before saving to avoid errors
            if os.path.exists(path):
                wandb.save(path, base_path=os.path.dirname(path))
                

class VelocityOnPolicyRunner(OnPolicyRunner):
  env: RslRlVecEnvWrapper

  def save(self, path: str, infos=None):
    """Save the model and training information."""
    super().save(path, infos)
    # if self.logger_type in ["wandb"]: ### OG LINE DELETE THE ONE BELOW IF NOTHING WORKS
    if getattr(self, "logger_type", None) in ["wandb"]:
      policy_path = path.split("model")[0]
      filename = os.path.basename(os.path.dirname(policy_path)) + ".onnx"
      if self.alg.policy.actor_obs_normalization:
        normalizer = self.alg.policy.actor_obs_normalizer
      else:
        normalizer = None
      export_velocity_policy_as_onnx(
        self.alg.policy,
        normalizer=normalizer,
        path=policy_path,
        filename=filename,
      )
      attach_onnx_metadata(
        self.env.unwrapped,
        wandb.run.name,  # type: ignore
        path=policy_path,
        filename=filename,
      )
      wandb.save(policy_path + filename, base_path=os.path.dirname(policy_path))

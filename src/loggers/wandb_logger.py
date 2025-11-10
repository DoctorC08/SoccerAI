import wandb
from typing import Any, Dict
import torch.nn as nn

class WandBLogger:
    def __init__(self, project: str, name: str, config: dict, reinit: Any = True, sweep: bool = False, **kwargs):
        self.sweep = sweep
        self.run = None
        if not sweep:
            self.run = wandb.init(
                project=project, 
                name=name, 
                config=config, 
                reinit=reinit,
                **kwargs
            )
            print(f"WandB run initialized: {self.run.name}")
        else:
            self.run = wandb.run 
            if self.run is None:
                raise RuntimeError("W&B sweep run is not initialized. Ensure wandb.init() is called in train_sweep.")
            
            self.run.name = name
            wandb.config.update(config)
            
            print(f"Sweep mode detected. Attached to W&B run: {self.run.name}")

    def log(self, data: Dict[str, float], step: int = None):
        #TODO: upload to wandb every N steps because logging is slow
        wandb.log(data, step=step)
    
    def watch_model(self, model: nn.Module, criterion=None, log: str = "all", log_freq=1000, idx=None):
        if criterion: 
            wandb.watch(model, criterion=criterion, log=log, log_freq=log_freq, idx=idx)
        else:
            wandb.watch(model, log=log, log_freq=log_freq, idx=idx)

    def update_config(self, new_config: dict):
        wandb.config.update(new_config)

    def close(self):
        if self.run: 
            self.run.finish()
        else:
            raise ValueError("No WandB run to finish.")
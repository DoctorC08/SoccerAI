import wandb
from typing import Any, Dict
import torch.nn as nn

class WandBLogger:
    def __init__(self, project: str, name: str, config: dict, reinit: Any =True, **kwargs):
        self.run = wandb.init(
            project=project, 
            name=name, 
            config=config, 
            reinit=reinit,
            **kwargs
        )

        print(f"WandB run initialized: {self.run.name}")

    def log(self, data: Dict[str, float], step: int = None):
        wandb.log(data, step=step)
    
    def watch_model(self, model: nn.Module, criterion=None, log: str = "all", log_freq=1000, idx=None):
        wandb.watch(model, criterion=criterion, log=log, log_freq=log_freq, idx=idx)

    def update_config(self, new_config: dict):
        wandb.config.update(new_config)

    def close(self):
        self.run.finish()
# utils/callbacks.py
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
from typing import List, Dict, Any
import torch 

class GradualUnfreezeCallback(Callback):
    """
    Callback to gradually unfreeze layers of a SwinV2 model and adjust learning rate.

    Expects the model name in pl_module.hparams to start with 'swinv2_'.
    Requires the optimizer to be configured with *all* model parameters initially.
    """
    def __init__(self, schedule: List[Dict[str, Any]]):
        """
        Args:
            schedule (List[Dict[str, Any]]): List of dictionaries, each containing:
                - 'epoch': The epoch number (0-indexed) at which to apply the change.
                - 'layers': A list of layer name prefixes to unfreeze (e.g., ['layers.3', 'norm']).
                - 'lr': The new learning rate to set for the *entire* optimizer.
        """
        super().__init__()
        # Sort schedule by epoch just in case
        self.schedule = sorted(schedule, key=lambda x: x['epoch'])
        self.current_stage = 0
        print(f"GradualUnfreezeCallback initialized with schedule: {self.schedule}")

    def on_train_epoch_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"):
        """Called when the train epoch begins."""
        current_epoch = trainer.current_epoch

        # Check if the model is SwinV2 (important check)
        if not pl_module.hparams.model_name.startswith('swinv2_'):
            if current_epoch == 0: # Print warning only once
                 print("Warning: GradualUnfreezeCallback is active, but the model name does not start with 'swinv2_'. Skipping unfreezing logic.")
            return

        # Find the next stage in the schedule
        next_stage_idx = -1
        for i, stage_info in enumerate(self.schedule):
            if current_epoch == stage_info['epoch']:
                 next_stage_idx = i
                 break

        if next_stage_idx != -1:
            stage_info = self.schedule[next_stage_idx]
            print(f"\n--- Gradual Unfreeze: Reached Epoch {current_epoch} ---")

            # 1. Unfreeze specified layers
            layers_to_unfreeze = stage_info['layers']
            print(f"Unfreezing layers starting with: {layers_to_unfreeze}")
            unfrozen_params_count = 0
            total_params_in_layers = 0

            for name, param in pl_module.model.named_parameters():
                 if any(name.startswith(prefix) for prefix in layers_to_unfreeze):
                    total_params_in_layers += param.numel()
                    if not param.requires_grad:
                        param.requires_grad = True
                        unfrozen_params_count += 1
                        # print(f"  Unfroze: {name}") # Optional: Verbose logging

            print(f"Unfroze {unfrozen_params_count} new parameter groups in specified layers ({total_params_in_layers} total params in these layers).")
            trainable_params = sum(p.numel() for p in pl_module.parameters() if p.requires_grad)
            print(f"Total trainable parameters now: {trainable_params}")


            # 2. Adjust Learning Rate
            new_lr = stage_info['lr']
            optimizer = trainer.optimizers[0] # Assuming one optimizer
            print(f"Setting optimizer learning rate to: {new_lr}")
            for param_group in optimizer.param_groups:
                 param_group['lr'] = new_lr

            self.current_stage = next_stage_idx + 1


class PrintCallback(Callback):
    def on_train_epoch_start(self, trainer, pl_module):
        print(f"\nEpoch {trainer.current_epoch} Start")

    # def on_train_epoch_end(self, trainer, pl_module):
    #     print(f"Epoch {trainer.current_epoch} End")

    # def on_validation_epoch_end(self, trainer, pl_module):
    #     print(f"Validation Epoch {trainer.current_epoch} End")
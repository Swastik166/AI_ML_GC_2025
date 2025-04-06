# models/classifier.py
import torch
import torch.nn as nn
import pytorch_lightning as pl
import timm
import torchvision.models as tv_models
from torchmetrics import MetricCollection, Accuracy, Precision, Recall, F1Score
from typing import Optional, Dict, List 

# --- Model Choices ---
DEFAULT_MODEL_CHOICES = [
    "resnet50", "resnet101",
    "efficientnet_b0", "efficientnet_b3", "efficientnet_b4",
    "vit_tiny_patch16_224", "vit_small_patch16_224", "vit_base_patch16_224",
    "swin_tiny_patch4_window7_224", "swin_small_patch4_window7_224", "swin_base_patch4_window7_224",
    "convnext_tiny", "convnext_small", "convnext_base", 'swinv2_base_window12to16_192to256.ms_in22k_ft_in1k',
    'swinv2_large_window12to16_192to256.ms_in22k_ft_in1k', 'coatnet_2_rw_224.sw_in12k_ft_in1k',
    "mobilenetv3_large_100" # Correct timm name
]

class GCClassifier(pl.LightningModule):
    """
    PyTorch Lightning Module for Image Classification.
    Uses timm for easy access to various pretrained models and supports
    fine-tuning or feature extraction (freezing backbone).
    Includes options for label smoothing and gradual unfreezing (for SwinV2).
    """
    def __init__(self,
                 model_name: str = "resnet50",
                 pretrained: bool = True,
                 freeze_backbone: bool = False,
                 num_classes: int = 200,
                 learning_rate: float = 1e-4,
                 optimizer_name: str = "adamw", # adamw, adam, sgd
                 lr_scheduler_name: Optional[str] = "reducelronplateau", # cosine, reducelronplateau, none
                 optimizer_kwargs: Optional[Dict] = None, 
                 lr_scheduler_kwargs: Optional[Dict] = None, 
                 label_smoothing: float = 0.0,
                 gradual_unfreeze_schedule: Optional[List[Dict]] = None
                 ):
        """
        Args:
            # ... (existing args) ...
            label_smoothing (float): Amount of label smoothing (0.0 to 1.0). 0.0 means no smoothing.
            gradual_unfreeze_schedule (Optional[List[Dict]]): Schedule for unfreezing SwinV2 layers.
                                                             Example: [{'epoch': 5, 'layers_to_unfreeze': ['layers.3', 'norm'], 'lr': 5e-5},
                                                                       {'epoch': 10, 'layers_to_unfreeze': ['layers.2'], 'lr': 1e-5}]
        """
        super().__init__()

        # Use self.hparams to access all init args
        self.save_hyperparameters()

        self.num_classes = num_classes 
        self.model = self._create_model()

        # --- Determine if Gradual Unfreezing is Active ---
        self.is_gradual_unfreezing = self.hparams.gradual_unfreeze_schedule is not None \
                                      and self.hparams.model_name.startswith('swinv2_')

        # --- Freezing Logic ---
        # If gradual unfreezing for swinv2, freeze backbone initially regardless of freeze_backbone flag.
        # The callback will handle unfreezing.
        if self.is_gradual_unfreezing:
            print("Gradual Unfreezing is active for SwinV2. Initializing with frozen backbone.")
            self._freeze_backbone(initial_freeze=True) # Force initial freeze
        elif self.hparams.freeze_backbone:
             self._freeze_backbone(initial_freeze=False) # Standard freezing

        # --- Loss Function ---
        print(f"Using Label Smoothing: {self.hparams.label_smoothing}")
        self.criterion = nn.CrossEntropyLoss(label_smoothing=self.hparams.label_smoothing)

        # --- Metrics ---
       
        metrics = MetricCollection({
            'acc': Accuracy(task="multiclass", num_classes=self.num_classes, top_k=1),
            'precision': Precision(task="multiclass", num_classes=self.num_classes, average='macro'),
            'recall': Recall(task="multiclass", num_classes=self.num_classes, average='macro'),
            'f1': F1Score(task="multiclass", num_classes=self.num_classes, average='macro')
        })
        self.train_metrics = metrics.clone(prefix='train_')
        self.val_metrics = metrics.clone(prefix='val_')


    def _create_model(self):
        
        model_name = self.hparams.model_name
        pretrained = self.hparams.pretrained
        num_classes = self.num_classes

        print(f"Creating model: {model_name} (pretrained={pretrained}, classes={num_classes})")

        # --- Try TIMM first ---
        try:
            model = timm.create_model(
                model_name,
                pretrained=pretrained,
                num_classes=num_classes # Directly set the final layer size
            )
            print(f"Loaded model from timm.")

            return model
        except Exception as e_timm:
            print(f"Could not load model '{model_name}' from timm: {e_timm}. Trying torchvision.")

        # --- Fallback to Torchvision ---
        try:
            weights = tv_models.get_model_weights(model_name).DEFAULT if pretrained else None
            print(f"Loading model from torchvision with weights: {'DEFAULT' if weights else None}")
            model = tv_models.get_model(model_name, weights=weights)

            # Replace the classifier layer
            in_features = self._get_torchvision_classifier_in_features(model)
            classifier_attr = self._find_classifier_layer_attr(model)

            if in_features and classifier_attr:
                setattr(model, classifier_attr, nn.Linear(in_features, num_classes))
                print(f"Replaced torchvision classifier layer '{classifier_attr}'.")
                return model
            else:
                 raise RuntimeError(f"Could not automatically find and replace classifier for torchvision model '{model_name}'.")

        except Exception as e_tv:
             # Combine error messages for better debugging
             raise RuntimeError(
                 f"Failed to load model '{model_name}' from both timm and torchvision.\n"
                 f"Timm Error: {e_timm}\n"
                 f"Torchvision Error: {e_tv}"
             )

    def _freeze_backbone(self, initial_freeze=False):
        """
        Freezes layers except the classifier.
        If initial_freeze is True (for gradual unfreezing), freezes everything except the head.
        """
        if initial_freeze:
            print("Initial freeze for gradual unfreezing: Freezing all layers except the head.")
        else:
            print("Freezing backbone layers...")

        classifier_params = set()
        classifier_param_names = set() 

        # Try to get classifier parameters directly using timm's helper
        try:
            classifier_module = self.model.get_classifier()
            if isinstance(classifier_module, nn.Module):
                 classifier_params.update(p for p in classifier_module.parameters())
                 classifier_param_names.update(name for name, _ in classifier_module.named_parameters())
                 print(f"Identified classifier module via get_classifier(): {type(classifier_module).__name__}")
            else:
                 raise AttributeError("get_classifier() did not return a Module")
        except AttributeError:
            # Fallback: Check common classifier layer names
            print("Could not get classifier module directly, using name-based fallback (fc, classifier, head).")
            classifier_keys = ['fc', 'classifier', 'head'] # Common names for the final layer(s)
            for name, param in self.model.named_parameters():
                # Check if the parameter name starts with one of the classifier keys
                # Assumes classifier module names are like 'head.xxx', 'fc.xxx'
                name_parts = name.split('.')
                if len(name_parts) > 0 and name_parts[0] in classifier_keys:
                    classifier_params.add(param)
                    classifier_param_names.add(name)

        if not classifier_params:
             print("Warning: Could not identify any classifier parameters for freezing. All layers will remain trainable unless gradual unfreezing overrides.")
             # If we are doing standard freeze_backbone and failed, don't freeze anything
             if not initial_freeze:
                 return
             # If we are doing initial freeze for gradual, still proceed to freeze all and let callback unfreeze head
             else:
                 print("Proceeding with initial freeze (will freeze all). Gradual unfreeze callback should unfreeze head.")


        # Freeze all parameters initially
        frozen_count = 0
        for name, param in self.model.named_parameters():
            param.requires_grad = False
            frozen_count += 1
        print(f"Initially froze {frozen_count} parameter groups.")

        # Unfreeze only the identified classifier parameters
        unfrozen_count = 0
        for name, param in self.model.named_parameters():
            if param in classifier_params:
                 param.requires_grad = True
                 unfrozen_count += 1
                 # print(f"  Unfreezing: {name}") # Optional: Verbose logging

        # Verify some parameters were unfrozen
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Successfully unfroze {unfrozen_count} classifier parameter groups (total {trainable_params} trainable parameters in the head).")
        if unfrozen_count == 0 and len(classifier_param_names) > 0:
             print(f"Warning: Identified classifier names {classifier_param_names} but corresponding parameters were not found/unfrozen.")


    def _get_torchvision_classifier_in_features(self, model):
        
        if hasattr(model, 'fc') and isinstance(model.fc, nn.Linear):
            return model.fc.in_features
        elif hasattr(model, 'classifier'):
            classifier = model.classifier
            if isinstance(classifier, nn.Linear):
                return classifier.in_features
            elif isinstance(classifier, nn.Sequential):
                for layer in reversed(classifier):
                    if isinstance(layer, nn.Linear):
                        return layer.in_features
        elif hasattr(model, 'head') and isinstance(model.head, nn.Linear):
             return model.head.in_features
        print(f"Warning: Could not determine in_features for torchvision model type {type(model).__name__}")
        return None

    def _find_classifier_layer_attr(self, model):
       
        if hasattr(model, 'fc') and isinstance(model.fc, nn.Linear):
            return 'fc'
        elif hasattr(model, 'classifier') and isinstance(model.classifier, (nn.Linear, nn.Sequential)):
            return 'classifier'
        elif hasattr(model, 'head') and isinstance(model.head, nn.Linear):
             return 'head'
        print(f"Warning: Could not find standard classifier attribute ('fc', 'classifier', 'head') for {type(model).__name__}")
        return None

    def forward(self, x):
       
        return self.model(x)

    def _common_step(self, batch, batch_idx):
        
        images, labels = batch
        logits = self(images)
        loss = self.criterion(logits, labels)
        preds = torch.argmax(logits, dim=1)
        return loss, preds, labels

    def training_step(self, batch, batch_idx):
       
        loss, preds, labels = self._common_step(batch, batch_idx)
        self.train_metrics.update(preds, labels)
        self.log('train_loss', loss, on_step=True, on_epoch=True, logger=True, prog_bar=True)
        self.log_dict(self.train_metrics, on_step=False, on_epoch=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        
        loss, preds, labels = self._common_step(batch, batch_idx)
        self.val_metrics.update(preds, labels)
        self.log('val_loss', loss, on_epoch=True, logger=True, prog_bar=True)
        self.log_dict(self.val_metrics, on_epoch=True, logger=True, prog_bar=True)
        return loss

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        
        if isinstance(batch, (list, tuple)) and len(batch) == 2:
             images, filenames = batch
        elif isinstance(batch, torch.Tensor):
             images = batch
             start_idx = batch_idx * images.size(0)
             filenames = [f"image_{start_idx + i}" for i in range(images.size(0))]
             print("Warning: Filenames not found in predict batch, generating dummy names.")
        else:
             raise TypeError(f"Unexpected batch type in predict_step: {type(batch)}")
        logits = self(images)
        preds = torch.argmax(logits, dim=1)
        return {"preds": preds, "filenames": filenames}


    def configure_optimizers(self):
        """Configure optimizer and learning rate scheduler."""
        lr = self.hparams.learning_rate
        opt_name = self.hparams.optimizer_name.lower()
        optimizer_kwargs = self.hparams.optimizer_kwargs or {}

    
        params_to_optimize = self.parameters() # Pass all parameters
        print("Optimizer will be configured with all model parameters.")
        if self.is_gradual_unfreezing or self.hparams.freeze_backbone:
             print("Note: Frozen parameters (requires_grad=False) will not be updated by the optimizer.")


        # --- Optimizer Selection ---
        if opt_name == "adam":
            optimizer = torch.optim.Adam(params_to_optimize, lr=lr, **optimizer_kwargs)
        elif opt_name == "adamw":
            optimizer = torch.optim.AdamW(params_to_optimize, lr=lr, **optimizer_kwargs)
        elif opt_name == "sgd":
            optimizer_kwargs.setdefault('momentum', 0.9)
            optimizer_kwargs.setdefault('weight_decay', 1e-4) # Adjusted default
            optimizer = torch.optim.SGD(params_to_optimize, lr=lr, **optimizer_kwargs)
        else:
            raise ValueError(f"Unsupported optimizer: {opt_name}")
        print(f"Using optimizer: {opt_name} with initial lr: {lr} and kwargs: {optimizer_kwargs}")

        # --- Scheduler Selection ---
        scheduler_name = self.hparams.lr_scheduler_name
        if scheduler_name is None or scheduler_name.lower() == "none":
            print("Using no Learning Rate Scheduler.")
            return optimizer

        print(f"Using LR Scheduler: {scheduler_name}")
        scheduler_kwargs = self.hparams.lr_scheduler_kwargs or {}

        if scheduler_name == "reducelronplateau":
            # ... (keep existing ReduceLROnPlateau config) ...
            scheduler_kwargs.setdefault('mode', 'min')
            scheduler_kwargs.setdefault('factor', 0.1)
            scheduler_kwargs.setdefault('patience', 5) # Consider matching EarlyStopping patience
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                **scheduler_kwargs
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": scheduler_kwargs.get("monitor", "val_loss"),
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        elif scheduler_name == "cosine":
             # Ensure trainer context is available or T_max is provided
             if 'T_max' not in scheduler_kwargs:
                 try:
                     if self.trainer and self.trainer.estimated_stepping_batches:
                         total_steps = self.trainer.estimated_stepping_batches
                         print(f"Cosine scheduler T_max estimated from trainer: {total_steps}")
                         scheduler_kwargs['T_max'] = total_steps
                     else:
                          raise ValueError("Cannot determine total steps for CosineAnnealingLR automatically. "
                                           "Ensure trainer is running or provide T_max in --lr_scheduler_kwargs.")
                 except AttributeError: # Handles case where trainer might not be attached yet
                      raise ValueError("Cannot determine total steps for CosineAnnealingLR. Provide T_max in --lr_scheduler_kwargs.")

             scheduler_kwargs.setdefault('eta_min', lr * 0.01) # Common default: 1% of initial LR
             scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                **scheduler_kwargs # Pass T_max and eta_min etc.
             )
             print(f"Cosine scheduler configured with: {scheduler_kwargs}")
             return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "step", # Cosine annealing updates per step
                },
             }
        else:
            raise ValueError(f"Unsupported LR scheduler: {scheduler_name}")
# main.py
import os
import argparse
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping
from pathlib import Path
import json # Import json for parsing schedule

from data.datamodule import GCDataModule
from models.classifier import GCClassifier, DEFAULT_MODEL_CHOICES
from utils.helpers import create_output_dir, save_label_map, plot_training_results
# Import the callback (we will create this file next)
from utils.callbacks import GradualUnfreezeCallback, PrintCallback

def get_args():
    parser = argparse.ArgumentParser(description="Train an Image Classification Model")

    # --- Data Args ---
  
    parser.add_argument('--data_dir', type=str, required=True, help='Root directory containing train, validation, test folders')
    parser.add_argument('--image_size', type=int, default=224, help='Target image size (height and width)')
    parser.add_argument('--num_workers', type=int, default=os.cpu_count() // 2, help='Number of dataloader workers')


    # --- Model Args ---

    parser.add_argument('--model_name', type=str, default='resnet50', choices=DEFAULT_MODEL_CHOICES + ['custom'],
                        help=f'Name of the pretrained model to use. Choices: {", ".join(DEFAULT_MODEL_CHOICES)}')
    parser.add_argument('--pretrained', action=argparse.BooleanOptionalAction, default=True, help='Use pretrained weights')
    parser.add_argument('--freeze_backbone', action=argparse.BooleanOptionalAction, default=False, help='Freeze layers except the final classifier (ignored if --gradual_unfreeze_schedule is used for swinv2)')


    # --- Training Args ---
  
    parser.add_argument('--epochs', type=int, default=20, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size per device')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Initial learning rate')
    parser.add_argument('--optimizer', type=str, default='adamw', choices=['adam', 'adamw', 'sgd'], help='Optimizer type')
    parser.add_argument('--lr_scheduler', type=str, default='reducelronplateau', choices=['reducelronplateau', 'cosine', 'none'], help='Learning rate scheduler')
    parser.add_argument('--num_augmentations', type=int, default=1, help='Number of augmented copies per training image')
    parser.add_argument('--patience', type=int, default=10, help='Patience for EarlyStopping (monitoring val_loss)')
    parser.add_argument('--label_smoothing', type=float, default=0.0, help='Amount of label smoothing (e.g., 0.1)') # <<< New argument
    parser.add_argument('--gradual_unfreeze_schedule', type=str, default=None, 
                        help='JSON string defining the gradual unfreeze schedule for SwinV2 models. '
                             'Example: \'[{"epoch": 5, "layers": ["layers.3", "norm"], "lr": 5e-5}, {"epoch": 10, "layers": ["layers.2"], "lr": 1e-5}]\'')


    # --- Infrastructure Args ---
    parser.add_argument('--output_base_dir', type=str, default='./training_outputs', help='Base directory to save run outputs')
    parser.add_argument('--accelerator', type=str, default='gpu', choices=['gpu', 'cpu', 'mps', 'tpu', 'auto'], help='Hardware accelerator')
    parser.add_argument('--devices', type=str, default='1', help='Number of devices (e.g., "1", "4") or specific IDs (e.g., "0,1", "[0, 2]")')
    parser.add_argument('--precision', type=str, default='32-true', help='Training precision (e.g., "32-true", "16-mixed")')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--run_name', type=str, default=None, help='Optional specific name for this run')


    args = parser.parse_args()

    # --- Process devices ---
    if args.accelerator in ['gpu', 'auto']:
         try:
             
             args.devices = int(args.devices)
             if args.devices == -1: args.devices = 'auto' # Allow -1 for auto GPU count
         except ValueError:

             pass
    elif args.accelerator == 'cpu':
        try: 
            args.devices = int(args.devices) if int(args.devices) > 0 else 1
        except ValueError:
            args.devices = 1 # Default to 1 CPU process if invalid string
    else:
         args.devices = 1 # Fallback


    print(f"Using accelerator: {args.accelerator}, devices: {args.devices}")

    # --- Parse Gradual Unfreeze Schedule ---
    if args.gradual_unfreeze_schedule:
        try:
            args.gradual_unfreeze_schedule = json.loads(args.gradual_unfreeze_schedule)
            # Basic validation
            if not isinstance(args.gradual_unfreeze_schedule, list):
                raise ValueError("Schedule must be a list of dictionaries.")
            for item in args.gradual_unfreeze_schedule:
                if not all(k in item for k in ['epoch', 'layers', 'lr']):
                    raise ValueError("Each schedule item must contain 'epoch', 'layers', and 'lr'.")
            # Sort schedule by epoch
            args.gradual_unfreeze_schedule.sort(key=lambda x: x['epoch'])
            print(f"Parsed gradual unfreeze schedule: {args.gradual_unfreeze_schedule}")
        except Exception as e:
             raise ValueError(f"Error parsing --gradual_unfreeze_schedule JSON: {e}")


    return args


def main():
    args = get_args()
    pl.seed_everything(args.seed, workers=True)

    # --- Create Output Directory ---
    setting_name_parts = []
    if args.model_name.startswith('swinv2_') and args.gradual_unfreeze_schedule:
        setting_name_parts.append("gradual_unfreeze")
    elif args.freeze_backbone:
        setting_name_parts.append("frozen")
    else:
        setting_name_parts.append("finetuned")
    setting_name_parts.extend([
        f"{args.image_size}px",
        f"lr{args.learning_rate}",
        f"bs{args.batch_size}",
        f"aug{args.num_augmentations}",
        f"ls{args.label_smoothing}" # Add label smoothing info
    ])
    setting_name = "_".join(setting_name_parts)

    # setting_name = f"{'frozen' if args.freeze_backbone else 'finetuned'}_{args.image_size}px_lr{args.learning_rate}_bs{args.batch_size}_augment{args.num_augmentations}"
    run_name = args.run_name if args.run_name else f"{args.model_name}_{setting_name}"
    output_dir = create_output_dir(args.output_base_dir, run_name)

    # --- Data Module ---
    datamodule = GCDataModule(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        image_size=args.image_size,
        num_augmentations=args.num_augmentations
    )
    # Don't call setup here if lazy loading is okay, trainer.fit will call it.
    # Call prepare_data if needed (e.g., downloading)
    # datamodule.prepare_data()
    # datamodule.setup('fit') # Setup explicitly to get num_classes early

    # --- Determine num_classes ---
    # Need to setup at least the train dataset to get class info
    print("Setting up datamodule to determine number of classes...")
    datamodule.setup('fit')
    num_classes = datamodule.num_classes
    if num_classes is None:
        raise RuntimeError("Could not determine the number of classes from the training data.")
    print(f"Determined number of classes: {num_classes}")

    # --- Save Label Map ---
    if datamodule.int_to_label:
        label_map_path = os.path.join(output_dir, "int_to_label.pkl")
        save_label_map(datamodule.int_to_label, label_map_path)
    else:
        print("Warning: int_to_label map not found in datamodule after setup.")


    # --- Model ---
    model = GCClassifier(
        model_name=args.model_name,
        pretrained=args.pretrained,
        freeze_backbone=args.freeze_backbone, # Pass it, but logic inside init handles override
        num_classes=num_classes,
        learning_rate=args.learning_rate,
        optimizer_name=args.optimizer,
        lr_scheduler_name=args.lr_scheduler,
        label_smoothing=args.label_smoothing,  
        gradual_unfreeze_schedule=args.gradual_unfreeze_schedule 
    )

    # --- Callbacks ---
    callbacks = []
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(output_dir, 'checkpoints'),
        filename='best_model-{epoch:02d}-{val_loss:.3f}-{val_acc:.4f}', # Increased precision
        save_top_k=1,
        monitor='val_acc', # Monitor validation accuracy
        mode='max',
        save_last=True
    )
    callbacks.append(checkpoint_callback)

    lr_monitor = LearningRateMonitor(logging_interval='epoch') # Log LR per epoch
    callbacks.append(lr_monitor)

    early_stopping_callback = EarlyStopping(
        monitor='val_loss', # Stop based on validation loss
        patience=args.patience,
        verbose=True,
        mode='min'
    )
    callbacks.append(early_stopping_callback)

    # Add Gradual Unfreezing Callback if applicable
    if model.is_gradual_unfreezing: # Check the flag set in model.__init__
        print("Adding GradualUnfreezeCallback.")
        gradual_unfreeze_callback = GradualUnfreezeCallback(
            schedule=args.gradual_unfreeze_schedule # Pass the parsed schedule
        )
        callbacks.append(gradual_unfreeze_callback)

    
    callbacks.append(PrintCallback())


    # --- Loggers ---
    tensorboard_save_dir = os.path.join(output_dir, 'tensorboard_logs')
    csv_save_dir = os.path.join(output_dir, 'csv_logs')
    # Use subdirectories for different runs if preferred, but name='' version='' saves directly in save_dir
    tb_logger = TensorBoardLogger(save_dir=tensorboard_save_dir, name='', version='')
    csv_logger = CSVLogger(save_dir=csv_save_dir, name='', version='')

    # --- Trainer ---
    trainer = pl.Trainer(
        accelerator=args.accelerator,
        devices=args.devices,
        max_epochs=args.epochs,
        logger=[tb_logger, csv_logger],
        callbacks=callbacks, # Pass the list of callbacks
        precision=args.precision,
        log_every_n_steps=max(1, 50 // args.batch_size), # Log ~50 steps per epoch roughly
        deterministic=True, # Ensure reproducibility if seed is set
        
        # strategy='ddp_find_unused_parameters_true' if args.devices > 1 and isinstance(args.devices, int) else None,
    )

    # --- Training ---
    print(f"\n--- Starting Training ---")
    print(f" Model: {args.model_name}")
    print(f" Classes: {num_classes}")
    print(f" Image Size: {args.image_size}x{args.image_size}")
    print(f" Pretrained: {args.pretrained}")
    #
    print(f" Initial State: {'Initially Frozen Backbone (Gradual Unfreeze)' if model.is_gradual_unfreezing else ('Frozen Backbone' if args.freeze_backbone else 'Full Finetuning')}")
    print(f" Epochs: {args.epochs}")
    print(f" Batch Size: {args.batch_size} (per device)")
    print(f" Base LR: {args.learning_rate}, Optimizer: {args.optimizer}, Scheduler: {args.lr_scheduler}")
    print(f" Label Smoothing: {args.label_smoothing}")
    if model.is_gradual_unfreezing:
        print(f" Gradual Unfreeze Schedule: {args.gradual_unfreeze_schedule}")
    print(f" Augmentations per image: {args.num_augmentations}")
    print(f" Early Stopping Patience: {args.patience}")
    # Use trainer attributes for resolved accelerator/devices
    print(f" Accelerator: {trainer.accelerator.__class__.__name__}, Devices: {trainer.num_devices}")
    print(f" Precision: {args.precision}")
    print(f" Output Dir: {output_dir}")
    print(f"-------------------------\n")

    trainer.fit(model, datamodule=datamodule)

    # --- Final Validation ---
    print("\n--- Running Final Validation on Best Checkpoint ---")
    best_ckpt_path = checkpoint_callback.best_model_path
    last_ckpt_path = os.path.join(checkpoint_callback.dirpath, "last.ckpt")
    validate_path = None

    if best_ckpt_path and os.path.exists(best_ckpt_path):
         print(f"Using best checkpoint: {best_ckpt_path}")
         validate_path = best_ckpt_path
    elif os.path.exists(last_ckpt_path):
         print(f"Best checkpoint '{best_ckpt_path}' not found or invalid. Using last checkpoint: {last_ckpt_path}")
         validate_path = last_ckpt_path # Trainer understands "last" but explicit path is safer
    else:
         print("No best or last checkpoint found for final validation.")

    if validate_path:
        try:
            val_results = trainer.validate(model=None, datamodule=datamodule, ckpt_path=validate_path) # Use model=None when using ckpt_path
            print(f"Validation Results ({os.path.basename(validate_path)}):", val_results)
        except Exception as e:
            print(f"Could not run final validation on {validate_path}: {e}")


    # --- Post-Training ---
   
    print(f"\n--- Training Finished ---")
    if checkpoint_callback.best_model_path and os.path.exists(checkpoint_callback.best_model_path):
        print(f"Best model checkpoint saved in: {checkpoint_callback.best_model_path}")
    else:
        print("Best model checkpoint not found or not saved.")
    print(f"Logs saved in: {output_dir}")
    print(f"To view TensorBoard logs: tensorboard --logdir {tensorboard_save_dir}")

    print("\n--- Generating Training Plots ---")
    csv_file_path = os.path.join(csv_logger.save_dir, 'metrics.csv') 
    plot_output_path = os.path.join(output_dir, 'training_plots.png')

    if os.path.exists(csv_file_path):
        plot_training_results(csv_log_path=csv_file_path, output_plot_path=plot_output_path)
    else:
        print(f"Could not find metrics file at {csv_file_path} to generate plots.")


if __name__ == '__main__':
    main()

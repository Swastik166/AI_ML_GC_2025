# predict.py
import os
import argparse
import pandas as pd
import torch
import pytorch_lightning as pl
from tqdm import tqdm

from data.datamodule import GCDataModule 
from data.dataset import TestDataset 
from models.classifier import GCClassifier 
from utils.helpers import load_label_map
from utils.augmentations import get_val_test_transforms 

def get_args():
    parser = argparse.ArgumentParser(description="Generate Predictions for Bird Classification")

    # --- Input Args ---
    parser.add_argument('--checkpoint_path', type=str, required=True, help='Path to the best model .ckpt file')
    parser.add_argument('--test_data_dir', type=str, required=True, help='Path to the test data folder (containing image1.jpg, etc.)')
    parser.add_argument('--label_map_path', type=str, required=True, help='Path to the saved integer-to-label map file (e.g., int_to_label.pkl/json)')

    # --- Output Args ---
    parser.add_argument('--output_csv', type=str, default='submission.csv', help='Path to save the output CSV file')

    # --- Inference Args ---
    parser.add_argument('--image_size', type=int, default=224, help='Image size used during training (must match)')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for prediction (can be larger than training)')
    parser.add_argument('--num_workers', type=int, default=os.cpu_count() // 2, help='Number of dataloader workers')
    parser.add_argument('--accelerator', type=str, default='gpu', choices=['gpu', 'cpu', 'mps', 'tpu', 'auto'], help='Hardware accelerator')
    parser.add_argument('--devices', type=str, default='1', help='Number or IDs of devices to use')
   

    args = parser.parse_args()

 
    if args.accelerator == 'gpu' or args.accelerator == 'auto':
         try:
             args.devices = int(args.devices)
         except ValueError:
             pass 
    else:
        try:
             args.devices = int(args.devices) if int(args.devices) > 0 else 1
        except ValueError:
            args.devices = 1

    return args

def main():
    args = get_args()

    # --- Load Label Map ---
    print(f"Loading label map from: {args.label_map_path}")
    try:
        int_to_label = load_label_map(args.label_map_path)
        print(f"Loaded {len(int_to_label)} labels.")
    except Exception as e:
        print(f"Error loading label map: {e}")
        return

    # --- Load Model ---
    print(f"Loading model from checkpoint: {args.checkpoint_path}")
    try:
        # Load the model class structure and weights
        model = GCClassifier.load_from_checkpoint(args.checkpoint_path, map_location='cpu') # Load to CPU first
        model.eval() # Set model to evaluation mode
    except Exception as e:
         print(f"Error loading model checkpoint: {e}")
         print("Make sure the checkpoint file exists and the BirdClassifier class definition is available.")
         return

    # --- Prepare Data ---
    print(f"Preparing test data from: {args.test_data_dir}")
    test_transforms = get_val_test_transforms(args.image_size) # Use the *validation* transforms for test
    test_dataset = TestDataset(args.test_data_dir, transform=test_transforms)

   
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )

    # --- Setup Trainer for Prediction ---
    trainer = pl.Trainer(
        accelerator=args.accelerator,
        devices=args.devices,
        logger=False, 
        callbacks=[],
        # precision=args.precision # Generally not needed/set for predict
    )

    # --- Run Prediction ---
    print(f"Running prediction using {args.accelerator} on devices: {args.devices}")
    # The predict method returns a list of outputs from predict_step (one item per batch)
    predictions_list = trainer.predict(model, dataloaders=test_dataloader, return_predictions=True)

    if predictions_list is None:
         print("Prediction returned None. Check dataloader and model.")
         return

    # --- Process Results ---
    print("Processing predictions...")
    all_filenames = []
    all_pred_indices = []

    
    for batch_output in tqdm(predictions_list, desc="Aggregating batches"):
        all_filenames.extend(batch_output['filenames'])
        # Ensure predictions are on CPU and converted to numpy/list
        all_pred_indices.extend(batch_output['preds'].cpu().tolist())

    # Convert predicted indices to original string labels
    predicted_labels = [int(int_to_label[idx]) for idx in all_pred_indices]

    # --- Create Submission File ---
    print(f"Creating submission file: {args.output_csv}")
    submission_df = pd.DataFrame({
        'ID': all_filenames,
        'label': predicted_labels
    })

    
    
    try:
         submission_df['sort_key'] = submission_df['ID'].str.extract(r'(\d+)').astype(int)
         submission_df = submission_df.sort_values(by='sort_key').drop(columns='sort_key')
    except Exception as e:
         print(f"Could not sort by numeric ID ({e}). Sorting alphabetically by ID.")
         submission_df = submission_df.sort_values(by='ID')


    submission_df.to_csv(args.output_csv, index=False)
    print(f"Submission CSV saved successfully to {args.output_csv}")


if __name__ == '__main__':
    main()
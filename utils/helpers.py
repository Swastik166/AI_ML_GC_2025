# utils/helpers.py
import os
import json
import pickle
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def create_output_dir(base_dir: str, run_name: str) -> str:
    """Creates the output directory for a specific run."""
    output_dir = Path(base_dir) / run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory created: {output_dir}")
    return str(output_dir)

def save_label_map(label_map: dict, filename: str):
    """Saves the integer-to-label mapping to a file (JSON or Pickle)."""
    filepath = Path(filename)
    if filepath.suffix == '.json':
        with open(filepath, 'w') as f:
            json.dump(label_map, f, indent=4)
        print(f"Label map saved to {filepath}")
    elif filepath.suffix == '.pkl':
        with open(filepath, 'wb') as f:
            pickle.dump(label_map, f)
        print(f"Label map saved to {filepath}")
    else:
        # Default to pickle if extension is unknown or missing
         if not filepath.suffix:
             filepath = filepath.with_suffix('.pkl')
         with open(filepath, 'wb') as f:
             pickle.dump(label_map, f)
         print(f"Label map saved to {filepath} (defaulted to pickle)")


def load_label_map(filename: str) -> dict:
    """Loads the integer-to-label mapping from a file."""
    filepath = Path(filename)
    if not filepath.exists():
        # Try adding default extension if needed
        if not filepath.suffix:
            filepath_pkl = filepath.with_suffix('.pkl')
            filepath_json = filepath.with_suffix('.json')
            if filepath_pkl.exists():
                filepath = filepath_pkl
            elif filepath_json.exists():
                filepath = filepath_json
            else:
                 raise FileNotFoundError(f"Label map file not found: {filename} (or .pkl/.json)")

    print(f"Loading label map from: {filepath}")
    if filepath.suffix == '.json':
        with open(filepath, 'r') as f:
            label_map = json.load(f)
            # JSON saves keys as strings, convert back to int if needed
            try:
                return {int(k): v for k, v in label_map.items()}
            except ValueError:
                 print("Warning: Could not convert JSON keys to integers. Assuming string keys are intended.")
                 return label_map # Return as is if keys aren't ints
    elif filepath.suffix == '.pkl':
        with open(filepath, 'rb') as f:
            label_map = pickle.load(f)
        return label_map
    else:
        raise ValueError(f"Unsupported file extension for label map: {filepath.suffix}")
    
    
    
    
def plot_training_results(csv_log_path: str, output_plot_path: str):
    """
    Reads metrics from a CSVLogger file and plots training/validation loss and accuracy.
    Handles cases where epoch-level train/val metrics might be on adjacent rows.

    Args:
        csv_log_path (str): Path to the metrics.csv file generated by CSVLogger.
        output_plot_path (str): Path to save the generated plot image (e.g., 'plots.png').
    """
    csv_path = Path(csv_log_path)
    if not csv_path.is_file():
        print(f"Warning: CSV log file not found at {csv_log_path}. Skipping plotting.")
        return

    try:
        metrics_df = pd.read_csv(csv_path, na_values=['', ' '])
        print(f"Successfully loaded CSV: {csv_log_path}")
        print(f"Columns found: {metrics_df.columns.tolist()}")
    except Exception as e:
        print(f"Error reading CSV log file {csv_log_path}: {e}. Skipping plotting.")
        return


    val_loss_col = 'val_loss'
    val_acc_col = 'val_acc'
    train_loss_col = 'train_loss_epoch'
    train_acc_col = 'train_acc'

    # --- Check if essential base columns exist ---
    base_required_cols = ['epoch', val_loss_col, train_loss_col]
    missing_base_cols = [col for col in base_required_cols if col not in metrics_df.columns]
    if missing_base_cols:
        print(f"Warning: Missing essential columns ({missing_base_cols}) in {csv_log_path}. Check logging keys in LightningModule. Skipping plotting.")
        return

    # --- Data Preparation ---
    # 1. Find epochs where validation definitely completed (val_loss is not NaN)
    try:
        metrics_df['epoch'] = pd.to_numeric(metrics_df['epoch'], errors='coerce')
        metrics_df.dropna(subset=['epoch'], inplace=True) 
        metrics_df['epoch'] = metrics_df['epoch'].astype(int)

        # Ensure validation loss is numeric
        metrics_df[val_loss_col] = pd.to_numeric(metrics_df[val_loss_col], errors='coerce')

        completed_epochs = metrics_df.loc[metrics_df[val_loss_col].notna(), 'epoch'].unique()
        if len(completed_epochs) == 0:
             print(f"Warning: No epochs found with non-NaN '{val_loss_col}' values in {csv_log_path}. Skipping plotting.")
             return
        print(f"Found completed epochs based on '{val_loss_col}': {completed_epochs.tolist()}")

    except Exception as e:
        print(f"Error processing epoch or validation loss columns: {e}. Skipping plotting.")
        return

    # 2. Filter the original DataFrame to include only rows from these completed epochs
    epoch_filtered_df = metrics_df[metrics_df['epoch'].isin(completed_epochs)].copy() 

    # 3. Convert metric columns to numeric, coercing errors
    metric_cols_to_convert = [val_loss_col, val_acc_col, train_loss_col, train_acc_col]
    for col in metric_cols_to_convert:
        if col in epoch_filtered_df.columns:
            epoch_filtered_df[col] = pd.to_numeric(epoch_filtered_df[col], errors='coerce')
        else:
            print(f"Note: Metric column '{col}' not found during numeric conversion.")


    # 4. Group by epoch and aggregate using mean. NaN values will be ignored by mean() for each group.
    # This handles cases where train/val metrics are on different rows within the same epoch group.
    try:
        epoch_data_agg = epoch_filtered_df.groupby('epoch').mean().reset_index()
        print(f"Aggregated data shape (rows, cols): {epoch_data_agg.shape}")
        # print("Aggregated data head:\n", epoch_data_agg.head()) 
    except Exception as e:
        print(f"Error during aggregation: {e}. Skipping plotting.")
        return


    # --- Check if aggregated data is usable ---
    if epoch_data_agg.empty:
        print(f"Warning: Aggregated data is empty after grouping epochs in {csv_log_path}. Skipping plotting.")
        return

    # Check again for necessary columns AFTER aggregation (mean might produce NaN if no value existed)
    plot_val_acc = val_acc_col in epoch_data_agg.columns and epoch_data_agg[val_acc_col].notna().any()
    plot_train_acc = train_acc_col in epoch_data_agg.columns and epoch_data_agg[train_acc_col].notna().any()
    plot_val_loss = val_loss_col in epoch_data_agg.columns and epoch_data_agg[val_loss_col].notna().any()
    plot_train_loss = train_loss_col in epoch_data_agg.columns and epoch_data_agg[train_loss_col].notna().any()

    if not (plot_val_loss and plot_train_loss):
         print(f"Warning: Missing aggregated loss data (need non-NaN '{train_loss_col}' and '{val_loss_col}'). Skipping plotting.")
         return


    epochs = epoch_data_agg['epoch']

    # --- Plotting ---
    plt.style.use('ggplot')
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))
    fig.suptitle(f'Training History ({csv_path.parent.parent.name})', fontsize=14) 


    # Loss Plot
    if plot_train_loss:
        axes[0].plot(epochs, epoch_data_agg[train_loss_col], label='Training Loss', marker='.')
    if plot_val_loss:
        axes[0].plot(epochs, epoch_data_agg[val_loss_col], label='Validation Loss', marker='.')
    axes[0].set_title('Loss vs. Epoch')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    if plot_train_loss or plot_val_loss:
        axes[0].legend()
    axes[0].grid(True)

    # Accuracy Plot
    if plot_train_acc:
        axes[1].plot(epochs, epoch_data_agg[train_acc_col], label='Training Accuracy', marker='.')
    if plot_val_acc:
        axes[1].plot(epochs, epoch_data_agg[val_acc_col], label='Validation Accuracy', marker='.')

    if plot_train_acc or plot_val_acc:
        axes[1].set_title('Accuracy vs. Epoch')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy')
        axes[1].legend()
        axes[1].grid(True)
    else:
        axes[1].set_title('Accuracy Plot Skipped')
        axes[1].text(0.5, 0.5, 'No accuracy data found', horizontalalignment='center', verticalalignment='center', transform=axes[1].transAxes)


    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) 
    try:
        # Ensure the output directory exists (though it should from main.py)
        Path(output_plot_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_plot_path)
        print(f"Training plots saved to: {output_plot_path}")
    except Exception as e:
        print(f"Error saving plot to {output_plot_path}: {e}")
    plt.close(fig)
import os
import numpy as np
import nibabel as nib
import SimpleITK as sitk
import matplotlib.pyplot as plt

def show_prediction(prediction_dir, file_name, save_path=None):
    """
    Displays slices of a NIfTI prediction file in a grid layout.

    Args:
        prediction_dir (str): Directory containing prediction files.
        file_name (str): Name of the NIfTI prediction file (e.g., 'BRATS_003.nii.gz').
        save_path (str, optional): If provided, saves the plot to this path instead of displaying it.
    """
    file_path = os.path.join(prediction_dir, file_name)
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return

    img = sitk.ReadImage(file_path)
    img_array = sitk.GetArrayFromImage(img)

    n_slices = img_array.shape[0]
    n_cols = 10
    n_rows = n_slices // n_cols + (n_slices % n_cols != 0)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, n_rows * 2))
    axes = axes.flatten()

    for i in range(n_slices):
        axes[i].imshow(img_array[i, :, :], cmap="gray")
        axes[i].set_title(f"Slice {i}", fontsize=6)
        axes[i].axis('off')

    for i in range(n_slices, len(axes)):
        axes[i].axis('off')

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        print(f"Saved prediction visualization to {save_path}")
    else:
        plt.show()

# === Example usage ===
if __name__ == "__main__":
    pred_dir = "/content/predictions"
    filename = "BRATS_003.nii.gz"
    output_path = "./report/predictions/BRATS_003_plot.png"
    show_prediction(pred_dir, filename, save_path=output_path)

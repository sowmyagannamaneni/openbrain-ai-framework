import nibabel as nib
import os
import matplotlib.pyplot as plt

def show_predictions(true_dir, pred_dir, num_samples=5):
    true_files = [f for f in os.listdir(true_dir) if f.endswith('.nii.gz')]
    pred_files = [f for f in os.listdir(pred_dir) if f.endswith('.nii.gz')]

    fig, axes = plt.subplots(num_samples, 3, figsize=(15, 5*num_samples))

    for i in range(num_samples):
        true_img = nib.load(os.path.join(true_dir, true_files[i])).get_fdata()
        pred_img = nib.load(os.path.join(pred_dir, pred_files[i])).get_fdata()

        slice_idx = true_img.shape[0] // 2

        axes[i, 0].imshow(true_img[slice_idx, :, :], cmap='gray')
        axes[i, 0].set_title('Original Image')

        axes[i, 1].imshow(true_img[slice_idx, :, :], cmap='gray')
        axes[i, 1].imshow(pred_img[slice_idx, :, :], cmap='jet', alpha=0.5)
        axes[i, 1].set_title('True Label + Prediction')

        axes[i, 2].imshow(pred_img[slice_idx, :, :], cmap='gray')
        axes[i, 2].set_title('Prediction')

    plt.tight_layout()
    plt.show()

show_predictions('/content/nnUNet_raw/Dataset001_BrainTumour/labelsTr', "/content/predictions")

import SimpleITK as sitk
import matplotlib.pyplot as plt
import os

def show_prediction(prediction_dir, file_name):
    file_path = os.path.join(prediction_dir, file_name)
    if not os.path.exists(file_path):
        print(f"File {file_path} does not exist.")
        return

    img = sitk.ReadImage(file_path)
    img_array = sitk.GetArrayFromImage(img)

    # Adjust subplot dimensions
    n_slices = img_array.shape[0]
    n_cols = 10
    n_rows = n_slices // n_cols + (n_slices % n_cols != 0)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 15))
    axes = axes.flatten()

    for i in range(n_slices):
        axes[i].imshow(img_array[i, :, :], cmap="gray")
        axes[i].axis('off')
        axes[i].set_title(f'Slice {i + 1}')

    # Hide any remaining axes
    for i in range(n_slices, len(axes)):
        axes[i].axis('off')

    plt.show()

# Call the function to show the prediction
show_prediction("/content/predictions", "BRATS_003.nii.gz")

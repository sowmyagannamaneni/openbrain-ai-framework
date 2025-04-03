from sklearn.metrics import confusion_matrix
import seaborn as sns
import nibabel as nib
import os
import matplotlib.pyplot as plt

def plot_confusion_matrix(true_dir, pred_dir, class_names):
    y_true = []
    y_pred = []

    for true_file in os.listdir(true_dir):
        if true_file.endswith(".nii.gz"):
            true_path = os.path.join(true_dir, true_file)
            pred_path = os.path.join(pred_dir, true_file)

            if not os.path.exists(pred_path):
                continue

            true_img = nib.load(true_path).get_fdata().astype(int).flatten()
            pred_img = nib.load(pred_path).get_fdata().astype(int).flatten()

            y_true.extend(true_img)
            y_pred.extend(pred_img)

    cm = confusion_matrix(y_true, y_pred, labels=range(len(class_names)))
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=class_names, yticklabels=class_names, cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()

class_names = ['Background', 'Whole Tumor', 'Tumor Core', 'Enhancing Tumor']
plot_confusion_matrix('/content/nnUNet_raw/Dataset001_BrainTumour/labelsTr', "/content/predictions", class_names)

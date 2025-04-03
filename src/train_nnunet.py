import os
import zipfile
import shutil
import nibabel as nib
import numpy as np
from batchgenerators.utilities.file_and_folder_operations import save_json, join

# === Extract BraTS dataset ===
zip_path = "/content/drive/MyDrive/dataset_brats/archive.zip"
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall("/content")

# === Set paths ===
SRC_TR = "/content/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData"
SRC_TS = "/content/BraTS2020_ValidationData/MICCAI_BraTS2020_ValidationData"
RAW_DIR = "/content/nnUNet_raw/Dataset001_BrainTumour"

# === Setup folders ===
os.makedirs(f"{RAW_DIR}/imagesTr", exist_ok=True)
os.makedirs(f"{RAW_DIR}/imagesTs", exist_ok=True)
os.makedirs(f"{RAW_DIR}/labelsTr", exist_ok=True)

modalities = {"flair": "0000", "t1": "0001", "t1ce": "0002", "t2": "0003"}
train_folders = sorted([f for f in os.listdir(SRC_TR) if not f.endswith("csv")])
val_folders = sorted([f for f in os.listdir(SRC_TS) if not f.endswith("csv")])

# === Copy and rename training data ===
count = 1
training = []
for patient in train_folders[:60]:
    label_name = f"BRATS_{count:03d}.nii"
    for mod, mod_code in modalities.items():
        img_name = f"BRATS_{count:03d}_{mod_code}.nii"
        shutil.copy(os.path.join(SRC_TR, patient, f"{patient}_{mod}.nii"), f"{RAW_DIR}/imagesTr/{img_name}")
        training.append({"image": f"./imagesTr/{img_name}", "label": f"./labelsTr/{label_name}"})
    shutil.copy(os.path.join(SRC_TR, patient, f"{patient}_seg.nii"), f"{RAW_DIR}/labelsTr/{label_name}")
    count += 1

# === Copy and rename test data ===
count = 1
test = []
for patient in val_folders[:30]:
    for mod, mod_code in modalities.items():
        img_name = f"BRATS_{count:03d}_{mod_code}.nii"
        shutil.copy(os.path.join(SRC_TS, patient, f"{patient}_{mod}.nii"), f"{RAW_DIR}/imagesTs/{img_name}")
    test.append({"image": f"./imagesTs/{img_name}"})
    count += 1

# === Fix label classes (replace label 4 → 3) ===
for file in os.listdir(f"{RAW_DIR}/labelsTr"):
    if file.endswith(".nii"):
        path = os.path.join(f"{RAW_DIR}/labelsTr", file)
        img = nib.load(path)
        data = img.get_fdata()
        data[data == 4] = 3
        nib.save(nib.Nifti1Image(data, img.affine, img.header), path)

# === Compress .nii to .nii.gz ===
def compress_nii_files(directory):
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".nii"):
                nii_path = os.path.join(root, file)
                with open(nii_path, 'rb') as f_in, gzip.open(nii_path + ".gz", 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
                os.remove(nii_path)

compress_nii_files(f"{RAW_DIR}/imagesTr")
compress_nii_files(f"{RAW_DIR}/imagesTs")
compress_nii_files(f"{RAW_DIR}/labelsTr")

# === Generate dataset.json ===
channels = {"0": "FLAIR", "1": "T1", "2": "T1ce", "3": "T2"}
labels = {
    "background": 0,
    "whole_tumor": [1, 2, 3],
    "tumor_core": [2, 3],
    "enhancing_tumor": [3]
}

save_json({
    "channel_names": channels,
    "labels": labels,
    "numTraining": len(training),
    "file_ending": ".nii.gz",
    "name": "BraTS2020",
    "description": "Brain Tumor Segmentation 2020",
    "reference": "https://www.med.upenn.edu/cbica/brats2020/",
    "licence": "see https://www.med.upenn.edu/sbia/brats2020/data.html",
    "release": "0",
    "train": training,
    "test": test,
    "regions_class_order": [1, 2, 3]
}, os.path.join(RAW_DIR, "dataset.json"))

# === Run nnUNet commands (plan → train → predict) ===
os.system("nnUNetv2_plan_and_preprocess -d 1 --verify_dataset_integrity")
os.system("nnUNetv2_train 1 3d_fullres 0")
os.system("nnUNetv2_find_best_configuration 1 -c 3d_fullres -h")
os.system("cp /content/nnUNet_results/Dataset001_BrainTumour/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_0/checkpoint_best.pth /content/nnUNet_results/Dataset001_BrainTumour/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_0/checkpoint_final.pth")
os.system("nnUNetv2_predict -i /content/nnUNet_raw/Dataset001_BrainTumour/imagesTs -o /content/predictions -d 1 -c 3d_fullres --save_probabilities -f 0")

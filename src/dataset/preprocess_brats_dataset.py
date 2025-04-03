import nibabel as nib
import numpy as np
import os
import pickle
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import pandas as pd
import zipfile
import shutil

# Paths and config
zip_path = '/content/drive/MyDrive/dataset_brats/archive.zip'
dataset_directory = '/content/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData'
test_size = 20
val_size = 10
train_size = 70

# Initialize the scaler
scaler = StandardScaler()

def extract_and_prepare_directories(zip_path, dataset_directory, test_size, val_size, train_size):
    print("Extracting zip file and preparing directories...")

    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall()
    print("=====> 20%: Zip file extracted.")

    patient_folders = [dir for dir in os.listdir(dataset_directory) if not dir.endswith('csv')]
    np.random.shuffle(patient_folders)
    print("=====> 40%: Patient folders listed and shuffled.")

    print(f"Test Size: {test_size}, Validation Size: {val_size}, Train Size: {train_size}")

    train_patients = patient_folders[:train_size]
    validation_patients = patient_folders[train_size:train_size + val_size]
    test_patients = patient_folders[train_size + val_size:train_size + val_size + test_size]
    print("=====> 60%: Patient folders split into train, validation, and test sets.")

    selected_patients = train_patients + validation_patients + test_patients
    non_selected_patients = list(set(patient_folders) - set(selected_patients))

    for patient in non_selected_patients:
        shutil.rmtree(os.path.join(dataset_directory, f"{patient}"))
    print("=====> 80%: Non-selected patient folders removed.")

    os.makedirs(name=f"train", exist_ok=True)
    print("=====> 100%: Directories prepared.")

    return train_patients, validation_patients, test_patients

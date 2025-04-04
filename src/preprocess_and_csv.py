import os
import random
import numpy as np
import torch
import pickle
import nibabel as nib
from scipy import ndimage
from scipy.ndimage.interpolation import zoom
from torch.utils.data import Dataset
import cv2
import pandas as pd

# Paths and config
HU_min, HU_max = -200, 250
data_mean = 50.21997497685108
data_std = 68.47153712416372

# Function to load image
def read_image(path):
    with open(path, 'rb') as file:
        img = pickle.load(file)
        return img

# Function to load NIfTI image
def load_nifti_image(filepath):
    image = nib.load(filepath)
    data = image.get_fdata()
    data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
    return data

# Function to normalize image
def normalize_image(image, data_type):      
    original_shape = image.shape
    reshaped_data = image.reshape(-1, original_shape[2])
    if data_type == 'train':
        scaled_data = scaler.fit_transform(reshaped_data)
    else:
        scaled_data = scaler.transform(reshaped_data)
    scaled_data = scaled_data.reshape(original_shape)
    return scaled_data

# Function to clean data
def clean_data(data):
    return np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)

# Function to prepare CSV for train/validation/test
def prepare_csv(patients_folders, data_type, dataset_directory):
    modality = "t1"
    csv_obj = {}
    image_paths = []
    mask_paths = []

    for patient in patients_folders:
        patient_number = patient.split('_')[2]
        image_folder = f"data/{patient_number}/images/"
        os.makedirs(name=image_folder, exist_ok=True)
        mask_folder = f"data/{patient_number}/masks/"
        os.makedirs(name=mask_folder, exist_ok=True)

        image_path = os.path.join(dataset_directory, patient, f"{patient}_{modality}.nii")
        mask_path = os.path.join(dataset_directory, patient, f"{patient}_seg.nii")

        image_data = load_nifti_image(image_path)
        mask_data = load_nifti_image(mask_path)

        img_arr = np.concatenate((image_data[:, :, 0:1], image_data[:, :, 0:1], image_data,
                                  image_data[:, :, -1:], image_data[:, :, -1:]), axis=-1)
        mask_arr = np.concatenate((mask_data[:, :, 0:1], mask_data[:, :, 0:1], mask_data,
                                   mask_data[:, :, -1:], mask_data[:, :, -1:]), axis=-1)

        for slice_indx in range(2, img_arr.shape[2]-2):
            slice_arr = img_arr[:,:,slice_indx-2: slice_indx+3]
            slice_arr = np.flip(np.rot90(slice_arr, k=1, axes=(0, 1)), axis=1)
            mask_arr_2D = mask_arr[:,:,slice_indx-2: slice_indx+3]
            mask_arr_2D = np.flip(np.rot90(mask_arr_2D, k=1, axes=(0, 1)), axis=1)

            img_name = f"2Dimage_{slice_indx:04d}.pkl"
            img_pkl = os.path.join(image_folder, img_name)
            mask_name = f"2Dmask_{slice_indx:04d}.pkl"
            mask_pkl = os.path.join(mask_folder, mask_name)

            with open(img_pkl, 'wb') as f:
                pickle.dump(slice_arr, f)
            with open(mask_pkl, 'wb') as f:
                pickle.dump(mask_arr_2D, f)

            image_paths.append(f"{patient_number}/images/{img_name}")
            mask_paths.append(f"{patient_number}/masks/{mask_name}")

    csv_obj = {'image_pth': image_paths, 'mask_pth': mask_paths}
    return csv_obj

# Function to create CSV files
def create_csv_file(csv_data, file_type):
    df = pd.DataFrame(csv_data)
    os.makedirs("data", exist_ok=True)
    df.to_csv(f"data/{file_type}.csv", index=False)

# Main function to run
if __name__ == "__main__":
    zip_path = "/content/drive/MyDrive/dataset_brats/archive.zip"
    dataset_directory = "/content/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData"

    test_size = 20
    val_size = 10
    train_size = 70

    from data_utils import extract_and_prepare_directories  # if separated
    train_patients, val_patients, test_patients = extract_and_prepare_directories(zip_path, dataset_directory, test_size, val_size, train_size)

    create_csv_file(prepare_csv(train_patients, 'train', dataset_directory), 'training')
    create_csv_file(prepare_csv(val_patients, 'validation', dataset_directory), 'validation')
    create_csv_file(prepare_csv(test_patients, 'test', dataset_directory), 'test')

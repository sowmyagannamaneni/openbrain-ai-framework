# nnUNet-BraTS2020

This repository contains the implementation of nnUNet for brain tumor segmentation using the BraTS2020 dataset.

## Setup and Installation

To run this code on Google Colab, follow these steps:

### 1. Create a New Notebook and Mount Google Drive

Create a new notebook in Google Colab and mount your Google Drive:

```python
from google.colab import drive
drive.mount('/content/drive')
```

### 2. Download the Dataset

Download the BraTS2020 dataset from https://www.kaggle.com/datasets/awsaf49/brats20-dataset-training-validation and upload it to your Google Drive or Colab environment. Set the dataset and validation paths accordingly:

```python
dataset_dir = '/content/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData'
dataset_val_dir = '/content/BraTS2020_ValidationData/MICCAI_BraTS2020_ValidationData'
```

### 3. Install Required Packages

Install the necessary packages using the following command:

```python
!pip install nnunetv2
```

After installing the packages, restart the runtime.

### 4. Run the Notebook

Open and run the provided Jupyter notebook:

1. Upload the notebook files to Colab.
2. Open the notebook in Colab.
3. Follow the instructions and run the cells step-by-step, ensuring the paths are correctly set for your dataset.


## References

- nnUNet: [GitHub Link](https://github.com/MIC-DKFZ/nnUNet)
- BraTS2020 Dataset: [BraTS2020 Data](https://www.med.upenn.edu/cbica/brats2020/data.html)

## Acknowledgements

This work is based on the nnUNet framework and utilizes the BraTS2020 dataset for brain tumor segmentation.

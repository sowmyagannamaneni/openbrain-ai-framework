# 🧠 NeuroAI: Open-Source Framework for Brain Segmentation and Abnormality Detection


> An open-source deep learning framework for 3D MRI-based brain tumor segmentation and anomaly detection using nnUNet and MASAM.

---

## 🗂️ Project Structure

- **src/** – Python scripts for data processing, model training, and inference.
- **models/** – Pretrained model weights.
- **results/** – Sample segmentation outputs and evaluation metrics.
- **report/** – Technical report summarizing findings and experiments.

---

## 📊 Features

- 🧠 **Segmentation**: Multi-class brain tumor segmentation (WT, TC, ET).
- 🔍 **Anomaly Detection**: Attention-enhanced MASAM model to flag subtle abnormalities.
- 📈 **Statistical Analysis**: Dice score, Jaccard Index, Precision, Recall, Confusion Matrix, and visualizations.
- 📊 **Clinical Readiness**: Case-wise evaluation, reproducible outputs.

---

## 🧪 Dataset

This project uses the [BraTS2020 dataset](https://www.med.upenn.edu/cbica/brats2020/data.html), which provides co-registered 3D MRI volumes in NIfTI format across 4 modalities:
- T1, T1ce, T2, FLAIR

Each case includes annotations for:
- Whole Tumor (WT)
- Tumor Core (TC)
- Enhancing Tumor (ET)

---


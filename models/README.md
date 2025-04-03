# Models Directory

This directory contains key models used for image segmentation and attention-based tasks.

## Files

### 1. `factt_model.py`
Contains the **Fact-TT model** that uses window partitioning and relative positional embeddings for attention-based image processing.

### 2. `sam_model.py`
Defines the **SAM (Segment Anything Model)** that predicts object masks using image and input prompts (points, boxes, or masks).

### 3. `masam_modules.py`
Includes modules for the **MASAM model**, such as **PromptEncoder**, **PositionEmbedding**, and **MLP Block** for attention-based segmentation.


## Dependencies
- **PyTorch**
- **NumPy**
- **Torchvision**
- **Scikit-learn**
- **Matplotlib/Seaborn**

## Usage
- **Training**: Run the corresponding training scripts (e.g., `train_masam.py`).
- **Evaluation**: Evaluate model performance using utility functions for metrics.

## Contributions
Feel free to contribute to the model development, training, or evaluation scripts. Please cite our work if using this code.



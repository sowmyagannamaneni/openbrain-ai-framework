import matplotlib.pyplot as plt
import re
import os

def read_training_log(file_path):
    epochs = []
    dice_scores = []
    with open(file_path, 'r') as file:
        for line in file:
            if "Pseudo dice" in line:
                match = re.search(r"Pseudo dice \[(.*?)\]", line)
                if match:
                    scores = match.group(1).split(', ')
                    avg_dice = sum([float(score) for score in scores]) / len(scores)
                    dice_scores.append(avg_dice)
            if "Epoch" in line:
                match = re.search(r"Epoch (\d+)", line)
                if match:
                    epochs.append(int(match.group(1)))
    return epochs, dice_scores

log_file_path = '/content/nnUNet_results/Dataset001_BrainTumour/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_0/training_log.txt'

if os.path.exists(log_file_path):
    epochs, dice_scores = read_training_log(log_file_path)
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, dice_scores, label='Pseudo Dice Score')
    plt.xlabel('Epochs')
    plt.ylabel('Pseudo Dice Score')
    plt.title('Pseudo Dice Score per Epoch')
    plt.legend()
    plt.grid(True)
    plt.show()

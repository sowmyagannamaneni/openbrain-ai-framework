# Perform Evaluation Metrics
def calculate_metrics(true_dir, pred_dir):
    metrics = {
        "precision": [],
        "recall": [],
        "f1_score": [],
        "accuracy": [],
        "jaccard_index": [],
        "dice_score": []
    }

    for true_file in os.listdir(true_dir):
        if true_file.endswith(".nii.gz"):
            true_path = os.path.join(true_dir, true_file)
            pred_path = os.path.join(pred_dir, true_file)

            if not os.path.exists(pred_path):
                print(f"Prediction file {pred_path} does not exist.")
                continue

            true_img = nib.load(true_path).get_fdata().astype(int).flatten()
            pred_img = nib.load(pred_path).get_fdata().astype(int).flatten()

            metrics["precision"].append(precision_score(true_img, pred_img, average='weighted', zero_division=0))
            metrics["recall"].append(recall_score(true_img, pred_img, average='weighted', zero_division=0))
            metrics["f1_score"].append(f1_score(true_img, pred_img, average='weighted', zero_division=0))
            metrics["accuracy"].append(accuracy_score(true_img, pred_img))
            metrics["jaccard_index"].append(jaccard_score(true_img, pred_img, average='weighted', zero_division=0))
            metrics["dice_score"].append(1 - dice(true_img, pred_img))

    for metric, values in metrics.items():
        print(f"{metric.capitalize()}: {np.mean(values):.4f}")

    return metrics

metrics = calculate_metrics('/content/nnUNet_raw/Dataset001_BrainTumour/labelsTr', "/content/predictions")

# Visualization of Metrics
def plot_metrics(metrics):
    fig, ax = plt.subplots()
    for metric, values in metrics.items():
        ax.plot(values, label=metric)
    ax.set_xlabel('Samples')
    ax.set_ylabel('Score')
    ax.legend(loc='best')
    plt.show()

plot_metrics(metrics)
# Print metrics without dice score
def print_metrics(metrics):
    for metric, values in metrics.items():
        if metric != "dice_score":
            print(f"{metric.capitalize()}: {np.mean(values):.4f}")

# Call printing function
print_metrics(metrics)

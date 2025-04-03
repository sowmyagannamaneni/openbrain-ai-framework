import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def plot_metric_distributions(metrics):
    metrics_df = pd.DataFrame(metrics)
    metrics_df = metrics_df.drop(columns=['dice_score'])

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Metric Distributions')

    for ax, metric in zip(axes.flatten(), metrics_df.columns):
        sns.boxplot(data=metrics_df[metric], ax=ax)
        ax.set_title(metric.capitalize())

    plt.tight_layout()
    plt.show()

plot_metric_distributions(metrics)

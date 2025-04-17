"""Plot a comparison of the speed of ADMET-AI versions for large-scale ADMET prediction using CSV data."""
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

FIGSIZE = (14, 10)
matplotlib.rcParams["font.size"] = 28
plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"


def plot_admet_speed(results_path: Path, save_path: Path, max_time: int = 800) -> None:
    """
    Plot a comparison of the speed of ADMET-AI versions for large-scale ADMET prediction.

    :param results_path: Path to a CSV file containing the ADMET website speed results.
                         Expected CSV columns: 'Website', '1 Molecule', '10 Molecules', '100 Molecules', '1,000 Molecules'
    :param save_path: Path to a PDF file where the plot will be saved.
    :param max_time: The maximum time to plot on the y-axis.
    """
    # Load speed results from CSV
    results = pd.read_csv(results_path)

    # Melt the DataFrame to have one row per Website per Molecule count.
    # 'Website' remains as identifier and other columns become two new columns: 'Number of Molecules' and 'Time (h)'
    results_melted = results.melt(
        id_vars="Website", var_name="Number of Molecules", value_name="Time (h)"
    )

    # Clean the "Number of Molecules" column:
    # Remove the text " Molecule(s)" and any commas to convert to an integer.
    # For example, "1 Molecule" -> 1, "1,000 Molecules" -> 1000.
    results_melted["Number of Molecules"] = (
        results_melted["Number of Molecules"]
        .str.replace(" Molecules", "", regex=False)
        .str.replace(" Molecule", "", regex=False)
        .str.replace(",", "", regex=False)
        .astype(int)
    )

    # Optionally, sort the DataFrame by "Number of Molecules" (if not already sorted)
    results_melted = results_melted.sort_values("Number of Molecules")

    # Create the plot with a line for each website
    plt.subplots(figsize=FIGSIZE)
    sns.lineplot(
        x="Number of Molecules",
        y="Time (h)",
        hue="Website",
        data=results_melted,
        marker="o",
        markersize=10,
        linewidth=3,
    )

    # Limit y-axis to max_time
    plt.ylim(0, max_time)

    # Set axis labels and title
    plt.xlabel("Number of Molecules")
    plt.ylabel("Time (h)")
    plt.title("ADMET Speed Comparison")

    # Save the plot as PDF
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, bbox_inches="tight")


if __name__ == "__main__":
    from tap import tapify
    tapify(plot_admet_speed)
"""Plot results from TDC Leaderboard - CSV dataset version."""
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

FIGSIZE = (14, 10)
matplotlib.rcParams["font.size"] = 18
plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"
SCATTER_SIZE = 150


def split_axes(axes: list[plt.Axes]) -> None:
    """Split axes (two columns) into two disjoint plots.

    :param axes: List of two axes (two columns) to split.
    """
    for ax in axes:
        ax.set_xlabel("")
        ax.set_ylabel("")
    axes[0].spines.right.set_visible(False)
    axes[1].spines.left.set_visible(False)
    axes[1].tick_params(axis="y", which="both", left=False, right=False)
    d = 0.5  # proportion of vertical to horizontal extent of the slanted line
    kwargs = dict(
        marker=[(-d, -1), (d, 1)],
        markersize=12,
        linestyle="none",
        color="k",
        mec="k",
        mew=1,
        clip_on=False,
    )
    axes[0].plot([1, 1], [0, 1], transform=axes[0].transAxes, **kwargs)
    axes[1].plot([0, 0], [0, 1], transform=axes[1].transAxes, **kwargs)


def plot_dataset_performance(results: pd.DataFrame, save_dir: Path) -> None:
    """Plot performance metrics by dataset."""
    matplotlib.rcParams["font.size"] = 18
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Performance by dataset
    fig, ax = plt.subplots(figsize=FIGSIZE)
    results_sorted = results.sort_values(by="Leaderboard Mean")
    sns.barplot(x="Leaderboard Mean", y="Dataset", data=results_sorted, ax=ax)
    ax.set_title("Leaderboard Mean Performance by Dataset")
    ax.set_xlabel("Performance (Various Metrics)")
    plt.tight_layout()
    plt.savefig(save_dir / "dataset_performance.pdf", bbox_inches="tight")
    plt.close()


def plot_mean_vs_ensemble(results: pd.DataFrame, save_dir: Path) -> None:
    """Plot comparison between mean and ensemble performance."""
    matplotlib.rcParams["font.size"] = 18
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Ensure both columns exist
    if "Leaderboard Mean" in results.columns and "Leaderboard Ensemble" in results.columns:
        # Create a melted dataframe for side-by-side bars
        plot_data = results.melt(
            id_vars=["Dataset", "Category", "Leaderboard Metric"], 
            value_vars=["Leaderboard Mean", "Leaderboard Ensemble"],
            var_name="Type",
            value_name="Value"
        )
        
        # Group by metric for separate plots
        metrics = plot_data["Leaderboard Metric"].unique()
        
        for metric in metrics:
            metric_data = plot_data[plot_data["Leaderboard Metric"] == metric]
            
            if metric == "MAE":
                fig, axes = plt.subplots(1, 2, sharey=True, figsize=FIGSIZE)
                fig.subplots_adjust(wspace=0.05)
                
                # Get value ranges for splitting
                values = metric_data["Value"].values
                median_val = np.median(values)
                max_val = np.max(values)
                
                # Split datasets into two groups for visualization
                lower_values = metric_data[metric_data["Value"] < median_val]
                higher_values = metric_data[metric_data["Value"] >= median_val]
                
                sns.barplot(x="Value", y="Dataset", hue="Type", data=lower_values, ax=axes[0])
                sns.barplot(x="Value", y="Dataset", hue="Type", data=higher_values, ax=axes[1])
                
                # Set limits for better visualization
                axes[0].set_xlim(0, median_val)
                axes[1].set_xlim(median_val * 0.9, max_val * 1.1)
                
                # Remove duplicate legend
                axes[0].legend().remove()
                
                # Split axes for better visualization
                split_axes(axes)
                fig.text(0.5, 0.04, f"{metric} Performance", ha="center")
                
            else:
                fig, ax = plt.subplots(figsize=FIGSIZE)
                sns.barplot(x="Value", y="Dataset", hue="Type", data=metric_data, ax=ax)
                ax.set_title(f"{metric} Performance: Mean vs Ensemble")
                ax.set_xlabel(f"{metric} Value")
                
            plt.tight_layout()
            plt.savefig(save_dir / f"mean_vs_ensemble_{metric.lower()}.pdf", bbox_inches="tight")
            plt.close()


def plot_performance_by_category(results: pd.DataFrame, save_dir: Path) -> None:
    """Plot performance metrics grouped by category."""
    matplotlib.rcParams["font.size"] = 18
    save_dir.mkdir(parents=True, exist_ok=True)
    
    if "Category" in results.columns:
        # Group by category
        categories = results["Category"].unique()
        
        # Create a plot showing average performance by category
        category_means = results.groupby("Category")["Leaderboard Mean"].mean().reset_index()
        category_means = category_means.sort_values(by="Leaderboard Mean")
        
        fig, ax = plt.subplots(figsize=FIGSIZE)
        sns.barplot(x="Leaderboard Mean", y="Category", data=category_means, ax=ax)
        ax.set_title("Average Performance by Category")
        ax.set_xlabel("Average Performance (Various Metrics)")
        plt.tight_layout()
        plt.savefig(save_dir / "category_performance.pdf", bbox_inches="tight")
        plt.close()
        
        # Create individual plots for each category
        for category in categories:
            category_data = results[results["Category"] == category]
            fig, ax = plt.subplots(figsize=FIGSIZE)
            sns.barplot(x="Leaderboard Mean", y="Dataset", data=category_data, ax=ax)
            ax.set_title(f"Performance for {category} Datasets")
            ax.set_xlabel("Performance (Various Metrics)")
            plt.tight_layout()
            plt.savefig(save_dir / f"performance_{category.lower()}.pdf", bbox_inches="tight")
            plt.close()


def plot_performance_improvement(results: pd.DataFrame, save_dir: Path) -> None:
    """Plot performance improvement compared to best reported performance."""
    matplotlib.rcParams["font.size"] = 18
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if the columns exist
    if "Leaderboard % Diff (10/04/23)" in results.columns:
        # Sort by improvement percentage
        results_sorted = results.sort_values(by="Leaderboard % Diff (10/04/23)")
        
        fig, ax = plt.subplots(figsize=FIGSIZE)
        sns.barplot(x="Leaderboard % Diff (10/04/23)", y="Dataset", data=results_sorted, ax=ax)
        ax.set_title("Performance Improvement Compared to Best (10/04/23)")
        ax.set_xlabel("Percentage Difference (%)")
        plt.tight_layout()
        plt.savefig(save_dir / "performance_improvement.pdf", bbox_inches="tight")
        plt.close()


def plot_metric_distribution(results: pd.DataFrame, save_dir: Path) -> None:
    """Plot distribution of metrics used across datasets."""
    matplotlib.rcParams["font.size"] = 18
    save_dir.mkdir(parents=True, exist_ok=True)
    
    if "Leaderboard Metric" in results.columns:
        # Count occurrences of each metric
        metric_counts = results["Leaderboard Metric"].value_counts().reset_index()
        metric_counts.columns = ["Metric", "Count"]
        
        fig, ax = plt.subplots(figsize=FIGSIZE)
        sns.barplot(x="Count", y="Metric", data=metric_counts, ax=ax)
        ax.set_title("Distribution of Evaluation Metrics")
        plt.tight_layout()
        plt.savefig(save_dir / "metric_distribution.pdf", bbox_inches="tight")
        plt.close()


def plot_tdc_results(results_path: Path, save_dir: Path) -> None:
    """
    Plot results from TDC Leaderboard using the dataset-only format.
    
    :param results_path: Path to a CSV file containing dataset results.
    :param save_dir: Path to a directory where the plots will be saved.
    """
    print(f"Reading results from CSV file: {results_path}")
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Load results from CSV
    try:
        results = pd.read_csv(results_path)
        print(f"Successfully loaded data with {len(results)} rows")
    except Exception as e:
        print(f"Error loading CSV: {e}")
        raise
    
    # Print columns to help with debugging
    print(f"Available columns: {results.columns.tolist()}")
    
    # If no "Sheet" column exists, assign a default value based on metric
    if "Sheet" not in results.columns:
        # Map metrics to sheet names
        metric_to_sheet = {
            "AUROC": "TDC Leaderboard Classification",
            "AUPRC": "TDC Leaderboard Classification",
            "Spearman": "TDC Leaderboard Regression",
            "MAE": "TDC Leaderboard Regression"
        }
        
        # Apply mapping if the column exists
        if "Leaderboard Metric" in results.columns:
            results["Sheet"] = results["Leaderboard Metric"].map(
                lambda x: metric_to_sheet.get(x, "TDC Leaderboard Regression")
            )
            print("Added 'Sheet' column based on Leaderboard Metric")
        else:
            # Default fallback
            results["Sheet"] = "TDC Leaderboard Regression"
            print("Added 'Sheet' column with default value")
    
    # Check for NA values in important columns
    for col in ["Dataset", "Leaderboard Mean", "Leaderboard Metric"]:
        if col in results.columns and results[col].isna().any():
            print(f"Warning: Column {col} contains NA values")
    
    # Generate dataset-level plots
    try:
        plot_dataset_performance(results, save_dir)
        print("Successfully created dataset performance plot")
    except Exception as e:
        print(f"Error creating dataset performance plot: {e}")
    
    try:
        plot_mean_vs_ensemble(results, save_dir)
        print("Successfully created mean vs ensemble plot")
    except Exception as e:
        print(f"Error creating mean vs ensemble plot: {e}")
    
    try:
        plot_performance_by_category(results, save_dir)
        print("Successfully created performance by category plot")
    except Exception as e:
        print(f"Error creating performance by category plot: {e}")
    
    try:
        plot_performance_improvement(results, save_dir)
        print("Successfully created performance improvement plot")
    except Exception as e:
        print(f"Error creating performance improvement plot: {e}")
    
    try:
        plot_metric_distribution(results, save_dir)
        print("Successfully created metric distribution plot")
    except Exception as e:
        print(f"Error creating metric distribution plot: {e}")
    
    print("Plotting complete. All plots saved to", save_dir)


if __name__ == "__main__":
    from tap import tapify
    tapify(plot_tdc_results)
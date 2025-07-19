from flwr_datasets.visualization import plot_label_distributions

def showDataDist(partition_name, column):
    """
    Visualizes the label distribution for a specified column in the given federated dataset partition.

    Args:
        partition_name (FederatedDataset): The federated dataset partition to visualize.
        column (str): The name of the column (label) for which to plot the distribution.

    Returns:
        None. Displays a plot of the label distribution for the specified column.
    """
    _ = plot_label_distributions(
        partitioner=partition_name.partitioners["train"],
        label_name=column,
        legend=True,
    )
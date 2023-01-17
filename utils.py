import matplotlib.pyplot as plt


def init_loss_graph():
    """Initialize loss graph in matplotlib."""
    plt.ion()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xlabel("step")
    ax.set_ylabel("loss")
    ax.set_title("Loss Graph")
    plt.show(block=False)
    return ax, fig


def downsample_metric_history(metric_history, downsample_factor):
    """Downsample metric history by averaging over a window of size downsample_factor."""
    downsampled_metric_history = []
    for i in range(0, len(metric_history), downsample_factor):
        downsampled_metric_history.append(sum(metric_history[i:i + downsample_factor]) / downsample_factor)
    return downsampled_metric_history

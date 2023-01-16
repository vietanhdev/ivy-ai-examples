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

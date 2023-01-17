import sys

sys.path.append(".")

import ivy
import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt

from utils import init_loss_graph, downsample_metric_history


# Draw losses on training - set to False for faster training
INTERATIVE_LOSS_GRAPH = False

# Data preparation
iris = datasets.load_iris()  # Load iris dataset
X = iris.data  # Shape: [N, 4]
Y = iris.target  # Shape: [N, 1]

# Convert Y to one-hot
Y = np.eye(3)[Y]  # Shape: [N, 3]

# Shuffle the data
np.random.seed(0)
idx = np.arange(X.shape[0])
np.random.shuffle(idx)
X = X[idx].astype(np.float32)
Y = Y[idx].astype(np.float32)

# Split the data into train (80%) and test (20%)
TRAIN_RATIO = 0.8
train_idx = int(X.shape[0] * TRAIN_RATIO)
X_train = X[:train_idx]
Y_train = Y[:train_idx]
X_test = X[train_idx:]
Y_test = Y[train_idx:]


class IrisClassificationModel(ivy.Module):
    """Define a simple model for Iris classification."""

    def __init__(self):
        self.linear0 = ivy.Linear(4, 32)
        self.linear1 = ivy.Linear(32, 3)
        ivy.Module.__init__(self)

    def _forward(self, x):
        x = self.linear0(x)
        x = ivy.relu(x)
        x = self.linear1(x)
        x = ivy.softmax(x)
        return x


ivy.set_backend("torch")  # change to other backend
model = IrisClassificationModel()
optimizer = ivy.Adam(0.01)

# Initialize loss graph in matplotlib
if INTERATIVE_LOSS_GRAPH:
    ax, fig = init_loss_graph()

# Train
batch_size = 32
idx = 0
last_loss_np = 0
losses = []
for step in range(400):
    idx = (idx + 1) % (X_train.shape[0] // batch_size)
    xx = ivy.array(X_train[idx * batch_size : idx * batch_size + batch_size])
    yy = ivy.array(Y_train[idx * batch_size : idx * batch_size + batch_size])

    def loss_fn(v):
        out = model(xx, v=v)
        out = ivy.mean(ivy.cross_entropy(yy, out))
        return out

    loss, grads = ivy.execute_with_gradients(loss_fn, model.v)
    model.v = optimizer.step(model.v, grads)

    # Update loss graph
    loss_np = ivy.to_numpy(loss).item()
    losses.append(loss_np)
    if step % 10 == 0:
        print("Step {} - loss {}".format(step, loss_np))
    if INTERATIVE_LOSS_GRAPH:
        ax.scatter(step, loss_np, c="b", marker=".")
        if step > 0:
            ax.add_line(plt.Line2D([step - 1, step], [last_loss_np, loss_np], c="b"))
        last_loss_np = loss_np
        fig.canvas.draw()
        fig.canvas.flush_events()

# Draw final loss graph
if not INTERATIVE_LOSS_GRAPH:
    ax, fig = init_loss_graph()
    # Downsample loss history
    losses = downsample_metric_history(losses, 10)
    for i, loss in enumerate(losses):
        ax.scatter(i, loss, c="b", marker=".")
        if i > 0:
            ax.add_line(plt.Line2D([i - 1, i], [losses[i - 1], loss], c="b"))
    fig.canvas.draw()
    fig.canvas.flush_events()

# Evaluate
correct = 0
for i in range(X_test.shape[0]):
    xx = ivy.array(X_test[i])
    yy = ivy.array(Y_test[i])
    out = model(xx)
    if ivy.argmax(out) == ivy.argmax(yy):
        correct += 1
print("Accuracy: {}".format(correct / X_test.shape[0]))

while True:
    # Keep the graph open
    plt.pause(0.5)

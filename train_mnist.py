import sys

sys.path.append(".")

import cv2
import ivy
import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt

from utils import init_loss_graph, downsample_metric_history


# Draw losses on training - set to False for faster training
INTERATIVE_LOSS_GRAPH = False
SHOW_SAMPLE_IMAGES = False

# Data preparation
digits = datasets.load_digits()
X = digits.data.reshape((-1, 8, 8, 1))  # Shape: [N, 8, 8]
Y = digits.target  # Shape: [N, 1]

# Convert Y to one-hot
Y = np.eye(10)[Y]  # Shape: [N, 10]

# Shuffle the data
np.random.seed(0)
idx = np.arange(X.shape[0])
np.random.shuffle(idx)
X = X[idx].astype(np.float32)
Y = Y[idx].astype(np.float32)


# Resize images to 28x28
Xs = []
for i in range(X.shape[0]):
    Xs.append(cv2.resize(X[i], (28, 28)))
X = np.stack(Xs, axis=0).reshape((-1, 28, 28, 1))

# Show some images
if SHOW_SAMPLE_IMAGES:
    for i in range(10):
        plt.subplot(2, 5, i + 1)
        plt.imshow(X[i].reshape((28, 28)), cmap="gray")
    plt.show()

# Preprocess for X
X = X * 2.0 / 255.0


# Split the data into train (80%) and test (20%)
TRAIN_RATIO = 0.8
train_idx = int(X.shape[0] * TRAIN_RATIO)
X_train = X[:train_idx]
Y_train = Y[:train_idx]
X_test = X[train_idx:]
Y_test = Y[train_idx:]


class DigitsClassificationModel(ivy.Module):
    """Define a simple CNN model for digits classification."""

    def __init__(self):
        self.conv0 = ivy.Conv2D(1, 8, [3, 3], [1, 1], "SAME")
        self.conv1 = ivy.Conv2D(8, 16, [3, 3], [1, 1], "SAME")
        self.conv2 = ivy.Conv2D(16, 64, [3, 3], [1, 1], "SAME")
        self.linear = ivy.Linear(64, 10)
        ivy.Module.__init__(self)

    def _forward(self, x):
        x = self.conv0(x)
        x = ivy.relu(x)
        x = ivy.max_pool2d(x, [2, 2], [2, 2], "SAME")
        x = self.conv1(x)
        x = ivy.relu(x)
        x = ivy.max_pool2d(x, [2, 2], [2, 2], "SAME")
        x = self.conv2(x)
        x = ivy.relu(x)

        # Global average pooling
        x = ivy.avg_pool2d(x, [x.shape[1], x.shape[2]], [1, 1], "VALID")
        x = ivy.reshape(x, (x.shape[0], -1))

        x = self.linear(x)
        x = ivy.softmax(x)

        return x


ivy.set_backend("torch")  # change to other backend
model = DigitsClassificationModel()
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
    if yy.shape[0] != batch_size:
        continue

    def loss_fn(v):
        out = model(xx, v=v)
        out = ivy.sum(ivy.cross_entropy(yy, out))
        return out

    loss, grads = ivy.execute_with_gradients(loss_fn, model.v)
    model.v = optimizer.step(model.v, grads)

    # Update loss graph
    loss_np = ivy.to_numpy(loss).item()
    losses.append(loss_np)
    if step % 5 == 0:
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
    loss = downsample_metric_history(losses, 15)
    for i, loss in enumerate(losses):
        ax.scatter(i, loss, c="b", marker=".")
        if i > 0:
            ax.add_line(plt.Line2D([i - 1, i], [losses[i - 1], loss], c="b"))
    fig.canvas.draw()
    fig.canvas.flush_events()

# Evaluate
correct = 0
for i in range(X_test.shape[0]):
    xx = ivy.array(X_test[i]).reshape((1, 28, 28, 1))
    yy = ivy.array(Y_test[i]).reshape((1, 10))
    out = model(xx)
    if ivy.argmax(out) == ivy.argmax(yy):
        correct += 1
print("Accuracy: {}".format(correct / X_test.shape[0]))

while True:
    # Keep the graph open
    plt.pause(0.5)

import ivy


class MyModel(ivy.Module):
    def __init__(self):
        self.linear0 = ivy.Linear(3, 64)
        self.linear1 = ivy.Linear(64, 1)
        ivy.Module.__init__(self)

    def _forward(self, x):
        x = ivy.relu(self.linear0(x))
        return ivy.sigmoid(self.linear1(x))


ivy.set_backend("torch")  # change to any backend!
model = MyModel()
optimizer = ivy.Adam(1e-2)
x_in = ivy.array([1.0, 2.0, 3.0])
target = ivy.array([0.0])


def loss_fn(v):
    out = model(x_in, v=v)
    return ivy.mean((out - target) ** 2)


for step in range(100):
    loss, grads = ivy.execute_with_gradients(loss_fn, model.v)
    model.v = optimizer.step(model.v, grads)
    print("step {} loss {}".format(step, ivy.to_numpy(loss).item()))

print("Finished training!")

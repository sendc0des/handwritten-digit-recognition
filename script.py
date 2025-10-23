import os
import numpy as np
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
from tensorflow import keras
import tensorflow as tf
import torch

np.random.seed(42)
tf.random.set_seed(42)
g = torch.Generator().manual_seed(2147483647)

class Linear:
    
    def __init__(self, fan_in, fan_out, bias=True):
        self.weight = torch.randn((fan_in, fan_out)) * (fan_in**-0.5)
        self.bias = torch.zeros(fan_out) if bias else None

    def __call__(self, x):
        self.out = x @ self.weight
        if self.bias is not None:
            self.out += self.bias
        return self.out
    
    def parameters(self):
        return [self.weight] + ([] if self.bias is None else [self.bias])
    
class ReLU:
    
    def __call__(self, x):
        self.out = torch.relu(x)
        return self.out
    
    def parameters(self):
        return []
    
class Sequential:

    def __init__(self, layers):
        self.layers = layers

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        self.out = x
        return self.out
    
    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]

# import the dataset
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()

# scale the values and reshape the tensors to make them 2D
X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32')  / 255.0

X_train = X_train.reshape(-1,28*28)
X_test = X_test.reshape(-1,28*28)

# convert the tensorflow tensors to pytorch tensors
numpy_arr = [X_train, y_train, X_test, y_test]
pytorch_tensors = [torch.from_numpy(t) for t in numpy_arr]
X_train, y_train, X_test, y_test = pytorch_tensors

# split the dataset into train, eval and test sets
n = int(0.9 * len(X_train))
Xtr, X_val, ytr, y_val = X_train[:n], X_train[n:], y_train[:n], y_train[n:]

# define the model
n_hidden = 128
num_digits = 10

model = Sequential([
    Linear(784, n_hidden), ReLU(),
    Linear(n_hidden, n_hidden), ReLU(),
    Linear(n_hidden, num_digits),
])

with torch.no_grad():
    model.layers[-1].weight *= 0.1

parameters = model.parameters()
for p in parameters:
    p.requires_grad = True

# train the model on the train set
max_steps = 10000
batch_size = 32
lossi = []

for i in range(max_steps):

    ix = torch.randint(0, Xtr.shape[0], (batch_size,), generator=g)
    Xb, yb = Xtr[ix], ytr[ix]

    logits = model(Xb)
    logits_maxes = logits.max(axis=1, keepdim=True).values
    stable_logits = logits - logits_maxes
    counts = logits.exp()
    probs = counts / counts.sum(axis=1, keepdim=True)
    loss = -probs[torch.arange(batch_size), yb.long()].log().mean()
    # loss = torch.nn.functional.cross_entropy(logits, yb.long())

    # backward pass
    for p in parameters:
        p.grad = None
    loss.backward()

    # update
    lr = 0.1
    for p in parameters:
        p.data += -lr * p.grad

    # track stats
    if i % 1000 == 0:
        print(f'{i:7d}/{max_steps:7d}: {loss.item():.4f}')
    lossi.append(loss.log10().item())

# evaluate loss on train set and val set
with torch.no_grad():
    def split_loss(split):
        x,y = {
            'train': (Xtr, ytr),
            'val': (X_val, y_val),
            'test': (X_test, y_test),
            }[split]
        logits = model(x)
        loss = torch.nn.functional.cross_entropy(logits, y)
        print(split, loss.item())

split_loss('train')
split_loss('val')
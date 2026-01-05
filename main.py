import math
import torch
import torchvision
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from einops import rearrange

from modules.activation import ReLU
from modules.mlp import MLP
from modules.conv import Conv2d


batch_size = 32

transforms = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize([0.1307], [0.3081]),           # this isnt good i think
])

mnist_train_data = torchvision.datasets.MNIST("./data", train=True, transform=transforms, download=True)
mnist_test_data = torchvision.datasets.MNIST("./data", train=False, transform=transforms, download=True)
train_data_loader = torch.utils.data.DataLoader(mnist_train_data, batch_size=batch_size, shuffle=True, num_workers=0)
test_data_loader = torch.utils.data.DataLoader(mnist_test_data, batch_size=batch_size, shuffle=True, num_workers=0)

in_size = 28 * 28
out_size = 10

model = MLP(in_size, [256], out_size)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
cross_entropy_loss = nn.CrossEntropyLoss()



losses = []
for batch in train_data_loader:
    optimizer.zero_grad()

    inputs, labels = batch
    inputs = rearrange(inputs, "B C W H -> B (C H W)")

    predictions = model(inputs)

    loss = cross_entropy_loss(predictions, labels)

    losses.append(loss.item())

    loss.backward()
    optimizer.step()

correct = 0 
total = 0
with torch.no_grad():
    for batch in test_data_loader:
        inputs, labels = batch
        inputs = rearrange(inputs, "B C W H -> B (C H W)")
        logprobs = model(inputs)
        predictions = torch.argmax(logprobs, dim=1)
        
        cor = torch.sum(torch.where(predictions == labels, 1, 0))
        correct += cor
        total += len(labels)

print(f"acc: {correct / total}")



plt.plot(losses)
plt.show()

import torch
import torchvision
import torch.nn as nn
import matplotlib.pyplot as plt
from einops import rearrange

from modules.activation import ReLU
from modules.mlp import MLP
from modules.conv import MaxPool2d


batch_size = 32

transforms = torchvision.transforms.Compose(
    [
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize([0.1307], [0.3081]),
    ]
)

mnist_train_data = torchvision.datasets.MNIST(
    "./data", train=True, transform=transforms, download=True
)
mnist_test_data = torchvision.datasets.MNIST(
    "./data", train=False, transform=transforms, download=True
)
train_data_loader = torch.utils.data.DataLoader(
    mnist_train_data, batch_size=batch_size, shuffle=True, num_workers=0
)
test_data_loader = torch.utils.data.DataLoader(
    mnist_test_data, batch_size=batch_size, shuffle=True, num_workers=0
)

in_size = 28 * 28
out_size = 10


class ConvTestModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(
            in_channels=1, out_channels=3, kernel_size=3, stride=1
        )
        self.conv2 = torch.nn.Conv2d(
            in_channels=3, out_channels=6, kernel_size=3, stride=1
        )
        self.maxpool = MaxPool2d(kernel_size=2, stride=2)
        self.activation = ReLU()
        self.mlp = MLP(216, [128], 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.activation(x)
        x = self.maxpool(x)
        x = self.conv2(x)
        x = self.activation(x)
        x = self.maxpool(x)
        x = rearrange(x, "B C H W -> B (C H W)")
        x = self.mlp(x)
        return x


model = ConvTestModel()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
cross_entropy_loss = nn.CrossEntropyLoss()

losses = []
num_epochs = 3
for _ in range(num_epochs):
    for batch in train_data_loader:
        optimizer.zero_grad()

        inputs, labels = batch

        predictions = model(inputs)

        loss = cross_entropy_loss(predictions, labels)

        losses.append(loss.item())

        loss.backward()
        optimizer.step()

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
correct = 0
total = 0
with torch.no_grad():
    for batch in test_data_loader:
        inputs, labels = batch
        logprobs = model(inputs)
        predictions = torch.argmax(logprobs, dim=1)

        cor = torch.sum(torch.where(predictions == labels, 1, 0))
        correct += cor
        total += len(labels)

print(f"acc: {correct / total}")


plt.plot(losses)
plt.show()

import torch
import torchvision
import torch.nn as nn
import matplotlib.pyplot as plt
from einops import rearrange
from tqdm import tqdm

from modules.resnet import ResNetCIFAR

batch_size = 256

transforms = torchvision.transforms.Compose(
    [
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize([0.4914, 0.4822, 0.4465], [0.247, 0.243, 0.161]),
    ]
)

mnist_train_data = torchvision.datasets.CIFAR10(
    "./data", train=True, transform=transforms, download=True
)
mnist_test_data = torchvision.datasets.CIFAR10(
    "./data", train=False, transform=transforms, download=True
)
train_data_loader = torch.utils.data.DataLoader(
    mnist_train_data, batch_size=batch_size, shuffle=True, num_workers=0
)
test_data_loader = torch.utils.data.DataLoader(
    mnist_test_data, batch_size=batch_size, shuffle=True, num_workers=0
)

model = ResNetCIFAR()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
cross_entropy_loss = nn.CrossEntropyLoss()

losses = []
num_epochs = 1
for _ in range(num_epochs):
    for batch in tqdm(train_data_loader):
        optimizer.zero_grad()

        inputs, labels = batch

        predictions = model(inputs)

        loss = cross_entropy_loss(predictions, labels)

        losses.append(loss.item())
        print(loss.item())

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

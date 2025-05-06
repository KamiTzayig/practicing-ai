import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


labels_map = {
    0: "T-Shirt",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle Boot",
}
model = NeuralNetwork()
model.load_state_dict(torch.load('model_weights.pth', weights_only=True))
model.eval()


test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor()
)

# inference the test data and show the image with the predicted guess and the actual label

for i in range(10):
    plt.subplot(2, 5, i+1)
    i = i+15
    plt.imshow(test_data[i][0].squeeze(), cmap="gray")
    prediction = model(test_data[i][0])
    predicted_class = prediction.argmax(1).item()
    plt.title(f"{labels_map[predicted_class]},{labels_map[test_data[i][1]]}")
plt.show()


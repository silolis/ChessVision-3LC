import torch.nn.functional as F
from torch import nn


# Define the CNN architecture
class ChessVisionModel(nn.Module):
    def __init__(self, dense_1_size, dense_2_size, num_classes):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 30, kernel_size=5, padding=2)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(30, 15, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.2)
        self.fc1 = nn.Linear(15 * 16 * 16, dense_1_size)
        self.fc2 = nn.Linear(dense_1_size, dense_2_size)
        self.fc3 = nn.Linear(dense_2_size, num_classes)

    def forward(self, x, return_embeddings=False):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.dropout(x)
        x = x.reshape(-1, 15 * 16 * 16)
        fc1 = F.relu(self.fc1(x))  # 1024
        fc2 = F.relu(self.fc2(fc1))  # 64
        fc3 = self.fc3(fc2)  # 13
        if not return_embeddings:
            return fc3
        else:
            return fc3, (fc1, fc2)

    # def embed1024(self, x):
    #     x = self.pool1(F.relu(self.conv1(x)))
    #     x = self.pool2(F.relu(self.conv2(x)))
    #     x = self.dropout(x)
    #     x = x.view(-1, 15 * 16 * 16)
    #     x = F.relu(self.fc1(x))
    #     return x

    # def embed64(self, x):
    #     x = self.pool1(F.relu(self.conv1(x)))
    #     x = self.pool2(F.relu(self.conv2(x)))
    #     x = self.dropout(x)
    #     x = x.view(-1, 15 * 16 * 16)
    #     x = F.relu(self.fc1(x))
    #     x = F.relu(self.fc2(x))
    #     return x

    # def embed13(self, x):
    #     x = self.pool1(F.relu(self.conv1(x)))
    #     x = self.pool2(F.relu(self.conv2(x)))
    #     x = self.dropout(x)
    #     x = x.view(-1, 15 * 16 * 16)
    #     x = F.relu(self.fc1(x))
    #     x = F.relu(self.fc2(x))
    #     x = self.fc3(x)
    #     return x


if __name__ == "__main__":
    # List all the layers of the model
    import timm

    model = timm.create_model("resnet18", pretrained=False, num_classes=13, in_chans=1)
    model.eval()
    for i, (name, param) in enumerate(model.named_modules()):
        print(f"Layer {i}: {name}")

    # # named parameters:
    # Layer 0: conv1.weight - torch.Size([30, 1, 5, 5])
    # Layer 1: conv1.bias - torch.Size([30])
    # Layer 2: conv2.weight - torch.Size([15, 30, 3, 3])
    # Layer 3: conv2.bias - torch.Size([15])
    # Layer 4: fc1.weight - torch.Size([1024, 3840])
    # Layer 5: fc1.bias - torch.Size([1024])
    # Layer 6: fc2.weight - torch.Size([64, 1024])
    # Layer 7: fc2.bias - torch.Size([64])
    # Layer 8: fc3.weight - torch.Size([13, 64])
    # Layer 9: fc3.bias - torch.Size([13])

    # # named modules:
    # Layer 0:
    # Layer 1: conv1
    # Layer 2: pool1
    # Layer 3: conv2
    # Layer 4: pool2
    # Layer 5: dropout
    # Layer 6: fc1
    # Layer 7: fc2
    # Layer 8: fc3

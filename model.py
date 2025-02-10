import torch
import torch.nn as nn
import torch.optim as optim

# Define an improved model architecture
class MyModel(nn.Module):
    def __init__(self, input_dim):
        super(MyModel, self).__init__()
        self.layer1 = nn.Linear(input_dim, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.relu1 = nn.ReLU()
        self.layer2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.relu2 = nn.ReLU()
        self.layer3 = nn.Linear(128, 64)
        self.relu3 = nn.ReLU()
        self.output_layer = nn.Linear(64, 1)

    def forward(self, x):
        x = self.layer1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.layer2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.layer3(x)
        x = self.relu3(x)
        x = self.output_layer(x)
        return x

def create_model(features):
    model = MyModel(features.shape[1])
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    return model, optimizer

if __name__ == '__main__':
    # create sample model with 228 input features
    model, _ = create_model(torch.zeros(1, 228))
    print(model)

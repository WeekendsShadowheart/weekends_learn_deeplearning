import torch
import torch.nn as nn
import torch.optim as optim 
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torchvision import datasets, transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

transform = transforms.Compose([
    transforms.ToTensor(),  
    transforms.Normalize((0.5,), (0.5,))  
])

train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=64, shuffle=True, num_workers=4, pin_memory=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)

class SimpleCNN(nn.Module):
    def __init__():
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3)
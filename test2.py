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

# 定义简单 CNN
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, stride=1, padding=1)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = x.view(-1, 64 * 7 * 7)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# 初始化模型、损失函数和优化器
model = SimpleCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)  # 调整 lr 更稳定

# 训练参数
epochs = 5
train_losses = []  # 用于保存每个 batch 的 loss

# 训练循环
model.train()
for i in range(epochs):
    total_loss = 0
    for batch_idx, (image, label) in enumerate(train_loader):
        image, label = image.to(device), label.to(device)  # 移动到 GPU

        outputs = model(image)
        loss = criterion(outputs, label)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        train_losses.append(loss.item())

        # 每 10 个 batch 显示一次训练进度
        if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == len(train_loader):
            progress = (batch_idx + 1) / len(train_loader) * 100
            print(f"\rEpoch {i+1}/{epochs} - Batch {batch_idx+1}/{len(train_loader)} ({progress:.1f}%) - Loss: {loss.item():.4f}", end='')

    print(f"\nEpoch {i+1} finished, total loss: {total_loss:.4f}")

# 测试集评估
model.eval()
correct = 0
total = 0

with torch.no_grad():
    for image, label in test_loader:
        image, label = image.to(device), label.to(device)
        output = model(image)
        total += image.size(0)
        _, predicted = torch.max(output, 1)
        correct += (predicted == label).sum().item()
    
print(f"Accuracy: {(correct / total):.4f}, correct: {correct:.4f}, total: {total:.4f}")

# 可视化训练损失
plt.figure(figsize=(8,5))
plt.plot(train_losses, label='Training Loss')
plt.xlabel('Batch')
plt.ylabel('Loss')
plt.title('Training Loss over Batches')
plt.legend()
plt.show()

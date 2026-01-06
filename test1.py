import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt


n = 10000
data = torch.randn(n, 2)
labels = (data[:, 0] ** 2 + data[:, 1] ** 2 < 1).float().unsqueeze(1)

class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(2, 16)
        self.fc2 = nn.Linear(16, 16)
        self.fc3 = nn.Linear(16, 16)
        self.fc4 = nn.Linear(16, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.sigmoid(self.fc4(x))
        return x

model = SimpleNN()

print(model)

criterion = nn.BCELoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

epochs = 3000
for epoch in range(epochs):
    outputs = model(data)
    loss = criterion(outputs, labels)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}')

def plot_decision_boundary(model, data):
    x_min, x_max = data[:, 0].min() - 1, data[:, 0].max() + 1
    y_min, y_max = data[:, 1].min() - 1, data[:, 1].max() + 1

    # 使用更细的步长生成网格，步长可以根据需要调整（0.05 或更小）
    xx, yy = torch.meshgrid(torch.arange(x_min, x_max, 0.0001), torch.arange(y_min, y_max, 0.05), indexing='ij')

    # 创建网格点
    grid = torch.cat([xx.reshape(-1, 1), yy.reshape(-1, 1)], dim=1)

    # 预测每个网格点的类别
    predictions = model(grid).detach().numpy().reshape(xx.shape)

    # 绘制决策边界：使用等高线绘制决策区域
    plt.contourf(xx, yy, predictions, levels=[0, 0.5, 1], cmap='coolwarm', alpha=0.7)

    # 绘制原始数据点
    #plt.scatter(data[:, 0], data[:, 1], c=labels.squeeze(), cmap='coolwarm', edgecolors='k')

    # 标题
    plt.title("Decision Boundary")

    # 显示图形
    plt.show()

plot_decision_boundary(model, data)

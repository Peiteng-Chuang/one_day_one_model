import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import SuperResolutionDataset
from model import SRResNet  
from torchvision import transforms
from torch.optim.lr_scheduler import StepLR

# 設定裝置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 資料轉換
transform = transforms.Compose([transforms.ToTensor()])

# 載入資料
dataset = SuperResolutionDataset(img100_dir="./data/img90", img30_dir="./data/img30", transform=transform)
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

# 初始化模型
model = SRResNet().to(device)

# 設定 Loss 和 Optimizer
# criterion = nn.L1Loss()
criterion = nn.MSELoss()        # 讓生成的高解析度圖像更接近 Ground Truth
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 訓練
num_epochs = 50
for epoch in range(num_epochs):
    for i, (lr, hr) in enumerate(dataloader):
        lr, hr = lr.to(device), hr.to(device)

        optimizer.zero_grad()
        output = model(lr)
        loss = criterion(output, hr)
        loss.backward()
        optimizer.step()

        if i % 10 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Step [{i}/{len(dataloader)}], Loss: {loss.item():.4f}")

# 儲存模型
torch.save(model.state_dict(), "super_res_model.pth")

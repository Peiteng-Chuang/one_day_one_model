import torch
from torchvision import transforms
from model import SRResNet  
from PIL import Image
import matplotlib.pyplot as plt

# 設定裝置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 讀取模型
model = SRResNet().to(device)
model.load_state_dict(torch.load("./checkpoint/super_res_model.pth", map_location=device))
model.eval()

# 資料轉換
transform = transforms.Compose([transforms.ToTensor()])

# 讀取圖像並處理
def super_resolve(img_path):
    img = Image.open(img_path).convert("RGB")
    img_lr = transform(img).unsqueeze(0).to(device)  # LR 圖像處理

    with torch.no_grad():
        output = model(img_lr).squeeze(0).cpu()  # 預測結果

    return img, transforms.ToPILImage()(output)  # 返回原圖與超解析度結果

# 測試
# img_path = "C:/Users/Peiteng.Chuang/Desktop/sr/img30/egdt_test_2_48_scale15.png"  # 使用一張 30x30 圖像
img_path = "./data/test/20241127-115647_img145_4_scale15.png"  # 使用一張 30x30 圖像
original_img, result_img = super_resolve(img_path)

# 顯示原圖與結果
fig, axes = plt.subplots(1, 2, figsize=(12, 6))
axes[0].imshow(original_img)
axes[0].set_title("Original Image (30x30)")
axes[0].axis("off")
axes[1].imshow(result_img)
axes[1].set_title("Super Resolved Image (90x90)")
axes[1].axis("off")

plt.show()

import torch
from torchvision import transforms
from model import SRResNet  
from PIL import Image

# 設定裝置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 讀取模型
model = SRResNet().to(device)
model.load_state_dict(torch.load("./checkpoint/super_res_model.pth", map_location=device))
model.eval()

# 資料轉換
transform = transforms.Compose([transforms.ToTensor()])

def super_resolve(img_path):
    img = Image.open(img_path).convert("RGB")
    img = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(img).squeeze(0).cpu()

    return transforms.ToPILImage()(output)

# 測試
img_path = "./data/test/20241127-115647_img145_4_scale15.png"
output_img = super_resolve(img_path)
output_img.show()

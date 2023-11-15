import torch

# 加载.pth文件
checkpoint = torch.load(r'experiments/pretrained_models/lab_32_gopro.pth')

# 打印键值名称

with open("key.txt", "w") as file:
    for key in checkpoint["params"].keys():
        file.write(key + "\n")

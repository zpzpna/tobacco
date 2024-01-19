import os
from torch import nn
import torchvision.models
from torch.utils.data import DataLoader
from torchvision import models, datasets, transforms
from torchinfo import summary
from pathlib import Path


# 注意，transform两种情况考虑！
# 1：自己训练模型：自己造transforms（manually）
# 2：使用别人的pretrained model ，自己的数据transforms需要和别人的transforms一致（automatically）
def create_dataloader(
        train_root: str,
        test_root: str,
        transforms,
        batchsize: int,
        numworkers: int = os.cpu_count()
):
    train_data = datasets.ImageFolder(train_root, transforms)
    test_data = datasets.ImageFolder(test_root, transforms)
    train_loader = DataLoader(train_data, batchsize, True, num_workers=numworkers, pin_memory=True)
    test_loader = DataLoader(test_data, batchsize, False, num_workers=numworkers, pin_memory=True)

    return train_loader, test_loader, train_data.classes


if __name__ == "__main__":
    print()
    # # manually transforms:
    # normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
    #                                  std=[0.229, 0.224, 0.225])
    # manual_transforms = transforms.Compose([transforms.Resize((224, 224)),  # 设置图片大小
    #                                         transforms.ToTensor(),  # 设置图片像素【0，1】
    #                                         normalize])  # 使得图片分布和ImageNet的分布一样
    #
    # weights = torchvision.models.EfficientNet_B0_Weights.DEFAULT  # Default 代表最好的参数
    # # auto_transforms = weights.transforms
    #
    model = torchvision.models.efficientnet_b2(weights=torchvision.models.EfficientNet_B2_Weights.DEFAULT)
    for params in model.parameters():
        params.requires_grad = False
    # model.heads = nn.Sequential(
    #     nn.Linear(768, 9))
    summary(model, (32, 3, 224, 224), col_names=["input_size", "output_size", "trainable"], row_settings=["var_names"])
    print(model.features[8])
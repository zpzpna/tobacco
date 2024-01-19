import torch
from torch.utils.data import Dataset
from pathlib import Path
import os
from PIL import Image


def find_class(data_path):
    classes = [entry.name for entry in os.scandir(data_path)]
    class_to_idx = {class_name: i for i, class_name in enumerate(classes)}
    return classes, class_to_idx


class CustomImageFolder(Dataset):
    def __init__(self, root_dir: str, transforms) -> None:
        super().__init__()
        self.paths = list(Path(root_dir).glob("*/*.jpg"))
        # print(self.paths[3692]) 这里是查找上传时出错的图片，设置了dataloader batch为1，shufle=0，然后看列表顺序来查找错误图片
        self.transform = transforms
        self.classes, self.class_to_idx = find_class(root_dir)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index) -> (torch.Tensor, int):
        img = Image.open(self.paths[index])
        class_name = self.paths[index].parent.name
        class_idx = self.class_to_idx[class_name]
        if self.transform:
            return self.transform(img), class_idx
        else:
            return img, class_idx


if __name__ == "__main__":
    import torchvision.transforms

    train_dir = Path.cwd() / "img_data"/ "train"
    test_dir = Path.cwd() / "img_data" / "test"
    transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize((256, 256)),
        torchvision.transforms.ToTensor()
    ])
    train_data = CustomImageFolder(train_dir,
                                   transforms)
    print(train_data)

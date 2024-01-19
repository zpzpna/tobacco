import torch.optim
import argparse
import engine, data_setup, utils
from pathlib import Path
from torchvision import models
from torch import nn
import os

os.environ["CUDA_VISIBLE_DEVICES"] = '1'
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-l", "--lr", type=float, default=5e-5)
    parser.add_argument("-d", "--device", type=str, default='cpu', choices=['cpu', 'cuda'])
    parser.add_argument("-e", "--epochs", type=int, default=5)
    parser.add_argument("-b", "--batch", type=int, default=32)
    opt = parser.parse_args()
    datadir = Path("img_data/")
    train_dir = datadir / "train"
    test_dir = datadir / "test"
    transforms = models.ViT_B_32_Weights.DEFAULT.transforms()  # 一定要有括号！transforms()
    train_loader, test_loader, class_name = data_setup.create_dataloader(train_dir.__str__(),
                                                                         test_dir.__str__(),
                                                                         transforms,
                                                                         opt.batch)

    all_model = ["efficientNet_b2", "vit_b32"]
    all_epoch = [10, 20]
    experiment_num = 1
    for model_name in all_model:
        for epochs in all_epoch:
            print("*" * 15)
            print(f"experiment_num: {experiment_num}")
            experiment_num += 1
            print("*" * 15)
            print(f"experiment_name: data_10_percent")
            print(f"model_name: {model_name}")
            print(f"epochs_num:{epochs}")
            if model_name == "vit_b32":
                model = models.vit_b_32(weights=models.ViT_B_32_Weights.DEFAULT)
                for params in model.parameters():
                    params.requires_grad = False
                model.heads = nn.Sequential(
                    nn.Linear(768, len(class_name)))
            else:
                model = models.efficientnet_b2(weights=models.EfficientNet_B2_Weights.DEFAULT)
                model.classifier = nn.Sequential(
                    nn.Dropout(0.3, True),
                    nn.Linear(1408, len(class_name)))

            optimizer = torch.optim.Adam(model.parameters(), opt.lr)
            loss_fn = torch.nn.CrossEntropyLoss()
            acc_fn = engine.accuracy
            device = opt.device
            # epochs = opt.epochs
            epochs = epochs
            model = model.to(device)
            results = engine.train(model, train_loader, test_loader, optimizer, loss_fn, acc_fn, device, epochs,
                                   writer=utils.create_write("data_100_percent_all", model_name,
                                                             extra=f"{epochs}_epochs_{opt.batch}_batch_{opt.lr}_lr"))

            model_dir = Path.cwd() / Path("saved_model")
            utils.save_model(model, model_dir.__str__(), model_name)

    # utils.save_loss_acc(results,"./loss_acc.json")

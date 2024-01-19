from torch.utils.tensorboard import SummaryWriter
import json
from pathlib import Path
from datetime import datetime
import os
import matplotlib.pyplot as plt
import torch


def save_model(model: torch.nn.Module,
               model_dir: str,
               model_name: str) -> None:
    model_name = Path(model_name)
    model_dir = Path(model_dir)

    model_path = model_dir / (model_name.__str__() + ".pth")
    model_dir.mkdir(exist_ok=True, parents=True)
    print("save model......")
    torch.save(obj=model.state_dict(),
               f=model_path.__str__())
    print(f"model has been saved in '{model_path}'")


def save_loss_acc(data: dict,
                  path) -> None:
    with open(path, "w") as wf:
        json.dump(obj=data, fp=wf)
    print(f"loss & acc has been saved in '{path}'")


def plot_loss_acc(epochs,
                  train_data,
                  test_data,
                  ax):
    plt.subplot(1, 2, ax)
    plt.plot(epochs, train_data, label="train_loss")
    plt.plot(epochs, test_data, label="test_loss")
    plt.legend()


def create_write(experiment_name: str,
                 model_name: str,
                 extra: str = None):
    fmt_time = datetime.now().strftime("%Y-%B-%d")
    if extra:
        log_dir = os.path.join("runs", fmt_time, experiment_name, model_name, extra)
    else:
        log_dir = os.path.join("runs", fmt_time, experiment_name, model_name)
    return SummaryWriter(log_dir)

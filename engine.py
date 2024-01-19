from torch.utils.tensorboard import SummaryWriter
import torch.utils.data
from utils import create_write
from tqdm import trange


def accuracy(y_pred, y_true):
    correct = torch.eq(y_pred, y_true).sum().item()
    acc = correct / len(y_true) * 100
    return acc


def train_step(model: torch.nn.Module,
               train_loader,
               loss_fn,
               optimizer: torch.optim.Optimizer,
               acc_fn,
               device):
    model.train()

    loss, acc = torch.tensor(0.0, device=device), 0
    for batch_num, (x_train, y_train) in enumerate(train_loader):
        x_train, y_train = x_train.to(device), y_train.to(device),

        logits = model(x_train)
        probs = torch.softmax(logits, dim=-1)
        y_train_pred = torch.argmax(probs, dim=-1).squeeze()

        loss_batch = loss_fn(logits, y_train)
        acc_batch = acc_fn(y_train_pred, y_train)

        optimizer.zero_grad()
        loss_batch.backward()
        optimizer.step()

        loss += loss_batch
        acc += acc_batch
    loss /= len(train_loader)
    acc /= len(train_loader)
    return loss, acc


def test_step(model: torch.nn.Module,
              test_loader,
              loss_fn,
              acc_fn,
              device):
    model.eval()

    with torch.inference_mode():
        loss, acc = torch.tensor(0.0, device=device), 0
        for batch_num, (x_test, y_test) in enumerate(test_loader):
            x_test, y_test = x_test.to(device), y_test.to(device),

            logits = model(x_test)
            probs = torch.softmax(logits, dim=-1)
            y_test_pred = torch.argmax(probs, dim=-1).squeeze()

            loss_batch = loss_fn(logits, y_test)
            acc_batch = acc_fn(y_test_pred, y_test)

            loss += loss_batch
            acc += acc_batch
        loss /= len(test_loader)
        acc /= len(test_loader)
        return loss, acc


def train(model: torch.nn.Module,
          train_loader,
          test_loader,
          optimizer,
          loss_fn,
          acc_fn,
          device,
          epochs,
          writer: torch.utils.tensorboard.writer.SummaryWriter = None):
    results = {
        "loss_train": [],
        "loss_test": [],
        "acc_train": [],
        "acc_test": [],
        "epochs": []
    }
    # writer = create_write("data_10_percent","effcientb0","5_epochs")
    for epoch in trange(epochs):
        loss_train, acc_train = train_step(model, train_loader, loss_fn, optimizer, acc_fn, device)
        loss_test, acc_test = test_step(model, test_loader, loss_fn, acc_fn, device)
        print(f"train_loss:{loss_train} | train_acc:{acc_train}|test_loss:{loss_test} | test_acc:{acc_test}")
        results["loss_train"].append(loss_train.item())
        results["loss_test"].append(loss_test.item())
        results["acc_train"].append(acc_train)
        results["acc_test"].append(acc_test)
        results["epochs"].append(epoch)
        if writer:
            writer.add_scalars("Loss",
                               {"train_loss": loss_train,
                                "test_loss": loss_test},
                               epoch)
            writer.add_scalars("Acc",
                               {"train_acc": acc_train,
                                "test_acc": acc_test},
                               epoch)
            writer.add_graph(model=model,
                             input_to_model=torch.randn(32, 3, 224, 224, device=device))
            writer.flush()
            writer.close()
    return results


# 使用ensorboard的命令行

"""
%%load_ext tensorboard
tensorboard --logdir runs --port=6006

上传：tensorboard dev upload --logdir runs \
            --name "sss"
            --description "xxx"
"""

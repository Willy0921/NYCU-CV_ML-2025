import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import os
import sys
import signal
import argparse
import time
import datetime
import logging
import random
import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from timm.data import create_transform
from model import (
    CustomResNeXt101,
    CustomResNeXt26_Simple,
    CustomResNeXt50_32x4d_Simple,
    CustomResNeXt101_32x8d_Simple,
    CustomSEResNeXt50_32x4d_Simple,
)
from torch.amp import autocast, GradScaler

global model, optimizer, current_epoch, EXP_DIR
EXP_DIR = None
LAST_CHECKPOINT_TIME = time.time()


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_data_loaders(data_dir, batch_size=32, input_size=512):
    train_transform = create_transform(
        input_size=input_size,
        is_training=True,
        auto_augment="rand-m9-mstd0.5",
        # color_jitter=0.4, # brightness, contrast, saturation, hue
        # re_prob=0.3,      # random erasing
        # re_mode="pixel",  # random erasing mode
        # re_count=1,       # random erasing count
        # scale=(0.08, 1.0), # input re-scale
        # ratio=(0.75, 1.33), # aspect ratio range
    )
    val_transform = create_transform(
        input_size=input_size,
        is_training=False,
    )

    train_dataset = torchvision.datasets.ImageFolder(
        root=os.path.join(data_dir, "train"), transform=train_transform
    )
    val_dataset = torchvision.datasets.ImageFolder(
        root=os.path.join(data_dir, "val"), transform=val_transform
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=4
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=4
    )
    return train_loader, val_loader


def save_checkpoint(model, optimizer, epoch, exp_dir, filename="checkpoint.pth"):
    global LAST_CHECKPOINT_TIME
    current_time = time.time()
    if current_time - LAST_CHECKPOINT_TIME >= 1800:  # save checkpoint every 30 minutes
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        }
        path = os.path.join(exp_dir, filename)
        torch.save(checkpoint, path)
        LAST_CHECKPOINT_TIME = current_time
        logging.info(f"Checkpoint saved at epoch {epoch} in {path}")


def load_checkpoint(model, optimizer, resume_dir, filename="checkpoint.pth"):
    path = os.path.join(resume_dir, filename)
    if os.path.exists(path):
        checkpoint = torch.load(path)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"]
        logging.info(
            f"Checkpoint loaded from {path}. Resuming from epoch {start_epoch}"
        )
        return start_epoch
    return 0


def handle_interrupt(signal_num, frame):
    logging.info("Interrupt received! Saving checkpoint before exiting...")
    global model, optimizer, current_epoch, EXP_DIR
    save_checkpoint(model, optimizer, current_epoch, EXP_DIR)
    sys.exit(0)


def evaluate_loss_and_acc(model, val_loader, device, criterion):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    avg_loss = total_loss / total
    acc = 100 * correct / total
    return avg_loss, acc


def plot_training_curves(train_loss, val_loss, train_acc, val_acc, exp_dir):
    epochs = range(1, len(train_loss) + 1)

    # plot loss curve
    plt.figure()
    plt.plot(epochs, train_loss, label="Train Loss")
    plt.plot(epochs, val_loss, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    loss_fig_path = os.path.join(exp_dir, "train_val_loss.png")
    plt.savefig(loss_fig_path)
    plt.close()
    logging.info(f"Training/Validation Loss curve saved to {loss_fig_path}")

    # plot accuracy curve
    plt.figure()
    plt.plot(epochs, train_acc, label="Train Acc")
    plt.plot(epochs, val_acc, label="Val Acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title("Training and Validation Accuracy")
    plt.legend()
    acc_fig_path = os.path.join(exp_dir, "train_val_acc.png")
    plt.savefig(acc_fig_path)
    plt.close()
    logging.info(f"Training/Validation Accuracy curve saved to {acc_fig_path}")


def train(
    model,
    train_loader,
    val_loader,
    device,
    exp_dir,
    epochs=10,
    lr=1e-4,
    resume=False,
    resume_dir=None,
):
    global current_epoch, optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-6
    )
    scaler = GradScaler()
    model.to(device)
    model = torch.compile(model)

    start_epoch = 0
    if resume:
        start_epoch = load_checkpoint(model, optimizer, resume_dir)

    # Initialize best
    best_val_acc = 0.0

    # Initialize history
    train_loss_history = []
    train_acc_history = []
    val_loss_history = []
    val_acc_history = []

    # Setup signal handler
    signal.signal(signal.SIGINT, handle_interrupt)  # handle Ctrl+C
    signal.signal(signal.SIGTERM, handle_interrupt)  # handle kill

    for epoch in range(start_epoch, epochs):
        current_epoch = epoch
        model.train()
        running_loss = 0.0
        correct, total = 0, 0
        start_time = time.time()

        try:
            for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
                images, labels = images.to(device), labels.to(device)

                optimizer.zero_grad()

                with autocast(device_type="cuda"):
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                running_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            scheduler.step()

            epoch_time = time.time() - start_time
            remaining_time = epoch_time * (epochs - epoch - 1)

            train_loss = running_loss / len(train_loader)
            train_acc = 100 * correct / total
            val_loss, val_acc = evaluate_loss_and_acc(
                model, val_loader, device, criterion
            )

            train_loss_history.append(train_loss)
            train_acc_history.append(train_acc)
            val_loss_history.append(val_loss)
            val_acc_history.append(val_acc)

            log_message = (
                "Epoch [{}/{}], Train Loss: {:.4f}, Val Loss: {:.4f}, Train Acc: {:.2f}%, Val Acc: {:.2f}%, Remaining: {:.2f} min"
            ).format(
                epoch + 1,
                epochs,
                train_loss,
                val_loss,
                train_acc,
                val_acc,
                remaining_time / 60,
            )
            logging.info(log_message)
            save_checkpoint(model, optimizer, epoch + 1, exp_dir)

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model_path = os.path.join(exp_dir, "best_model.pth")
                torch.save(model.state_dict(), best_model_path)
                logging.info(
                    f"Best model updated at epoch {epoch+1} with Val Acc: {val_acc:.2f}%"
                )
        except Exception as e:
            logging.error(f"Unexpected error: {e}. Saving checkpoint before exiting...")
            save_checkpoint(model, optimizer, epoch, exp_dir)
            break

    # training curve
    plot_training_curves(
        train_loss_history,
        val_loss_history,
        train_acc_history,
        val_acc_history,
        exp_dir,
    )


def evaluate(model, val_loader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    val_acc = 100 * correct / total
    return val_acc


def main():
    parser = argparse.ArgumentParser(description="Train ResNeXt using timm")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to dataset")
    parser.add_argument(
        "--epochs", type=int, default=50, help="Number of training epochs"
    )
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument(
        "--resume", action="store_true", help="Resume training from checkpoint"
    )
    parser.add_argument(
        "--resume_dir", type=str, help="Path to directory containing checkpoint"
    )
    args = parser.parse_args()

    set_seed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    exp_base_dir = "./exp"
    if not os.path.exists(exp_base_dir):
        os.makedirs(exp_base_dir)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = os.path.join(exp_base_dir, timestamp)
    os.makedirs(exp_dir)
    global EXP_DIR
    EXP_DIR = exp_dir

    log_file = os.path.join(exp_dir, "experiment.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler(sys.stdout)],
    )
    logging.info("batch_size: " + str(args.batch_size))
    logging.info("lr: " + str(args.lr))
    logging.info("epochs: " + str(args.epochs))
    logging.info("Experiment directory: " + exp_dir)

    train_loader, val_loader = get_data_loaders(args.data_dir, args.batch_size)

    num_classes = len(train_loader.dataset.classes)
    logging.info(f"Number of classes: {num_classes}")

    # model = CustomResNeXt101_32x8d_Simple(num_classes=num_classes)
    # model = CustomResNeXt50_32x4d_Simple(num_classes=num_classes)
    model = CustomSEResNeXt50_32x4d_Simple(num_classes)
    logging.info("model name: " + model.__class__.__name__)
    logging.info(
        f"Number of parameters in model: {sum(p.numel() for p in model.parameters())/1e6} M"
    )
    train(
        model,
        train_loader,
        val_loader,
        device,
        exp_dir,
        args.epochs,
        args.lr,
        args.resume,
        args.resume_dir,
    )

    model_save_path = os.path.join(exp_dir, "last_epoch_model.pth")
    torch.save(model.state_dict(), model_save_path)
    logging.info("Model saved to " + model_save_path)


if __name__ == "__main__":
    main()

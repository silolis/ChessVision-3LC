import argparse
from pathlib import Path
from typing import Any

import timm
import tlc
import torch
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import tqdm
from torch.utils.data import DataLoader

from chessvision.piece_classification.model import ChessVisionModel
from chessvision.piece_classification.training_utils import EarlyStopping
from chessvision.utils import classifier_weights_dir

mp.set_start_method("spawn", force=True)
CHECKPOINT_PATH = Path(classifier_weights_dir) / "checkpoint.pth"
CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)

DATASET_ROOT = "C:/Data/chessboard-squares"
tlc.register_url_alias("CHESSPIECES_DATASET_ROOT", DATASET_ROOT)

TRAIN_DATASET_PATH = DATASET_ROOT + "/training"
VAL_DATASET_PATH = DATASET_ROOT + "/validation"

TRAIN_DATASET_NAME = "chesspieces-train"
VAL_DATASET_NAME = "chesspieces-val"


train_transforms = transforms.Compose(
    [
        transforms.Resize((64, 64)),
        transforms.Grayscale(),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.95, 1.05), shear=0),
        transforms.ToTensor(),
        transforms.Normalize([0.564], [0.246]),
    ]
)

val_transforms = transforms.Compose(
    [
        transforms.Resize((64, 64)),
        transforms.Grayscale(),
        transforms.ToTensor(),
        transforms.Normalize([0.564], [0.246]),
    ]
)

label_names = [
    "White Bishop",
    "White King",
    "White Knight",
    "White Pawn",
    "White Queen",
    "White Rook",
    "Black Bishop",
    "Black King",
    "Black Knight",
    "Black Pawn",
    "Black Queen",
    "Black Rook",
    "Empty Square",
    "Unknown",
]


train_dataset = datasets.ImageFolder(TRAIN_DATASET_PATH)
val_dataset = datasets.ImageFolder(VAL_DATASET_PATH)

sample_structure = (tlc.PILImage("image"), tlc.CategoricalLabel("label", classes=label_names))


def train_trans(sample):
    return train_transforms(sample[0]), sample[1]


def val_trans(sample):
    return val_transforms(sample[0]), sample[1]


tlc_train_dataset = (
    tlc.Table.from_torch_dataset(
        dataset=train_dataset,
        dataset_name=TRAIN_DATASET_NAME,
        table_name="train",
        structure=sample_structure,
    )
    .map(train_trans)
    .map_collect_metrics(val_trans)
    .latest()
)

tlc_val_dataset = (
    tlc.Table.from_torch_dataset(
        dataset=val_dataset,
        dataset_name=VAL_DATASET_NAME,
        table_name="val",
        structure=sample_structure,
    )
    .map(val_trans)
    .latest()
)

print(f"Using training dataset: {tlc_train_dataset.url}")
print(f"Using validation dataset: {tlc_val_dataset.url}")


# Training step
def train(
    model: nn.Module,
    train_loader: Any,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for data, target in tqdm.tqdm(train_loader, desc="Training", total=len(train_loader)):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        _, predicted = output.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()

    train_loss = running_loss / len(train_loader)
    accuracy = 100.0 * correct / total
    return train_loss, accuracy


# Validation step
def validate(model, val_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in tqdm.tqdm(val_loader, desc="Validation", total=len(val_loader)):
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            running_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()

    val_loss = running_loss / len(val_loader)
    accuracy = 100.0 * correct / total
    return val_loss, accuracy


# Hyperparameters
DENSE_1_SIZE = 128
DENSE_2_SIZE = 64
NUM_CLASSES = 13
INITIAL_LR = 0.001
LR_SCHEDULER_STEP_SIZE = 4
LR_SCHEDULER_GAMMA = 0.1
MAX_EPOCHS = 10
EARLY_STOPPING_PATIENCE = 4
HIDDEN_LAYER_INDEX = 7


def load_checkpoint(model: torch.nn.Module, optimizer: torch.optim.Optimizer | None = None, filename="checkpoint.pth"):
    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    epoch = checkpoint["epoch"]
    best_val_loss = checkpoint["best_val_loss"]
    return model, optimizer, epoch, best_val_loss


def save_checkpoint(model, optimizer, epoch, best_val_loss, filename="checkpoint.pth"):
    torch.save(
        {
            "epoch": epoch,
            "best_val_loss": best_val_loss,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        filename,
    )


def main(args):
    # Training variables
    early_stopping = EarlyStopping(patience=EARLY_STOPPING_PATIENCE, verbose=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model(args.model_id).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=INITIAL_LR)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=LR_SCHEDULER_STEP_SIZE, gamma=LR_SCHEDULER_GAMMA)
    start_epoch = 0
    best_val_loss = float("inf")

    if CHECKPOINT_PATH.exists():
        model, optimizer, start_epoch, best_val_loss = load_checkpoint(model, optimizer, str(CHECKPOINT_PATH))
        start_epoch += 1  # Increment to continue from next epoch

    # Create datasets
    run_parameters = {
        "DENSE_1_SIZE": DENSE_1_SIZE,
        "DENSE_2_SIZE": DENSE_2_SIZE,
        "NUM_CLASSES": NUM_CLASSES,
        "INITIAL_LR": INITIAL_LR,
        "LR_SCHEDULER_STEP_SIZE": LR_SCHEDULER_STEP_SIZE,
        "LR_SCHEDULER_GAMMA": LR_SCHEDULER_GAMMA,
        "MAX_EPOCHS": MAX_EPOCHS,
        "EARLY_STOPPING_PATIENCE": EARLY_STOPPING_PATIENCE,
        "HIDDEN_LAYER_INDEX": HIDDEN_LAYER_INDEX,
    }

    run = tlc.init(
        project_name=args.project_name,
        description=args.description,
        parameters=run_parameters,
        # if_exists="reuse",
    )
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    sampler = tlc_train_dataset.create_sampler()
    train_data_loader = DataLoader(
        tlc_train_dataset,
        batch_size=32,
        shuffle=False,
        sampler=sampler,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
    )
    val_data_loader = DataLoader(tlc_val_dataset, batch_size=32, shuffle=False, num_workers=0, pin_memory=True)
    metrics_collection_dataloader_args = {
        "batch_size": 512,
    }
    chessvision_metrics_collector = tlc.ClassificationMetricsCollector(
        classes=label_names,
    )

    metrics_collectors = [
        chessvision_metrics_collector,
    ]
    if args.compute_embeddings:
        embeddings_collector = tlc.EmbeddingsMetricsCollector(layers=[HIDDEN_LAYER_INDEX])
        metrics_collectors.append(embeddings_collector)

    predictor = tlc.Predictor(model, layers=[HIDDEN_LAYER_INDEX])

    for epoch in range(start_epoch, start_epoch + MAX_EPOCHS):
        train_loss, train_acc = train(model, train_data_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_data_loader, criterion, device)
        scheduler.step()

        tlc.collect_metrics(
            tlc_val_dataset,
            metrics_collectors=metrics_collectors,
            predictor=predictor,
            split="val",
            constants={"epoch": epoch},
            dataloader_args=metrics_collection_dataloader_args,
        )

        tlc.collect_metrics(
            tlc_train_dataset,
            metrics_collectors=metrics_collectors,
            predictor=predictor,
            split="train",
            constants={"epoch": epoch},
            dataloader_args=metrics_collection_dataloader_args,
        )

        tlc.log(
            {
                "train_loss": train_loss,
                "val_loss": val_loss,
                "train_acc": train_acc,
                "val_acc": val_acc,
                "epoch": epoch,
                "learning_rate": optimizer.param_groups[0]["lr"],
            }
        )

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accuracies.append(train_acc)
        val_accuracies.append(val_acc)

        print(
            f"Epoch: {epoch + 1}, "
            f"Train Loss: {train_loss:.4f}, "
            f"Val Loss: {val_loss:.4f}, "
            f"Train Acc: {train_acc:.2f}%, "
            f"Val Acc: {val_acc:.2f}%"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            print("Saving model...")
            save_checkpoint(model, optimizer, epoch, best_val_loss, str(CHECKPOINT_PATH))

        early_stopping(val_loss, model)

        if early_stopping.early_stop:
            print("Early stopping")
            break

    if args.compute_embeddings:
        print("Reducing embeddings...")
        run.reduce_embeddings_per_dataset(n_components=2, method="pacmap")

    print("Training completed")


def get_model(model_id: str) -> nn.Module:
    if model_id == "chessvision-cnn":
        model = ChessVisionModel(DENSE_1_SIZE, DENSE_2_SIZE, NUM_CLASSES)
    elif model_id == "resnet18":
        model = timm.create_model("resnet18", pretrained=False, num_classes=NUM_CLASSES, in_chans=1)
    elif model_id == "efficientnet":
        model = timm.create_model("efficientnet_b0", pretrained=False, num_classes=NUM_CLASSES, in_chans=1)
    else:
        raise ValueError(f"Unknown model id: {model_id}")
    return model


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--project-name", type=str, default="chessvision-classification")
    argparser.add_argument("--description", type=str, default="Train a chess piece classifier")
    argparser.add_argument("--model-id", type=str, help="The model id to train", default="resnet18")
    argparser.add_argument("--compute-embeddings", action="store_true")
    args = argparser.parse_args()
    main(args)
x

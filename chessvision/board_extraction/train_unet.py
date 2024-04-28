from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

import tlc
import torch
import torch.nn as nn

# import wandb
from evaluate import evaluate
from torch import optim
from torch.utils.data import DataLoader, random_split
from torchvision.transforms import ToTensor
from tqdm import tqdm
from unet import UNet
from utils.data_loading import BasicDataset
from utils.dice_score import dice_loss

from chessvision.board_extraction.loss_collector import LossCollector
from chessvision.utils import best_extractor_weights, extractor_weights_dir

# Ensure dataset has been downloaded and exists at DATASET_ROOT. See playground.ipynb for details.
DATASET_ROOT = "C:/Data/chessboard-segmentation"
tlc.register_url_alias(
    "CHESSVISION_SEGMENTATION_DATA_ROOT",
    DATASET_ROOT,
)
tlc.register_url_alias(
    "CHESSVISION_SEGMENTATION_PROJECT_ROOT",
    "C:/Users/gudbrand/AppData/Local/3LC/3LC/projects/chessboard-segmentation",
)

dir_img = Path(DATASET_ROOT) / "images/"
dir_mask = Path(DATASET_ROOT) / "masks/"
assert dir_img.exists()
assert dir_mask.exists()
dir_checkpoint = extractor_weights_dir

USE_WANDB = False

def load_checkpoint(model: torch.nn.Module, checkpoint_path: str):
    state_dict = torch.load(checkpoint_path)
    # del state_dict["mask_values"]
    model.load_state_dict(state_dict)
    logging.info(f"Model loaded from {checkpoint_path}")
    return model

def _save_checkpoint(model: torch.nn.Module, checkpoint_path: str):
    state_dict = model.state_dict()
    torch.save(state_dict, checkpoint_path)

class TransformSampleToModel:
    """Convert a dict of PIL images to a dict of tensors of the right type."""

    def __call__(self, sample: dict[str, Any]) -> dict[str, torch.Tensor]:
        image, mask = sample["image"], sample["mask"]
        return {
            "image": ToTensor()(image),
            "mask": ToTensor()(mask).long(),
        }

class PrepareModelOutputsForLogging:
    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold

    def __call__(self, batch, predictor_output: tlc.PredictorOutput):
        predictions_tensor = predictor_output.forward

        for i in range(len(predictions_tensor)):
            predictions_tensor[i] = (torch.sigmoid(predictions_tensor[i]) > self.threshold) * 255

        return batch, predictions_tensor.squeeze(1)


def train_model(
    model,
    device,
    epochs: int = 5,
    batch_size: int = 1,
    learning_rate: float = 1e-5,
    val_percent: float = 0.1,
    save_checkpoint: bool = True,
    img_scale: float = 0.5,
    amp: bool = False,
    weight_decay: float = 1e-8,
    momentum: float = 0.999,
    gradient_clipping: float = 1.0,
):
    # 1. Create dataset
    dataset = BasicDataset(dir_img, dir_mask, img_scale)

    # 2. Split into train / validation partitions
    n_val = int(len(dataset) * val_percent)
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(0))

    # 2.1 Create 3LC datasets & training run
    run = tlc.init("chessboard-segmentation")

    sample_structure = {
        "image": tlc.PILImage("image"),
        "mask": tlc.SegmentationPILImage("mask", classes={0: "background", 255: "chessboard"}),
    }

    tlc_val_dataset = tlc.Table.from_torch_dataset(
        dataset=val_set,
        dataset_name="chessboard-segmentation-val",
        structure=sample_structure,
    ).map(TransformSampleToModel()).latest()

    tlc_train_dataset = tlc.Table.from_torch_dataset(
        dataset=train_set,
        dataset_name="chessboard-segmentation-train",
        structure=sample_structure,
    ).map(TransformSampleToModel()).latest()

    # 3. Create data loaders
    loader_args = {"batch_size": batch_size, "num_workers": 0, "pin_memory": True}
    train_loader = DataLoader(
        tlc_train_dataset,
        shuffle=False,
        sampler=tlc_train_dataset.create_sampler(),
        **loader_args,
    )
    val_loader = DataLoader(
        tlc_val_dataset,
        shuffle=False,
        drop_last=True,
        **loader_args,
    )

    # (Initialize logging)
    if USE_WANDB:
        experiment = wandb.init(project="U-Net", resume="allow", anonymous="must")
        experiment.config.update(
            dict(
                epochs=epochs,
                batch_size=batch_size,
                learning_rate=learning_rate,
                val_percent=val_percent,
                save_checkpoint=save_checkpoint,
                img_scale=img_scale,
                amp=amp,
            )
        )

    logging.info(
        f"""Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {learning_rate}
        Training size:   {n_train}
        Validation size: {n_val}
        Checkpoints:     {save_checkpoint}
        Device:          {device.type}
        Images scaling:  {img_scale}
        Mixed Precision: {amp}
    """
    )

    # 4. Set up the optimizer, the loss, the learning rate scheduler and the loss scaling for AMP
    optimizer = optim.RMSprop(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay, momentum=momentum, foreach=True
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, "max", patience=5)  # goal: maximize Dice score
    grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)
    criterion = nn.BCEWithLogitsLoss()
    global_step = 0

    # 5. Begin training
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0
        with tqdm(total=n_train, desc=f"Epoch {epoch}/{epochs}", unit="img") as pbar:
            for batch in train_loader:
                images, true_masks = batch["image"], batch["mask"]

                assert images.shape[1] == model.n_channels, (
                    f"Network has been defined with {model.n_channels} input channels, "
                    f"but loaded images have {images.shape[1]} channels. Please check that "
                    "the images are loaded correctly."
                )

                images = images.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
                true_masks = true_masks.to(device=device, dtype=torch.long)

                with torch.autocast(device.type if device.type != "mps" else "cpu", enabled=amp):
                    masks_pred = model(images)
                    loss = criterion(masks_pred, true_masks.float())
                    loss += dice_loss(
                        torch.sigmoid(masks_pred),
                        true_masks.float(),
                        multiclass=False,
                        reduce_batch_first=False,
                    )

                optimizer.zero_grad(set_to_none=True)
                grad_scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
                grad_scaler.step(optimizer)
                grad_scaler.update()

                pbar.update(images.shape[0])
                global_step += 1
                epoch_loss += loss.item()
                if USE_WANDB:
                    experiment.log({"train loss": loss.item(), "step": global_step, "epoch": epoch})

                pbar.set_postfix(**{"loss (batch)": loss.item()})

                # Evaluation round
                division_step = n_train // (5 * batch_size)
                if division_step > 0:
                    if global_step % division_step == 0:
                        if USE_WANDB:
                            histograms = {}
                            for tag, value in model.named_parameters():
                                tag = tag.replace("/", ".")
                                if not (torch.isinf(value) | torch.isnan(value)).any():
                                    histograms["Weights/" + tag] = wandb.Histogram(value.data.cpu())
                                if not (torch.isinf(value.grad) | torch.isnan(value.grad)).any():
                                    histograms["Gradients/" + tag] = wandb.Histogram(value.grad.data.cpu())

                        val_score = evaluate(model, val_loader, device, amp)
                        scheduler.step(val_score)

                        tlc.log({"val_dice": val_score.item(), "step": global_step})
                        logging.info(f"Validation Dice score: {val_score}")

                        if USE_WANDB:
                            experiment.log(
                                {
                                    "learning rate": optimizer.param_groups[0]["lr"],
                                    "validation Dice": val_score,
                                    "images": wandb.Image(
                                        images[0].cpu(),
                                        masks={
                                            "predicted": {
                                                "mask_data": (torch.sigmoid(masks_pred[0]) > 0.5)
                                                .float()
                                                .cpu()
                                                .numpy()
                                                .squeeze(0),
                                                "class_labels": {0: "background", 1: "chessboard"},
                                            },
                                            "true": {
                                                "mask_data": true_masks[0].float().cpu().numpy(),
                                                "class_labels": {0: "background", 1: "chessboard"},
                                            },
                                        },
                                    ),
                                    "step": global_step,
                                    "epoch": epoch,
                                    **histograms,
                                }
                            )

        if save_checkpoint:
            Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)
            _save_checkpoint(model, str(Path(dir_checkpoint) / f"checkpoint_epoch{epoch}.pth"))
            logging.info(f"Checkpoint {epoch} saved!")

        tlc.log({"train_loss": epoch_loss / n_train, "epoch": epoch})

        # Collect per-sample metrics using tlc every 5 epochs
        if epoch in [1, 5, 10, 15, 20]:
            predictor = tlc.Predictor(
                model=model,
                layers=[52],
            )

            collectors = [
                LossCollector(),
                tlc.SegmentationMetricsCollector(
                    label_map={0: "background", 255: "chessboard"},
                    preprocess_fn=PrepareModelOutputsForLogging(threshold=0.75)
                ),
                tlc.EmbeddingsMetricsCollector(
                    layers=[52],
                    reshape_strategy={52: "mean"}
                )
            ]

            tlc.collect_metrics(
                tlc_train_dataset,
                metrics_collectors=collectors,
                predictor=predictor,
                split="train",
                constants={"step": global_step, "epoch": epoch},
                dataloader_args={"batch_size": 4},
            )
            tlc.collect_metrics(
                tlc_val_dataset,
                metrics_collectors=collectors,
                predictor=predictor,
                split="val",
                constants={"step": global_step, "epoch": epoch},
                dataloader_args={"batch_size": 4},
            )

    print("Training completed. Reducing embeddings to 3 dimensions using pacmap...")
    run.reduce_embeddings_by_foreign_table_url(
        tlc_train_dataset.url,
        delete_source_tables=True,
        method="pacmap",
        n_components=2,
    )


def get_args():
    parser = argparse.ArgumentParser(description="Train the UNet on images and target masks")
    parser.add_argument("--epochs", "-e", metavar="E", type=int, default=5, help="Number of epochs")
    parser.add_argument("--batch-size", "-b", dest="batch_size", metavar="B", type=int, default=1, help="Batch size")
    parser.add_argument(
        "--learning-rate", "-l", metavar="LR", type=float, default=1e-5, help="Learning rate", dest="lr"
    )
    parser.add_argument("--load", "-f", type=str, default=False, help="Load model from a .pth file")
    parser.add_argument("--scale", "-s", type=float, default=0.5, help="Downscaling factor of the images")
    parser.add_argument(
        "--validation",
        "-v",
        dest="val",
        type=float,
        default=10.0,
        help="Percent of the data that is used as validation (0-100)",
    )
    parser.add_argument("--amp", action="store_true", default=False, help="Use mixed precision")
    parser.add_argument("--bilinear", action="store_true", default=False, help="Use bilinear upsampling")
    parser.add_argument("--classes", "-c", type=int, default=2, help="Number of classes")

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device {device}")

    # Change here to adapt to your data
    # n_channels=3 for RGB images
    # n_classes is the number of probabilities you want to get per pixel
    model = UNet(n_channels=3, n_classes=args.classes, bilinear=args.bilinear)
    model = model.to(memory_format=torch.channels_last)

    logging.info(
        f"Network:\n"
        f"\t{model.n_channels} input channels\n"
        f"\t{model.n_classes} output channels (classes)\n"
        f'\t{"Bilinear" if model.bilinear else "Transposed conv"} upscaling'
    )

    if args.load:
        load_checkpoint(model, best_extractor_weights)

    model.to(device=device)
    try:
        train_model(
            model=model,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            device=device,
            img_scale=args.scale,
            val_percent=args.val / 100,
            amp=args.amp,
        )
    except torch.cuda.OutOfMemoryError:
        logging.error(
            "Detected OutOfMemoryError! "
            "Enabling checkpointing to reduce memory usage, but this slows down training. "
            "Consider enabling AMP (--amp) for fast and memory efficient training"
        )
        torch.cuda.empty_cache()
        model.use_checkpointing()
        train_model(
            model=model,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            device=device,
            img_scale=args.scale,
            val_percent=args.val / 100,
            amp=args.amp,
        )

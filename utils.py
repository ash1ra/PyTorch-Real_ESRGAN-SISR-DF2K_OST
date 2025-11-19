import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Literal, overload

import matplotlib.pyplot as plt
import seaborn as sns
import torch
from PIL import Image, ImageDraw, ImageFont
from safetensors.torch import load_file, save_file
from torch import Tensor, nn, optim
from torch.amp import GradScaler
from torch.optim.lr_scheduler import MultiStepLR
from torchvision.io import decode_image
from torchvision.transforms import v2 as transforms

import config


@dataclass
class Metrics:
    epochs: int = field(default=0)
    generator_learning_rates: list[float] = field(default_factory=list)
    discriminator_learning_rates: list[float] = field(default_factory=list)
    generator_train_losses: list[float] = field(default_factory=list)
    discriminator_train_losses: list[float] = field(default_factory=list)
    generator_val_losses: list[float] = field(default_factory=list)
    generator_val_psnrs: list[float] = field(default_factory=list)
    generator_val_ssims: list[float] = field(default_factory=list)


logger = config.create_logger("INFO", __file__)


def create_hr_and_lr_imgs(
    img_path: str | Path,
    scaling_factor: Literal[2, 4, 8],
    crop_size: int | None = None,
    test_mode: bool = False,
) -> tuple[Tensor, Tensor]:
    img_tensor = decode_image(Path(img_path).__fspath__())

    if test_mode:
        _, height, width = img_tensor.shape

        height_remainder = height % scaling_factor
        width_remainder = width % scaling_factor

        top_bound = height_remainder // 2
        left_bound = width_remainder // 2

        bottom_bound = top_bound + (height - height_remainder)
        right_bound = left_bound + (width - width_remainder)

        hr_img_tensor = img_tensor[:, top_bound:bottom_bound, left_bound:right_bound]
    elif crop_size:
        augmentation_transforms = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomChoice(
                    [
                        transforms.RandomRotation(degrees=[0, 0]),
                        transforms.RandomRotation(degrees=[90, 90]),
                        transforms.RandomRotation(degrees=[180, 180]),
                        transforms.RandomRotation(degrees=[270, 270]),
                    ]
                ),
                transforms.RandomCrop(size=(crop_size, crop_size)),
            ]
        )

        hr_img_tensor = augmentation_transforms(img_tensor)

    lr_transforms = transforms.Compose(
        [
            transforms.Resize(
                size=(
                    hr_img_tensor.shape[1] // scaling_factor,
                    hr_img_tensor.shape[2] // scaling_factor,
                ),
                interpolation=transforms.InterpolationMode.BICUBIC,
                antialias=True,
            )
        ]
    )

    normalize_transforms = transforms.Compose(
        [
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )

    lr_img_tensor = lr_transforms(hr_img_tensor)

    hr_img_tensor = normalize_transforms(hr_img_tensor)
    lr_img_tensor = normalize_transforms(lr_img_tensor)

    return hr_img_tensor, lr_img_tensor


@overload
def convert_img(
    img: Tensor,
    source: Literal["[-1, 1]", "[0, 1]", "imagenet", "uint8"],
    target: Literal["pil"],
) -> Image.Image: ...


@overload
def convert_img(
    img: Tensor,
    source: Literal["[-1, 1]", "[0, 1]", "imagenet", "uint8"],
    target: Literal["[-1, 1]", "[0, 1]", "imagenet", "uint8", "y-channel"],
) -> Tensor: ...


def convert_img(
    img: Tensor,
    source: Literal["[-1, 1]", "[0, 1]", "imagenet", "uint8"],
    target: Literal["[-1, 1]", "[0, 1]", "imagenet", "uint8", "pil", "y-channel"],
) -> Tensor | Image.Image:
    if single_tensor := img.dim() == 3:
        img.unsqueeze_(0)

    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]
    ycbcr_weights = [0.299, 0.587, 0.114]

    imagenet_mean_tensor = torch.tensor(imagenet_mean, device=img.device).view(
        1, 3, 1, 1
    )
    imagenet_std_tensor = torch.tensor(imagenet_std, device=img.device).view(1, 3, 1, 1)
    ycbcr_weights_tensor = torch.tensor(ycbcr_weights, device=img.device).view(
        1, 3, 1, 1
    )

    imagenet_norm_transform = transforms.Normalize(mean=imagenet_mean, std=imagenet_std)
    to_pil_img_transform = transforms.ToPILImage()

    match source:
        case "[0, 1]":
            pass
        case "[-1, 1]":
            img = (img + 1.0) / 2.0
        case "imagenet":
            img = img * imagenet_std_tensor + imagenet_mean_tensor
        case "uint8":
            img = img.to(torch.float32) / 255.0
        case _:
            raise ValueError(f"Unknown source format: {source}")

    match target:
        case "[0, 1]":
            pass
        case "[-1, 1]":
            img = img * 2.0 - 1.0
        case "imagenet":
            img = imagenet_norm_transform(img)
        case "uint8":
            img = (img.clamp(0.0, 1.0) * 255.0).to(torch.uint8)
        case "pil":
            img = to_pil_img_transform(img[0])
        case "y-channel":
            img = torch.sum(img * ycbcr_weights_tensor, dim=1, keepdim=True)
        case _:
            raise ValueError(f"Unknown target format: {target}")

    if single_tensor and target != "pil":
        img.squeeze_(0)

    return img


def compare_imgs(
    lr_img_tensor: Tensor,
    sr_img_tensor: Tensor,
    output_path: str | Path,
    hr_img_tensor: Tensor | None = None,
    scaling_factor: Literal[2, 4, 8] = 4,
    orientation: Literal["horizontal", "vertical"] = "vertical",
) -> None:
    bicubic_label = "Bicubic"
    sr_label = "SRGAN"
    hr_label = "Original"

    lr_img = convert_img(lr_img_tensor, "[-1, 1]", "pil")
    sr_img = convert_img(sr_img_tensor, "[-1, 1]", "pil")

    bicubic_img = transforms.Resize(
        size=(sr_img_tensor.shape[2], sr_img_tensor.shape[3]),
        interpolation=transforms.InterpolationMode.BICUBIC,
    )(lr_img)

    width, height = sr_img.size

    if orientation == "horizontal" and isinstance(hr_img_tensor, Tensor):
        hr_img = convert_img(hr_img_tensor, "[-1, 1]", "pil")

        total_width = width * 3 + 50
        total_height = height

        comparison_img = Image.new("RGB", (total_width, total_height), color="white")

        comparison_img.paste(bicubic_img, (0, 50))
        comparison_img.paste(sr_img, (width + 25, 50))
        comparison_img.paste(hr_img, (width * 2 + 50, 50))
    else:
        total_width = width
        total_height = height * 2 + 100

        comparison_img = Image.new("RGB", (total_width, total_height), color="white")

        comparison_img.paste(bicubic_img, (0, 50))
        comparison_img.paste(sr_img, (0, height + 100))

    draw = ImageDraw.Draw((comparison_img))

    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/TTF/JetBrainsMonoNerdFont-Regular.ttf", size=36
        )
    except OSError:
        font = ImageFont.load_default()

    bicubic_text_width = draw.textlength(bicubic_label, font=font)
    sr_text_width = draw.textlength(sr_label, font=font)
    hr_text_width = draw.textlength(hr_label, font=font)

    if orientation == "horizontal":
        draw.text(
            ((width - bicubic_text_width) / 2, 5),
            bicubic_label,
            fill="black",
            font=font,
        )

        draw.text(
            ((width - sr_text_width) / 2 + width + 25, 5),
            sr_label,
            fill="black",
            font=font,
        )

        draw.text(
            ((width - hr_text_width) / 2 + width * 2 + 50, 5),
            hr_label,
            fill="black",
            font=font,
        )
    else:
        draw.text(
            ((width - bicubic_text_width) / 2, 5),
            bicubic_label,
            fill="black",
            font=font,
        )

        draw.text(
            ((width - sr_text_width) / 2, height + 55),
            sr_label,
            fill="black",
            font=font,
        )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    comparison_img.save(output_path, format="PNG")


def _save_optimizer_state(
    optimizer: optim.Optimizer,
    checkpoint_dir_path: str | Path,
    prefix: str,
) -> dict:
    checkpoint_dir_path = Path(checkpoint_dir_path)

    optimizer_state = optimizer.state_dict()
    optimizer_tensors = {}
    optimizer_metadata = {"param_groups": optimizer_state["param_groups"]}

    optimizer_state_buffers = optimizer_state["state"]
    optimizer_metadata["state"] = {}

    for param_id, buffers in optimizer_state_buffers.items():
        param_id_str = str(param_id)
        optimizer_metadata["state"][param_id_str] = {}

        for buffer_name, value in buffers.items():
            if isinstance(value, torch.Tensor):
                tensor_key = f"state_{param_id_str}_{buffer_name}"
                optimizer_tensors[tensor_key] = value
            else:
                optimizer_metadata["state"][param_id_str][buffer_name] = value

    if optimizer_tensors:
        save_file(
            optimizer_tensors, checkpoint_dir_path / f"{prefix}_optimizer.safetensors"
        )

    return optimizer_metadata


def _load_optimizer_state(
    optimizer: optim.Optimizer,
    checkpoint_dir_path: str | Path,
    prefix: str,
    full_metadata: dict,
    device: str,
) -> None:
    checkpoint_dir_path = Path(checkpoint_dir_path)

    optimizer_metadata_key = f"{prefix}_optimizer_metadata"
    if optimizer_metadata_key not in full_metadata:
        logger.warning(
            f"Metadata for '{optimizer_metadata_key}' not found in training_state.json"
        )
        return

    optimizer_metadata = full_metadata[optimizer_metadata_key]

    optimizer_tensors_path = checkpoint_dir_path / f"{prefix}_optimizer.safetensors"
    if optimizer_tensors_path.exists():
        optimizer_tensors = load_file(filename=optimizer_tensors_path, device=device)
    else:
        optimizer_tensors = {}
        logger.warning(f"Optimizer tensor file not found: {optimizer_tensors_path}")

    optimizer_state_buffers = {
        int(param_id): buffers
        for param_id, buffers in optimizer_metadata["state"].items()
    }

    for tensor_key, tensor_value in optimizer_tensors.items():
        parts = tensor_key.split("_")
        if len(parts) < 3 or parts[0] != "state":
            logger.warning(
                f"Unrecognized tensor key in {prefix}_optimizer: {tensor_key}"
            )
            continue

        param_id = int(parts[1])
        buffer_name = "_".join(parts[2:])

        if param_id not in optimizer_state_buffers:
            optimizer_state_buffers[param_id] = {}

        optimizer_state_buffers[param_id][buffer_name] = tensor_value

    optimizer_state_to_load = {
        "param_groups": optimizer_metadata["param_groups"],
        "state": optimizer_state_buffers,
    }

    try:
        optimizer.load_state_dict(optimizer_state_to_load)
    except Exception as e:
        logger.error(f"Failed to load state_dict for {prefix}_optimizer: {e}")
        logger.warning(f"Continuing without loading {prefix}_optimizer state.")


def save_checkpoint(
    checkpoint_dir_path: str | Path,
    epoch: int,
    generator: nn.Module,
    generator_optimizer: optim.Optimizer,
    metrics: Metrics,
    generator_scaler: GradScaler | None = None,
    generator_scheduler: MultiStepLR | None = None,
    discriminator: nn.Module | None = None,
    discriminator_optimizer: optim.Optimizer | None = None,
    discriminator_scaler: GradScaler | None = None,
    discriminator_scheduler: MultiStepLR | None = None,
) -> None:
    checkpoint_dir_path = Path(checkpoint_dir_path)
    checkpoint_dir_path.mkdir(parents=True, exist_ok=True)

    save_file(generator.state_dict(), checkpoint_dir_path / "generator.safetensors")

    generator_optimizer_metadata = _save_optimizer_state(
        generator_optimizer,
        checkpoint_dir_path,
        "generator",
    )

    if discriminator and discriminator_optimizer:
        save_file(
            discriminator.state_dict(),
            checkpoint_dir_path / "discriminator.safetensors",
        )
        discriminator_optimizer_metadata = _save_optimizer_state(
            discriminator_optimizer,
            checkpoint_dir_path,
            "discriminator",
        )

    full_metadata = {
        "epoch": epoch,
        "metrics": asdict(metrics),
        "generator_optimizer_metadata": generator_optimizer_metadata,
        "discriminator_optimizer_metadata": discriminator_optimizer_metadata
        if discriminator
        else None,
        "generator_scaler_state_dict": generator_scaler.state_dict()
        if generator_scaler
        else None,
        "discriminator_scaler_state_dict": discriminator_scaler.state_dict()
        if discriminator_scaler
        else None,
        "generator_scheduler_state_dict": generator_scheduler.state_dict()
        if generator_scheduler
        else None,
        "discriminator_scheduler_state_dict": discriminator_scheduler.state_dict()
        if discriminator_scheduler
        else None,
    }

    with open(checkpoint_dir_path / "training_state.json", "w") as f:
        json.dump(full_metadata, f, indent=4)

    logger.debug(f'Checkpoint was saved to "{checkpoint_dir_path}" after {epoch} epoch')


def load_checkpoint(
    checkpoint_dir_path: str | Path,
    generator: nn.Module,
    test_mode: bool = False,
    metrics: Metrics | None = None,
    generator_optimizer: optim.Optimizer | None = None,
    generator_scaler: GradScaler | None = None,
    generator_scheduler: MultiStepLR | None = None,
    discriminator: nn.Module | None = None,
    discriminator_optimizer: optim.Optimizer | None = None,
    discriminator_scaler: GradScaler | None = None,
    discriminator_scheduler: MultiStepLR | None = None,
    device: Literal["cuda", "cpu"] = "cpu",
) -> int:
    checkpoint_dir_path = Path(checkpoint_dir_path)

    generator_path = checkpoint_dir_path / "generator.safetensors"
    discriminator_path = checkpoint_dir_path / "discriminator.safetensors"
    state_path = checkpoint_dir_path / "training_state.json"

    if generator_path.exists():
        generator.load_state_dict(load_file(filename=generator_path, device=device))
    else:
        logger.warning(
            f"Checkpoint (generator.safetensors or training_state.json) was not found at '{checkpoint_dir_path}', starting from 1 epoch"
        )
        return 1

    if test_mode:
        logger.info(
            f"Loaded Generator weights from '{checkpoint_dir_path}' (test_mode=True)"
        )
        return 1

    if state_path.exists():
        with open(state_path, "r") as f:
            full_metadata = json.load(f)

        if metrics and "metrics" in full_metadata:
            metrics_dict = full_metadata["metrics"]
            metrics.epochs = metrics_dict["epochs"]
            metrics.generator_learning_rates = metrics_dict["generator_learning_rates"]
            metrics.discriminator_learning_rates = metrics_dict[
                "discriminator_learning_rates"
            ]
            metrics.generator_train_losses = metrics_dict["generator_train_losses"]
            metrics.discriminator_train_losses = metrics_dict[
                "discriminator_train_losses"
            ]
            metrics.generator_val_losses = metrics_dict["generator_val_losses"]
            metrics.generator_val_psnrs = metrics_dict["generator_val_psnrs"]
            metrics.generator_val_ssims = metrics_dict["generator_val_ssims"]

        if generator_optimizer:
            _load_optimizer_state(
                generator_optimizer,
                checkpoint_dir_path,
                "generator",
                full_metadata,
                device,
            )

        if generator_scaler and full_metadata.get("generator_scaler_state_dict"):
            generator_scaler.load_state_dict(
                full_metadata["generator_scaler_state_dict"]
            )

        if generator_scheduler and full_metadata.get("generator_scheduler_state_dict"):
            generator_scheduler.load_state_dict(
                full_metadata["generator_scheduler_state_dict"]
            )

        if discriminator and discriminator_optimizer:
            discriminator.load_state_dict(
                load_file(filename=discriminator_path, device=device)
            )

            _load_optimizer_state(
                discriminator_optimizer,
                checkpoint_dir_path,
                "discriminator",
                full_metadata,
                device,
            )

            if discriminator_scaler and full_metadata.get(
                "discriminator_scaler_state_dict"
            ):
                discriminator_scaler.load_state_dict(
                    full_metadata["discriminator_scaler_state_dict"]
                )

            if discriminator_scheduler and full_metadata.get(
                "discriminator_scheduler_state_dict"
            ):
                discriminator_scheduler.load_state_dict(
                    full_metadata["discriminator_scheduler_state_dict"]
                )

        logger.info(f'Checkpoint was loaded from "{checkpoint_dir_path}"')

        return full_metadata["epoch"]
    else:
        logger.error("State path does not exists, can not load model parameters")
        return 0


def format_time(total_seconds: float) -> str:
    if total_seconds < 0:
        total_seconds = 0

    hours = int(total_seconds // 3600)
    minutes = int((total_seconds % 3600) // 60)
    seconds = int(total_seconds % 60)

    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def create_hyperparameters_str() -> str:
    return f"Scaling factor: {config.SCALING_FACTOR} | Crop size: {config.CROP_SIZE} | Batch size: {config.TRAIN_BATCH_SIZE} | Generator learning rate: {config.GENERATOR_LEARNING_RATE} | Discriminator learning rate: {config.DISCRIMINATOR_LEARNING_RATE}| Epochs: {config.EPOCHS} | Number of workers: {config.NUM_WORKERS} | Dev mode: {config.DEV_MODE}"


def plot_training_metrics(
    metrics: Metrics,
    hyperparameters_str: str,
    model_type: Literal["psnr", "esrgan"],
) -> None:
    sns.set_style("whitegrid")
    sns.set_palette("deep")
    palette = sns.color_palette("deep")

    epochs = list(range(0, len(metrics.generator_train_losses)))

    fig, axs = plt.subplots(2, 2, figsize=(14, 10))

    if model_type == "psnr":
        fig.suptitle("PSNR-based Generator Training Metrics", fontsize=18)
    else:
        fig.suptitle("ESRGAN Training Metrics", fontsize=18)

    fig.text(0.5, 0.94, hyperparameters_str, ha="center", va="top", fontsize=10)

    sns.lineplot(
        x=epochs,
        y=metrics.generator_train_losses,
        label="Generator train loss",
        ax=axs[0, 0],
        linewidth=2.5,
        color=palette[0],
    )

    if model_type == "esrgan":
        sns.lineplot(
            x=epochs,
            y=metrics.generator_val_losses,
            label="Generator val loss",
            ax=axs[0, 0],
            linewidth=2.5,
            color=palette[1],
        )
        axs[0, 0].set_title("Generator training and validation losses")
        axs[0, 0].set_xlabel("Epoch")
        axs[0, 0].set_ylabel("Loss")

        sns.lineplot(
            x=epochs,
            y=metrics.discriminator_train_losses,
            ax=axs[0, 1],
            linewidth=2.5,
            color=palette[2],
        )
        axs[0, 1].set_title("Discriminator training loss")
        axs[0, 1].set_xlabel("Epoch")
        axs[0, 1].set_ylabel("Loss")

    elif model_type == "psnr":
        axs[0, 0].set_title("Generator training loss")
        axs[0, 0].set_xlabel("Epoch")
        axs[0, 0].set_ylabel("Loss")

        sns.lineplot(
            x=epochs,
            y=metrics.generator_val_losses,
            label="Generator val loss",
            ax=axs[0, 1],
            linewidth=2.5,
            color=palette[1],
        )

        axs[0, 1].set_title("Generator val loss")
        axs[0, 1].set_xlabel("Epoch")
        axs[0, 1].set_ylabel("Loss")

    sns.lineplot(
        x=epochs,
        y=metrics.generator_val_psnrs,
        ax=axs[1, 0],
        linewidth=2.5,
        color=palette[1],
    )
    axs[1, 0].set_title("Validation Peak Signal-to-Noise Ratio")
    axs[1, 0].set_xlabel("Epoch")
    axs[1, 0].set_ylabel("PSNR")

    sns.lineplot(
        x=epochs,
        y=metrics.generator_val_ssims,
        ax=axs[1, 1],
        linewidth=2.5,
        color=palette[1],
    )
    axs[1, 1].set_title("Validation Structural Similarity Index Measure")
    axs[1, 1].set_xlabel("Epoch")
    axs[1, 1].set_ylabel("SSIM")

    plt.tight_layout(rect=[0, 0.03, 1, 0.94])

    output_path = (
        Path("images")
        / f"training_metrics_{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}.png"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")

    plt.show()

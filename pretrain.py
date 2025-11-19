from time import time
from typing import Literal

import torch
from torch import nn, optim
from torch.amp import GradScaler, autocast
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure

import config
from data_processing import SRDataset
from models import Generator
from utils import (
    Metrics,
    convert_img,
    create_hyperparameters_str,
    format_time,
    load_checkpoint,
    plot_training_metrics,
    save_checkpoint,
)

logger = config.create_logger("INFO", __file__)


def train_step(
    data_loader: DataLoader,
    generator: nn.Module,
    content_loss_fn: nn.Module,
    generator_optimizer: optim.Optimizer,
    generator_scaler: GradScaler | None = None,
    generator_scheduler: MultiStepLR | None = None,
    device: Literal["cuda", "cpu"] = "cpu",
) -> float:
    total_generator_loss = 0.0

    generator.train()

    for i, (hr_img_tensor, lr_img_tensor) in enumerate(data_loader):
        hr_img_tensor = hr_img_tensor.to(device, non_blocking=True)
        lr_img_tensor = lr_img_tensor.to(device, non_blocking=True)

        with autocast(device, enabled=(generator_scaler is not None)):
            sr_img_tensor = generator(lr_img_tensor)

            content_loss = content_loss_fn(sr_img_tensor, hr_img_tensor)

        total_generator_loss += content_loss.item()

        generator_optimizer.zero_grad()

        if generator_scaler:
            generator_scaler.scale(content_loss).backward()
            generator_scaler.unscale_(generator_optimizer)
            clip_grad_norm_(generator.parameters(), max_norm=1.0)
            generator_scaler.step(generator_optimizer)
            generator_scaler.update()
        else:
            content_loss.backward()
            clip_grad_norm_(generator.parameters(), max_norm=1.0)
            generator_optimizer.step()

        if i % config.PRINT_FREQUENCY == 0:
            logger.debug(f"Processing batch {i}/{len(data_loader)}...")

    total_generator_loss /= len(data_loader)

    return total_generator_loss


def validation_step(
    data_loader: DataLoader,
    generator: nn.Module,
    content_loss_fn: nn.Module,
    psnr_metric: PeakSignalNoiseRatio,
    ssim_metric: StructuralSimilarityIndexMeasure,
    device: Literal["cpu", "cuda"] = "cpu",
) -> tuple[float, float, float]:
    total_content_loss = 0.0

    generator.eval()

    with torch.inference_mode():
        for hr_img_tensor, lr_img_tensor in data_loader:
            hr_img_tensor = hr_img_tensor.to(device, non_blocking=True)
            lr_img_tensor = lr_img_tensor.to(device, non_blocking=True)

            sr_img_tensor = generator(lr_img_tensor)

            content_loss = content_loss_fn(sr_img_tensor, hr_img_tensor)

            y_hr_tensor = convert_img(hr_img_tensor, "[-1, 1]", "y-channel")
            y_sr_tensor = convert_img(sr_img_tensor, "[-1, 1]", "y-channel")

            sf = config.SCALING_FACTOR
            y_hr_tensor = y_hr_tensor[:, :, sf:-sf, sf:-sf]
            y_sr_tensor = y_sr_tensor[:, :, sf:-sf, sf:-sf]

            psnr_metric.update(y_sr_tensor, y_hr_tensor)  # type: ignore
            ssim_metric.update(y_sr_tensor, y_hr_tensor)  # type: ignore

            total_content_loss += content_loss.item()

        total_content_loss /= len(data_loader)
        total_psnr = psnr_metric.compute().item()  # type: ignore
        total_ssim = ssim_metric.compute().item()  # type: ignore

        psnr_metric.reset()
        ssim_metric.reset()

    return total_content_loss, total_psnr, total_ssim


def train(
    train_data_loader: DataLoader,
    val_data_loader: DataLoader,
    generator: nn.Module,
    content_loss_fn: nn.Module,
    generator_optimizer: optim.Optimizer,
    start_epoch: int,
    epochs: int,
    metrics: Metrics,
    psnr_metric: PeakSignalNoiseRatio,
    ssim_metric: StructuralSimilarityIndexMeasure,
    generator_scaler: GradScaler | None = None,
    generator_scheduler: MultiStepLR | None = None,
    device: Literal["cuda", "cpu"] = "cpu",
) -> None:
    if not metrics.epochs:
        metrics.epochs = epochs - start_epoch + 1

    if start_epoch > 1 and metrics.generator_val_psnrs:
        best_val_psnr = max(metrics.generator_val_psnrs)
    else:
        best_val_psnr = 0.0

    dashes_count = 54

    logger.info("-" * dashes_count)
    logger.info("Model parameters:")
    logger.info(f"Scaling factor: {config.SCALING_FACTOR}")
    logger.info(f"Crop size: {config.CROP_SIZE}")
    logger.info(f"Batch size: {config.TRAIN_BATCH_SIZE}")
    logger.info(f"Learning rate: {config.GENERATOR_LEARNING_RATE}")
    logger.info(f"Epochs: {config.EPOCHS}")
    logger.info(f"Number of workers: {config.NUM_WORKERS}")
    logger.info("-" * dashes_count)
    logger.info("PSNR-based Generator:")
    logger.info(f"Count of channels: {config.GENERATOR_CHANNELS_COUNT}")
    logger.info(f"Count of growths channels: {config.GENERATOR_GROWTHS_CHANNELS_COUNT}")
    logger.info(
        f"Count of residual dense blocks: {config.GENERATOR_RES_DENSE_BLOCKS_COUNT}"
    )
    logger.info(
        f"Count of residual in residual dense blocks: {config.GENERATOR_RRDB_COUNT}"
    )
    logger.info(f"Large kernel size: {config.GENERATOR_LARGE_KERNEL_SIZE}")
    logger.info(f"Small kernel size: {config.GENERATOR_SMALL_KERNEL_SIZE}")
    logger.info("-" * dashes_count)
    logger.info("Starting model training...")

    try:
        training_start_time = time()
        for epoch in range(start_epoch, epochs + 1):
            epoch_start_time = time()

            generator_train_loss = train_step(
                data_loader=train_data_loader,
                generator=generator,
                content_loss_fn=content_loss_fn,
                generator_optimizer=generator_optimizer,
                generator_scaler=generator_scaler,
                generator_scheduler=generator_scheduler,
                device=device,
            )

            generator_val_loss, generator_val_psnr, generator_val_ssim = (
                validation_step(
                    data_loader=val_data_loader,
                    generator=generator,
                    content_loss_fn=content_loss_fn,
                    psnr_metric=psnr_metric,
                    ssim_metric=ssim_metric,
                    device=device,
                )
            )

            if generator_scheduler:
                generator_scheduler.step()

            end_time = time() - epoch_start_time
            epoch_time = format_time(end_time)
            elapsed_time = format_time(time() - training_start_time)
            remaining_time = format_time(end_time * (epochs - epoch))

            generator_optimizer_lr = generator_optimizer.param_groups[0]["lr"]

            metrics.generator_learning_rates.append(generator_optimizer_lr)
            metrics.generator_train_losses.append(generator_train_loss)
            metrics.generator_val_losses.append(generator_val_loss)
            metrics.generator_val_psnrs.append(generator_val_psnr)
            metrics.generator_val_ssims.append(generator_val_ssim)

            logger.info(
                f"Epoch: {epoch}/{epochs} ({epoch_time} | {elapsed_time}/{remaining_time}) | Generator LR: {generator_optimizer_lr}"
            )
            logger.info(
                f"Generator Train Loss: {generator_train_loss:.4f} | Generator Val Loss: {generator_val_loss:.4f} | Generator Val PSNR: {generator_val_psnr:.4f} | Generator Val SSIM: {generator_val_ssim:.4f}"
            )

            if generator_val_psnr > best_val_psnr:
                best_val_psnr = generator_val_psnr
                logger.debug(
                    f"New best model found with val loss: {best_val_psnr:.4f} at epoch {epoch}"
                )
                save_checkpoint(
                    checkpoint_dir_path=config.BEST_PSNR_CHECKPOINT_DIR_PATH,
                    epoch=epoch,
                    generator=generator,
                    generator_optimizer=generator_optimizer,
                    metrics=metrics,
                    generator_scaler=generator_scaler,
                    generator_scheduler=generator_scheduler,
                )

            save_checkpoint(
                checkpoint_dir_path=config.PSNR_CHECKPOINT_DIR_PATH,
                epoch=epoch,
                generator=generator,
                generator_optimizer=generator_optimizer,
                metrics=metrics,
                generator_scaler=generator_scaler,
                generator_scheduler=generator_scheduler,
            )

        plot_training_metrics(metrics, create_hyperparameters_str(), model_type="psnr")

    except KeyboardInterrupt:
        logger.info("Saving model's weights and finish training...")
        save_checkpoint(
            checkpoint_dir_path=config.PSNR_CHECKPOINT_DIR_PATH,
            epoch=epoch,
            generator=generator,
            generator_optimizer=generator_optimizer,
            metrics=metrics,
            generator_scaler=generator_scaler,
            generator_scheduler=generator_scheduler,
        )

        plot_training_metrics(metrics, create_hyperparameters_str(), model_type="psnr")


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_dataset = SRDataset(
        data_path=config.TRAIN_DATASET_PATH,
        scaling_factor=config.SCALING_FACTOR,
        crop_size=config.CROP_SIZE,
        dev_mode=config.DEV_MODE,
    )

    val_dataset = SRDataset(
        data_path=config.VAL_DATASET_PATH,
        scaling_factor=config.SCALING_FACTOR,
        crop_size=config.CROP_SIZE,
        test_mode=True,
        dev_mode=config.DEV_MODE,
    )

    train_data_loader = DataLoader(
        dataset=train_dataset,
        batch_size=config.REAL_TRAIN_BATCH_SIZE,
        shuffle=True,
        pin_memory=True if device == "cuda" else False,
        num_workers=config.NUM_WORKERS,
    )

    val_data_loader = DataLoader(
        dataset=val_dataset,
        batch_size=config.TEST_BATCH_SIZE,
        shuffle=False,
        pin_memory=True if device == "cuda" else False,
        num_workers=config.NUM_WORKERS,
    )

    generator = Generator(
        channels_count=config.GENERATOR_CHANNELS_COUNT,
        growth_channels_count=config.GENERATOR_GROWTHS_CHANNELS_COUNT,
        large_kernel_size=config.GENERATOR_LARGE_KERNEL_SIZE,
        small_kernel_size=config.GENERATOR_SMALL_KERNEL_SIZE,
        res_dense_blocks_count=config.GENERATOR_RES_DENSE_BLOCKS_COUNT,
        rrdb_count=config.GENERATOR_RRDB_COUNT,
        scaling_factor=config.SCALING_FACTOR,
    ).to(device)

    content_loss_fn = nn.L1Loss()

    metrics = Metrics()
    psnr_metric = PeakSignalNoiseRatio(data_range=1.0).to(device)
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)

    generator_optimizer = optim.Adam(
        generator.parameters(), lr=config.GENERATOR_LEARNING_RATE
    )
    generator_scaler = GradScaler(device) if device == "cuda" else None
    generator_scheduler = MultiStepLR(
        optimizer=generator_optimizer,
        milestones=config.SCHEDULER_MILESTONES,
        gamma=config.SCHEDULER_SCALING_VALUE,
    )

    start_epoch = 1
    if config.LOAD_PSNR_CHECKPOINT:
        if (
            config.BEST_PSNR_CHECKPOINT_DIR_PATH.exists()
            or config.PSNR_CHECKPOINT_DIR_PATH.exists()
        ):
            if (
                config.LOAD_BEST_PSNR_CHECKPOINT
                and config.BEST_PSNR_CHECKPOINT_DIR_PATH.exists()
            ):
                checkpoint_dir_path_to_load = config.BEST_PSNR_CHECKPOINT_DIR_PATH
                logger.info(
                    f'Loading best checkpoint from "{checkpoint_dir_path_to_load}"...'
                )
            elif config.PSNR_CHECKPOINT_DIR_PATH.exists():
                checkpoint_dir_path_to_load = config.PSNR_CHECKPOINT_DIR_PATH
                logger.info(
                    f'Loading checkpoint from "{checkpoint_dir_path_to_load}"...'
                )

            start_epoch = load_checkpoint(
                checkpoint_dir_path=checkpoint_dir_path_to_load,
                generator=generator,
                generator_optimizer=generator_optimizer,
                metrics=metrics,
                generator_scaler=generator_scaler,
                generator_scheduler=generator_scheduler,
                device=device,
            )

            if generator_scheduler and start_epoch > 1:
                epochs_to_skip = start_epoch - 1

                for _ in range(epochs_to_skip):
                    generator_scheduler.step()

                logger.info(f"Schedulers advanced to epoch {start_epoch}")
        else:
            logger.warning(
                "No checkpoints were found, start training from the beginning..."
            )

    train(
        train_data_loader=train_data_loader,
        val_data_loader=val_data_loader,
        generator=generator,
        content_loss_fn=content_loss_fn,
        generator_optimizer=generator_optimizer,
        start_epoch=start_epoch,
        epochs=config.EPOCHS,
        metrics=metrics,
        psnr_metric=psnr_metric,
        ssim_metric=ssim_metric,
        generator_scaler=generator_scaler,
        generator_scheduler=generator_scheduler,
        device=device,
    )


if __name__ == "__main__":
    main()

from pathlib import Path
from typing import Literal

import torch
import torch.nn.functional as F
from safetensors.torch import load_file
from torch import Tensor, nn
from torchvision.io import decode_image, write_png
from torchvision.transforms import v2 as transforms

import config
from models import Generator
from utils import compare_imgs, convert_img, load_checkpoint

logger = config.create_logger("INFO", __file__)


def apply_network_interpolation(
    model: nn.Module,
    params_psnr: dict,
    params_esrgan: dict,
    alpha: float,
) -> None:
    logger.info("Creating network interpolated model...")

    params_interpolated = dict()

    for key in params_psnr.keys():
        if key in params_esrgan:
            tensor_psnr = params_psnr[key].float()
            tensor_esrgan = params_esrgan[key].float()

            params_interpolated[key] = (1 - alpha) * tensor_psnr + alpha * tensor_esrgan
        else:
            logger.warning(
                f'Key "{key}" not found in ESRGAN model. Copying from PSNR-oriented model...'
            )
            params_interpolated[key] = params_psnr[key].float()

    model.load_state_dict(params_interpolated)


def upscale_img_tiled(
    model: nn.Module,
    lr_img_tensor: Tensor,
    scale_factor: Literal[2, 4, 8] = 4,
    tile_size: int = 512,
    tile_overlap: int = 64,
    device: Literal["cuda", "cpu"] = "cpu",
) -> Tensor:
    batch_size, channels, height_original, width_original = lr_img_tensor.shape

    height_target = height_original * scale_factor
    width_target = width_original * scale_factor

    border_pad = tile_overlap // 2

    lr_img_tensor_padded = F.pad(
        lr_img_tensor, (border_pad, border_pad, border_pad, border_pad), "reflect"
    )

    _, _, height_padded, width_padded = lr_img_tensor_padded.shape

    step_size = tile_size - tile_overlap

    pad_right = (step_size - (width_padded - tile_size) % step_size) % step_size
    pad_bottom = (step_size - (height_padded - tile_size) % step_size) % step_size

    lr_img_tensor_padded = F.pad(
        lr_img_tensor_padded, (0, pad_right, 0, pad_bottom), "reflect"
    )

    _, _, height_final, width_final = lr_img_tensor_padded.shape

    logger.info(
        f"Original LR: {width_original}x{height_original} | Target SR: {width_target}x{height_target}"
    )

    final_img_canvas = torch.zeros(
        (batch_size, channels, height_final * scale_factor, width_final * scale_factor),
        dtype=lr_img_tensor.dtype,
        device="cpu",
    )

    count_canvas = torch.zeros_like(final_img_canvas, device="cpu")

    for height in range(0, height_final - tile_size + 1, step_size):
        for width in range(0, width_final - tile_size + 1, step_size):
            lr_img_tensor_tile = lr_img_tensor_padded[
                :, :, height : height + tile_size, width : width + tile_size
            ]

            with torch.inference_mode():
                sr_img_tensor_tile = model(lr_img_tensor_tile).cpu()

            final_height_start = height * scale_factor
            final_width_start = width * scale_factor
            final_height_end = (height + tile_size) * scale_factor
            final_width_end = (width + tile_size) * scale_factor

            final_img_canvas[
                :,
                :,
                final_height_start:final_height_end,
                final_width_start:final_width_end,
            ] += sr_img_tensor_tile

            count_canvas[
                :,
                :,
                final_height_start:final_height_end,
                final_width_start:final_width_end,
            ] += 1

    logger.info("All tiles processed. Blending results...")

    output_padded = final_img_canvas / count_canvas

    final_border_pad = border_pad * scale_factor

    final_output = output_padded[
        :,
        :,
        final_border_pad : final_border_pad + height_target,
        final_border_pad : final_border_pad + width_target,
    ]

    return final_output


def inference(
    model: nn.Module,
    input_path: Path,
    output_path: Path,
    scaling_factor: Literal[2, 4, 8] = 4,
    use_downscale: bool = False,
    use_tiling: bool = True,
    create_comparisson: bool = False,
    comparisson_path: Path | None = None,
    orientation: Literal["horizontal", "vertical"] = "vertical",
    use_network_interpolation: bool = False,
    alpha: float = 0.8,
    device: Literal["cpu", "cuda"] = "cpu",
) -> None:
    if not input_path.exists():
        raise FileNotFoundError(f"Input image not found: {input_path}")

    if input_path.suffix.lower() not in (".jpg", ".jpeg", ".png"):
        raise ValueError("Input image must be in JPG or PNG format")

    lr_img_tensor_uint8 = decode_image(str(input_path))

    original_lr_img_tensor_uint8 = lr_img_tensor_uint8

    if use_downscale:
        logger.info(f"Downscaling image by {scaling_factor} times...")

        _, lr_img_height, lr_img_width = lr_img_tensor_uint8.shape

        height_remainder = lr_img_height % scaling_factor
        width_remainder = lr_img_width % scaling_factor

        if height_remainder != 0 or width_remainder != 0:
            pad_top = height_remainder // 2
            pad_left = width_remainder // 2
            pad_bottom = pad_top + (lr_img_height - height_remainder)
            pad_right = pad_left + (lr_img_width - width_remainder)

            lr_img_tensor_uint8 = lr_img_tensor_uint8[
                :, pad_top:pad_bottom, pad_left:pad_right
            ]

        _, lr_img_height, lr_img_width = lr_img_tensor_uint8.shape

        lr_img_height_final = lr_img_height // scaling_factor
        lr_img_width_final = lr_img_width // scaling_factor

        lr_img_tensor_uint8 = transforms.Resize(
            size=(lr_img_height_final, lr_img_width_final),
            interpolation=transforms.InterpolationMode.BICUBIC,
            antialias=True,
        )(lr_img_tensor_uint8)

    lr_img_tensor = (
        convert_img(lr_img_tensor_uint8, source="uint8", target="[-1, 1]")
        .unsqueeze(0)
        .to(device)
    )

    if use_network_interpolation:
        if (
            config.BEST_REAL_ESRNET_CHECKPOINT_DIR_PATH / "generator.safetensors"
        ).exists() and (
            config.BEST_REAL_ESRGAN_CHECKPOINT_DIR_PATH / "generator.safetensors"
        ).exists():
            params_psnr = load_file(
                filename=config.BEST_REAL_ESRNET_CHECKPOINT_DIR_PATH
                / "generator.safetensors",
                device=device,
            )
            params_esrgan = load_file(
                filename=config.BEST_REAL_ESRGAN_CHECKPOINT_DIR_PATH
                / "generator.safetensors",
                device=device,
            )

            apply_network_interpolation(
                model=model,
                params_psnr=params_psnr,
                params_esrgan=params_esrgan,
                alpha=alpha,
            )
        else:
            logger.error(
                "PSNR-oriented Generator weights or ESRGAN Generator weights not found"
            )
            raise FileNotFoundError

    if use_tiling:
        logger.info(
            f"Starting tiled inference with tile size: {config.TILE_SIZE} and tile overlap: {config.TILE_OVERLAP}..."
        )

        sr_img_tensor = upscale_img_tiled(
            model=model,
            lr_img_tensor=lr_img_tensor,
            scale_factor=scaling_factor,
            tile_size=config.TILE_SIZE,
            tile_overlap=config.TILE_OVERLAP,
            device=device,
        )
    else:
        with torch.inference_mode():
            sr_img_tensor = model(lr_img_tensor).cpu()

    if create_comparisson and comparisson_path:
        if comparisson_path.parent.exists():
            logger.info("Creating comparison image...")

            original_lr_img_tensor = convert_img(
                original_lr_img_tensor_uint8, source="uint8", target="[-1, 1]"
            )

            compare_imgs(
                lr_img_tensor=lr_img_tensor,
                sr_img_tensor=sr_img_tensor,
                hr_img_tensor=original_lr_img_tensor
                if orientation == "horizontal"
                else None,
                output_path=comparisson_path,
                scaling_factor=scaling_factor,
                orientation=orientation,
            )
        else:
            logger.error("Comparison image path not found")
            raise FileNotFoundError

    sr_img_tensor_uint8 = convert_img(sr_img_tensor, source="[-1, 1]", target="uint8")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_png(sr_img_tensor_uint8.squeeze(0), str(output_path))

    logger.info(f"Upscaled image was saved to {output_path}")


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"

    generator = Generator(
        channels_count=config.GENERATOR_CHANNELS_COUNT,
        growth_channels_count=config.GENERATOR_GROWTHS_CHANNELS_COUNT,
        large_kernel_size=config.GENERATOR_LARGE_KERNEL_SIZE,
        small_kernel_size=config.GENERATOR_SMALL_KERNEL_SIZE,
        res_dense_blocks_count=config.GENERATOR_RES_DENSE_BLOCKS_COUNT,
        rrdb_count=config.GENERATOR_RRDB_COUNT,
        scaling_factor=config.SCALING_FACTOR,
    ).to(device)

    _ = load_checkpoint(
        checkpoint_dir_path=config.BEST_REAL_ESRGAN_CHECKPOINT_DIR_PATH,
        generator=generator,
        test_mode=True,
    )

    logger.info(
        f"Model {config.BEST_REAL_ESRGAN_CHECKPOINT_DIR_PATH} loaded on {device}"
    )

    generator.eval()

    inference(
        model=generator,
        input_path=config.INFERENCE_INPUT_IMG_PATH,
        output_path=config.INFERENCE_OUTPUT_IMG_PATH,
        scaling_factor=config.SCALING_FACTOR,
        use_downscale=True,
        use_tiling=False,
        create_comparisson=True,
        comparisson_path=config.INFERECE_COMPARISON_IMG_PATH,
        orientation="vertical",
        use_network_interpolation=True,
        alpha=0.8,
        device=device,
    )


if __name__ == "__main__":
    main()

from pathlib import Path
from typing import Literal

from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset
from utils import create_hr_and_lr_imgs

from config import create_logger

logger = create_logger("INFO", __file__)


class SRDataset(Dataset):
    def __init__(
        self,
        data_path: str | Path,
        scaling_factor: Literal[2, 4, 8],
        crop_size: int | None = None,
        test_mode: bool = False,
        dev_mode: bool = False,
    ) -> None:
        self.scaling_factor = scaling_factor
        self.crop_size = crop_size
        self.test_mode = test_mode
        self.imgs = []

        data_path = Path(data_path)

        if not data_path.exists():
            logger.error(f'Path does not exists: "{data_path}"')
            raise FileNotFoundError

        if data_path.is_dir():
            logger.info(f'Creating dataset from directory ("{data_path}")...')
            img_paths = list(data_path.iterdir())
        elif data_path.is_file():
            logger.info(f'Creating dataset from file ("{data_path}")...')
            try:
                with open(data_path, "r") as f:
                    img_paths = [Path(line.strip()) for line in f.readlines() if line]
            except FileNotFoundError:
                logger.error(f'File with imgs list was not found ("{data_path}")')

        try:
            for img_path in img_paths:
                if img_path.suffix.lower() in (".jpg", ".jpeg", ".png"):
                    with Image.open(img_path) as img:
                        if img.mode == "RGB":
                            if test_mode:
                                self.imgs.append(img_path)
                            else:
                                width, height = img.size
                                if width >= self.crop_size and height >= self.crop_size:
                                    self.imgs.append(img_path)
        except FileNotFoundError:
            logger.error(f'Image at path "{img_path}" was not found, skipping...')

        if dev_mode:
            self.imgs = self.imgs[: int(len(self.imgs) * 0.1)]

    def __len__(self) -> int:
        return len(self.imgs)

    def __getitem__(self, i: int) -> tuple[Tensor, Tensor]:
        return create_hr_and_lr_imgs(
            img_path=self.imgs[i],
            scaling_factor=self.scaling_factor,
            crop_size=self.crop_size,
            test_mode=self.test_mode,
        )

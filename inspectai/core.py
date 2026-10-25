"""Normal-patch memory and nearest-neighbour scoring (PatchCore-inspired)."""
from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
import time

import numpy as np
from PIL import Image, ImageOps
from sklearn.neighbors import NearestNeighbors


MODEL_NAME = 'ResNet-18 / ImageNet1K-V1 / layer2+layer3'
MAX_BYTES = 8 * 1024 * 1024
MAX_PIXELS = 12_000_000


@dataclass(frozen=True)
class Config:
    size: int = 224
    grid: int = 14
    bank_size: int = 768
    seed: int = 17
    normal_quantile: float = 0.95


def open_image(data: bytes) -> Image.Image:
    """Reject oversized or malformed uploads before decoding to RGB."""
    if not data or len(data) > MAX_BYTES:
        raise ValueError('Image must be between 1 byte and 8 MB.')
    try:
        with Image.open(BytesIO(data)) as im:
            if im.width * im.height > MAX_PIXELS or min(im.size) < 48:
                raise ValueError('Image resolution must be at least 48 px per side and at most 12 MP.')
            return ImageOps.exif_transpose(im).convert('RGB')
    except (OSError, Image.DecompressionBombError) as exc:
        raise ValueError('Could not read this image. Use a PNG or JPEG.') from exc


class Extractor:
    """Frozen pretrained CNN. The first use may download official weights."""

    def __init__(self, config: Config = Config()):
        import torch
        from torchvision.models import ResNet18_Weights, resnet18

        self.torch = torch
        self.functional = torch.nn.functional
        self.config = config
        weights = ResNet18_Weights.IMAGENET1K_V1
        self.model = resnet18(weights=weights).eval().cpu()
        # Keep the model view aligned with the displayed heatmap: no extra zoom/crop.
        self.preprocess = weights.transforms(resize_size=config.size, crop_size=config.size)

    def patches(self, image: Image.Image) -> np.ndarray:
        torch, F = self.torch, self.functional
        image = ImageOps.fit(image.convert('RGB'), (self.config.size, self.config.size))
        x = self.preprocess(image).unsqueeze(0)
        with torch.inference_mode():
            m = self.model
            x = m.maxpool(m.relu(m.bn1(m.conv1(x))))
            x = m.layer1(x)
            middle = m.layer2(x)
            deep = m.layer3(middle)
            middle = F.adaptive_avg_pool2d(middle, (self.config.grid, self.config.grid))
            deep = F.adaptive_avg_pool2d(deep, (self.config.grid, self.config.grid))
            middle = F.normalize(middle, dim=1)
            deep = F.normalize(deep, dim=1)
            features = torch.cat((middle, deep), dim=1)
        return features[0].permute(1, 2, 0).reshape(-1, features.shape[1]).numpy().astype('float32')


@dataclass
class Inspection:
    score: float
    patch_map: np.ndarray
    seconds: float


class Inspector:
    def __init__(self, extractor, bank: np.ndarray, config: Config = Config()):
        if bank.ndim != 2 or bank.shape[0] == 0 or not np.isfinite(bank).all():
            raise ValueError('Reference bank is empty or invalid.')
        self.extractor = extractor
        self.bank = bank.astype('float32')
        self.config = config
        self.neighbors = NearestNeighbors(n_neighbors=1, algorithm='brute', metric='euclidean', n_jobs=1)
        self.neighbors.fit(self.bank)

    def inspect(self, image: Image.Image) -> Inspection:
        start = time.perf_counter()
        patches = self.extractor.patches(image)
        if patches.shape[1] != self.bank.shape[1] or patches.shape[0] != self.config.grid ** 2:
            raise ValueError('Feature dimensions do not match the reference bank.')
        distances, _ = self.neighbors.kneighbors(patches)
        patch_map = distances[:, 0].reshape(self.config.grid, self.config.grid)
        return Inspection(float(patch_map.max()), patch_map, time.perf_counter() - start)


def build_bank(extractor, normal_images: list[Image.Image], config: Config = Config()) -> np.ndarray:
    if not 3 <= len(normal_images) <= 100:
        raise ValueError('Provide between 3 and 100 normal reference images.')
    patches = np.concatenate([extractor.patches(image) for image in normal_images]).astype('float32')
    if not np.isfinite(patches).all():
        raise ValueError('Non-finite reference features.')
    rng = np.random.default_rng(config.seed)
    sample = rng.choice(len(patches), size=min(config.bank_size, len(patches)), replace=False)
    return patches[np.sort(sample)]


def calibrate(inspector: Inspector, normal_validation: list[Image.Image]) -> tuple[float, list[float]]:
    if len(normal_validation) < 5:
        raise ValueError('Provide at least five separate normal validation images.')
    scores = [inspector.inspect(image).score for image in normal_validation]
    return float(np.quantile(scores, inspector.config.normal_quantile)), scores


def heatmap(image: Image.Image, patch_map: np.ndarray, threshold: float) -> Image.Image:
    """Fixed threshold-based colour scale; bright yellow marks the largest distances."""
    if threshold <= 0:
        raise ValueError('Threshold must be positive.')
    base = ImageOps.fit(image.convert('RGB'), (320, 320))
    normalized = np.clip(patch_map / (2 * threshold), 0, 1)
    field = Image.fromarray(np.uint8(normalized * 255)).resize(base.size, Image.Resampling.BICUBIC)
    values = np.asarray(field).astype('float32') / 255
    colours = np.stack((np.full_like(values, 255), 55 + 180 * values, 25 + 45 * values), axis=-1)
    original = np.asarray(base, dtype='float32')
    opacity = (0.72 * np.clip((values - 0.2) / 0.8, 0, 1))[..., None]
    blended = original * (1 - opacity) + colours * opacity
    return Image.fromarray(np.uint8(np.clip(blended, 0, 255)))

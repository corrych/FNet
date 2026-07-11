from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
from segmentation_models_pytorch.base import initialization as init
from segmentation_models_pytorch.encoders import get_encoder
from torch import Tensor
from decoder import Decoder


def load_encoder_checkpoint(
    encoder: torch.nn.Module,
    checkpoint_path: str,
    *,
    encoder_name: str,
    encoder_depth: int,
    in_channels: int,
) -> None:
    path = Path(checkpoint_path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(
            f"Encoder checkpoint not found: {path}. "
            "Create it with prepare_encoder_weights.py on a networked machine."
        )

    checkpoint = torch.load(path, map_location="cpu")
    if not isinstance(checkpoint, dict) or "encoder_state_dict" not in checkpoint:
        raise ValueError(
            "Expected an encoder checkpoint created by prepare_encoder_weights.py."
        )

    expected_config = {
        "encoder_name": encoder_name,
        "encoder_depth": encoder_depth,
        "in_channels": in_channels,
    }
    for key, expected_value in expected_config.items():
        saved_value = checkpoint.get(key)
        if saved_value is not None and saved_value != expected_value:
            raise ValueError(
                f"Encoder checkpoint {key}={saved_value!r} does not match "
                f"the requested value {expected_value!r}."
            )

    encoder.load_state_dict(checkpoint["encoder_state_dict"], strict=True)


class SegmentationModel(torch.nn.Module):
    encoder: torch.nn.Module
    decoder: torch.nn.Module

    def initialize(self) -> None:
        init.initialize_decoder(self.decoder)

    def forward(self, x: Tensor):
        features = self.encoder(x)
        return self.decoder(*features)

    @torch.inference_mode()
    def predict(self, x: Tensor):
        if self.training:
            self.eval()
        return self.forward(x)


class FNet(SegmentationModel):
    """
    FNet with an offline-loaded encoder and a freshly initialized decoder.

    The encoder checkpoint is independent from the decoder, so decoder changes
    do not affect loading ImageNet-pretrained encoder parameters.
    """

    def __init__(
        self,
        encoder_name: str = "tu-caformer_s36",
        encoder_depth: int = 5,
        encoder_checkpoint: Optional[str] = None,
        in_channels: int = 3,
        classes: int = 1,
    ) -> None:
        super().__init__()

        if classes < 1:
            raise ValueError("classes must be a positive integer")

        self.encoder = get_encoder(
            encoder_name,
            in_channels=in_channels,
            depth=encoder_depth,
            weights=None,
        )
        self.decoder = Decoder(out_channels=classes, return_dict=True)

        self.initialize()
        if encoder_checkpoint:
            load_encoder_checkpoint(
                self.encoder,
                encoder_checkpoint,
                encoder_name=encoder_name,
                encoder_depth=encoder_depth,
                in_channels=in_channels,
            )


def build_model(
    encoder_name: str = "tu-caformer_s36",
    encoder_depth: int = 5,
    encoder_checkpoint: Optional[str] = None,
    in_channels: int = 3,
    classes: int = 1,
) -> FNet:
    return FNet(
        encoder_name=encoder_name,
        encoder_depth=encoder_depth,
        encoder_checkpoint=encoder_checkpoint,
        in_channels=in_channels,
        classes=classes,
    )

__all__ = ["SegmentationModel", "FNet", "build_model"]

from functools import partial

from pytorch_toolbelt.modules import ABN, ACT_SWISH, ACT_RELU, UnetDecoderBlock
from pytorch_toolbelt.modules import encoders as E
from pytorch_toolbelt.modules.decoders import UNetDecoder, UNetDecoderV2
from pytorch_toolbelt.modules.encoders import EncoderModule
from torch import nn
from torch.nn import functional as F
from typing import Optional, List, Union, Callable

from .common import disaster_type_classifier, damage_types_classifier
from ..dataset import (
    OUTPUT_MASK_4_KEY,
    OUTPUT_MASK_8_KEY,
    OUTPUT_MASK_16_KEY,
    OUTPUT_MASK_32_KEY,
    OUTPUT_MASK_KEY,
    DISASTER_TYPE_KEY,
    DISASTER_TYPES,
    DAMAGE_TYPE_KEY,
    DAMAGE_TYPES,
)

__all__ = [
    "resnet18_unet32",
    "resnet34_unet32",
    "resnet50_unet64",
    "seresnext50_unet64",
    "seresnext101_unet64",
    "efficient_unet_b1",
    "efficient_unet_b3",
    "efficient_unet_b4",
    "densenet121_unet32",
    "densenet201_unet32",
]


class UnetSegmentationModel(nn.Module):
    def __init__(
        self,
        encoder: EncoderModule,
        num_classes: int,
        disaster_type_classes: int,
        damage_type_classes: int,
        dropout=0.25,
        abn_block: Union[ABN, Callable[[int], nn.Module]] = ABN,
        unet_channels: Union[int, List[int]] = 32,
        full_size_mask=True,
    ):
        super().__init__()
        self.encoder = encoder

        def final_block(input_features, output_features):
            return nn.Sequential(
                UnetDecoderBlock(input_features, 0, input_features, abn_block=abn_block, scale_factor=2),
                nn.Conv2d(input_features, output_features, kernel_size=3, padding=1),
            )

        self.decoder = UNetDecoder(
            feature_maps=encoder.output_filters,
            decoder_features=unet_channels,
            mask_channels=num_classes,
            dropout=dropout,
            abn_block=abn_block,
            final_block=final_block,
        )

        self.full_size_mask = full_size_mask
        if disaster_type_classes is not None:
            self.disaster_type_classifier = disaster_type_classifier(
                encoder.output_filters[-1], disaster_type_classes, dropout=dropout
            )
        else:
            self.disaster_type_classifier = None

        if damage_type_classes is not None:
            self.damage_types_classifier = damage_types_classifier(
                encoder.output_filters[-1], damage_type_classes, dropout=dropout
            )
        else:
            self.damage_types_classifier = None

    def forward(self, x):
        enc_features = self.encoder(x)

        # Decode mask
        mask = self.decoder(enc_features)

        if self.full_size_mask and (mask.size(2) != x.size(2) or mask.size(3) != x.size(3)):
            mask = F.interpolate(mask, size=x.size()[2:], mode="bilinear", align_corners=False)

        output = {OUTPUT_MASK_KEY: mask}

        if self.disaster_type_classifier is not None:
            disaster_type = self.disaster_type_classifier(enc_features[-1])
            output[DISASTER_TYPE_KEY] = disaster_type

        if self.damage_types_classifier is not None:
            damage_types = self.damage_types_classifier(enc_features[-1])
            output[DAMAGE_TYPE_KEY] = damage_types

        return output


def resnet18_unet32(input_channels=6, num_classes=5, dropout=0.0, pretrained=True):
    encoder = E.Resnet18Encoder(pretrained=pretrained, layers=[0, 1, 2, 3, 4])
    encoder.change_input_channels(input_channels)
    return UnetSegmentationModel(
        encoder,
        num_classes=num_classes,
        disaster_type_classes=len(DISASTER_TYPES),
        damage_type_classes=len(DAMAGE_TYPES),
        unet_channels=[32, 32, 64, 128, 256],
        dropout=dropout,
        abn_block=partial(ABN, activation=ACT_RELU),
    )


def resnet34_unet32(input_channels=6, num_classes=5, dropout=0.0, pretrained=True):
    encoder = E.Resnet34Encoder(pretrained=pretrained)
    encoder.change_input_channels(input_channels)
    return UnetSegmentationModel(
        encoder,
        num_classes=num_classes,
        disaster_type_classes=len(DISASTER_TYPES),
        damage_type_classes=len(DAMAGE_TYPES),
        unet_channels=32,
        dropout=dropout,
        abn_block=partial(ABN, activation=ACT_RELU),
    )


def resnet50_unet64(input_channels=6, num_classes=5, dropout=0.0, pretrained=True):
    encoder = E.Resnet50Encoder(pretrained=pretrained)
    encoder.change_input_channels(input_channels)
    return UnetSegmentationModel(
        encoder,
        num_classes=num_classes,
        disaster_type_classes=len(DISASTER_TYPES),
        damage_type_classes=len(DAMAGE_TYPES),
        unet_channels=[64, 128, 256, 512],
        dropout=dropout,
        abn_block=partial(ABN, activation=ACT_RELU),
    )


def seresnext50_unet64(input_channels=6, num_classes=5, dropout=0.0, pretrained=True):
    encoder = E.SEResNeXt50Encoder(pretrained=pretrained)
    encoder.change_input_channels(input_channels)
    return UnetSegmentationModel(
        encoder,
        num_classes=num_classes,
        disaster_type_classes=len(DISASTER_TYPES),
        damage_type_classes=len(DAMAGE_TYPES),
        unet_channels=64,
        dropout=dropout,
        abn_block=partial(ABN, activation=ACT_RELU),
    )


def seresnext101_unet64(input_channels=6, num_classes=5, dropout=0.0, pretrained=True):
    encoder = E.SEResNeXt101Encoder(pretrained=pretrained)
    encoder.change_input_channels(input_channels)
    return UnetSegmentationModel(
        encoder,
        num_classes=num_classes,
        disaster_type_classes=len(DISASTER_TYPES),
        damage_type_classes=len(DAMAGE_TYPES),
        unet_channels=64,
        dropout=dropout,
        abn_block=partial(ABN, activation=ACT_RELU),
    )


def efficient_unet_b1(input_channels=6, num_classes=5, dropout=0.0, pretrained=True):
    encoder = E.EfficientNetB1Encoder()
    encoder.change_input_channels(input_channels)
    return UnetSegmentationModel(
        encoder,
        num_classes=num_classes,
        disaster_type_classes=len(DISASTER_TYPES),
        damage_type_classes=len(DAMAGE_TYPES),
        unet_channels=32,
        dropout=dropout,
        abn_block=partial(ABN, activation="swish"),
    )


def efficient_unet_b3(input_channels=6, num_classes=5, dropout=0.0, pretrained=True):
    encoder = E.EfficientNetB3Encoder()
    encoder.change_input_channels(input_channels)
    return UnetSegmentationModel(
        encoder,
        num_classes=num_classes,
        disaster_type_classes=len(DISASTER_TYPES),
        damage_type_classes=len(DAMAGE_TYPES),
        unet_channels=32,
        dropout=dropout,
        abn_block=partial(ABN, activation=ACT_SWISH),
    )


def efficient_unet_b4(input_channels=6, num_classes=5, dropout=0.0, pretrained=True):
    encoder = E.EfficientNetB4Encoder()
    encoder.change_input_channels(input_channels)
    return UnetSegmentationModel(
        encoder,
        num_classes=num_classes,
        disaster_type_classes=len(DISASTER_TYPES),
        damage_type_classes=len(DAMAGE_TYPES),
        unet_channels=32,
        dropout=dropout,
        abn_block=partial(ABN, activation=ACT_SWISH),
    )


def densenet121_unet32(input_channels=6, num_classes=5, dropout=0.0, pretrained=True):
    encoder = E.DenseNet121Encoder(pretrained=pretrained)
    encoder.change_input_channels(input_channels)
    return UnetSegmentationModel(
        encoder,
        num_classes=num_classes,
        disaster_type_classes=len(DISASTER_TYPES),
        damage_type_classes=len(DAMAGE_TYPES),
        unet_channels=32,
        dropout=dropout,
        abn_block=partial(ABN, activation=ACT_RELU),
    )


def densenet201_unet32(input_channels=6, num_classes=5, dropout=0.0, pretrained=True):
    encoder = E.DenseNet201Encoder(pretrained=pretrained)
    encoder.change_input_channels(input_channels)
    return UnetSegmentationModel(
        encoder,
        num_classes=num_classes,
        disaster_type_classes=len(DISASTER_TYPES),
        damage_type_classes=len(DAMAGE_TYPES),
        unet_channels=32,
        dropout=dropout,
        abn_block=partial(ABN, activation=ACT_RELU),
    )

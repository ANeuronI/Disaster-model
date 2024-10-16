from functools import partial

from pytorch_toolbelt.modules import ABN, ACT_RELU
from pytorch_toolbelt.modules import encoders as E
from pytorch_toolbelt.modules.decoders import FPNSumDecoder, FPNCatDecoder
from pytorch_toolbelt.modules.encoders import EncoderModule
from torch import nn
from torch.nn import functional as F

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
    "FPNSumSegmentationModel",
    "FPNCatSegmentationModel",
    "resnet18_fpncat128",
    "resnet34_fpncat128",
    "resnet101_fpncat256",
    "resnet152_fpncat256",
    "resnet101_fpncat256",
    "seresnext50_fpncat128",
    "seresnext101_fpncat256",
    "seresnext101_fpnsum256",
    "effnetB4_fpncat128",
    "densenet121_fpnsum128",
]


class FPNSumFinalBlock(nn.Module):
    def __init__(self, input_channels, output_channels, abn_block=ABN):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, input_channels, kernel_size=3, padding=1, bias=False)
        self.abn1 = abn_block(input_channels)
        self.conv2 = nn.Conv2d(input_channels, input_channels, kernel_size=3, padding=1, bias=False)
        self.abn2 = abn_block(input_channels)

        self.final = nn.Conv2d(input_channels, output_channels, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.abn1(x)
        x = self.conv2(x)
        x = self.abn2(x)
        x = self.final(x)
        return x


class FPNSumSegmentationModel(nn.Module):
    def __init__(
        self,
        encoder: EncoderModule,
        num_classes: int,
        disaster_type_classes: int,
        damage_type_classes: int,
        dropout=0.25,
        abn_block=ABN,
        full_size_mask=True,
        fpn_channels=256,
    ):
        super().__init__()
        self.encoder = encoder

        self.decoder = FPNSumDecoder(
            feature_maps=encoder.output_filters,
            output_channels=num_classes,
            dsv_channels=None,
            fpn_channels=fpn_channels,
            abn_block=abn_block,
            dropout=dropout,
            final_block=FPNSumFinalBlock,
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

        if self.full_size_mask:
            mask = F.interpolate(mask, size=x.size()[2:], mode="bilinear", align_corners=False)

        output = {OUTPUT_MASK_KEY: mask}

        if self.disaster_type_classifier is not None:
            disaster_type = self.disaster_type_classifier(enc_features[-1])
            output[DISASTER_TYPE_KEY] = disaster_type

        if self.damage_types_classifier is not None:
            damage_types = self.damage_types_classifier(enc_features[-1])
            output[DAMAGE_TYPE_KEY] = damage_types

        return output


class FPNCatFinalBlock(nn.Module):
    def __init__(self, input_channels, output_channels, abn_block=ABN):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, input_channels, kernel_size=3, padding=1, bias=False)
        self.abn1 = abn_block(input_channels)
        self.conv2 = nn.Conv2d(input_channels, output_channels, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.abn1(x)
        x = self.conv2(x)
        return x


class FPNCatSegmentationModel(nn.Module):
    def __init__(
        self,
        encoder: EncoderModule,
        num_classes: int,
        disaster_type_classes: int,
        damage_type_classes: int,
        dropout=0.25,
        abn_block=ABN,
        fpn_channels=256,
        full_size_mask=True,
    ):
        super().__init__()
        self.encoder = encoder

        self.decoder = FPNCatDecoder(
            feature_maps=encoder.output_filters,
            output_channels=num_classes,
            dsv_channels=None,
            fpn_channels=fpn_channels,
            abn_block=abn_block,
            dropout=dropout,
            final_block=partial(FPNCatFinalBlock, abn_block=abn_block),
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

        if self.full_size_mask:
            mask = F.interpolate(mask, size=x.size()[2:], mode="bilinear", align_corners=False)

        output = {OUTPUT_MASK_KEY: mask}

        if self.disaster_type_classifier is not None:
            disaster_type = self.disaster_type_classifier(enc_features[-1])
            output[DISASTER_TYPE_KEY] = disaster_type

        if self.damage_types_classifier is not None:
            damage_types = self.damage_types_classifier(enc_features[-1])
            output[DAMAGE_TYPE_KEY] = damage_types

        return output


def resnet18_fpncat128(num_classes=5, dropout=0.0, pretrained=True):
    encoder = E.Resnet18Encoder(pretrained=pretrained)
    encoder.change_input_channels(6)
    return FPNCatSegmentationModel(
        encoder,
        num_classes=num_classes,
        disaster_type_classes=len(DISASTER_TYPES),
        damage_type_classes=len(DAMAGE_TYPES),
        fpn_channels=128,
        dropout=dropout,
    )


def resnet34_fpncat128(num_classes=5, dropout=0.0, pretrained=True):
    encoder = E.Resnet34Encoder(pretrained=pretrained)
    encoder.change_input_channels(6)
    return FPNCatSegmentationModel(
        encoder,
        num_classes=num_classes,
        disaster_type_classes=len(DISASTER_TYPES),
        damage_type_classes=len(DAMAGE_TYPES),
        fpn_channels=128,
        dropout=dropout,
    )


def seresnext50_fpncat128(num_classes=5, dropout=0.0, pretrained=True):
    encoder = E.SEResNeXt50Encoder(pretrained=pretrained)
    encoder.change_input_channels(6)
    return FPNCatSegmentationModel(
        encoder,
        num_classes=num_classes,
        disaster_type_classes=len(DISASTER_TYPES),
        damage_type_classes=len(DAMAGE_TYPES),
        fpn_channels=128,
        dropout=dropout,
        abn_block=partial(ABN, activation=ACT_RELU),
    )


def seresnext101_fpncat256(num_classes=5, dropout=0.0, pretrained=True):
    encoder = E.SEResNeXt101Encoder(pretrained=pretrained)
    encoder.change_input_channels(6)
    return FPNCatSegmentationModel(
        encoder,
        num_classes=num_classes,
        disaster_type_classes=len(DISASTER_TYPES),
        damage_type_classes=len(DAMAGE_TYPES),
        fpn_channels=256,
        dropout=dropout,
    )


def seresnext101_fpnsum256(num_classes=5, dropout=0.0, pretrained=True):
    encoder = E.SEResNeXt101Encoder(pretrained=pretrained)
    encoder.change_input_channels(6)
    return FPNSumSegmentationModel(
        encoder,
        num_classes=num_classes,
        disaster_type_classes=len(DISASTER_TYPES),
        damage_type_classes=len(DAMAGE_TYPES),
        fpn_channels=256,
        dropout=dropout,
    )


def resnet101_fpncat256(num_classes=5, dropout=0.0, pretrained=True):
    encoder = E.Resnet101Encoder(pretrained=pretrained)
    encoder.change_input_channels(6)
    return FPNCatSegmentationModel(
        encoder,
        num_classes=num_classes,
        disaster_type_classes=len(DISASTER_TYPES),
        damage_type_classes=len(DAMAGE_TYPES),
        fpn_channels=256,
        dropout=dropout,
    )


def resnet152_fpncat256(num_classes=5, dropout=0.0, pretrained=True):
    encoder = E.Resnet152Encoder(pretrained=pretrained)
    encoder.change_input_channels(6)
    return FPNCatSegmentationModel(
        encoder,
        num_classes=num_classes,
        disaster_type_classes=len(DISASTER_TYPES),
        damage_type_classes=len(DAMAGE_TYPES),
        fpn_channels=256,
        dropout=dropout,
    )


def effnetB4_fpncat128(num_classes=5, dropout=0.0, pretrained=True):
    encoder = E.EfficientNetB4Encoder(abn_params={"activation": "swish"}, pretrained=pretrained)
    encoder.change_input_channels(6)
    return FPNCatSegmentationModel(
        encoder,
        num_classes=num_classes,
        disaster_type_classes=len(DISASTER_TYPES),
        damage_type_classes=len(DAMAGE_TYPES),
        fpn_channels=128,
        dropout=dropout,
    )


def densenet121_fpnsum128(input_channels=6, num_classes=5, dropout=0.0, pretrained=True):
    encoder = E.DenseNet121Encoder(pretrained=pretrained)
    encoder.change_input_channels(input_channels)
    return FPNSumSegmentationModel(
        encoder,
        num_classes=num_classes,
        disaster_type_classes=len(DISASTER_TYPES),
        damage_type_classes=len(DAMAGE_TYPES),
        fpn_channels=128,
        dropout=dropout,
        abn_block=partial(ABN, activation=ACT_RELU),
    )

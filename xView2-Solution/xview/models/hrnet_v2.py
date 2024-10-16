from typing import List

import torch
from pytorch_toolbelt.modules import encoders as E, ABN
from pytorch_toolbelt.modules.decoders import DecoderModule
from pytorch_toolbelt.modules.encoders import EncoderModule
from torch import nn
from torch.nn import functional as F

from .common import disaster_type_classifier, damage_types_classifier
from ..dataset import OUTPUT_MASK_KEY, DISASTER_TYPE_KEY, DISASTER_TYPES, DAMAGE_TYPE_KEY, DAMAGE_TYPES

__all__ = ["HRNetSegmentationModelV2", "hrnet18v2", "hrnet34v2", "hrnet48v2"]


class HRNetDecoder(DecoderModule):
    def __init__(self, feature_maps: List[int], output_channels: int, first_bottleneck=256, bottleneck=128, abn_block=ABN):
        super().__init__()
        features = feature_maps[-1]
        self.last_layer = nn.Sequential(
            nn.Conv2d(in_channels=features, out_channels=first_bottleneck, kernel_size=3, padding=1, bias=False),
            abn_block(first_bottleneck),

            nn.Conv2d(in_channels=first_bottleneck, out_channels=bottleneck, kernel_size=3, padding=1, bias=False),
            abn_block(bottleneck),

            nn.Conv2d(in_channels=bottleneck, out_channels=bottleneck, kernel_size=3, padding=1, bias=False),
            abn_block(bottleneck),

            nn.Conv2d(in_channels=bottleneck, out_channels=bottleneck, kernel_size=3, padding=1, bias=False),
            abn_block(bottleneck),

            nn.Conv2d(in_channels=bottleneck, out_channels=output_channels, kernel_size=1),
        )

    def forward(self, features):
        return self.last_layer(features[-1])


class HRNetSegmentationModelV2(nn.Module):
    def __init__(
        self,
        encoder: EncoderModule,
        num_classes: int,
        disaster_type_classes: int,
        damage_type_classes: int,
        full_size_mask=True,
        abn_block=ABN,
    ):
        super().__init__()
        self.encoder = encoder

        feature_maps = [2 * fm for fm in encoder.output_filters]

        self.decoder = HRNetDecoder(feature_maps=feature_maps, output_channels=num_classes, abn_block=abn_block)

        self.full_size_mask = full_size_mask
        if disaster_type_classes is not None:
            self.disaster_type_classifier = disaster_type_classifier(feature_maps[-1], disaster_type_classes)
        else:
            self.disaster_type_classifier = None

        if damage_type_classes is not None:
            self.damage_types_classifier = damage_types_classifier(feature_maps[-1], damage_type_classes)
        else:
            self.damage_types_classifier = None

    def forward(self, x):
        batch_size = x.size(0)
        pre, post = x[:, 0:3, ...], x[:, 3:6, ...]

        x = torch.cat([pre, post], dim=0)
        features = self.encoder(x)
        features = [torch.cat([f[0:batch_size], f[batch_size: batch_size * 2]], dim=1) for f in features]

        # Decode mask
        mask = self.decoder(features)

        if self.full_size_mask:
            mask = F.interpolate(mask, size=x.size()[2:], mode="bilinear", align_corners=False)

        output = {OUTPUT_MASK_KEY: mask}

        if self.disaster_type_classifier is not None:
            disaster_type = self.disaster_type_classifier(features[-1])
            output[DISASTER_TYPE_KEY] = disaster_type

        if self.damage_types_classifier is not None:
            damage_types = self.damage_types_classifier(features[-1])
            output[DAMAGE_TYPE_KEY] = damage_types

        return output


def hrnet18v2(num_classes=5, dropout=0.0, pretrained=True):
    encoder = E.HRNetV2Encoder18(pretrained=pretrained)
    return HRNetSegmentationModelV2(
        encoder,
        num_classes=num_classes,
        disaster_type_classes=len(DISASTER_TYPES),
        damage_type_classes=len(DAMAGE_TYPES),
    )


def hrnet34v2(num_classes=5, dropout=0.0, pretrained=True):
    encoder = E.HRNetV2Encoder34(pretrained=pretrained)
    return HRNetSegmentationModelV2(
        encoder,
        num_classes=num_classes,
        disaster_type_classes=len(DISASTER_TYPES),
        damage_type_classes=len(DAMAGE_TYPES),
    )


def hrnet48v2(num_classes=5, dropout=0.0, pretrained=True):
    encoder = E.HRNetV2Encoder48(pretrained=pretrained)
    return HRNetSegmentationModelV2(
        encoder,
        num_classes=num_classes,
        disaster_type_classes=len(DISASTER_TYPES),
        damage_type_classes=len(DAMAGE_TYPES),
    )

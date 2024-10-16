from torch import nn

from . import fpn, unet, hrnet, unetv2, cls, fpn_v2, hrnet_v2, fpn_v3, unetv3

__all__ = ["get_model"]


def get_model(model_name: str, dropout=0.0, pretrained=True, classifiers=True) -> nn.Module:
    registry = {
        # FPN family using concatenation
        "resnet18_fpncat128": fpn.resnet18_fpncat128,
        "resnet34_fpncat128": fpn.resnet34_fpncat128,
        "resnet101_fpncat256": fpn.resnet101_fpncat256,
        "resnet152_fpncat256": fpn.resnet152_fpncat256,
        "seresnext50_fpncat128": fpn.seresnext50_fpncat128,
        "effnetB4_fpncat128": fpn.effnetB4_fpncat128,
        "seresnext101_fpncat256": fpn.seresnext101_fpncat256,

        # FPN V2
        "resnet34_fpncatv2_256": fpn_v2.resnet34_fpncatv2_256,
        "resnet34_fpncatv2_256_nearest": fpn_v2.resnet34_fpncatv2_256_nearest,
        "densenet201_fpncatv2_256": fpn_v2.densenet201_fpncatv2_256,
        "resnet101_fpncatv2_256": fpn_v2.resnet101_fpncatv2_256,
        "efficientb4_fpncatv2_256": fpn_v2.efficientb4_fpncatv2_256,
        "inceptionv4_fpncatv2_256": fpn_v2.inceptionv4_fpncatv2_256,

        # FPN V3
        "resnet18_fpncatv3_128": fpn_v3.resnet18_fpncatv3_128,
        "resnet50_fpncatv3_256": fpn_v3.resnet50_fpncatv3_256,

        # FPN family using summation
        "seresnext101_fpnsum256": fpn.seresnext101_fpnsum256,
        "densenet121_fpnsum128": fpn.densenet121_fpnsum128,

        # UNet
        "resnet18_unet32": unet.resnet18_unet32,
        "resnet34_unet32": unet.resnet34_unet32,
        "resnet50_unet64": unet.resnet50_unet64,
        "seresnext50_unet64": unet.seresnext50_unet64,
        "seresnext101_unet64": unet.seresnext101_unet64,
        "densenet121_unet32": unet.densenet121_unet32,
        "densenet201_unet32": unet.densenet201_unet32,

        # UnetV2
        "resnet18_unet_v2": unetv2.resnet18_unet_v2,
        "resnet34_unet_v2": unetv2.resnet34_unet_v2,
        "resnet50_unet_v2": unetv2.resnet50_unet_v2,
        "resnet101_unet_v2": unetv2.resnet101_unet_v2,
        "seresnext50_unet_v2": unetv2.seresnext50_unet_v2,
        "seresnext101_unet_v2": unetv2.seresnext101_unet_v2,
        "densenet121_unet_v2": unetv2.densenet121_unet_v2,
        "densenet169_unet_v2": unetv2.densenet169_unet_v2,
        "efficientb3_unet_v2": unetv2.efficientb3_unet_v2,

        # UnetV3
        "resnet18_unet_v3": unetv3.resnet18_unet_v3,
        "resnet34_unet_v3": unetv3.resnet34_unet_v3,
        "resnet101_unet_v3": unetv3.resnet101_unet_v3,

        # Efficient net
        "efficient_unet_b1": unet.efficient_unet_b1,
        "efficient_unet_b3": unet.efficient_unet_b3,
        "efficient_unet_b4": unet.efficient_unet_b4,

        "resnet34_cls": cls.resnet34_cls,
        "resnet18_cls": cls.resnet18_cls
    }

    return registry[model_name.lower()](dropout=dropout, pretrained=pretrained, classifiers=classifiers)

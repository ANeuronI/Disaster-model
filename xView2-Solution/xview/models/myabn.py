import torch
from pytorch_toolbelt.modules import instantiate_activation_block, ACT_LEAKY_RELU, ACT_RELU, ACT_RELU6
from torch import nn


class MyABN(nn.Module):
    _version = 2
    __constants__ = [
        "track_running_stats",
        "momentum",
        "eps",
        "weight",
        "bias",
        "running_mean",
        "running_var",
        "num_batches_tracked",
        "num_features",
        "affine",
    ]

    """Activated Batch Normalization
    This gathers a `BatchNorm` and an activation function in a single module
    """

    def __init__(
        self,
        num_features: int,
        eps=1e-5,
        momentum=0.1,
        affine=True,
        track_running_stats=True,
        activation="leaky_relu",
        slope=0.01,
    ):
        params = {}
        if activation in {ACT_LEAKY_RELU}:
            params["negative_slope"] = slope
        if activation in {ACT_LEAKY_RELU, ACT_RELU, ACT_RELU6}:
            params["inplace"] = True

        super(MyABN, self).__init__()
        self.bn = nn.BatchNorm2d(num_features, eps, momentum, affine, track_running_stats)
        self.act = instantiate_activation_block(activation, **params)
        self.activation_name = activation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.bn(x)

        # Dirty hack to make MKLLDDNN happy
        if self.activation_name == ACT_LEAKY_RELU:
            x = x.to_dense()

        x = self.act(x)

        if self.activation_name == ACT_LEAKY_RELU:
            x = x.to_mkldnn()

        return x

    def _load_from_state_dict(
        self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
    ):
        local_unexpected_keys = []
        local_missing_keys = []
        local_error_msgs = []
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict, local_missing_keys, local_unexpected_keys, local_error_msgs)
        if len(local_unexpected_keys):
            print(local_missing_keys, local_unexpected_keys, local_error_msgs)
            self.bn._load_from_state_dict(
                state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
            )
            print(missing_keys, unexpected_keys, error_msgs)

        if len(local_missing_keys):
            print(local_missing_keys)
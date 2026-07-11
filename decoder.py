import torch
import torch.nn as nn


class ConvBnAct(nn.Module):
    def __init__(self, in_channel, out_channel, kernel, stride, padding, dilation=1, bias=False, act=True):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channel,
            out_channel,
            kernel,
            stride,
            padding=padding,
            dilation=dilation,
            bias=bias,
        )
        self.bn = nn.BatchNorm2d(out_channel)
        self.act = nn.ReLU(inplace=True) if act else nn.Identity()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x

    def forward_fuse(self, x):
        return self.act(self.conv(x))


class PN(nn.Module):
    def __init__(self, large_channels, small_channels, scale_size, out_channels):
        super().__init__()
        self.large_res = ConvBnAct(large_channels, small_channels, 1, 1, 0)
        self.downsample = ConvBnAct(
            large_channels,
            small_channels,
            kernel=scale_size,
            stride=scale_size,
            padding=0,
        )
        self.gating = nn.Sequential(
            nn.Conv2d(small_channels * 2, small_channels, 1),
            nn.BatchNorm2d(small_channels),
            nn.Sigmoid(),
        )
        self.upscale = nn.UpsamplingBilinear2d(scale_factor=scale_size)
        self.cnn_out = ConvBnAct(small_channels, out_channels, 3, 1, 1)

    def forward(self, xlarge, xsmall):
        reslarge = self.large_res(xlarge)
        downsampled = self.downsample(xlarge)

        concat_feat = torch.cat([downsampled, xsmall], dim=1)
        gate = self.gating(concat_feat)
        merged = gate * downsampled + (1 - gate) * xsmall

        upscaled = self.upscale(merged)
        residual = upscaled + reslarge
        out = self.cnn_out(residual)
        return out


class PredictionHead(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, upsampling=1):
        conv2d = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
        )
        upsampling = (
            nn.UpsamplingBilinear2d(scale_factor=upsampling)
            if upsampling > 1
            else nn.Identity()
        )
        super().__init__(conv2d, upsampling)


class OcclusionAwarePNBranch(nn.Module):
    """
    Single-branch decoder for occlusion-aware building extraction.

    It keeps adjacent-scale PN decoding as the main path, and injects direct
    deep semantic context from s4 -> s1.
    """

    def __init__(self, out_channels=1):
        super().__init__()

        self.layer3 = PN(320, 512, 2, 256)

        self.layer2_1 = PN(128, 256, 2, 128)
        self.layer2_2 = PN(128, 512, 4, 256)

        self.layer1_1 = PN(64, 256, 2, 64)
        self.layer1_2 = PN(64, 256, 4, 64)


        
        self.fuse_s1 = ConvBnAct(64 * 2, 64, 3, 1, 1)

        self.upscale = nn.UpsamplingBilinear2d(scale_factor=2)
        self.head = PredictionHead(
            in_channels=64,
            out_channels=out_channels,
            kernel_size=3,
            upsampling=2,
        )

    def forward(self, s1, s2, s3, s4):

        x3 = self.layer3(s3,s4)

        x2_1 = self.layer2_1(s2,x3)
        x2_2 = self.layer2_2(x2_1,s4)

        x1_1 = self.layer1_1(s1,x2_2)
        x1_2 = self.layer1_2(x1_1,x3)

        o1 = self.fuse_s1(torch.cat([x1_1, x1_2], dim=1))
        feature = self.upscale(o1)
        logits = self.head(feature)
        return logits


class Decoder(nn.Module):
    """
    Single-branch occlusion-aware decoder.

    The decoder predicts the final mask directly. It no longer uses online
    hard labels, hard gates, or residual branch supervision.
    """

    def __init__(self, out_channels=1, return_dict=True):
        super().__init__()
        self.decoder_branch = OcclusionAwarePNBranch(out_channels=out_channels)
        self.return_dict = return_dict

    def forward(self, *features):
        features = features[1:]
        features = features[::-1]

        s4, s3, s2, s1 = features[0], features[1], features[2], features[3]

        final_logits = self.decoder_branch(s1, s2, s3, s4)
        final_mask = torch.sigmoid(final_logits)

        if not self.return_dict:
            return final_mask

        return {"final_mask": final_mask}


__all__ = [
    "ConvBnAct",
    "PN",
    "PredictionHead",
    "OcclusionAwarePNBranch",
    "Decoder",
]

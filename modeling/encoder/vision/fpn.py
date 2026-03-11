import torch
import torch.nn.functional as F
from torchvision.ops import FeaturePyramidNetwork
from collections import OrderedDict

class EfficientFeaturePyramidNetwork(FeaturePyramidNetwork):

    def __init__(
        self,
        in_channels_list,
        out_channels,
        extra_blocks=None,
        norm_layer=None,
        output_level="res3"
    ):
        super().__init__(
            in_channels_list,
            out_channels,
            extra_blocks,
            norm_layer,
        )
        self.output_level = output_level

        # Ensure inner_blocks Conv2d weights are fully contiguous
        for idx, block in enumerate(self.inner_blocks):
            if isinstance(block, torch.nn.Conv2d):
                # Recreate Conv2d with fresh contiguous weights
                new_block = torch.nn.Conv2d(
                    block.in_channels,
                    block.out_channels,
                    block.kernel_size,
                    stride=block.stride,
                    padding=block.padding,
                    dilation=block.dilation,
                    bias=(block.bias is not None),
                    padding_mode=block.padding_mode,
                ).to(memory_format=torch.contiguous_format).requires_grad_(True)
                
                # Copy weights/bias from original block
                new_block.weight.data.copy_(block.weight.data)
                if block.bias is not None:
                    new_block.bias.data.copy_(block.bias.data)
                
                self.inner_blocks[idx] = new_block

        # Register backward hooks to ensure .grad is contiguous
        for block in self.inner_blocks:
            if isinstance(block, torch.nn.Conv2d):
                block.weight.register_hook(lambda grad: grad.contiguous())

    def forward(self, x):
        """
        Computes the FPN for a set of feature maps.

        Args:
            x (OrderedDict[Tensor]): feature maps for each feature level.

        Returns:
            results (OrderedDict[Tensor]): feature maps after FPN layers.
                They are ordered from the highest resolution first.
        """
        names = list(x.keys())
        x = [v.contiguous(memory_format=torch.contiguous_format) for v in x.values()]

        last_inner = self.get_result_from_inner_blocks(x[-1], -1)
        results = []
        results.append(self.get_result_from_layer_blocks(last_inner, -1))

        if names[-1] != self.output_level:
            for idx in range(len(x) - 2, -1, -1):
                inner_lateral = self.get_result_from_inner_blocks(x[idx], idx)
                feat_shape = inner_lateral.shape[-2:]
                inner_top_down = F.interpolate(last_inner, size=feat_shape, mode="nearest")
                last_inner = (inner_lateral + inner_top_down).contiguous()
                results.insert(0, self.get_result_from_layer_blocks(last_inner, idx))

                # Stop early if we've reached desired output level
                if names[idx] == self.output_level:
                    names = names[idx:]
                    break
        else:
            names = names[-1:]

        if self.extra_blocks is not None:
            results, names = self.extra_blocks(results, x, names)

        # Reformat to OrderedDict
        out = OrderedDict({
            k: v.contiguous(memory_format=torch.contiguous_format)
            for k, v in zip(names, results)
        })

        return out

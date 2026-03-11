# Adapted from https://github.com/openai/CLIP/blob/main/clip/model.py

import torch
from torch import nn

import clip
from clip.model import ModifiedResNet


class CLIPTransform(nn.Module):

    def __init__(self):
        super().__init__()
        _mean = [0.48145466, 0.4578275, 0.40821073]
        _std = [0.26862954, 0.26130258, 0.27577711]
        self.register_buffer("mean", torch.tensor(_mean).reshape(1, -1, 1, 1))
        self.register_buffer("std", torch.tensor(_std).reshape(1, -1, 1, 1))

    def forward(self, img):
        return (img - self.mean) / self.std


def load_clip():
    # Use local cache from master directory to avoid redownloading
    import os
    
    # First try to use the cached model from master directory
    master_cache = os.path.join(os.path.dirname(__file__), "../../../pretrain_unimanual/openai")
    local_cache = os.path.expanduser("~/.cache/clip")
    
    # print("Loading CLIP visual encoder...")
    if os.path.exists(os.path.join(master_cache, "RN50.pt")):
        # print(f"Using cached CLIP model from: {master_cache}")
        clip_model, clip_transforms = clip.load("RN50", download_root=master_cache)
    else:
        print(f"Fallback to default cache directory: {local_cache}")
        os.makedirs(local_cache, exist_ok=True)
        clip_model, clip_transforms = clip.load("RN50", download_root=local_cache)
    
    state_dict = clip_model.state_dict()
    layers = tuple([len(set(k.split(".")[2] for k in state_dict if k.startswith(f"visual.layer{b}")))
                    for b in [1, 2, 3, 4]])
    output_dim = state_dict["text_projection"].shape[1]
    heads = state_dict["visual.layer1.0.conv1.weight"].shape[0] * 32 // 64
    backbone = ModifiedResNetFeatures(layers, output_dim, heads)
    backbone.load_state_dict(clip_model.visual.state_dict())
    # normalize = clip_transforms.transforms[-1]
    normalize = CLIPTransform()
    print("CLIP visual encoder loaded successfully.")
    return backbone, normalize


class ModifiedResNetFeatures(ModifiedResNet):
    def __init__(self, layers, output_dim, heads, input_resolution=224, width=64):
        super().__init__(layers, output_dim, heads, input_resolution, width)

    def forward(self, x: torch.Tensor):
        x = x.type(self.conv1.weight.dtype)
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x0 = self.relu3(self.bn3(self.conv3(x)))
        x = self.avgpool(x0)
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        return {
            "res1": x0,
            "res2": x1,
            "res3": x2,
            "res4": x3,
            "res5": x4,
        }

import einops
from torch import nn
from torchvision.ops import Conv2dNormActivation

from ..vision.fpn import EfficientFeaturePyramidNetwork
from .base_encoder import Encoder as BaseEncoder


class Encoder(BaseEncoder):

    def __init__(self,
                 backbone="clip",
                 embedding_dim=60,
                 nhist=1,
                 num_attn_heads=9,
                 num_vis_instr_attn_layers=2,
                 fps_subsampling_factor=5,
                 finetune_backbone=False,
                 finetune_text_encoder=False,
                 rot_dim=3):
        super().__init__(
            backbone=backbone,
            embedding_dim=embedding_dim,
            nhist=nhist,
            num_attn_heads=num_attn_heads,
            num_vis_instr_attn_layers=num_vis_instr_attn_layers,
            fps_subsampling_factor=fps_subsampling_factor,
            finetune_backbone=finetune_backbone,
            finetune_text_encoder=finetune_text_encoder
        )

        # Postprocess scene features
        if self._backbone_name == 'clip':
            self.feature_pyramid = EfficientFeaturePyramidNetwork(
                [64, 256, 512, 1024, 2048],
                embedding_dim, output_level="res4"
            )
            self.rgb2d_proj = nn.Conv2d(2048, embedding_dim, 1)

        # Camera ids
        self.camera_ids = nn.Embedding(5, embedding_dim)

        # Proprioception learnable projection if no 3D is used
        self.rot_dim = rot_dim
        self.proprio_feat = nn.Linear(3 + rot_dim, embedding_dim)

    def encode_proprio(self, proprio, context_feats, context_pos):
        """
        Compute proprioception features.

        Args:
            - proprio: (B, nhist, 3+)
            - context_feats: (B, npt, C)
            - context_pos: (B, npt, 3)

        Returns:
            - gripper_feats: (B, nhist, F)
        """
        return self.proprio_feat(proprio[..., :3 + self.rot_dim])

    def encode_clip(self, rgb3d, rgb2d, pcd, text):
        """
        Compute visual features/pos embeddings.

        Args:
            - rgb3d: (B, ncam3d, 3, H, W), rgb obs of 3D cameras
            - rgb2d: (B, ncam2d, 3, H, W), rgb obs of 2D cameras
            - pcd: (B, ncam3d, 3, H, W) or None
            - text: [str] of len=B, text instruction

        Returns:
            - rgb3d_feats: (B, Np, F)
            - rgb2d_feats: (B, ncam2d, F)
            - pcd: (B, Np, 3)
            - instr_feats: (B, L, F)
        """
        # Encode language
        instruction = self.text_encoder(text)
        instr_feats = self.instruction_encoder(instruction)

        # 3D camera features (not 3D, we just keep the naming convention)
        rgb3d_feats = None
        if rgb3d is not None:
            num_cameras = rgb3d.shape[1]
            # Pass each view independently through backbone
            rgb3d = einops.rearrange(rgb3d, "bt ncam c h w -> (bt ncam) c h w")
            rgb3d = self.normalize(rgb3d)
            rgb3d_feats = self.backbone(rgb3d)
            # Pass visual features through feature pyramid network
            rgb3d_feats = self.feature_pyramid(rgb3d_feats)["res4"]
            # Add camera id embeddings
            rgb3d_feats = einops.rearrange(
                rgb3d_feats,
                "(bt ncam) c h w -> bt ncam c h w", ncam=num_cameras
            )
            rgb3d_feats = rgb3d_feats + self.camera_ids.weight[:num_cameras][
                None, :, :, None, None
            ]
            # Merge different cameras
            rgb3d_feats = einops.rearrange(
                rgb3d_feats, "bt ncam c h w -> bt (ncam h w) c"
            )
            # Attention from vision to language
            rgb3d_feats = self.vl_attention(seq1=rgb3d_feats, seq2=instr_feats)[-1]

        # 2D camera features
        rgb2d_feats = None

        return rgb3d_feats, rgb2d_feats, None, instr_feats

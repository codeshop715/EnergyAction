import torch
from torch import nn

from ...utils.layers import AttentionModule
from ..vision import fetch_visual_encoders
from ..text import fetch_text_encoders, fetch_tokenizers


class BaseEncoder(nn.Module):

    def __init__(self,
                 backbone="clip",
                 embedding_dim=60,
                 nhist=1,
                 num_attn_heads=9,
                 num_vis_instr_attn_layers=2,
                 fps_subsampling_factor=5,
                 finetune_backbone=False,
                 finetune_text_encoder=False):
        super().__init__()
        self.subsampling_factor = fps_subsampling_factor
        self._backbone_name = backbone

        # Instruction encoder
        self.text_encoder, _dim = fetch_text_encoders(backbone)
        self.tokenizer = fetch_tokenizers(backbone)
        if self.text_encoder is not None:  # is None when using a VLM
            for p in self.text_encoder.parameters():
                p.requires_grad = finetune_text_encoder
            self.instruction_encoder = nn.Linear(_dim, embedding_dim)

        # Scene encoder
        self.backbone, self.normalize = fetch_visual_encoders(backbone)
        for p in self.backbone.parameters():
            p.requires_grad = finetune_backbone

        # Attention from vision to language
        if backbone == 'clip':
            self.vl_attention = AttentionModule(
                num_layers=num_vis_instr_attn_layers, d_model=embedding_dim,
                dim_fw=4 * embedding_dim, n_heads=num_attn_heads, pre_norm=False
            )

    def forward(self, rgb3d, rgb2d, pcd, instruction, proprio):
        """
        Encode different modalities, independent of denoising step.

        Args:
            - rgb3d: (B, ncam3d, 3, H, W)
            - rgb2d: (B, ncam2d, 3, H, W)
            - pcd: (B, ncam3d, 3, H, W)
            - instruction: (B, nt), tokens
            - proprio: (B, nhist, 3+6+X)

        Returns:
            - rgb3d_feats: (B, N, F)
            - pcd: (B, N, 3)
            - rgb2d_feats: (B, N2d, F)
            - rgb2d_pos: (B, N2d, 3)
            - instr_feats: (B, L, F)
            - instr_pos: (B, L, 3)
            - proprio_feats: (B, nhist, F)
            - fps_scene_feats: (B, n, F), n < N
            - fps_scene_pos: (B, n, 3), n < N
        """
        vl_enc_fn = {
            'clip': self.encode_clip,
        }[self._backbone_name]
        # Compute scene features/positional embeddings, language embeddings
        rgb3d_feats, rgb2d_feats, pcd, instr_feats = vl_enc_fn(
            rgb3d, rgb2d, pcd, instruction
        )
        rgb2d_pos = None

        # print(proprio.shape)
        # Use the current end-effector position as language 'position'
        # Handle flattened proprio: if it's 2D, reshape it back to 3D
        if len(proprio.shape) == 2:
            # Assume it was flattened from [B, nhist, nhand*action_dim]
            # For single arm, nhand=1, so we need to unflatten it
            # We know action_dim from the context, but let's assume a reasonable split
            B, flat_dim = proprio.shape
            # Try to infer nhist and action_dim
            # Common case: nhist=3, action_dim=8 for single arm
            if flat_dim % 8 == 0:
                nhist = flat_dim // 8
                proprio = proprio.reshape(B, nhist, 8)
            else:
                # Fallback: treat as single timestep
                proprio = proprio.unsqueeze(1)  # [B, 1, flat_dim]
        # if len(proprio.shape) == 4:
        #     proprio = proprio.flatten(1, 2)
        instr_pos = proprio[:, -1:, :3].repeat(1, instr_feats.size(1), 1)
        # print(instr_pos.shape)

        # Encode proprioception
        proprio_feats = self.encode_proprio(proprio, rgb3d_feats, pcd)

        # Point subsampling based on scene features
        fps_scene_feats, fps_scene_pos = self.run_dps(rgb3d_feats, pcd)

        return (
            rgb3d_feats, pcd,
            rgb2d_feats, rgb2d_pos,
            instr_feats, instr_pos,
            proprio_feats,
            fps_scene_feats, fps_scene_pos
        )

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
        return None

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
        return None, None, None, None

    def run_dps(self, features, pos):
        # features (B, Np, F)
        # context_pos (B, Np, 3)
        # outputs of analogous shape, with smaller Np
        if self.subsampling_factor == 1:
            return features, pos

        bs, npts, ch = features.shape
        sampled_inds = density_based_sampler(features, self.subsampling_factor)

        # Sample features
        expanded_inds = sampled_inds.unsqueeze(-1).expand(-1, -1, ch)  # B Np F
        sampled_features = torch.gather(features, 1, expanded_inds)

        # If positions are None, return
        if pos is None:
            return sampled_features, None

        # Else sample positions
        expanded_inds = sampled_inds.unsqueeze(-1).expand(-1, -1, 3)  # B Np 3
        sampled_pos = torch.gather(pos, 1, expanded_inds)
        return sampled_features, sampled_pos


@torch.no_grad()
def density_based_sampler(features, subsample_factor, k=8):
    """
    Args:
        features: Tensor of shape (B, N, C)
        subsample_factor: downsampling factor, e.g., 4 keeps 25% of the points
        k: number of neighbors to compute local density (default: 8)

    Returns:
        sampled_inds: LongTensor (B, N//factor) with sampled point indices
    """
    B, N, C = features.shape
    # (B, N, N) pairwise distances in feature space
    dists = torch.cdist(features, features, p=2)  # L2 distance

    # Get average distance to k nearest neighbors (as inverse density estimate)
    knn_dists, _ = dists.topk(k=k, dim=-1, largest=False)
    density = knn_dists.mean(dim=-1)  # (B, N), higher = more sparse

    # Choose top M points with highest avg distance (i.e. lowest density)
    M = int(N // subsample_factor)
    sampled_inds = density.topk(M, dim=-1, largest=True).indices  # (B, M)

    return sampled_inds


# For backward compatibility
Encoder = BaseEncoder
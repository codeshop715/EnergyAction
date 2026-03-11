import numpy as np
import torch


class RLBenchDepth2Cloud:

    def __init__(self, shape):
        self.uniforms = torch.from_numpy(
            self._create_uniform_pixel_coords_image(shape)
        ).permute(2, 0, 1).cuda(non_blocking=True)  # (3, H, W)

    @staticmethod
    def _create_uniform_pixel_coords_image(resolution):
        h, w = resolution
        u = np.tile(np.arange(w), (h, 1))
        v = np.tile(np.arange(h), (w, 1)).T
        ones = np.ones_like(u)
        return np.stack((u, v, ones), axis=-1)  # (H, W, 3)

    @staticmethod
    def _get_cam_proj_mat_inv_b(extrinsics, intrinsics):
        # extrinsics: (B, 4, 4), intrinsics: (B, 3, 3)
        C = extrinsics[:, :3, 3:]  # (B, 3, 1)
        R = extrinsics[:, :3, :3]  # (B, 3, 3)
        R_inv = R.transpose(1, 2)  # (B, 3, 3)
        t_inv = -torch.bmm(R_inv, C)  # (B, 3, 1)
        ext_inv = torch.cat([R_inv, t_inv], dim=-1)  # (B, 3, 4)
        cam_proj = torch.bmm(intrinsics, ext_inv)  # (B, 3, 4)
        cam_proj_homo = torch.cat([
            cam_proj,
            torch.tensor(
                [0, 0, 0, 1],
                dtype=cam_proj.dtype,
                device=cam_proj.device
            )[None, None].expand(cam_proj.size(0), 1, 4)
        ], dim=1)  # (B, 4, 4)
        cam_proj_inv = torch.linalg.inv(cam_proj_homo.float())[:, :3]
        return cam_proj_inv.to(cam_proj_homo.dtype)  # (B, 3, 4)

    def unproject(self, depth, extrinsics, intrinsics):
        # depth is (B, H, W), extrinsics (B, 4, 4), intrinsics (B, 3, 3)
        # output is (B, 3, H, W)
        b, h, w = depth.shape
        uv1 = self.uniforms[None]  # (1, 3, H, W)
        pc = uv1 * depth[:, None]  # (B, 3, H, W)
        pc = torch.cat([pc, torch.ones_like(pc[:, :1])], dim=1)  # (B, 4, H, W)
        pc = pc.reshape(b, 4, -1)  # (B, 4, HW)

        cam_proj_inv = self._get_cam_proj_mat_inv_b(extrinsics, intrinsics)
        world_pc = torch.bmm(cam_proj_inv, pc)  # (B, 3, HW)
        return world_pc.reshape(b, 3, h, w)

    def __call__(self, depth, extrinsics, intrinsics):
        # depth is (B Nc H W), extrinsics (B Nc 4 4), intrinsics (B Nc 3 3)
        # output is (B Nc 3 H W)
        # Nc is the number of cameras
        b, nc, h, w = depth.shape
        pc = self.unproject(
            depth.view(-1, h, w),
            extrinsics.view(-1, 4, 4),
            intrinsics.view(-1, 3, 3)
        )
        return pc.reshape(b, nc, 3, h, w)

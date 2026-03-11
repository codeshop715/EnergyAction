import torch

from .base import BaseTrainTester


class RLBenchTrainTester(BaseTrainTester):

    @torch.no_grad()
    def prepare_batch(self, sample, augment=False):
        # For bimanual case, data is already split in the dataset class
        if "left_action" in sample:
            sample["left_action"] = self.preprocessor.process_actions(
                sample["left_action"]
            )
            sample["right_action"] = self.preprocessor.process_actions(
                sample["right_action"]
            )
            sample["left_proprioception"] = self.preprocessor.process_proprio(
                sample["left_proprioception"]
            )
            sample["right_proprioception"] = self.preprocessor.process_proprio(
                sample["right_proprioception"]
            )
        else: # Standard single-arm case
            sample["action"] = self.preprocessor.process_actions(sample["action"])
            sample["proprioception"] = self.preprocessor.process_proprio(
                sample["proprioception"]
            )

        rgbs, pcds = self.preprocessor.process_obs(
            sample["rgb"], sample.get("rgb2d"),
            sample.get("depth"), sample.get("extrinsics"), sample.get("intrinsics"),
            augment=augment
        )
        # Create action_mask with proper dimensions
        if "left_action" in sample:
            # For bimanual: mask shape should be (B, T, 1) for single-arm models
            # left_action shape: (B, T, 8) -> mask shape: (B, T, 1)
            batch_size, traj_len = sample["left_action"].shape[:2]
            action_mask = torch.zeros(
                (batch_size, traj_len, 1),
                dtype=bool,
                device='cuda'
            )
        else:
            # For single-arm: mask shape (B, T) is fine
            action_mask = torch.zeros(
                sample["action"].shape[:-1],
                dtype=bool,
                device='cuda'
            )
        
        sample.update({
            "rgb": rgbs,
            "pcd": pcds,
            # Placeholder for action mask, can be generated inside the model if needed
            "action_mask": action_mask,
        })

        # Tokenization will be handled in the training loop
        return sample

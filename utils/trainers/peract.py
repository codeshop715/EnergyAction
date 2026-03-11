import torch

from .base import BaseTrainTester


class PeractTrainTester(BaseTrainTester):

    @torch.no_grad()
    def prepare_batch(self, sample, augment=False):
        # Check if this is a bimanual dataset by looking for split action keys
        if "left_action" in sample and "right_action" in sample:
            # Bimanual case
            sample["left_action"] = self.preprocessor.process_actions(sample["left_action"])
            sample["right_action"] = self.preprocessor.process_actions(sample["right_action"])
            
            # Process proprioception for both arms
            sample["left_proprioception"] = self.preprocessor.process_proprio(sample["left_proprioception"])
            sample["right_proprioception"] = self.preprocessor.process_proprio(sample["right_proprioception"])
            
            # Process shared observations
            rgbs, pcds = self.preprocessor.process_obs(
                sample["rgb"], sample["pcd"],
                augment=augment
            )
            

            # Create trajectory masks for both arms
            # For bimanual with single-arm models: left_action shape (B, T, 8) -> mask shape (B, T, 1)
            # The extra dimension is needed because each arm's DenoiseActor expects (B, T, nhand=1) mask
            batch_size, traj_len = sample["left_action"].shape[:2]
            left_action_mask = torch.zeros((batch_size, traj_len, 1), dtype=bool, device='cuda')
            right_action_mask = torch.zeros((batch_size, traj_len, 1), dtype=bool, device='cuda')
            
            # Return as dictionary for bimanual case
            return {
                "left_action": sample["left_action"],
                "right_action": sample["right_action"],
                "trajectory_mask": left_action_mask,  # Use left mask as default
                "left_proprioception": sample["left_proprioception"],
                "right_proprioception": sample["right_proprioception"],
                "rgb3d": rgbs,
                "rgb2d": None,
                "pcd": pcds,
                "instr": sample["instr"],
                "instruction": sample["instr"]  # Add both keys for compatibility
            }
        else:
            # Regular single-arm case
            sample["action"] = self.preprocessor.process_actions(sample["action"])
            proprio = self.preprocessor.process_proprio(sample["proprioception"])
            rgbs, pcds = self.preprocessor.process_obs(
                sample["rgb"], sample["pcd"],
                augment=augment
            )
            return (
                sample["action"],
                torch.zeros(sample["action"].shape[:-1], dtype=bool, device='cuda'),
                rgbs,
                None,
                pcds,
                sample["instr"],
                proprio
            )

import json
import numpy as np

from torch.utils.data import Dataset

from .utils import to_tensor, read_zarr_with_cache, to_relative_action


class BaseDataset(Dataset):
    """Base dataset."""

    def __init__(
        self,
        root,  # the directory path of the dataset
        instructions,  # path to instruction file
        copies=None,  # copy the dataset for less loader restarts
        relative_action=False,  # whether to return relative actions
        mem_limit=8,  # cache limit per dataset class in GigaBytes
        actions_only=False,  # return actions without observations
        chunk_size=4,  # chunk size for zarr
        max_demos_per_task=-1  # limit demos per task (-1 = use all)
    ):
        super().__init__()
        self.copies = self.train_copies if copies is None else copies
        self._relative_action = relative_action
        self._actions_only = actions_only
        self.chunk_size = chunk_size
        self.max_demos_per_task = max_demos_per_task

        # Load instructions
        self._instructions = self._load_instructions(instructions)

        # Load all annotations lazily
        self.annos = read_zarr_with_cache(root, mem_gb=mem_limit)
        # Sanity check
        len_ = len(self.annos['action'])
        for key in self.annos:
            assert len(self.annos[key]) == len_, f'length mismatch in {key}'
        print(f"Found {len(self.annos['action'])} samples")
        
        # Initialize sampled indices to None (use all data by default)
        self._sampled_indices = None
        
        # Apply demo sampling if requested
        if self.max_demos_per_task > 0:
            self._sample_demos_per_task()

    def _load_instructions(self, instruction_file):
        return json.load(open(instruction_file))
    
    def _sample_demos_per_task(self):
        """Sample max_demos_per_task demos from each task (approximation based on keyposes)."""
        import numpy as np
        
        # Get task IDs
        task_ids = np.array(self.annos['task_id'])
        unique_tasks = np.unique(task_ids)
        
        print(f"Sampling approximately {self.max_demos_per_task} demos per task...")
        
        # Calculate sampling ratio: 20 demos out of 100 = 0.2
        # Since we don't have episode_id, we sample keyposes proportionally
        total_samples_before = len(task_ids)
        selected_indices = []
        
        np.random.seed(42)  # For reproducibility
        
        for task_id in unique_tasks:
            # Find all keyposes for this task
            task_mask = (task_ids == task_id)
            task_indices = np.where(task_mask)[0]
            n_keyposes = len(task_indices)
            
            # Estimate average demos in original dataset (assume 100 demos/task)
            estimated_original_demos = 100
            estimated_keyposes_per_demo = n_keyposes / estimated_original_demos
            
            # Calculate target number of keyposes for max_demos_per_task
            target_keyposes = int(self.max_demos_per_task * estimated_keyposes_per_demo)
            target_keyposes = max(1, min(target_keyposes, n_keyposes))  # Clamp to valid range
            
            # Random sample
            sampled = np.random.choice(task_indices, size=target_keyposes, replace=False)
            selected_indices.extend(sampled)
            
            ratio = target_keyposes / n_keyposes
            print(f"  Task {task_id}: {n_keyposes} -> {target_keyposes} keyposes ({ratio*100:.1f}%)")
        
        # Sort indices to maintain temporal order
        selected_indices = sorted(selected_indices)
        
        # Create a new dictionary with sampled data
        # We can't modify zarr Group directly, so we read the data and create numpy arrays
        sampled_annos = {}
        for key in self.annos.keys():
            original_data = self.annos[key]
            # For zarr arrays, we need to read the data first, then index it
            if hasattr(original_data, 'shape'):  # zarr array or numpy array
                # Read only the selected indices and convert to numpy array
                sampled_annos[key] = np.array(original_data[selected_indices])
            else:
                # For lists or other types
                sampled_annos[key] = [original_data[i] for i in selected_indices]
        
        # Replace self.annos with the sampled data
        self.annos = sampled_annos
        
        total_samples_after = len(selected_indices)
        print(f"Dataset reduced: {total_samples_before} -> {total_samples_after} samples ({total_samples_after/total_samples_before*100:.1f}%)")

    def _get_attr_by_idx(self, idx, attr, filter_cam=False):
        t = to_tensor(self.annos[attr][idx:idx + self.chunk_size])
        if filter_cam and self.camera_inds is not None:
            t = t[:, self.camera_inds]
        return t

    def _get_task(self, idx):
        return ["task"] * self.chunk_size

    def _get_instr(self, idx):
        return ["instruction"] * self.chunk_size

    def _get_rgb(self, idx, key='rgb'):
        return self._get_attr_by_idx(idx, key, True)

    def _get_depth(self, idx, key='depth'):
        return self._get_attr_by_idx(idx, key, True)

    def _get_proprioception(self, idx):
        return self._get_attr_by_idx(idx, 'proprioception', False)

    def _get_action(self, idx):
        if self._relative_action:
            if 'rel_action' in self.annos:
                return self._get_attr_by_idx(idx, 'rel_action', False)
            else:
                action = self._get_attr_by_idx(idx, 'action', False)
                prop = self._get_proprioception(idx)[[-1]]
                action = to_relative_action(action, prop, self.quat_format)
        else:
            action = self._get_attr_by_idx(idx, 'action', False)
        return action

    def __getitem__(self, idx):
        """
        self.annos: {
            action: (N, T, 8) float
            depth: (N, n_cam, H, W) float16
            proprioception: (N, nhist, 8) float
            rgb: (N, n_cam, 3, H, W) uint8
        }
        In addition self.annos may contain fields for task/instruction ids
        """
        # First detect which copy we fall into
        idx = idx % (len(self.annos['action']) // self.chunk_size)
        # and then which chunk
        idx = idx * self.chunk_size
        if self._actions_only:
            return {"action": self._get_action(idx)}
        return {
            "task": self._get_task(idx),
            "instr": self._get_instr(idx),  # [str]
            "rgb": self._get_rgb(idx),  # tensor(n_cam, 3, H, W)
            "depth": self._get_depth(idx),  # tensor(n_cam, H, W)
            "proprioception": self._get_proprioception(idx),  # tensor(1, 8)
            "action": self._get_action(idx)  # tensor(T, 8)
        }

    def __len__(self):
        return self.copies * (len(self.annos['action']) // self.chunk_size)

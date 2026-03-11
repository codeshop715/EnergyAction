class DataPreprocessor:

    def __init__(self, keypose_only=False, num_history=1,
                 custom_imsize=None, depth2cloud=None):
        self.keypose_only = keypose_only
        self.num_history = num_history
        self.custom_imsize = custom_imsize
        self.depth2cloud = depth2cloud

    def process_actions(self, actions):
        """Action shape: (B, T, nhand, 3+rot+1)."""
        actions = actions.cuda(non_blocking=True)
        # print(f"[process_actions] actions.shape: {actions.shape}")
        # if len(actions.shape) == 3:  # (B, T, action_dim)
        #     actions = actions.unsqueeze(2)  # → (B, T, 1, action_dim)
        if self.keypose_only:
            actions = actions[:, [-1]]
        return actions

    def process_proprio(self, proprio):
        """Proprio shape: (B, nhist, nhand, 3+rot+1)."""
        proprio = proprio.cuda(non_blocking=True)
        nhist_ = proprio.size(1)
        assert nhist_ >= self.num_history, "not enough proprio timesteps"
        # the first proprio is the current state
        proprio = proprio[:, :max(self.num_history, 1)]
        return proprio

    def process_obs(self, rgbs, pcds):
        pass

def compute_metrics(pred, gt):
    # pred/gt are (B, L, 3+rot+1)
    pos_l2 = ((pred[..., :3] - gt[..., :3]) ** 2).sum(-1).sqrt()
    # symmetric quaternion eval
    quat_l1 = (pred[..., 3:-1] - gt[..., 3:-1]).abs().sum(-1)
    quat_l1_ = (pred[..., 3:-1] + gt[..., 3:-1]).abs().sum(-1)
    select_mask = (quat_l1 < quat_l1_).float()
    quat_l1 = (select_mask * quat_l1 + (1 - select_mask) * quat_l1_)
    # gripper openess
    openess = ((pred[..., -1:] >= 0.5) == (gt[..., -1:] >= 0.5)).bool()
    tr = 'traj_'

    # Trajectory metrics
    ret_1, ret_2 = {
        tr + 'pos_l2': pos_l2.mean(),
        tr + 'pos_acc_001': (pos_l2 < 0.01).float().mean(),
        tr + 'rot_l1': quat_l1.mean(),
        tr + 'rot_acc_0025': (quat_l1 < 0.025).float().mean(),
        tr + 'gripper': openess.flatten().float().mean()
    }, {
        tr + 'pos_l2': pos_l2.mean(-1),
        tr + 'pos_acc_001': (pos_l2 < 0.01).float().mean(-1),
        tr + 'rot_l1': quat_l1.mean(-1),
        tr + 'rot_acc_0025': (quat_l1 < 0.025).float().mean(-1)
    }

    return ret_1, ret_2

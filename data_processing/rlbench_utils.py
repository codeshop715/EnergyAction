import os
import pickle

import numpy as np
from scipy.interpolate import CubicSpline, interp1d
from scipy.spatial.transform import Rotation as R


def image_to_float_array(image, scale_factor):
  """Recovers the depth values from an image.

  Reverses the depth to image conversion performed by FloatArrayToRgbImage or
  FloatArrayToGrayImage.

  The image is treated as an array of fixed point depth values.  Each
  value is converted to float and scaled by the inverse of the factor
  that was used to generate the Image object from depth values.  If
  scale_factor is specified, it should be the same value that was
  specified in the original conversion.

  The result of this function should be equal to the original input
  within the precision of the conversion.

  Args:
    image: Depth image output of FloatArrayTo[Format]Image.
    scale_factor: Fixed point scale factor.

  Returns:
    A 2D floating point numpy array representing a depth image.

  """
  image_array = np.array(image)
  image_shape = image_array.shape

  channels = image_shape[2] if len(image_shape) > 2 else 1
  assert 2 <= len(image_shape) <= 3
  if channels == 3:
    # RGB image needs to be converted to 24 bit integer.
    float_array = np.sum(image_array * [65536, 256, 1], axis=2)
  else:
    float_array = image_array.astype(np.float32)
  scaled_array = float_array / scale_factor
  return scaled_array


def _is_stopped(demo, i, obs, delta=0.1):
    next_is_not_final = i == (len(demo) - 2)
    gripper_state_no_change = i < (len(demo) - 2) and (
        obs.gripper_open == demo[i + 1].gripper_open
        and obs.gripper_open == demo[max(0, i - 1)].gripper_open
        and demo[max(0, i - 2)].gripper_open == demo[max(0, i - 1)].gripper_open
    )
    small_delta = np.allclose(obs.joint_velocities, 0, atol=delta)
    return (
        small_delta
        and (not next_is_not_final)
        and gripper_state_no_change
    )


def _is_stopped_right(demo, i, obs, delta=0.1):
    next_is_not_final = i == (len(demo) - 2)
    gripper_state_no_change = i < (len(demo) - 2) and (
        obs.gripper_open == demo[i + 1].right.gripper_open
        and obs.gripper_open == demo[max(0, i - 1)].right.gripper_open
        and demo[max(0, i - 2)].right.gripper_open == demo[max(0, i - 1)].right.gripper_open
    )
    small_delta = np.allclose(obs.joint_velocities, 0, atol=delta)
    return small_delta and (not next_is_not_final) and gripper_state_no_change


def _is_stopped_left(demo, i, obs, delta=0.1):
    next_is_not_final = i == (len(demo) - 2)
    gripper_state_no_change = i < (len(demo) - 2) and (
        obs.gripper_open == demo[i + 1].left.gripper_open
        and obs.gripper_open == demo[max(0, i - 1)].left.gripper_open
        and demo[max(0, i - 2)].left.gripper_open == demo[max(0, i - 1)].left.gripper_open
    )
    small_delta = np.allclose(obs.joint_velocities, 0, atol=delta)
    return small_delta and (not next_is_not_final) and gripper_state_no_change


def _keypoint_discovery_bimanual(demo, stopping_delta=0.1):
    episode_keypoints = []
    right_prev_gripper_open = demo[0].right.gripper_open
    left_prev_gripper_open = demo[0].left.gripper_open
    stopped_buffer = 0
    for i, obs in enumerate(demo._observations):
        right_stopped = _is_stopped_right(demo, i, obs.right, stopping_delta)
        left_stopped = _is_stopped_left(demo, i, obs.left, stopping_delta)
        stopped = (stopped_buffer <= 0) and right_stopped and left_stopped
        stopped_buffer = 4 if stopped else stopped_buffer - 1
        # if change in gripper, or end of episode.
        last = i == (len(demo) - 1)
        right_state_changed = obs.right.gripper_open != right_prev_gripper_open
        left_state_changed = obs.left.gripper_open != left_prev_gripper_open
        state_changed = right_state_changed or left_state_changed
        if i != 0 and (state_changed or last or stopped):
            episode_keypoints.append(i)

        right_prev_gripper_open = obs.right.gripper_open
        left_prev_gripper_open = obs.left.gripper_open
    if (
        len(episode_keypoints) > 1
        and (episode_keypoints[-1] - 1) == episode_keypoints[-2]
    ):
        episode_keypoints.pop(-2)
    # print("Found %d keypoints." % len(episode_keypoints), episode_keypoints)
    return episode_keypoints


def _keypoint_discovery_unimanual(demo, stopping_delta=0.1):
    episode_keypoints = []
    prev_gripper_open = demo[0].gripper_open
    stopped_buffer = 0
    for i, obs in enumerate(demo):
        stopped = _is_stopped(demo, i, obs, stopping_delta)
        stopped = (stopped_buffer <= 0) and stopped
        stopped_buffer = 4 if stopped else stopped_buffer - 1
        # if change in gripper, or end of episode.
        last = i == (len(demo) - 1)
        if i != 0 and (obs.gripper_open != prev_gripper_open or last or stopped):
            episode_keypoints.append(i)
        prev_gripper_open = obs.gripper_open
    if (
        len(episode_keypoints) > 1
        and (episode_keypoints[-1] - 1) == episode_keypoints[-2]
    ):
        episode_keypoints.pop(-2)
    # print("Found %d keypoints." % len(episode_keypoints), episode_keypoints)
    return episode_keypoints


def _keypoint_discovery_heuristic(demo, stopping_delta=0.1, bimanual=False):
    if bimanual:
        return _keypoint_discovery_bimanual(demo, stopping_delta)
    else:
        return _keypoint_discovery_unimanual(demo, stopping_delta)


def keypoint_discovery(demo, method="heuristic", bimanual=False):
    episode_keypoints = []
    if method == "heuristic":
        stopping_delta = 0.1
        return _keypoint_discovery_heuristic(demo, stopping_delta, bimanual)

    elif method == "random":
        # Randomly select keypoints.
        episode_keypoints = np.random.choice(
            range(len(demo)),
            size=20, replace=False
        )
        episode_keypoints.sort()
        return episode_keypoints

    elif method == "fixed_interval":
        # Fixed interval.
        episode_keypoints = []
        segment_length = len(demo) // 20
        for i in range(0, len(demo), segment_length):
            episode_keypoints.append(i)
        return episode_keypoints

    else:
        raise NotImplementedError


def _store_instructions(root, task, splits):
    # both root and path are strings
    var2text = {}
    for split in splits:
        folder = f'{root}/{split}/{task}/all_variations/episodes'
        eps = {ep for ep in os.listdir(folder) if ep.startswith('ep')}
        for ep in eps:
            # Load variation
            with open(f'{folder}/{ep}/variation_number.pkl', 'rb') as f:
                var_ = pickle.load(f)
            if var_ in var2text:
                continue
            # Read different descriptions
            with open(f'{folder}/{ep}/variation_descriptions.pkl', 'rb') as f:
                var2text[var_] = pickle.load(f)
    return var2text


def store_instructions(root, task_list, splits=['train', 'val']):
    # {task: {var: [text]}}
    return {task: _store_instructions(root, task, splits) for task in task_list}


def interpolate_trajectory(traj, num_steps):
    """
    Vectorized 6D interpolation.
    traj: (T, D)
    Returns: (num_steps, D)
    """
    T = traj.shape[0]
    if T == 1:
        return np.repeat(traj, num_steps, axis=0)

    t_orig = np.linspace(0, 1, T)
    t_target = np.linspace(0, 1, num_steps)

    interp_fn = CubicSpline(t_orig, traj[..., :-1], axis=0)
    main_ = interp_fn(t_target)
    interp_fn = interp1d(t_orig, traj[..., -1:], axis=0, kind='nearest')
    last_ = interp_fn(t_target)

    return np.concatenate((main_, last_), 1)


def quat_to_euler_np(quats, order='xyz', degrees=False):
    """
    Convert Nx4 numpy array of quaternions to Nx3 Euler angles.
    Input: quats of shape (N, 4) in [x, y, z, w] order
    Output: angles of shape (N, 3)
    """
    r = R.from_quat(quats)
    return r.as_euler(order, degrees=degrees)


def euler_to_quat_np(eulers, order='xyz', degrees=False):
    """
    Convert Nx3 numpy array of Euler angles to Nx4 quaternions.
    Input: eulers of shape (N, 3)
    Output: quats of shape (N, 4) in [x, y, z, w] order
    """
    r = R.from_euler(order, eulers, degrees=degrees)
    return r.as_quat()

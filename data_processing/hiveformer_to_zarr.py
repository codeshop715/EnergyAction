import argparse
import json
import os
import pickle

from numcodecs import Blosc
import zarr
import numpy as np
from PIL import Image
from tqdm import tqdm

from data_processing.rlbench_utils import (
    keypoint_discovery,
    image_to_float_array,
    quat_to_euler_np,
    interpolate_trajectory,
    euler_to_quat_np
)
from utils.common_utils import str2bool


NCAM = 2
NHAND = 1
IM_SIZE = 256
DEPTH_SCALE = 2**24 - 1
INTERP_LEN = 50


def parse_arguments():
    parser = argparse.ArgumentParser()
    # Tuples: (name, type, default)
    arguments = [
        ('root', str, '/data/group_data/katefgroup/datasets/hiveformer_clean/'),
        ('tgt', str, '/data/user_data/ngkanats/zarr_datasets/hiveformer/'),
        ('store_trajectory', str2bool, True),
        ('split', str, 'train')
    ]
    for arg in arguments:
        parser.add_argument(f'--{arg[0]}', type=arg[1], default=arg[2])

    return parser.parse_args()


def all_tasks_main(task, store_trajectory, split):
    # Check if the zarr already exists
    filename = f"{STORE_PATH}/{task}/{split}.zarr"
    if os.path.exists(filename):
        print(f"Zarr file {filename} already exists. Skipping...")
        return None

    cameras = ["wrist", "front"]

    # Initialize zarr
    compressor = Blosc(cname='lz4', clevel=1, shuffle=Blosc.SHUFFLE)
    with zarr.open_group(filename, mode="w") as zarr_file:

        def _create(field, shape, dtype):
            zarr_file.create_dataset(
                field,
                shape=(0,) + shape,
                chunks=(1,) + shape,
                compressor=compressor,
                dtype=dtype
            )

        _create("rgb", (NCAM, 3, IM_SIZE, IM_SIZE), "uint8")
        _create("depth", (NCAM, IM_SIZE, IM_SIZE), "float16")
        _create("proprioception", (3, NHAND, 8), "float32")
        _create(
            "action",
            (1 if not store_trajectory else INTERP_LEN, NHAND, 8),
            "float32"
        )
        _create("extrinsics", (NCAM, 4, 4), "float16")
        _create("intrinsics", (NCAM, 3, 3), "float16")
        _create("task_id", (), "uint8")
        _create("variation", (), "uint8")

        # Loop through episodes
        variations = os.listdir(f'{ROOT}/{task}/')
        for var_name in variations:
            var_int = var_name[len('variation'):]
            ep_folder = f'{ROOT}/{task}/{var_name}/episodes'
            episodes = sorted([
                ep for ep in os.listdir(ep_folder) if ep.startswith('episode')]
            )
            for ep in tqdm(episodes):
                # Read low-dim file from RLBench
                ld_file = f"{ep_folder}/{ep}/low_dim_obs.pkl"
                with open(ld_file, 'rb') as f:
                    demo = pickle.load(f)

                # Keypose discovery
                key_frames = keypoint_discovery(demo, bimanual=False)
                key_frames.insert(0, 0)

                # Loop through keyposes and store:
                # RGB (keyframes, cameras, 3, 256, 256)
                rgb = np.stack([
                    np.stack([
                        np.array(Image.open(
                            f"{ep_folder}/{ep}/{cam}_rgb/{k}.png"
                        ))
                        for cam in cameras
                    ])
                    for k in key_frames[:-1]
                ])
                rgb = rgb.transpose(0, 1, 4, 2, 3)

                # Depth (keyframes, cameras, 256, 256)
                depth_list = []
                for k in key_frames[:-1]:
                    cam_d = []
                    for cam in cameras:
                        depth = image_to_float_array(Image.open(
                            f"{ep_folder}/{ep}/{cam}_depth/{k}.png"
                        ), DEPTH_SCALE)
                        near = demo[k].misc[f'{cam}_camera_near']
                        far = demo[k].misc[f'{cam}_camera_far']
                        depth = near + depth * (far - near)
                        cam_d.append(depth)
                    depth_list.append(np.stack(cam_d).astype(np.float16))
                depth = np.stack(depth_list)

                # Proprioception (keyframes, 3, 1, 8)
                states = np.stack([np.concatenate([
                    demo[k].gripper_pose, [demo[k].gripper_open]
                ]) for k in key_frames]).astype(np.float32)
                # Store current eef pose as well as two previous ones
                prop = states[:-1]
                prop_1 = np.concatenate([prop[:1], prop[:-1]])
                prop_2 = np.concatenate([prop_1[:1], prop_1[:-1]])
                prop = np.concatenate([prop_2, prop_1, prop], 1)
                prop = prop.reshape(len(prop), 3, NHAND, 8)

                # Action (keyframes, 1, 1, 8)
                if not store_trajectory:
                    actions = states[1:].reshape(len(states[1:]), 1, NHAND, 8)
                else:
                    states = np.stack([np.concatenate([
                        demo[k].gripper_pose, [demo[k].gripper_open]
                    ]) for k in np.arange(len(demo))]).astype(np.float32)
                    actions = np.ascontiguousarray([
                        _interpolate(states[prev:next_ + 1], INTERP_LEN)
                        for prev, next_ in zip(key_frames[:-1], key_frames[1:])
                    ])
                    actions = actions.reshape(-1, INTERP_LEN, NHAND, 8)

                # Extrinsics (keyframes, cameras, 4, 4)
                extrinsics = np.stack([
                    np.stack([
                        demo[k].misc[f'{cam}_camera_extrinsics'].astype(np.float16)
                        for cam in cameras
                    ])
                    for k in key_frames[:-1]
                ])

                # Intrinsics (keyframes, cameras, 3, 3)
                intrinsics = np.stack([
                    np.stack([
                        demo[k].misc[f'{cam}_camera_intrinsics'].astype(np.float16)
                        for cam in cameras
                    ])
                    for k in key_frames[:-1]
                ])

                # Task id (keyframes,)
                task_id = np.array([0] * len(key_frames[:-1]))
                task_id = task_id.astype(np.uint8)

                # Variation (keyframes,)
                var_ = np.array([int(var_int)] * len(key_frames[:-1]))
                var_ = var_.astype(np.uint8)

                # Write
                zarr_file['rgb'].append(rgb)
                zarr_file['depth'].append(depth)
                zarr_file['proprioception'].append(prop)
                zarr_file['action'].append(actions)
                zarr_file['extrinsics'].append(extrinsics)
                zarr_file['intrinsics'].append(intrinsics)
                zarr_file['task_id'].append(task_id)
                zarr_file['variation'].append(var_)


def _interpolate(traj, num_steps):
    # Convert to Euler
    traj = np.concatenate((
        traj[:, :3],
        quat_to_euler_np(traj[:, 3:7]),
        traj[:, 7:]
    ), 1)
    # Interpolate
    traj = interpolate_trajectory(traj, num_steps)
    # Convert to quaternion
    traj = np.concatenate((
        traj[:, :3],
        euler_to_quat_np(traj[:, 3:6]),
        traj[:, 6:]
    ), 1)
    return traj


def store_instructions(root, task):
    # both root and path are strings
    var2text = {}
    task_folder = f'{root}/{task}/'
    for var_f in os.listdir(task_folder):
        if var_f.endswith('pkl'):
            continue
        int_var_ = int(var_f[len('variation'):])
        if int_var_ in var2text:
            continue
        # Read different descriptions
        with open(f'{root}/{task}/{var_f}/variation_descriptions.pkl', 'rb') as f:
            var2text[int_var_] = pickle.load(f)
    return var2text


if __name__ == "__main__":
    tasks = ["close_door",]
    args = parse_arguments()
    ROOT = f'{args.root}/{args.split}/'
    STORE_PATH = args.tgt
    # Create zarr data
    for task in tasks:
        print(task)
        all_tasks_main(task, args.store_trajectory, args.split)
        # Store instructions as json
        os.makedirs('instructions/hiveformer', exist_ok=True)
        instr_dict = {task: store_instructions(ROOT, task)}
        with open(f'instructions/hiveformer/{task}.json', 'w') as fid:
            json.dump(instr_dict, fid)

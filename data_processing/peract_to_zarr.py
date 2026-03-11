import argparse
import json
import os
from pathlib import Path
import pickle

import blosc
from numcodecs import Blosc
import zarr
import numpy as np
import torch
from tqdm import tqdm

from data_processing.rlbench_utils import store_instructions


STORE_EVERY = 1
NCAM = 4
NHAND = 1
IM_SIZE = 256


def parse_arguments():
    parser = argparse.ArgumentParser()
    # Tuples: (name, type, default)
    arguments = [
        ('root', str, '/data/group_data/katefgroup/VLA/Peract_packaged/'),
        ('tgt', str, '/data/user_data/ngkanats/zarr_datasets/Peract_dat_zarr/')
    ]
    for arg in arguments:
        parser.add_argument(f'--{arg[0]}', type=arg[1], default=arg[2])

    return parser.parse_args()


def to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.numpy()
    elif isinstance(x, np.ndarray):
        return x
    else:
        return np.array(x)


def all_tasks_main(split, tasks):
    # Check if the zarr already exists
    filename = f"{STORE_PATH}/{split}.zarr"
    if os.path.exists(filename):
        print(f"Zarr file {filename} already exists. Skipping...")
        return None

    cameras = ["left_shoulder", "right_shoulder", "wrist", "front"]
    task2id = {task: t for t, task in enumerate(tasks)}

    # Initialize zarr
    compressor = Blosc(cname='lz4', clevel=1, shuffle=Blosc.SHUFFLE)
    with zarr.open_group(filename, mode="w") as zarr_file:

        def _create(field, shape, dtype):
            zarr_file.create_dataset(
                field,
                shape=(0,) + shape,
                chunks=(STORE_EVERY,) + shape,
                compressor=compressor,
                dtype=dtype
            )

        _create("rgb", (NCAM, 3, IM_SIZE, IM_SIZE), "uint8")
        _create("pcd", (NCAM, 3, IM_SIZE, IM_SIZE), "float16")
        _create("proprioception", (3, NHAND, 8), "float32")
        _create("action", (1, NHAND, 8), "float32")
        _create("task_id", (), "uint8")
        _create("variation", (), "uint8")

        # Loop through episodes
        for task in tasks:
            print(task)
            episodes = []
            for var in range(0, 199):
                _path = Path(f'{ROOT}{split}/{task}+{var}/')
                if not _path.is_dir():
                    continue
                episodes.extend([
                    (ep, var) for ep in sorted(_path.glob("*.dat"))
                ])
            for ep, var in tqdm(episodes):
                # Read
                with open(ep, "rb") as f:
                    content = pickle.loads(blosc.decompress(f.read()))
                # Map [-1, 1] to [0, 255] uint8
                rgb = (127.5 * (content[1][:, :, 0] + 1)).astype(np.uint8)
                # Keep point cloud as it's hard to reverse
                pcd = content[1][:, :, 1].astype(np.float16)
                # Store current eef pose as well as two previous ones
                prop = np.stack([
                    to_numpy(tens).astype(np.float32) for tens in content[4]
                ])
                prop_1 = np.concatenate([prop[:1], prop[:-1]])
                prop_2 = np.concatenate([prop_1[:1], prop_1[:-1]])
                prop = np.concatenate([prop_2, prop_1, prop], 1)
                prop = prop.reshape(len(prop), 3, NHAND, 8)
                # Next keypose (concatenate curr eef to form a "trajectory")
                actions = np.stack([
                    to_numpy(tens).astype(np.float32) for tens in content[2]
                ]).reshape(len(content[2]), 1, NHAND, 8)
                # Task ids and variation ids
                tids = np.array([task2id[task]] * len(content[0])).astype(np.uint8)
                _vars = np.array([var] * len(content[0])).astype(np.uint8)

                # write
                zarr_file['rgb'].append(rgb)
                zarr_file['pcd'].append(pcd)
                zarr_file['proprioception'].append(prop)
                zarr_file['action'].append(actions)
                zarr_file['task_id'].append(tids)
                zarr_file['variation'].append(_vars)
                assert all(
                    len(zarr_file['action']) == len(zarr_file[key])
                    for key in zarr_file.keys()
                )


if __name__ == "__main__":
    tasks = [
        "place_cups", "close_jar", "insert_onto_square_peg",
        "light_bulb_in", "meat_off_grill", "open_drawer",
        "place_shape_in_shape_sorter", "place_wine_at_rack_location",
        "push_buttons", "put_groceries_in_cupboard",
        "put_item_in_drawer", "put_money_in_safe", "reach_and_drag",
        "slide_block_to_color_target", "stack_blocks", "stack_cups",
        "sweep_to_dustpan_of_size", "turn_tap"
    ]
    args = parse_arguments()
    ROOT = args.root
    STORE_PATH = args.tgt
    # Create zarr data
    for split in ['train', 'val']:
        all_tasks_main(split, tasks)
    # Store instructions as json (can be run independently)
    os.makedirs('instructions/peract', exist_ok=True)
    instr_dict = store_instructions(ROOT, tasks, ['train', 'val', 'test'])
    with open('instructions/peract/instructions.json', 'w') as fid:
        json.dump(instr_dict, fid)

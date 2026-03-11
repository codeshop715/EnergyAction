import argparse
import os

import numpy as np
import zarr
from numcodecs import Blosc
from tqdm import tqdm

from utils.common_utils import str2bool


def parse_arguments():
    parser = argparse.ArgumentParser()
    # Tuples: (name, type, default)
    arguments = [
        # Dataset/loader arguments
        ('src', str, '/data/user_data/ngkanats/zarr_datasets/Peract2_dense_zarr/train.zarr'),
        ('tgt', str, '/data/user_data/ngkanats/zarr_datasets/Peract2_dense_zarr/train_rechunked4.zarr'),
        ('chunk_size', int, 4),
        ('shuffle', str2bool, False)
    ]
    for arg in arguments:
        parser.add_argument(f'--{arg[0]}', type=arg[1], default=arg[2])

    return parser.parse_args()


def rechunk_zarr_group(
    old_zarr_path,
    new_zarr_path,
    chunk_size=4,
    shuffle=False,
    compressor=Blosc(cname='lz4', clevel=1, shuffle=Blosc.SHUFFLE)
):
    # Load old Zarr group (read-only)
    old_group = zarr.open_group(old_zarr_path, mode='r')
    if shuffle:
        inds = np.random.permutation(len(old_group['action']))
    else:
        inds = np.arange(len(old_group['action']))

    # Create new Zarr group (overwrite if exists)
    if os.path.exists(new_zarr_path):
        print(f"Deleting existing {new_zarr_path}")
        import shutil
        shutil.rmtree(new_zarr_path)

    new_group = zarr.open_group(new_zarr_path, mode='w')

    # Copy datasets with new chunking & compression
    for array_name in old_group.array_keys():
        old_array = old_group[array_name]
        shape = old_array.shape
        dtype = old_array.dtype

        # Choose chunk shape: match all dims except dim 0, set to chunk_size
        chunk_shape = (min(chunk_size, shape[0]),) + shape[1:]

        print(f"Rechunking {array_name} | shape={shape}, chunks={chunk_shape}")

        new_array = new_group.create_dataset(
            name=array_name,
            shape=shape,
            dtype=dtype,
            chunks=chunk_shape,
            compressor=compressor,
            overwrite=True,
        )

        # Copy over data in chunks (from old_array)
        for i in tqdm(range(0, shape[0], chunk_size), desc=f"Copying {array_name}"):
            end = min(i + chunk_size, shape[0])
            new_array[i:end] = old_array[inds[i:end]]

    print("✅ Rechunking complete.")


if __name__ == '__main__':
    args = parse_arguments()
    rechunk_zarr_group(
        old_zarr_path=args.src,
        new_zarr_path=args.tgt,
        chunk_size=args.chunk_size,
        shuffle=args.shuffle
    )

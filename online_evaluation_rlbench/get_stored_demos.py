import pickle
from os import listdir
from os.path import join, exists

import numpy as np
from natsort import natsorted

from rlbench.backend.const import *


def get_stored_demos(amount=1,
                     dataset_root='/data/group_data/katefgroup/VLA/peract_test/',
                     variation_number=0,
                     task_name='close_jar',
                     random_selection=False,
                     from_episode_number=0):

    task_root = join(dataset_root, task_name)
    if not exists(task_root):
        raise RuntimeError("Can't find the demos for %s at: %s" % (
            task_name, task_root))

    if variation_number == -1:
        # All variations
        examples_path = join(
            task_root, VARIATIONS_ALL_FOLDER,
            EPISODES_FOLDER)
        examples = listdir(examples_path)
    else:
        # Sample an amount of examples for the variation of this task
        examples_path = join(
            task_root, VARIATIONS_FOLDER % variation_number,
            EPISODES_FOLDER)
        examples = listdir(examples_path)

    # hack: ignore .DS_Store files from macOS zips
    examples = [e for e in examples if '.DS_Store' not in e]

    if amount == -1:
        amount = len(examples)
    if amount > len(examples):
        raise RuntimeError(
            'You asked for %d examples, but only %d were available.' % (
                amount, len(examples)))
    if random_selection:
        selected_examples = np.random.choice(examples, amount, replace=False)
    else:
        selected_examples = natsorted(
            examples)[from_episode_number:from_episode_number+amount]

    # Process these examples (e.g. loading observations)
    demos = []
    for example in selected_examples:
        example_path = join(examples_path, example)
        with open(join(example_path, LOW_DIM_PICKLE), 'rb') as f:
            obs = pickle.load(f)

        if variation_number == -1:
            with open(join(example_path, VARIATION_NUMBER), 'rb') as f:
                obs.variation_number = pickle.load(f)
        else:
            obs.variation_number = variation_number

        demos.append(obs)
    return demos

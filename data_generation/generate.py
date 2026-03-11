import open3d  # DON'T DELETE THIS!
from multiprocessing import Process, Manager

from pyrep.const import RenderMode

from rlbench import ObservationConfig
from rlbench.action_modes.action_mode import MoveArmThenGripper
from rlbench.action_modes.arm_action_modes import JointVelocity
from rlbench.action_modes.gripper_action_modes import Discrete
from rlbench.backend.utils import (
    task_file_to_task_class,
    float_array_to_rgb_image
)
import rlbench.backend.task as task

import os
import pickle
from PIL import Image
from rlbench.backend.const import *
import numpy as np
import random

from data_generation.customized_rlbench import CustomizedEnvironment

from absl import app
from absl import flags


MESH_POINT_FOLDER = 'mesh_points'
MESH_POINT_FORMAT = '%d.pkl'

FLAGS = flags.FLAGS

flags.DEFINE_string('save_path',
                    'data/train_dataset/microsteps/seed{seed}',
                    'Where to save the demos.')
flags.DEFINE_list('tasks', [],
                  'The tasks to collect. If empty, all tasks are collected.')
flags.DEFINE_list('image_size', [128, 128],
                  'The size of the images tp save.')
flags.DEFINE_enum('renderer',  'opengl3', ['opengl', 'opengl3'],
                  'The renderer to use. opengl does not include shadows, '
                  'but is faster.')
flags.DEFINE_integer('processes', 1,
                     'The number of parallel processes during collection.')
flags.DEFINE_integer('episodes_per_task', 10,
                     'The number of episodes to collect per task.')
flags.DEFINE_integer('variations', -1,
                     'Number of variations to collect per task. -1 for all.')
flags.DEFINE_integer('offset', 0,
                     'First variation id.')
flags.DEFINE_boolean('state', False,
                     'Record the state (not available for all tasks).')
flags.DEFINE_integer('seed', 0,
                     'Seed of randomness')


def check_and_make(dir):
    os.makedirs(dir, exist_ok=True)


class DemoSaver:

    def __init__(self, demo, example_path):
        self.demo = demo
        self.example_path = example_path

    def store(self, folder, attr):
        # Create folder
        path_ = os.path.join(self.example_path, folder)
        os.makedirs(path_, exist_ok=True)
        # Loop over demo and store
        for i, obs in enumerate(self.demo):
            # Read image
            img = obs.__getattribute__(attr)
            if 'rgb' in attr:
                img = Image.fromarray(img)
            elif 'depth' in attr:
                img = float_array_to_rgb_image(img, scale_factor=DEPTH_SCALE)
            elif 'mask' in attr:
                img = Image.fromarray((img * 255).astype(np.uint8))
            # Save image
            img.save(os.path.join(path_, IMAGE_FORMAT % i))
            # Set to None for pickling later
            obs.__setattr__(attr, None)


def save_demo(demo, example_path):
    ds = DemoSaver(demo, example_path)
    paths_attrs = [
        (LEFT_SHOULDER_RGB_FOLDER, 'left_shoulder_rgb'),
        (LEFT_SHOULDER_DEPTH_FOLDER, 'left_shoulder_depth'),
        (LEFT_SHOULDER_MASK_FOLDER, 'left_shoulder_mask'),
        (RIGHT_SHOULDER_RGB_FOLDER, 'right_shoulder_rgb'),
        (RIGHT_SHOULDER_DEPTH_FOLDER, 'right_shoulder_depth'),
        (RIGHT_SHOULDER_MASK_FOLDER, 'right_shoulder_mask'),
        (OVERHEAD_RGB_FOLDER, 'overhead_rgb'),
        (OVERHEAD_DEPTH_FOLDER, 'overhead_depth'),
        (OVERHEAD_MASK_FOLDER, 'overhead_mask'),
        (WRIST_RGB_FOLDER, 'wrist_rgb'),
        (WRIST_DEPTH_FOLDER, 'wrist_depth'),
        (WRIST_MASK_FOLDER, 'wrist_mask'),
        (FRONT_RGB_FOLDER, 'front_rgb'),
        (FRONT_DEPTH_FOLDER, 'front_depth'),
        (FRONT_MASK_FOLDER, 'front_mask')
    ]
    # Save image data first and then None them
    for folder, attr in paths_attrs:
        ds.store(folder, attr)

    # Store object point clouds
    mesh_point_path = os.path.join(example_path, MESH_POINT_FOLDER)
    os.makedirs(mesh_point_path, exist_ok=True)
    for i, obs in enumerate(demo):
        mesh_points = obs.mesh_points
        with open(os.path.join(mesh_point_path, MESH_POINT_FORMAT % i), 'wb') as f:
            pickle.dump(mesh_points, f)
        obs.__delattr__('mesh_points')

    # Save the low-dimension data
    with open(os.path.join(example_path, LOW_DIM_PICKLE), 'wb') as f:
        pickle.dump(demo, f)


def run(i, lock, task_index, variation_count, results, file_lock, tasks):
    """Each thread will choose one task and variation, and then gather
    all the episodes_per_task for that variation."""

    # Initialize each thread with random seed
    np.random.seed(FLAGS.seed)
    random.seed(FLAGS.seed)
    num_tasks = len(tasks)

    img_size = list(map(int, FLAGS.image_size))

    obs_config = ObservationConfig()
    obs_config.set_all(True)
    obs_config.right_shoulder_camera.image_size = img_size
    obs_config.left_shoulder_camera.image_size = img_size
    obs_config.overhead_camera.image_size = img_size
    obs_config.wrist_camera.image_size = img_size
    obs_config.front_camera.image_size = img_size

    # Store depth as 0 - 1
    obs_config.right_shoulder_camera.depth_in_meters = False
    obs_config.left_shoulder_camera.depth_in_meters = False
    obs_config.overhead_camera.depth_in_meters = False
    obs_config.wrist_camera.depth_in_meters = False
    obs_config.front_camera.depth_in_meters = False

    # We want to save the masks as rgb encodings.
    obs_config.left_shoulder_camera.masks_as_one_channel = False
    obs_config.right_shoulder_camera.masks_as_one_channel = False
    obs_config.overhead_camera.masks_as_one_channel = False
    obs_config.wrist_camera.masks_as_one_channel = False
    obs_config.front_camera.masks_as_one_channel = False

    # No need to save point cloud, we'll unproject them from depth
    obs_config.left_shoulder_camera.point_cloud = False
    obs_config.right_shoulder_camera.point_cloud = False
    obs_config.overhead_camera.point_cloud = False
    obs_config.wrist_camera.point_cloud = False
    obs_config.front_camera.point_cloud = False

    if FLAGS.renderer == 'opengl':
        obs_config.right_shoulder_camera.render_mode = RenderMode.OPENGL
        obs_config.left_shoulder_camera.render_mode = RenderMode.OPENGL
        obs_config.overhead_camera.render_mode = RenderMode.OPENGL
        obs_config.wrist_camera.render_mode = RenderMode.OPENGL
        obs_config.front_camera.render_mode = RenderMode.OPENGL

    rlbench_env = CustomizedEnvironment(
        action_mode=MoveArmThenGripper(JointVelocity(), Discrete()),
        obs_config=obs_config,
        headless=True
    )
    rlbench_env.launch()
    task_env = None
    tasks_with_problems = results[i] = ''

    while True:
        # Figure out what task/variation this thread is going to do
        with lock:

            if task_index.value >= num_tasks:
                print('Process', i, 'finished')
                break

            my_variation_count = variation_count.value
            t = tasks[task_index.value]
            task_env = rlbench_env.get_task(t)
            var_target = task_env.variation_count()
            if FLAGS.variations >= 0:
                var_target = np.minimum(FLAGS.variations+FLAGS.offset, var_target)
            if my_variation_count >= var_target:
                # If we have reached the required number of variations for this
                # task, then move on to the next task.
                variation_count.value = my_variation_count = FLAGS.offset
                task_index.value += 1

            variation_count.value += 1
            if task_index.value >= num_tasks:
                print('Process', i, 'finished')
                break
            t = tasks[task_index.value]

        task_env = rlbench_env.get_task(t)
        task_env.set_variation(my_variation_count)
        descriptions, obs = task_env.reset()

        variation_path = os.path.join(
            FLAGS.save_path, task_env.get_name(),
            VARIATIONS_FOLDER % my_variation_count
        )
        print(variation_path)

        check_and_make(variation_path)

        with open(os.path.join(variation_path, VARIATION_DESCRIPTIONS), 'wb') as f:
            pickle.dump(descriptions, f)

        episodes_path = os.path.join(variation_path, EPISODES_FOLDER)
        check_and_make(episodes_path)

        abort_variation = False
        print("episode per task", FLAGS.episodes_per_task)
        for ex_idx in range(FLAGS.episodes_per_task):
            print('Process', i, '// Task:', task_env.get_name(),
                  '// Variation:', my_variation_count, '// Demo:', ex_idx)
            attempts = 10
            while attempts > 0:
                episode_path = os.path.join(episodes_path, EPISODE_FOLDER % ex_idx)
                if os.path.exists(episode_path):
                    break
                try:
                    print("starting demo")
                    demo, = task_env.get_demos(amount=1, live_demos=True)
                    print("demo collected")
                except Exception as e:
                    attempts -= 1
                    if attempts > 0:
                        print('Process %d failed collecting task %s (variation: %d, '
                              'example: %d). Retrying...\n%s\n' % (
                                  i, task_env.get_name(), my_variation_count, ex_idx,
                                  str(e)))
                        continue
                    problem = (
                        'Process %d failed collecting task %s (variation: %d, '
                        'example: %d). Skipping this task/variation.\n%s\n' % (
                            i, task_env.get_name(), my_variation_count, ex_idx,
                            str(e))
                    )
                    print(problem)
                    tasks_with_problems += problem
                    abort_variation = True
                    break
                with file_lock:
                    print("saving demo")
                    save_demo(demo, episode_path)
                break
            if abort_variation:
                break

    results[i] = tasks_with_problems
    rlbench_env.shutdown()


def main(argv):

    FLAGS.save_path = FLAGS.save_path.format(seed=FLAGS.seed)

    task_files = [t.replace('.py', '') for t in os.listdir(task.TASKS_PATH)
                  if t != '__init__.py' and t.endswith('.py')]

    if len(FLAGS.tasks) > 0:
        for t in FLAGS.tasks:
            if t not in task_files:
                raise ValueError('Task %s not recognised!.' % t)
        task_files = FLAGS.tasks

    tasks = [task_file_to_task_class(t) for t in task_files]

    manager = Manager()

    result_dict = manager.dict()
    file_lock = manager.Lock()

    task_index = manager.Value('i', 0)
    variation_count = manager.Value('i', FLAGS.offset)
    lock = manager.Lock()

    check_and_make(FLAGS.save_path)

    processes = [Process(
        target=run, args=(
            i, lock, task_index, variation_count, result_dict, file_lock,
            tasks))
        for i in range(FLAGS.processes)]
    
    
    [t.start() for t in processes]
    [t.join() for t in processes]

    print('Data collection done!')
    for i in range(FLAGS.processes):
        print(result_dict[i])


if __name__ == '__main__':
  app.run(main)

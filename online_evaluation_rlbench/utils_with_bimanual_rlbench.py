import os
import glob
import random

from tqdm import tqdm
import numpy as np
import torch
import torch.nn.functional as F

from rlbench.observation_config import ObservationConfig, CameraConfig
from rlbench.environment import Environment
from rlbench.action_modes.action_mode import BimanualMoveArmThenGripper
from rlbench.action_modes.gripper_action_modes import BimanualDiscrete, assert_action_shape
from rlbench.action_modes.arm_action_modes import BimanualEndEffectorPoseViaPlanning
from rlbench.backend.exceptions import InvalidActionError
from pyrep.errors import IKError, ConfigurationPathError
from pyrep.const import RenderMode

from modeling.encoder.text import fetch_tokenizers
from online_evaluation_rlbench.get_stored_demos import get_stored_demos


def task_file_to_task_class(task_file):
    import importlib

    name = task_file.replace(".py", "")
    class_name = "".join([w[0].upper() + w[1:] for w in name.split("_")])
    mod = importlib.import_module("rlbench.bimanual_tasks.%s" % name)
    mod = importlib.reload(mod)
    task_class = getattr(mod, class_name)
    return task_class


class Mover:

    def __init__(self, task, max_tries=1):
        self._task = task
        self._last_action = None
        self._max_tries = max_tries

    def __call__(self, action, collision_checking=False):
        # action is an array (2, 8)
        # print(f"DEBUG: move() called with action.shape = {action.shape}")
        obs = None
        terminate = None
        reward = 0

        # Try to reach the desired pose without changing the gripper state
        target = action.copy()
        if self._last_action is not None:
            action[:, 7] = self._last_action[:, 7].copy() # copy gripper state
        for _ in range(self._max_tries):
            action_collision = np.ones((action.shape[0], action.shape[1]+1))
            action_collision[:, :-1] = action
            if collision_checking:
                action_collision[:, -1] = 0
            # Peract2 takes (right, left) action, but we predict (left, right)
            action_collision = action_collision[::-1]
            # print(f"DEBUG: action_collision.shape before ravel = {action_collision.shape}")
            action_collision = action_collision.ravel()
            # print(f"DEBUG: action_collision final shape = {action_collision.shape}, expected 18")
            # obs, reward, terminate = self._task.step(action_collision, ret_obs=True)
            obs, reward, terminate = self._task.step(action_collision)

            # Check if we reached the desired pose (planner may be inaccurate)
            l_pos = obs.left.gripper_pose[:3]
            r_pos = obs.right.gripper_pose[:3]
            l_dist_pos = np.sqrt(np.square(target[0, :3] - l_pos).sum())
            r_dist_pos = np.sqrt(np.square(target[1, :3] - r_pos).sum())
            criteria = (l_dist_pos < 5e-3, r_dist_pos < 5e-3)

            if all(criteria) or reward == 1:
                break

        # Then execute with gripper action (open/close))
        action = target
        if (
            not reward == 1.0
            and self._last_action is not None
            and (  # if any gripper state has changed, re-execute
                action[0, 7] != self._last_action[0, 7]
                or action[1, 7] != self._last_action[1, 7]
            )
        ):
            action_collision = np.ones((action.shape[0], action.shape[1]+1))
            action_collision[:, :-1] = action
            if collision_checking:
                action_collision[:, -1] = 0
            action_collision = action_collision[::-1]
            action_collision = action_collision.ravel()
            obs, reward, terminate = self._task.step(action_collision)

        # Store the last action action for the gripper state
        self._last_action = action.copy()

        return obs, reward, terminate


class Actioner:

    def __init__(self, policy=None, backbone='clip'):
        self._policy = policy.cuda()
        self._policy.eval()
        self._instr = None
        self.tokenizer = fetch_tokenizers(backbone)

    def load_episode(self, descriptions):
        instr = [random.choice(descriptions)]
        self._instr = self.tokenizer(instr).cuda(non_blocking=True)

    def predict(self, rgbs, pcds, left_gripper, right_gripper, prediction_len=1):
        """
        Args:
            rgbs: (1, ncam, 3, H, W)
            pcds: (1, ncam, 3, H, W)
            left_gripper: (1, nhist, 8)
            right_gripper: (1, nhist, 8)
            prediction_len: int

        Returns:
            trajectory: (1, prediction_len, nhand=2, 8)
        """
        trajectory_mask = torch.zeros((1, prediction_len, 1), dtype=bool, device='cuda')
        
        return self._policy(
            None,
            trajectory_mask,
            rgbs,
            None,
            pcds,
            self._instr,
            left_proprioception=left_gripper,
            right_proprioception=right_gripper,
            run_inference=True
        )


class RLBenchEnv:

    def __init__(
        self,
        data_path,
        task_str=None,
        image_size=(256, 256),
        apply_rgb=False,
        apply_depth=False,
        apply_pc=False,
        headless=False,
        apply_cameras=("over_shoulder_left", "over_shoulder_right", "wrist_left", "wrist_right", "front"),
        collision_checking=False
    ):

        # setup required inputs
        self.data_path = data_path
        self.apply_cameras = apply_cameras

        # setup RLBench environments
        self.obs_config = self.create_obs_config(
            image_size, apply_rgb, apply_depth, apply_pc, apply_cameras
        )

        self.action_mode = BimanualMoveArmThenGripper(
            arm_action_mode=BimanualEndEffectorPoseViaPlanning(collision_checking=collision_checking),
            gripper_action_mode=HandoverDiscrete() if 'handover' in task_str else BimanualDiscrete()
        )
        self.env = Environment(
            self.action_mode, str(data_path), self.obs_config,
            headless=headless, robot_setup="dual_panda"
        )

    def get_rgb_pcd_gripper_from_obs(self, obs):
        """
        Return rgb, pcd, and gripper from a given observation
        :param obs: an Observation from the env
        :return: rgb, pcd, left_gripper (1, 8), right_gripper (1, 8)
        """
        rgb = torch.stack([
            torch.from_numpy(np.array(obs.perception_data["{}_rgb".format(cam)])).float().permute(2, 0, 1) / 255.0
            for cam in self.apply_cameras
        ]).unsqueeze(0)
        pcd = torch.stack([
            torch.from_numpy(np.array(obs.perception_data["{}_point_cloud".format(cam)])).float().permute(2, 0, 1)
            for cam in self.apply_cameras
        ]).unsqueeze(0)
        
        left_gripper = torch.from_numpy(np.concatenate([
            obs.left.gripper_pose, [obs.left.gripper_open]
        ])).float().unsqueeze(0)
        
        right_gripper = torch.from_numpy(np.concatenate([
            obs.right.gripper_pose, [obs.right.gripper_open]
        ])).float().unsqueeze(0)

        return rgb, pcd, left_gripper, right_gripper

    def evaluate_task_on_multiple_variations(
        self,
        task_str,
        max_steps,
        actioner,
        max_tries=1,
        prediction_len=1,
        num_history=1
    ):
        self.env.launch()
        task_type = task_file_to_task_class(task_str)
        task = self.env.get_task(task_type)
        task_variations = glob.glob(
            os.path.join(self.data_path, task_str, "variation*")
        )
        task_variations = [
            int(n.split('/')[-1].replace('variation', ''))
            for n in task_variations
        ]

        var_success_rates = {}
        var_num_valid_demos = {}

        for variation in tqdm(task_variations):
            task.set_variation(variation)
            success_rate, valid, num_valid_demos = (
                self._evaluate_task_on_one_variation(
                    task_str=task_str,
                    task=task,
                    max_steps=max_steps,
                    variation=variation,
                    actioner=actioner,
                    max_tries=max_tries,
                    prediction_len=prediction_len,
                    num_history=num_history
                )
            )
            if valid:
                var_success_rates[variation] = success_rate
                var_num_valid_demos[variation] = num_valid_demos

        self.env.shutdown()

        var_success_rates["mean"] = (
            sum(var_success_rates.values()) /
            sum(var_num_valid_demos.values())
        )

        return var_success_rates

    @torch.no_grad()
    def _evaluate_task_on_one_variation(
        self,
        task_str,  # this is str
        task,  # this instance of TaskEnvironment
        max_steps,
        variation,
        actioner,
        max_tries=1,
        prediction_len=50,
        num_history=1
    ):
        success_rate = 0
        total_reward = 0
        var_demos = get_stored_demos(
            amount=-1,
            dataset_root=self.data_path,
            variation_number=variation,
            task_name=task_str,
            random_selection=False,
            from_episode_number=0
        )

        for demo_id, demo in enumerate(var_demos):

            left_grippers = torch.Tensor([]).cuda(non_blocking=True)
            right_grippers = torch.Tensor([]).cuda(non_blocking=True)
            descriptions, obs = task.reset_to_demo(demo)
            actioner.load_episode(descriptions)

            move = Mover(task, max_tries=max_tries)
            max_reward = 0.0

            for step_id in range(max_steps):

                # Fetch the current observation and predict one action
                rgb, pcd, left_gripper, right_gripper = self.get_rgb_pcd_gripper_from_obs(obs)
                rgbs_input = rgb.cuda(non_blocking=True)
                pcds_input = pcd.cuda(non_blocking=True)
                left_gripper = left_gripper.cuda(non_blocking=True)
                right_gripper = right_gripper.cuda(non_blocking=True)
                
                left_grippers = torch.cat([left_grippers, left_gripper.unsqueeze(1)], 1)
                right_grippers = torch.cat([right_grippers, right_gripper.unsqueeze(1)], 1)

                left_gripper_input = left_grippers[:, -num_history:]
                right_gripper_input = right_grippers[:, -num_history:]
                
                npad = num_history - left_gripper_input.shape[1]
                left_gripper_input = F.pad(
                    left_gripper_input, (0, 0, npad, 0), mode='replicate'
                )
                right_gripper_input = F.pad(
                    right_gripper_input, (0, 0, npad, 0), mode='replicate'
                )
                
                # if step_id == 0 and demo_id == 0:
                #     print("\n" + "=" * 80)
                #     print("📊 RUNTIME INPUT FORMAT VERIFICATION (Episode 0, Step 0)")
                #     print("=" * 80)
                #     print(f"   Proprioception history length: {num_history}")
                #     print(f"   Left gripper input shape: {left_gripper_input.shape}")
                #     print(f"   Right gripper input shape: {right_gripper_input.shape}")
                #     print(f"   Expected: (batch=1, nhist={num_history}, dim=8)")
                #     
                #     assert left_gripper_input.shape == (1, num_history, 8), \
                #         f"Left gripper shape mismatch! Got {left_gripper_input.shape}"
                #     assert right_gripper_input.shape == (1, num_history, 8), \
                #         f"Right gripper shape mismatch! Got {right_gripper_input.shape}"
                #     print("   ✅ Input shapes match training format")
                #     
                #     if npad > 0:
                #         print(f"   Padding applied: {npad} steps (replicate mode)")
                #         first_elem = left_gripper_input[0, 0]
                #         for i in range(npad):
                #             assert torch.equal(left_gripper_input[0, i], first_elem), \
                #                 f"Padding error at position {i}"
                #         print("   ✅ Replicate padding verified")
                #     else:
                #         print("   No padding needed (full history available)")
                #     print("=" * 80 + "\n")

                output = actioner.predict(
                    rgbs_input,
                    pcds_input,
                    left_gripper_input,
                    right_gripper_input,
                    prediction_len=prediction_len
                )
                
                # if step_id == 0 and demo_id == 0:
                #     print("=" * 80)
                #     print("📤 RUNTIME OUTPUT FORMAT VERIFICATION (Episode 0, Step 0)")
                #     print("=" * 80)
                #     print(f"   Model output shape: {output.shape}")
                #     print(f"   Expected: (batch=1, pred_len={prediction_len}, nhand=2, dim=8)")
                    
                #     assert output.shape[0] == 1, f"Batch size mismatch! Got {output.shape[0]}"
                #     assert output.shape[1] == prediction_len, \
                #         f"Prediction length mismatch! Got {output.shape[1]}, expected {prediction_len}"
                #     assert output.shape[2] == 2, f"Number of hands mismatch! Got {output.shape[2]}"
                #     assert output.shape[3] == 8, f"Action dimension mismatch! Got {output.shape[3]}"
                #     print("   ✅ Output shape matches training format")
                #     print("=" * 80 + "\n")

                # Update the observation based on the predicted action
                try:
                    # Execute predicted action (following the reference implementation)
                    actions = output[-1].cpu().numpy()  # Expected shape: (T, nhand, 8) or (nhand, 8)
                    
                    # Handle different output shapes
                    if len(actions.shape) == 3:  # (T, nhand, 8)
                        actions = actions[-1]  # Take last timestep: (nhand, 8)
                    
                    # For bimanual tasks, ensure we have exactly 2 arms (left and right)
                    if actions.shape[0] != 2:
                        raise ValueError(f"Expected 2 arms for bimanual task, got {actions.shape[0]} arms with shape {actions.shape}")
                    
                    actions[..., -1] = actions[..., -1].round()
                    # print("actions.shape")
                    # print(actions.shape)
                    # # print("obs.shape")
                    # # print(obs.shape)
                    # print("reward")
                    # print(reward)
                    # For bimanual tasks, pass the bimanual action with shape (2, 8)
                    # The move function expects (2, 8) and will handle collision checking
                    obs, reward, _ = move(actions, collision_checking=False)
                    # print("obs.shape")
                    # print(obs.shape)
                    # print("reward")
                    # print(reward)
                    max_reward = max(max_reward, reward)

                    if reward == 1:
                        success_rate += 1
                        break

                except (IKError, ConfigurationPathError, InvalidActionError) as e:
                    print(task_str, demo, step_id, success_rate, e)
                    reward = 0
                except RuntimeError as e:
                    # Handle V-REP/CoppeliaSim errors (e.g., proximity sensor failures)
                    if "V-REP" in str(e) or "Return value: -1" in str(e):
                        print(f"{task_str} Demo {demo_id} Step {step_id}: V-REP error (likely sensor failure), treating as failed step: {e}")
                        reward = 0
                        # Break out of the step loop for this demo, move to next demo
                        break
                    else:
                        # Re-raise if it's not a V-REP error
                        raise

            total_reward += max_reward

            print(
                task_str,
                "Variation",
                variation,
                "Demo",
                demo_id,
                "Reward",
                f"{reward:.2f}",
                "max_reward",
                f"{max_reward:.2f}",
                f"SR: {success_rate}/{demo_id+1}",
                f"SR: {total_reward:.2f}/{demo_id+1}",
                "# valid demos", demo_id + 1
            )

        # Compensate for failed demos
        valid = len(var_demos) > 0

        return success_rate, valid, len(var_demos)

    def create_obs_config(
        self, image_size, apply_rgb, apply_depth, apply_pc, apply_cameras, **kwargs
    ):
        """
        Set up observation config for RLBench environment.
            :param image_size: Image size.
            :param apply_rgb: Applying RGB as inputs.
            :param apply_depth: Applying Depth as inputs.
            :param apply_pc: Applying Point Cloud as inputs.
            :param apply_cameras: Desired cameras.
            :return: observation config
        """
        # Define a config for an unused camera with all applications as False.
        unused_cams = CameraConfig()
        unused_cams.set_all(False)

        # Define a config for a used camera with the given image size and flags
        used_cams = CameraConfig(
            rgb=apply_rgb,
            point_cloud=apply_pc,
            depth=apply_depth,
            mask=False,
            image_size=image_size,
            render_mode=RenderMode.OPENGL3,  # note OPENGL3 for Peract2!
            **kwargs
        )

        # apply_cameras is a tuple with the names(str) of all the cameras
        camera_names = apply_cameras
        cameras = {}
        for name in camera_names:
            cameras[name] = used_cams

        obs_config = ObservationConfig(
            camera_configs=cameras,
            joint_forces=False,
            joint_positions=False,
            joint_velocities=True,
            task_low_dim_state=False,
            gripper_touch_forces=False,
            gripper_pose=True,
            gripper_open=True,
            gripper_matrix=True,
            gripper_joint_positions=True
        )

        return obs_config


class HandoverDiscrete(BimanualDiscrete):
    """
    A custom gripper action mode for the handover task.
    It forces one gripper to release so that the other grasps.
    """

    def action(self, scene, action):
        assert_action_shape(action, self.action_shape(scene.robot))
        if 0.0 > action[0] > 1.0:
            raise InvalidActionError(
                'Gripper action expected to be within 0 and 1.')

        if 0.0 > action[1] > 1.0:
            raise InvalidActionError(
                'Gripper action expected to be within 0 and 1.')

        right_open_condition = all(
            x > 0.9 for x in scene.robot.right_gripper.get_open_amount())

        left_open_condition = all(
            x > 0.9 for x in scene.robot.left_gripper.get_open_amount())

        right_current_ee = 1.0 if right_open_condition else 0.0
        left_current_ee = 1.0 if left_open_condition else 0.0

        right_action = float(action[0] > 0.5)
        left_action = float(action[1] > 0.5)

        if right_current_ee != right_action or left_current_ee != left_action:
            if not self._detach_before_open:
                self._actuate(scene, action)

        # Move objects between grippers
        if right_current_ee != right_action:
            if right_action == 0.0 and self._attach_grasped_objects:
                left_grasped_objects = scene.robot.left_gripper.get_grasped_objects()
                for g_obj in scene.task.get_graspable_objects():
                    if g_obj in left_grasped_objects:
                        scene.robot.left_gripper.release()
                        scene.robot.right_gripper.grasp(g_obj)
                    else:
                        scene.robot.right_gripper.grasp(g_obj)
            else:
                scene.robot.right_gripper.release()
        if left_current_ee != left_action:
            if left_action == 0.0 and self._attach_grasped_objects:
                right_grasped_objects = scene.robot.right_gripper.get_grasped_objects()
                for g_obj in scene.task.get_graspable_objects():
                    if g_obj in right_grasped_objects:
                        scene.robot.right_gripper.release()
                        scene.robot.left_gripper.grasp(g_obj)
                    else:
                        scene.robot.left_gripper.grasp(g_obj)
            else:
                scene.robot.left_gripper.release()

        if right_current_ee != right_action or left_current_ee != left_action:
            if self._detach_before_open:
                self._actuate(scene, action)
            if right_action == 1.0 or left_action == 1.0:
                # Step a few more times to allow objects to drop
                for _ in range(10):
                    scene.pyrep.step()
                    scene.task.step()

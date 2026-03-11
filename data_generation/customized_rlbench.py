from scipy.spatial.transform import Rotation as R

from pyrep.objects.shape import Shape
from rlbench.environment import Environment
from rlbench.backend.scene import Scene
from rlbench.sim2real.domain_randomization_scene import DomainRandomizationScene


class CustomizedScene(Scene):

    def get_observation(self):
        obs = super().get_observation()

        mesh_points = {}
        # self._robot_shapes
        joints = self.robot.arm.get_visuals()
        # add self._workspace to the list of objects
        # base_object = self.task._base_object
        joints.append(Shape('diningTable_visible'))#self._workspace)
        for curr_obj in joints:
            obj_name = curr_obj.get_name()
            obj_vertices, _, _ = curr_obj.get_mesh_data()
            obj_pose = curr_obj.get_pose()
            position = obj_pose[:3]
            quaternion = obj_pose[3:7]
            rotation_matrix = R.from_quat(quaternion).as_matrix()
            rotated_vertices = (rotation_matrix @ obj_vertices.T).T
            curr_points = rotated_vertices + position.reshape(1, 3)
            mesh_points[obj_name] = curr_points
        for (curr_obj, obj_type) in self.task._initial_objs_in_scene:
            if isinstance(curr_obj, Shape):
                obj_name = curr_obj.get_name()
                obj_color = str(curr_obj.get_color())
                obj_vertices, _, _ = curr_obj.get_mesh_data()
                obj_pose = curr_obj.get_pose()
                position = obj_pose[:3]
                quaternion = obj_pose[3:7]
                rotation_matrix = R.from_quat(quaternion).as_matrix()
                rotated_vertices = (rotation_matrix @ obj_vertices.T).T
                curr_points = rotated_vertices + position.reshape(1, 3)
                mesh_points[obj_name+"_"+obj_color] = curr_points
        obs.mesh_points = mesh_points
        obs = self.task.decorate_observation(obs)
        return obs


class CustomizedDomainRandomizationScene(CustomizedScene, DomainRandomizationScene):
    pass


class CustomizedEnvironment(Environment):

    def launch(self):
        super().launch()
        if self._randomize_every is None:
            self._scene = CustomizedScene(
                self._pyrep, self._robot, self._obs_config, self._robot_setup)
        else:
            self._scene = CustomizedDomainRandomizationScene(
                self._pyrep, self._robot, self._obs_config, self._robot_setup,
                self._randomize_every, self._frequency,
                self._visual_randomization_config,
                self._dynamics_randomization_config)

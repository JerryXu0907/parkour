import random
from collections import OrderedDict, defaultdict
from isaacgym.torch_utils import torch_rand_float, get_euler_xyz, quat_from_euler_xyz, tf_apply, quat_rotate_inverse
from isaacgym import gymtorch, gymapi, gymutil
import torch
import torch.nn.functional as F
import torchvision.transforms as T

from legged_gym.envs.base.legged_robot import LeggedRobot
from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs.base.base_task import BaseTask
# from legged_gym.utils.terrain.terrain import Terrain
from legged_gym.utils.terrain.terrain_extreme import Terrain
from legged_gym.utils.math import quat_apply_yaw, wrap_to_pi, torch_rand_sqrt_float
from legged_gym.utils.helpers import class_to_dict
from .legged_robot_config import LeggedRobotCfg
import cv2
import numpy as np

'''
done    __init__
done    step
done    post_physics_step
done    check_termination
        _fill_extras
done    reset_idx

done    compute_observations
        |
        |-  _get_proprioception_obs
        |-  _get_prop_history_obs
        |-  _get_goal_obs

done    _process_rigid_shape_props
done    _process_dof_props
done    _process_rigid_body_props
done    _post_physics_step_callback
no      _resample_commands
done    _compute_torques
        _reset_dofs (no need for now)
done    _reset_root_states
        _update_terrain_curriculum
done    _get_noise_scale_vec
done    _init_buffers
done    _create_envs
        _draw_debug_vis
            |- draw goals
            |- draw delta v, delta d

TODO:
        _init_goals (add to reset_idx)
        _reset_goals
        
'''

'''
__init__                    checked 
step                        checked
_update_goals               checked
post_physics_step           checked
check_termination           checked
reset_idx                   checked
create_sim                  checked
set_camera                  no need
_process_rigid_shape_props  checked
_process_dof_props          checked
_process_rigid_body_props   checked
_post_physics_step_callback checked
_resample_commands          no need
_compute_torques            checked
_reset_dofs                 checked
_reset_root_states          checked
_push_robots                no need
_update_terrain_curriculum  no need
_init_buffers
_create_envs                checked
_get_env_origins            checked
_init_height_points         Note: extreme_parkour added noise in height measurement
_get_heights                checked
'''

def euler_from_quaternion(quat_angle):
        """
        Convert a quaternion into euler angles (roll, pitch, yaw)
        roll is rotation around x in radians (counterclockwise)
        pitch is rotation around y in radians (counterclockwise)
        yaw is rotation around z in radians (counterclockwise)
        """
        x = quat_angle[:,0]; y = quat_angle[:,1]; z = quat_angle[:,2]; w = quat_angle[:,3]
        t0 = +2.0 * (w * x + y * z)
        t1 = +1.0 - 2.0 * (x * x + y * y)
        roll_x = torch.atan2(t0, t1)
     
        t2 = +2.0 * (w * y - z * x)
        t2 = torch.clip(t2, -1, 1)
        pitch_y = torch.asin(t2)
     
        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 - 2.0 * (y * y + z * z)
        yaw_z = torch.atan2(t3, t4)
     
        return roll_x, pitch_y, yaw_z # in radians

class LeggedRobotParkour(LeggedRobot):

    # ------------init--------------
    def __init__(self, cfg: LeggedRobotCfg, sim_params, physics_engine, sim_device, headless):
        self.all_obs_components = cfg.env.obs_components
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)
        print(self.debug_viz)
        self._prepare_termination_function()
        self.reset_idx(torch.arange(self.num_envs, device=self.device))
        self.post_physics_step()
    
    def pre_physics_step(self, actions):
        self.volume_sample_points_refreshed = False
        actions_preprocessed = False
        if isinstance(self.cfg.normalization.clip_actions, (tuple, list)):
            self.cfg.normalization.clip_actions = torch.tensor(
                self.cfg.normalization.clip_actions,
                device= self.device,
            )
        if isinstance(getattr(self.cfg.normalization, "clip_actions_low", None), (tuple, list)):
            self.cfg.normalization.clip_actions_low = torch.tensor(
                self.cfg.normalization.clip_actions_low,
                device= self.device
            )
        if isinstance(getattr(self.cfg.normalization, "clip_actions_high", None), (tuple, list)):
            self.cfg.normalization.clip_actions_high = torch.tensor(
                self.cfg.normalization.clip_actions_high,
                device= self.device
            )
        if getattr(self.cfg.normalization, "clip_actions_method", None) == "tanh":
            clip_actions = self.cfg.normalization.clip_actions
            self.actions = (torch.tanh(actions) * clip_actions).to(self.device)
            actions_preprocessed = True
        elif getattr(self.cfg.normalization, "clip_actions_method", None) == "hard":
            if self.cfg.control.control_type == "P":
                actions_low = getattr(
                    self.cfg.normalization, "clip_actions_low",
                    self.dof_pos_limits[:, 0] - self.default_dof_pos,
                )
                actions_high = getattr(
                    self.cfg.normalization, "clip_actions_high",
                    self.dof_pos_limits[:, 1] - self.default_dof_pos,
                )
                self.actions = torch.clip(actions, actions_low, actions_high)
            else:
                raise NotImplementedError()
            actions_preprocessed = True
        if getattr(self.cfg.normalization, "clip_actions_delta", None) is not None:
            self.actions = torch.clip(
                self.actions,
                self.last_actions - self.cfg.normalization.clip_actions_delta,
                self.last_actions + self.cfg.normalization.clip_actions_delta,
            )
        
        if not actions_preprocessed:
            return super().pre_physics_step(actions)

    def post_decimation_step(self, dec_i):
        return_ = super().post_decimation_step(dec_i)
        self.max_torques = torch.maximum(
            torch.max(torch.abs(self.torques), dim= -1)[0],
            self.max_torques,
        )
        ### The set torque limit is usally smaller than the robot dataset
        self.torque_exceed_count_substep[(torch.abs(self.torques) > self.torque_limits).any(dim= -1)] += 1
        ### Hack to check the torque limit exceeding by your own value.
        # self.torque_exceed_count_envstep[(torch.abs(self.torques) > 38.).any(dim= -1)] += 1
        
        ### count how many times in the episode the robot is out of dof pos limit (summing all dofs)
        self.out_of_dof_pos_limit_count_substep += self._reward_dof_pos_limits().int()
        
        return return_
    
    def post_physics_step(self):
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_force_sensor_tensor(self.sim)

        self.episode_length_buf += 1
        self.common_step_counter += 1

        # prepare quantities
        self.base_quat[:] = self.root_states[:, 3:7]
        self.base_lin_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        self.base_ang_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
        self.projected_gravity[:] = quat_rotate_inverse(self.base_quat, self.gravity_vec)
        
        # self.roll, self.pitch, self.yaw = get_euler_xyz(self.base_quat)
        self.roll, self.pitch, self.yaw = euler_from_quaternion(self.base_quat)

        contact = torch.norm(self.contact_forces[:, self.feet_indices], dim=-1) > 2.
        self.contact_filt = torch.logical_or(contact, self.last_contacts) 
        self.last_contacts = contact

        self._update_goals()
        self._post_physics_step_callback()

        # compute observations, rewards, resets, ...
        self.check_termination()
        self.compute_reward()
        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        self.reset_idx(env_ids)
        self.compute_observations() # in some cases a simulation step might be required to refresh some obs (for example body positions)

        self.last_actions[:] = self.actions[:]
        self.last_dof_vel[:] = self.dof_vel[:]
        self.last_root_vel[:] = self.root_states[:, 7:13]
        self.last_root_pos[:] = self.root_states[:, :3]
        self.last_torques[:] = self.torques[:]

        self.debug_viz=True
        if self.viewer and self.enable_viewer_sync and self.debug_viz:
            self._draw_debug_vis()
            if self.cfg.depth.use_camera:
                window_name = "Depth Image"
                cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
                cv2.imshow("Depth Image", self.depth_buffer[self.lookat_id, -1].cpu().numpy() + 0.5)
                cv2.waitKey(1)

    def check_termination(self):
        super().check_termination()
        for i in range(len(self.termination_functions)):
            self.termination_functions[i]()

    def get_obs_segment_from_components(self, components):
        """ Observation segment is defined as a list of lists/ints defining the tensor shape with
        corresponding order.
        """
        segments = OrderedDict()
        if "proprioception" in components:
            segments["proprioception"] = (46,)
        if "height_measurements" in components:
            segments["height_measurements"] = (132,)
        if "forward_depth" in components:
            segments["forward_depth"] = (1, *self.cfg.sensor.forward_camera.resolution)
        if "base_pose" in components:
            segments["base_pose"] = (6,) # xyz + rpy
        if "robot_config" in components:
            """ Related to robot_config_buffer attribute, Be careful to change. """
            # robot shape friction
            # CoM (Center of Mass) x, y, z
            # base mass (payload)
            # motor strength for each joint
            segments["robot_config"] = (1 + 3 + 1 + 12,)
        if "goal" in components:
            segments['goal'] = (3+3+3,)
        if "prop_history" in components:
            segments["prop_history"] = (10 * 46,)
        return segments
    
    def get_num_obs_from_components(self, components):
        obs_segments = self.get_obs_segment_from_components(components)
        num_obs = 0
        for k, v in obs_segments.items():
            num_obs += np.prod(v)
        return num_obs

    def reset_idx(self, env_ids):
        """ Reset some environments.
            Calls self._reset_dofs(env_ids), self._reset_root_states(env_ids), and self._resample_commands(env_ids)
            [Optional] calls self._update_terrain_curriculum(env_ids), self.update_command_curriculum(env_ids) and
            Logs episode info
            Resets some buffers

        Args:
            env_ids (list[int]): List of environment ids which must be reset
        """
        if len(env_ids) == 0:
            return
        
        # reset the terrain
        # random initialization of the terrain

        # update curriculum
        if self.cfg.terrain.curriculum:
            self._update_terrain_curriculum(env_ids)
        
        self._fill_extras(env_ids)

        # reset robot states
        self._reset_dofs(env_ids)           # dof states no need to change, or we can add some initial velocities to the dofs (limbs)
        self._reset_root_states(env_ids)    # this is the part for changing the init root_states
        self._resample_commands(env_ids)

        # TODO: might have some problems regarding this
        self.gym.simulate(self.sim)
        self.gym.fetch_results(self.sim, True)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        self._reset_buffers(env_ids)
        self._reset_goals(env_ids)

    # -------------------------------
    def _init_buffers(self):
        # update obs_scales components incase there will be one-by-one scaling
        for k in self.all_obs_components:
            if isinstance(getattr(self.obs_scales, k, None), (tuple, list)):
                setattr(
                    self.obs_scales,
                    k,
                    torch.tensor(getattr(self.obs_scales, k, 1.), dtype= torch.float32, device= self.device)
                )
        self.obs_history_buf = torch.zeros(self.num_envs, self.cfg.env.history_len, self.cfg.env.n_proprio, dtype= torch.float32, device=self.device)
        super()._init_buffers()
        self._init_goals()

        # additional gym GPU state tensors
        force_sensor_tensor = self.gym.acquire_force_sensor_tensor(self.sim)
        rigid_body_state_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_force_sensor_tensor(self.sim)

        # additional wrapper tensors
        self.rigid_body_states = gymtorch.wrap_tensor(rigid_body_state_tensor).view(self.num_envs, -1, 13)
        # self.force_sensor_tensor = gymtorch.wrap_tensor(force_sensor_tensor).view(self.num_envs, 4, 6) # for feet only, see create_env()

        # additional data initializations
        self.reach_goal_timer = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
        self.sensor_tensor_dict = defaultdict(list)
        self.init_base_vel = torch.zeros(self.num_envs, 6, dtype=torch.float, device=self.device, requires_grad=False)
        self.max_speed_cmd = torch.ones(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False) * 3.0
        # gym sensing tensors
        for env_i, env_handle in enumerate(self.envs):
            if "forward_depth" in self.all_obs_components:
                self.sensor_tensor_dict["forward_depth"].append(gymtorch.wrap_tensor(
                    self.gym.get_camera_image_gpu_tensor(
                        self.sim,
                        env_handle,
                        self.sensor_handles[env_i]["forward_camera"],
                        gymapi.IMAGE_DEPTH,
                )))
            if "forward_color" in self.all_obs_components:
                self.sensor_tensor_dict["forward_color"].append(gymtorch.wrap_tensor(
                    self.gym.get_camera_image_gpu_tensor(
                        self.sim,
                        env_handle,
                        self.sensor_handles[env_i]["forward_camera"],
                        gymapi.IMAGE_COLOR,
                )))

        # projected gravity bias (if needed)
        if getattr(self.cfg.domain_rand, "randomize_gravity_bias", False):
            print("Initializing gravity bias for domain randomization")
            # add cross trajectory domain randomization on projected gravity bias
            # uniform sample from range
            self.gravity_bias = torch.rand(self.num_envs, 3, dtype= torch.float, device= self.device, requires_grad= False)
            self.gravity_bias[:, 0] *= self.cfg.domain_rand.gravity_bias_range["x"][1] - self.cfg.domain_rand.gravity_bias_range["x"][0]
            self.gravity_bias[:, 0] += self.cfg.domain_rand.gravity_bias_range["x"][0]
            self.gravity_bias[:, 1] *= self.cfg.domain_rand.gravity_bias_range["y"][1] - self.cfg.domain_rand.gravity_bias_range["y"][0]
            self.gravity_bias[:, 1] += self.cfg.domain_rand.gravity_bias_range["y"][0]
            self.gravity_bias[:, 2] *= self.cfg.domain_rand.gravity_bias_range["z"][1] - self.cfg.domain_rand.gravity_bias_range["z"][0]
            self.gravity_bias[:, 2] += self.cfg.domain_rand.gravity_bias_range["z"][0]

        self.max_power_per_timestep = torch.zeros(self.num_envs, dtype= torch.float32, device= self.device)
        all_obs_components = self.all_obs_components
        
        self.contour_detection_kernel = torch.zeros(
            (8, 1, 3, 3),
            dtype= torch.float32,
            device= self.device,
        )
        # emperical values to be more sensitive to vertical edges
        self.contour_detection_kernel[0, :, 1, 1] = 0.5
        self.contour_detection_kernel[0, :, 0, 0] = -0.5
        self.contour_detection_kernel[1, :, 1, 1] = 0.1
        self.contour_detection_kernel[1, :, 0, 1] = -0.1
        self.contour_detection_kernel[2, :, 1, 1] = 0.5
        self.contour_detection_kernel[2, :, 0, 2] = -0.5
        self.contour_detection_kernel[3, :, 1, 1] = 1.2
        self.contour_detection_kernel[3, :, 1, 0] = -1.2
        self.contour_detection_kernel[4, :, 1, 1] = 1.2
        self.contour_detection_kernel[4, :, 1, 2] = -1.2
        self.contour_detection_kernel[5, :, 1, 1] = 0.5
        self.contour_detection_kernel[5, :, 2, 0] = -0.5
        self.contour_detection_kernel[6, :, 1, 1] = 0.1
        self.contour_detection_kernel[6, :, 2, 1] = -0.1
        self.contour_detection_kernel[7, :, 1, 1] = 0.5
        self.contour_detection_kernel[7, :, 2, 2] = -0.5

        self.max_torques = torch.zeros_like(self.torques[..., 0])
        self.torque_exceed_count_substep = torch.zeros_like(self.torques[..., 0], dtype= torch.int32) # The number of substeps that the torque exceeds the limit
        self.torque_exceed_count_envstep = torch.zeros_like(self.torques[..., 0], dtype= torch.int32) # The number of envsteps that the torque exceeds the limit
        self.out_of_dof_pos_limit_count_substep = torch.zeros_like(self.torques[..., 0], dtype= torch.int32) # The number of substeps that the dof pos exceeds the limit
        
    def _init_height_points(self):
        y = torch.tensor(self.cfg.terrain.measured_points_y, device=self.device, requires_grad=False)
        x = torch.tensor(self.cfg.terrain.measured_points_x, device=self.device, requires_grad=False)
        grid_x, grid_y = torch.meshgrid(x, y)

        self.num_height_points = grid_x.numel()
        points = torch.zeros(self.num_envs, self.num_height_points, 3, device=self.device, requires_grad=False)
        for i in range(self.num_envs):
            offset = torch_rand_float(-self.cfg.terrain.measure_horizontal_noise, self.cfg.terrain.measure_horizontal_noise, (self.num_height_points,2), device=self.device).squeeze()
            xy_noise = torch_rand_float(-self.cfg.terrain.measure_horizontal_noise, self.cfg.terrain.measure_horizontal_noise, (self.num_height_points,2), device=self.device).squeeze() + offset
            points[i, :, 0] = grid_x.flatten() + xy_noise[:, 0]
            points[i, :, 1] = grid_y.flatten() + xy_noise[:, 1]
        return points

    def _create_envs(self):
        if self.cfg.domain_rand.randomize_motor:
            mtr_rng = self.cfg.domain_rand.leg_motor_strength_range
            self.motor_strength = torch_rand_float(
                mtr_rng[0],
                mtr_rng[1],
                (self.num_envs, self.num_actions),
                device=self.device,
            )
        
        return_ = super()._create_envs()
        
        front_hip_names = getattr(self.cfg.asset, "front_hip_names", ["FR_hip_joint", "FL_hip_joint"])
        self.front_hip_indices = torch.zeros(len(front_hip_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i, name in enumerate(front_hip_names):
            self.front_hip_indices[i] = self.dof_names.index(name)

        rear_hip_names = getattr(self.cfg.asset, "rear_hip_names", ["RR_hip_joint", "RL_hip_joint"])
        self.rear_hip_indices = torch.zeros(len(rear_hip_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i, name in enumerate(rear_hip_names):
            self.rear_hip_indices[i] = self.dof_names.index(name)

        hip_names = front_hip_names + rear_hip_names
        self.hip_indices = torch.zeros(len(hip_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i, name in enumerate(hip_names):
            self.hip_indices[i] = self.dof_names.index(name)
        
        return return_
    
    def _create_terrain(self):
        mesh_type = self.cfg.terrain.mesh_type
        self.terrain = Terrain(self.cfg.terrain, self.num_envs)
        if mesh_type=='plane':
            self._create_ground_plane()
        elif mesh_type=='heightfield':
            self._create_heightfield()
        elif mesh_type=='trimesh':
            self._create_trimesh()
        elif mesh_type is not None:
            raise ValueError("Terrain mesh type not recognised. Allowed types are [None, plane, heightfield, trimesh]")

    def _update_goals(self):
        next_flag = self.reach_goal_timer > self.cfg.env.reach_goal_delay / self.dt
        self.reach_goal_timer[next_flag] = 0

        ground_goals = torch.abs(self.env_goals[:, -1]) < 0.05
        self.target_pos_rel = self.env_goals - self.root_states[:, :3]
        self.target_pos_rel[ground_goals, -1] = 0.

        self.reached_goal_ids = torch.norm(self.target_pos_rel, dim=1) < self.cfg.env.next_goal_threshold
        self.reach_goal_timer[self.reached_goal_ids] += 1
        # self.target_pos_rel = quat_rotate_inverse(self.base_quat, self.target_pos_rel)
        # print(self.target_pos_rel)
        # x = input()

        norm = torch.norm(self.target_pos_rel, dim=-1, keepdim=True)
        target_vec_norm = self.target_pos_rel / (norm + 1e-5)
        self.target_yaw = torch.atan2(target_vec_norm[:, 1], target_vec_norm[:, 0])
    
    def _reset_buffers(self, env_ids):
        super()._reset_buffers(env_ids)

        # additional buffer reset
        self.last_root_vel[env_ids] = 0.
        self.reach_goal_timer[env_ids] = 0.
        self.init_base_vel[env_ids] = self.root_states[env_ids, 7:13]
        self.obs_history_buf[env_ids] = 0.

        if hasattr(self, "actions_history_buffer"):
            self.actions_history_buffer[:, env_ids] = 0.
            self.action_delayed_frames[env_ids] = self.cfg.control.action_history_buffer_length
        if hasattr(self, "forward_depth_buffer"):
            self.forward_depth_buffer[:, env_ids] = 0.
            self.forward_depth_delayed_frames[env_ids] = self.cfg.sensor.forward_camera.buffer_length
        if hasattr(self, "proprioception_buffer"):
            self.proprioception_buffer[:, env_ids] = 0.
            self.proprioception_delayed_frames[env_ids] = self.cfg.sensor.proprioception.buffer_length
        
        if hasattr(self, "velocity_sample_points"): self.velocity_sample_points[env_ids] = 0.

        if getattr(self.cfg.domain_rand, "randomize_gravity_bias", False):
            assert hasattr(self, "gravity_bias")
            self.gravity_bias[env_ids] = torch.rand_like(self.gravity_bias[env_ids])
            self.gravity_bias[env_ids, 0] *= self.cfg.domain_rand.gravity_bias_range["x"][1] - self.cfg.domain_rand.gravity_bias_range["x"][0]
            self.gravity_bias[env_ids, 0] += self.cfg.domain_rand.gravity_bias_range["x"][0]
            self.gravity_bias[env_ids, 1] *= self.cfg.domain_rand.gravity_bias_range["y"][1] - self.cfg.domain_rand.gravity_bias_range["y"][0]
            self.gravity_bias[env_ids, 1] += self.cfg.domain_rand.gravity_bias_range["y"][0]
            self.gravity_bias[env_ids, 2] *= self.cfg.domain_rand.gravity_bias_range["z"][1] - self.cfg.domain_rand.gravity_bias_range["z"][0]
            self.gravity_bias[env_ids, 2] += self.cfg.domain_rand.gravity_bias_range["z"][0]

        self.max_power_per_timestep[env_ids] = 0.
    
    def _get_noise_scale_vec(self, cfg):
        noise_vec = torch.zeros_like(self.obs_buf[0])
        self.add_noise = cfg.noise.add_noise
        
        segment_start_idx = 0
        obs_segments = self.get_obs_segment_from_components(cfg.env.obs_components)
        # write noise for each corresponding component.
        # for k, v in obs_segments.items():
        #     segment_length = np.prod(v)
        #     # write sensor scale to provided noise_vec
        #     # for example "_write_forward_depth_noise"
        #     getattr(self, "_write_" + k + "_noise")(noise_vec[segment_start_idx: segment_start_idx + segment_length])
        #     segment_start_idx += segment_length

        return noise_vec

    def _compute_torques(self, actions):    
        actions_scaled = actions * self.cfg.control.action_scale
        control_type = self.cfg.control.control_type
        if control_type=="P":
            if not self.cfg.domain_rand.randomize_motor:  # TODO add strength to gain directly
                torques = self.p_gains*(actions_scaled + self.default_dof_pos - self.dof_pos) - self.d_gains*self.dof_vel
            else:
                torques = self.motor_strength[0] * self.p_gains*(actions_scaled + self.default_dof_pos - self.dof_pos) - self.motor_strength[1] * self.d_gains*self.dof_vel
                
        elif control_type=="V":
            torques = self.p_gains*(actions_scaled - self.dof_vel) - self.d_gains*(self.dof_vel - self.last_dof_vel)/self.sim_params.dt
        elif control_type=="T":
            torques = actions_scaled
        else:
            raise NameError(f"Unknown controller type: {control_type}")
        return torch.clip(torques, -self.torque_limits, self.torque_limits)
    
    # ---------- Setting Goals ---------
    def _init_goals(self):
        self.terrain_goals = torch.from_numpy(self.terrain.goals[:, :, 0]).to(self.device).to(torch.float)
        self.env_goals = self.terrain_goals[self.terrain_levels, self.terrain_types] #- self.root_states[:, :3]
        self.env_goals_rel = self.terrain_goals[self.terrain_levels, self.terrain_types] - self.root_states[:, :3]
        self.init_velocities = self.root_states[:, 7:10]
        self.goal_velocities = torch_rand_float(*self.cfg.domain_rand.init_base_vel_range, self.env_goals.shape, device=self.device)

    def _reset_goals(self, env_ids):
        self.env_goals[env_ids] = self.terrain_goals[self.terrain_levels[env_ids], self.terrain_types[env_ids]] #- self.root_states[env_ids, :3]
        self.env_goals_rel[env_ids] = self.terrain_goals[self.terrain_levels[env_ids], self.terrain_types[env_ids]] - self.root_states[env_ids, :3]
        self.init_velocities[env_ids] = self.root_states[env_ids, 7:10]
        self.goal_velocities[env_ids] = torch_rand_float(*self.cfg.domain_rand.init_base_vel_range, self.env_goals[env_ids].shape, device=self.device)

    # ----Process robot physical properties----
    def _process_rigid_shape_props(self, props, env_id):
        props = super()._process_rigid_shape_props(props, env_id)
        if env_id == 0:
            all_obs_components = self.cfg.env.obs_components
            if "robot_config" in all_obs_components:
                all_obs_components
                self.robot_config_buffer = torch.empty(
                    self.num_envs, 1 + 3 + 1 + 12,
                    dtype= torch.float32,
                    device= self.device,
                )
        
        if hasattr(self, "robot_config_buffer"):
            self.robot_config_buffer[env_id, 0] = props[0].friction
        return props
    
    def _process_dof_props(self, props, env_id):
        props = super()._process_dof_props(props, env_id)
        if env_id == 0:
            if hasattr(self.cfg.control, "torque_limits"):
                if not isinstance(self.cfg.control.torque_limits, (tuple, list)):
                    self.torque_limits = torch.ones(self.num_dof, dtype= torch.float, device= self.device, requires_grad= False)
                    self.torque_limits *= self.cfg.control.torque_limits
                else:
                    self.torque_limits = torch.tensor(self.cfg.control.torque_limits, dtype= torch.float, device= self.device, requires_grad= False)
        return props

    def _process_rigid_body_props(self, props, env_id):
        props = super()._process_rigid_body_props(props, env_id)

        if self.cfg.domain_rand.randomize_com:
            rng_com_x = self.cfg.domain_rand.com_range.x
            rng_com_y = self.cfg.domain_rand.com_range.y
            rng_com_z = self.cfg.domain_rand.com_range.z
            rand_com = np.random.uniform(
                [rng_com_x[0], rng_com_y[0], rng_com_z[0]],
                [rng_com_x[1], rng_com_y[1], rng_com_z[1]],
                size=(3,),
            )
            props[0].com += gymapi.Vec3(*rand_com)

        if hasattr(self, "robot_config_buffer"):
            self.robot_config_buffer[env_id, 1] = props[0].com.x
            self.robot_config_buffer[env_id, 2] = props[0].com.y
            self.robot_config_buffer[env_id, 3] = props[0].com.z
            self.robot_config_buffer[env_id, 4] = props[0].mass
            self.robot_config_buffer[env_id, 5:5+12] = self.motor_strength[env_id] if hasattr(self, "motor_strength") else 1.
        return props

    # ---------Observation---------------
    def compute_observations(self):
        for key in self.sensor_handles[0].keys():
            if "camera" in key:
                # NOTE: Different from the documentation and examples from isaacgym
                # gym.fetch_results() must be called before gym.start_access_image_tensors()
                # refer to https://forums.developer.nvidia.com/t/camera-example-and-headless-mode/178901/10
                self.gym.fetch_results(self.sim, True)
                self.gym.step_graphics(self.sim)
                self.gym.render_all_camera_sensors(self.sim)
                self.gym.start_access_image_tensors(self.sim)
                break
        add_noise = self.add_noise; self.add_noise = False
        # super().compute_observations()
        # self.obs_super_impl = self.obs_buf
        self.obs_buf = self._get_obs_from_components(
            self.cfg.env.obs_components,
            privileged= False,
        )
        self.add_noise = add_noise

    def _get_obs_from_components(self, components: list, privileged= False):
        obs_segments = self.get_obs_segment_from_components(components)
        obs = []
        for k, v in obs_segments.items():
            # if k == "proprioception":
            #     obs.append(self._get_proprioception_obs(privileged))
            # elif k == "height_measurements":
            #     obs.append(self._get_height_measurements_obs(privileged))
            # else:
            #     # get the observation from specific component name
            #     # such as "_get_forward_depth_obs"
            obs.append(
                getattr(self, "_get_" + k + "_obs")(privileged) * \
                getattr(self.obs_scales, k, 1.)
            )
        obs = torch.cat(obs, dim= 1)
        return obs
    
    def _get_proprioception_obs(self, privileged= False):
        imu_obs = torch.stack((self.roll, self.pitch), dim=1)
        self.delta_yaw = self.target_yaw - self.yaw     
        obs_buf = torch.cat((self.base_ang_vel  * self.obs_scales.ang_vel,   # 3
                            imu_obs,    # 2
                            self.delta_yaw[:, None],            # 1       
                            # self.commands[:, 0:1],  #[1,1]
                            (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,   #  12
                            self.dof_vel * self.obs_scales.dof_vel,     #  12
                            self.last_actions,                          #  12
                            self.contact_filt.float()-0.5,              #  4
                            ),dim=-1)
        self.obs_history_buf = torch.where(
            (self.episode_length_buf <= 1)[:, None, None], 
            torch.stack([obs_buf] * self.cfg.env.history_len, dim=1),
            torch.cat([
                self.obs_history_buf[:, 1:],
                obs_buf.unsqueeze(1)
            ], dim=1)
        )
        return obs_buf
    
    def _get_prop_history_obs(self, privileged=False):
        return self.obs_history_buf.reshape(self.num_envs, -1)
    
    def _get_height_measurements_obs(self, privileged= False):
        heights = torch.clip(self.root_states[:, 2].unsqueeze(1) - 0.3 - self.measured_heights, -1, 1.)
        return heights
    
    def _get_forward_depth_obs(self, privileged= False):
        return torch.stack(self.sensor_tensor_dict["forward_depth"]).flatten(start_dim= 1)

    def _get_base_pose_obs(self, privileged= False):
        roll, pitch, yaw = get_euler_xyz(self.root_states[:, 3:7])
        roll[roll > np.pi] -= np.pi * 2 # to range (-pi, pi)
        pitch[pitch > np.pi] -= np.pi * 2 # to range (-pi, pi)
        yaw[yaw > np.pi] -= np.pi * 2 # to range (-pi, pi)
        return torch.cat([
            self.root_states[:, :3] - self.env_origins,
            torch.stack([roll, pitch, yaw], dim= -1),
        ], dim= -1)
    
    def _get_robot_config_obs(self, privileged= False):
        return self.robot_config_buffer

    def _get_goal_obs(self, privileged=False):
        goal_obs_buf = torch.cat([self.env_goals_rel, 
                                  self.init_velocities * self.obs_scales.lin_vel,
                                  self.goal_velocities * self.obs_scales.lin_vel], dim=-1)
        return goal_obs_buf

    # -------------Termination----------------
    def _prepare_termination_function(self):
        self.termination_functions = []
        for key in self.cfg.termination.termination_terms:
            name = '_termination_' + key
            self.termination_functions.append(getattr(self, name))
    
    def _termination_roll(self):
        roll_cutoff = torch.abs(self.roll) > self.cfg.termination.termination_threshold['roll']
        self.reset_buf |= roll_cutoff
    
    def _termination_pitch(self):
        pitch_cutoff = torch.abs(self.pitch) > self.cfg.termination.termination_threshold['pitch']
        self.reset_buf |= pitch_cutoff
    
    def _termination_height(self):
        if isinstance(self.cfg.termination.termination_threshold['height'], (tuple, list)):
            z = self.root_states[:, 2] - self.env_origins[:, 2]
            height_low_cutoff = z < self.cfg.termination.termination_threshold['height'][0]
            height_high_cutoff = z > self.cfg.termination.termination_threshold['height'][1]
            self.reset_buf |= height_low_cutoff
            self.reset_buf |= height_high_cutoff
        else:
            height_cutoff = self.root_states[:, 2] < self.cfg.termination.termination_threshold['height']
            self.reset_buf |= height_cutoff

    def _termination_goal(self):
        reach_goal_cutoff = self.reached_goal_ids
        reach_goal_x_cutoff = self.root_states[:, 0] > self.env_goals[:, 0]
        self.reset_buf |= reach_goal_cutoff
        self.reset_buf |= reach_goal_x_cutoff

    # ----------Rewards-----------------
    def _reward_tracking_goal_vel(self):
        norm = torch.norm(self.target_pos_rel, dim=-1, keepdim=True)
        target_vec_norm = self.target_pos_rel / (norm + 1e-5)
        cur_vel = self.root_states[:, 7:9]
        rew = torch.minimum(torch.sum(target_vec_norm[:, :2] * cur_vel, dim=-1), self.max_speed_cmd)
        return rew

    def _reward_tracking_yaw(self):
        rew = torch.exp(-torch.abs(self.target_yaw - self.yaw))
        return rew
    
    def _reward_delta_torques(self):
        return torch.sum(torch.square(self.torques - self.last_torques), dim=1)
    
    def _reward_hip_pos(self):
        return torch.sum(torch.square(self.dof_pos[:, self.hip_indices] - self.default_dof_pos[:, self.hip_indices]), dim=1)

    def _reward_feet_stumble(self):
        # Penalize feet hitting vertical surfaces
        rew = torch.any(torch.norm(self.contact_forces[:, self.feet_indices, :2], dim=2) >\
             4 *torch.abs(self.contact_forces[:, self.feet_indices, 2]), dim=1)
        return rew.float()

    def _reward_feet_edge(self):
        feet_pos_xy = ((self.rigid_body_states[:, self.feet_indices, :2] + self.terrain.cfg.border_size) / self.cfg.terrain.horizontal_scale).round().long()  # (num_envs, 4, 2)
        feet_pos_xy[..., 0] = torch.clip(feet_pos_xy[..., 0], 0, self.x_edge_mask.shape[0]-1)
        feet_pos_xy[..., 1] = torch.clip(feet_pos_xy[..., 1], 0, self.x_edge_mask.shape[1]-1)
        feet_at_edge = self.x_edge_mask[feet_pos_xy[..., 0], feet_pos_xy[..., 1]]
    
        self.feet_at_edge = self.contact_filt & feet_at_edge
        rew = (self.terrain_levels > 3) * torch.sum(self.feet_at_edge, dim=-1)
        return rew

    def _reward_lin_pos_x(self):
        return (self.root_states[:, :3] - self.last_root_pos)[:, 0]
    
    def _reward_goal_vel_align(self):
        rew = torch.sum(torch.square(self.root_states[:, 7:10] - self.goal_velocities), dim=1)
        rew = torch.exp(-rew/self.cfg.rewards.tracking_sigma)
        return_rew = torch.zeros_like(rew)
        return_rew[self.reached_goal_ids] = rew[self.reached_goal_ids]
        return return_rew

    def _reward_tracking_lin_vel(self):
        # Tracking of linear velocity commands (xy axes)
        lin_vel_error = torch.square(1.5 - self.root_states[:, 7]) + torch.square(self.root_states[:, 8])
        return torch.exp(-lin_vel_error/self.cfg.rewards.tracking_sigma)
    
    def _reward_lin_vel_l2norm(self):
        return torch.norm((self.commands[:, :2] - self.base_lin_vel[:, :2]), dim= 1)

    def _reward_world_vel_l2norm(self):
        lin_vel_error = torch.square(1.5 - self.root_states[:, 7]) + torch.square(self.root_states[:, 8])
        return torch.sqrt(lin_vel_error)

    def _reward_tracking_world_vel(self):
        world_vel_error = torch.square(1.5 - self.root_states[:, 7]) + torch.square(self.root_states[:, 8])
        return torch.exp(-world_vel_error/self.cfg.rewards.tracking_sigma)

    def _reward_legs_energy(self):
        return torch.sum(torch.square(self.torques * self.dof_vel), dim=1)

    def _reward_legs_energy_substeps(self):
        # (n_envs, n_substeps, n_dofs) 
        # square sum -> (n_envs, n_substeps)
        # mean -> (n_envs,)
        return torch.mean(torch.sum(torch.square(self.substep_torques * self.substep_dof_vel), dim=-1), dim=-1)

    def _reward_legs_energy_abs(self):
        return torch.sum(torch.abs(self.torques * self.dof_vel), dim=1)

    def _reward_alive(self):
        return 1.

    def _reward_dof_error(self):
        dof_error = torch.sum(torch.square(self.dof_pos - self.default_dof_pos), dim=1)
        return dof_error

    def _reward_lin_cmd(self):
        """ This reward term does not depend on the policy, depends on the command """
        return torch.norm(self.commands[:, :2], dim= 1)

    def _reward_lin_vel_x(self):
        return self.root_states[:, 7]
    
    def _reward_lin_vel_y_abs(self):
        return torch.abs(self.root_states[:, 8])
    
    def _reward_lin_vel_y_square(self):
        return torch.square(self.root_states[:, 8])

    def _reward_lin_pos_y(self):
        return torch.abs((self.root_states[:, :3] - self.env_origins)[:, 1])
    
    def _reward_yaw_abs(self):
        """ Aiming for the robot yaw to be zero (pointing to the positive x-axis) """
        yaw = get_euler_xyz(self.root_states[:, 3:7])[2]
        yaw[yaw > np.pi] -= np.pi * 2 # to range (-pi, pi)
        yaw[yaw < -np.pi] += np.pi * 2 # to range (-pi, pi)
        return torch.abs(yaw)

    def _reward_penetrate_depth(self):
        if not self.check_BarrierTrack_terrain(): return torch.zeros_like(self.root_states[:, 0])
        self.refresh_volume_sample_points()
        penetration_depths = self.terrain.get_penetration_depths(self.volume_sample_points.view(-1, 3)).view(self.num_envs, -1)
        penetration_depths *= torch.norm(self.velocity_sample_points, dim= -1) + 1e-3
        return torch.sum(penetration_depths, dim= -1)

    def _reward_penetrate_volume(self):
        if not self.check_BarrierTrack_terrain(): return torch.zeros_like(self.root_states[:, 0])
        self.refresh_volume_sample_points()
        penetration_mask = self.terrain.get_penetration_mask(self.volume_sample_points.view(-1, 3)).view(self.num_envs, -1)
        penetration_mask *= torch.norm(self.velocity_sample_points, dim= -1) + 1e-3
        return torch.sum(penetration_mask, dim= -1)

    def _reward_tilt_cond(self):
        """ Conditioned reward term in terms of whether the robot is engaging the tilt obstacle
        Use positive factor to enable rolling angle when incountering tilt obstacle
        """
        if not self.check_BarrierTrack_terrain(): return torch.zeros_like(self.root_states[:, 0])
        roll, pitch, yaw = get_euler_xyz(self.root_states[:, 3:7])
        pi = torch.acos(torch.zeros(1)).item() * 2 # which is 3.1415927410125732
        roll[roll > pi] -= pi * 2 # to range (-pi, pi)
        roll[roll < -pi] += pi * 2 # to range (-pi, pi)
        if hasattr(self, "volume_sample_points"):
            self.refresh_volume_sample_points()
            stepping_obstacle_info = self.terrain.get_stepping_obstacle_info(self.volume_sample_points.view(-1, 3))
        else:
            stepping_obstacle_info = self.terrain.get_stepping_obstacle_info(self.root_states[:, :3])
        stepping_obstacle_info = stepping_obstacle_info.view(self.num_envs, -1, stepping_obstacle_info.shape[-1])
        # Assuming that each robot will only be in one obstacle or non obstacle.
        robot_stepping_obstacle_id = torch.max(stepping_obstacle_info[:, :, 0], dim= -1)[0]
        tilting_mask = robot_stepping_obstacle_id == self.terrain.track_options_id_dict["tilt"]
        return_ = torch.where(tilting_mask, torch.clip(torch.abs(roll), 0, torch.pi/2), -torch.clip(torch.abs(roll), 0, torch.pi/2))
        return return_

    def _reward_hip_pos(self):
        return torch.sum(torch.square(self.dof_pos[:, self.hip_indices] - self.default_dof_pos[:, self.hip_indices]), dim=1)

    def _reward_front_hip_pos(self):
        """ Reward the robot to stop moving its front hips """
        return torch.sum(torch.square(self.dof_pos[:, self.front_hip_indices] - self.default_dof_pos[:, self.front_hip_indices]), dim=1)

    def _reward_rear_hip_pos(self):
        """ Reward the robot to stop moving its rear hips """
        return torch.sum(torch.square(self.dof_pos[:, self.rear_hip_indices] - self.default_dof_pos[:, self.rear_hip_indices]), dim=1)
    
    def _reward_down_cond(self):
        if not self.check_BarrierTrack_terrain(): return torch.zeros_like(self.root_states[:, 0])
        if not hasattr(self, "volume_sample_points"): return torch.zeros_like(self.root_states[:, 0])
        self.refresh_volume_sample_points()
        engaging_obstacle_info = self.terrain.get_engaging_block_info(
            self.root_states[:, :3],
            self.volume_sample_points - self.root_states[:, :3].unsqueeze(-2), # (n_envs, n_points, 3)
        )
        roll, pitch, yaw = get_euler_xyz(self.root_states[:, 3:7])
        pi = torch.acos(torch.zeros(1)).item() * 2 # which is 3.1415927410125732
        pitch[pitch > pi] -= pi * 2 # to range (-pi, pi)
        pitch[pitch < -pi] += pi * 2 # to range (-pi, pi)
        engaging_mask = (engaging_obstacle_info[:, 1 + self.terrain.track_options_id_dict["jump"]] > 0) \
            & (engaging_obstacle_info[:, 1 + self.terrain.max_track_options + 2] < 0.)
        pitch_err = torch.abs(pitch - 0.2)
        return torch.exp(-pitch_err/self.cfg.rewards.tracking_sigma) * engaging_mask # the higher positive factor, the more you want the robot to pitch down 0.2 rad

    def _reward_jump_x_vel_cond(self):
        if not self.check_BarrierTrack_terrain(): return torch.zeros_like(self.root_states[:, 0])
        if not hasattr(self, "volume_sample_points"): return torch.zeros_like(self.root_states[:, 0])
        self.refresh_volume_sample_points()
        engaging_obstacle_info = self.terrain.get_engaging_block_info(
            self.root_states[:, :3],
            self.volume_sample_points - self.root_states[:, :3].unsqueeze(-2), # (n_envs, n_points, 3)
        )
        engaging_mask = (engaging_obstacle_info[:, 1 + self.terrain.track_options_id_dict["jump"]] > 0) \
            & (engaging_obstacle_info[:, 1 + self.terrain.max_track_options + 2] > 0.) \
            & (engaging_obstacle_info[:, 0] > 0) # engaging jump-up, not engaging jump-down, positive distance.
        roll, pitch, yaw = get_euler_xyz(self.root_states[:, 3:7])
        pi = torch.acos(torch.zeros(1)).item() * 2 # which is 3.1415927410125732
        pitch[pitch > pi] -= pi * 2 # to range (-pi, pi)
        pitch[pitch < -pi] += pi * 2 # to range (-pi, pi)
        pitch_up_mask = pitch < -0.75 # a hack value

        return torch.clip(self.base_lin_vel[:, 0], max= 1.5) * engaging_mask * pitch_up_mask

    def _reward_sync_legs_cond(self):
        """ A hack to force same actuation on both rear legs when jump. """
        if not self.check_BarrierTrack_terrain(): return torch.zeros_like(self.root_states[:, 0])
        if not hasattr(self, "volume_sample_points"): return torch.zeros_like(self.root_states[:, 0])
        self.refresh_volume_sample_points()
        engaging_obstacle_info = self.terrain.get_engaging_block_info(
            self.root_states[:, :3],
            self.volume_sample_points - self.root_states[:, :3].unsqueeze(-2), # (n_envs, n_points, 3)
        )
        engaging_mask = (engaging_obstacle_info[:, 1 + self.terrain.track_options_id_dict["jump"]] > 0) \
            & (engaging_obstacle_info[:, 1 + self.terrain.max_track_options + 2] > 0.) \
            & (engaging_obstacle_info[:, 0] > 0) # engaging jump-up, not engaging jump-down, positive distance.
        rr_legs = torch.clone(self.actions[:, 6:9]) # shoulder, thigh, calf
        rl_legs = torch.clone(self.actions[:, 9:12]) # shoulder, thigh, calf
        rl_legs[:, 0] *= -1 # flip the sign of shoulder action
        return torch.norm(rr_legs - rl_legs, dim= -1) * engaging_mask
    
    def _reward_sync_all_legs_cond(self):
        """ A hack to force same actuation on both front/rear legs when jump. """
        if not self.check_BarrierTrack_terrain(): return torch.zeros_like(self.root_states[:, 0])
        if not hasattr(self, "volume_sample_points"): return torch.zeros_like(self.root_states[:, 0])
        self.refresh_volume_sample_points()
        engaging_obstacle_info = self.terrain.get_engaging_block_info(
            self.root_states[:, :3],
            self.volume_sample_points - self.root_states[:, :3].unsqueeze(-2), # (n_envs, n_points, 3)
        )
        engaging_mask = (engaging_obstacle_info[:, 1 + self.terrain.track_options_id_dict["jump"]] > 0) \
            & (engaging_obstacle_info[:, 1 + self.terrain.max_track_options + 2] > 0.) \
            & (engaging_obstacle_info[:, 0] > 0) # engaging jump-up, not engaging jump-down, positive distance.
        right_legs = torch.clone(torch.cat([
            self.actions[:, 0:3],
            self.actions[:, 6:9],
        ], dim= -1)) # shoulder, thigh, calf
        left_legs = torch.clone(torch.cat([
            self.actions[:, 3:6],
            self.actions[:, 9:12],
        ], dim= -1)) # shoulder, thigh, calf
        left_legs[:, 0] *= -1 # flip the sign of shoulder action
        left_legs[:, 3] *= -1 # flip the sign of shoulder action
        return torch.norm(right_legs - left_legs, p= 1, dim= -1) * engaging_mask
    
    def _reward_sync_all_legs(self):
        right_legs = torch.clone(torch.cat([
            self.actions[:, 0:3],
            self.actions[:, 6:9],
        ], dim= -1)) # shoulder, thigh, calf
        left_legs = torch.clone(torch.cat([
            self.actions[:, 3:6],
            self.actions[:, 9:12],
        ], dim= -1)) # shoulder, thigh, calf
        left_legs[:, 0] *= -1 # flip the sign of shoulder action
        left_legs[:, 3] *= -1 # flip the sign of shoulder action
        return torch.norm(right_legs - left_legs, p= 1, dim= -1)
    
    def _reward_dof_error_cond(self):
        """ Force dof error when not engaging obstacle """
        if not self.check_BarrierTrack_terrain(): return torch.zeros_like(self.root_states[:, 0])
        if not hasattr(self, "volume_sample_points"): return torch.zeros_like(self.root_states[:, 0])
        self.refresh_volume_sample_points()
        engaging_obstacle_info = self.terrain.get_engaging_block_info(
            self.root_states[:, :3],
            self.volume_sample_points - self.root_states[:, :3].unsqueeze(-2), # (n_envs, n_points, 3)
        )
        engaging_mask = (engaging_obstacle_info[:, 1] > 0)
        return torch.sum(torch.square(self.dof_pos - self.default_dof_pos), dim=1) * engaging_mask
        
    def _reward_leap_bonous_cond(self):
        """ counteract the tracking reward loss during leap"""
        if not self.check_BarrierTrack_terrain(): return torch.zeros_like(self.root_states[:, 0])
        if not hasattr(self, "volume_sample_points"): return torch.zeros_like(self.root_states[:, 0])
        self.refresh_volume_sample_points()
        engaging_obstacle_info = self.terrain.get_engaging_block_info(
            self.root_states[:, :3],
            self.volume_sample_points - self.root_states[:, :3].unsqueeze(-2), # (n_envs, n_points, 3)
        )
        engaging_mask = (engaging_obstacle_info[:, 1 + self.terrain.track_options_id_dict["leap"]] > 0) \
            & (-engaging_obstacle_info[:, 1 + self.terrain.max_track_options + 1] < engaging_obstacle_info[:, 0]) \
            & (engaging_obstacle_info[:, 0] < 0.) # engaging jump-up, not engaging jump-down, positive distance.

        world_vel_error = torch.sum(torch.square(self.commands[:, :2] - self.root_states[:, 7:9]), dim= 1)
        return (1 - torch.exp(-world_vel_error/self.cfg.rewards.tracking_sigma)) * engaging_mask # reverse version of tracking reward

    def _reward_exceed_torque_limits_i(self):
        """ Indicator function """
        max_torques = torch.abs(self.substep_torques).max(dim= 1)[0]
        exceed_torque_each_dof = max_torques > self.torque_limits
        exceed_torque = exceed_torque_each_dof.any(dim= 1)
        return exceed_torque.to(torch.float32)
    
    def _reward_exceed_torque_limits_square(self):
        """ square function for exceeding part """
        exceeded_torques = torch.abs(self.substep_torques) - self.torque_limits
        exceeded_torques[exceeded_torques < 0.] = 0.
        # sum along decimation axis and dof axis
        return torch.square(exceeded_torques).sum(dim= 1).sum(dim= 1)
    
    def _reward_exceed_torque_limits_l1norm(self):
        """ square function for exceeding part """
        exceeded_torques = torch.abs(self.substep_torques) - self.torque_limits
        exceeded_torques[exceeded_torques < 0.] = 0.
        # sum along decimation axis and dof axis
        return torch.norm(exceeded_torques, p= 1, dim= -1).sum(dim= 1)
    
    def _reward_exceed_dof_pos_limits(self):
        return self.substep_exceed_dof_pos_limits.to(torch.float32).sum(dim= -1).mean(dim= -1)

    # -----------Debug-------------------
    def _draw_debug_vis(self):
        super()._draw_debug_vis()
        for i in range(self.num_envs):
            # sphere_geom = gymutil.WireframeSphereGeometry(0.1, 32, 32, None, color=(1, 0, 0))
            # goal = self.terrain_goals[self.terrain_levels[i], self.terrain_types[i]].cpu().numpy()
            # pose = gymapi.Transform(gymapi.Vec3(goal[0], goal[1], goal[2]), r=None)
            # gymutil.draw_lines(sphere_geom, self.gym, self.viewer, self.envs[i], pose)
            
            if not self.cfg.depth.use_camera:
                sphere_geom_arrow = gymutil.WireframeSphereGeometry(0.02, 16, 16, None, color=(1, 0.35, 0.25))
                pose_robot = self.root_states[i, :3].cpu().numpy()      # in world frame
                for j in range(5):
                    norm = torch.norm(self.target_pos_rel, dim=-1, keepdim=True)    # target_pos_rel is in world frame
                    target_vec_norm = self.target_pos_rel / (norm + 1e-5)
                    pose_arrow = pose_robot[:3] + 0.1*(j+3) * target_vec_norm[i, :3].cpu().numpy()
                    pose = gymapi.Transform(gymapi.Vec3(pose_arrow[0], pose_arrow[1], pose_arrow[2]), r=None)
                    gymutil.draw_lines(sphere_geom_arrow, self.gym, self.viewer, self.envs[i], pose)
            sphere_geom = gymutil.WireframeSphereGeometry(0.1, 32, 32, None, color=(1, 0, 0))

            for t in range(10):
                goal = self.terrain_goals[t, self.terrain_types[i]].cpu().numpy()
                pose = gymapi.Transform(gymapi.Vec3(goal[0], goal[1], goal[2]), r=None)
                gymutil.draw_lines(sphere_geom, self.gym, self.viewer, self.envs[i], pose)

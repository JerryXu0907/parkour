# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO
from legged_gym.envs.base.base_config import BaseConfig
from posixpath import relpath
from torch.nn.modules.activation import ReLU
from torch.nn.modules.pooling import MaxPool2d

import torch.nn as nn
import numpy as np

class Go1ParkourCfg(LeggedRobotCfg):
    class env:
        num_envs = 4096
        obs_components = [
            "proprioception", # 46
            "height_measurements", # 187
            "base_pose",        # explicit
            "robot_config",     # latent
            "goal",             # goal
            "prop_history",     # history
        ]
        n_proprio = 46
        n_scan = 132
        n_base = 6
        n_priv_latent = n_robot_config = 1 + 3 + 1 + 12
        n_goal = 3 + 3 + 3
        n_priv = n_priv_explicit = n_base# + n_goal
        # n_priv = 3+3 +3
        # n_priv_latent = 4 + 1 + 12 +12
        # n_proprio = 3 + 2 + 3 + 4 + 36 + 5
        history_len = 10
        num_observations = n_proprio + n_scan + history_len*n_proprio + n_priv_latent + n_goal + n_priv_explicit  #n_scan + n_proprio + n_priv #187 + 47 + 5 + 12 
        num_privileged_obs = None # if not None a priviledge_obs_buf will be returned by step() (critic obs for assymetric training). None is returned otherwise 
        num_actions = 12
        env_spacing = 3.  # not used with heightfields/trimeshes 
        send_timeouts = True # send time out information to the algorithm
        episode_length_s = 20 # episode length in seconds
        obs_type = "og"
        history_encoding = True
        include_foot_contacts = True
        randomize_start_pos = False#True
        randomize_start_vel = False#True
        randomize_start_yaw = False#True
        rand_yaw_range = 1.2
        randomize_start_y = False
        rand_y_range = 0.5
        randomize_start_pitch = False
        rand_pitch_range = 1.6
        contact_buf_len = 100
        next_goal_threshold = 0.2
        reach_goal_delay = 0.1
        num_future_goal_obs = 2

    class depth:
        use_camera = False
        camera_num_envs = 192
        camera_terrain_num_rows = 10
        camera_terrain_num_cols = 20
        position = [0.27, 0, 0.03]  # front camera
        angle = [-5, 5]  # positive pitch down
        update_interval = 5  # 5 works without retraining, 8 worse
        original = (106, 60)
        resized = (87, 58)
        horizontal_fov = 87
        buffer_len = 2
        near_clip = 0
        far_clip = 2
        dis_noise = 0.0
        scale = 1
        invert = True

    class normalization:
        class obs_scales:
            lin_vel = 2.0
            ang_vel = 0.25
            dof_pos = 1.0
            dof_vel = 0.05
            height_measurements = 5.0
        clip_observations = 100.
        clip_actions = 1.2

    class noise:
        add_noise = False
        noise_level = 1.0 # scales other values
        quantize_height = True
        class noise_scales:
            rotation = 0.0
            dof_pos = 0.01
            dof_vel = 0.05
            lin_vel = 0.05
            ang_vel = 0.05
            gravity = 0.02
            height_measurements = 0.02

    class terrain:
        mesh_type = 'trimesh' # "heightfield" # none, plane, heightfield or trimesh
        hf2mesh_method = "grid"  # grid or fast
        max_error = 0.1 # for fast
        max_error_camera = 2

        y_range = [-0.4, 0.4]
        
        edge_width_thresh = 0.05
        horizontal_scale = 0.05 # [m] influence computation time by a lot
        horizontal_scale_camera = 0.1
        vertical_scale = 0.005 # [m]
        border_size = 5 # [m]
        height = [0.02, 0.06]
        simplify_grid = False
        gap_size = [0.02, 0.1]
        stepping_stone_distance = [0.02, 0.08]
        downsampled_scale = 0.075
        curriculum = True

        all_vertical = False
        no_flat = True
        
        static_friction = 1.0
        dynamic_friction = 1.0
        restitution = 0.
        measure_heights = True
        measured_points_x = [-0.45, -0.3, -0.15, 0, 0.15, 0.3, 0.45, 0.6, 0.75, 0.9, 1.05, 1.2] # 1mx1.6m rectangle (without center line)
        measured_points_y = [-0.75, -0.6, -0.45, -0.3, -0.15, 0., 0.15, 0.3, 0.45, 0.6, 0.75]
        measure_horizontal_noise = 0.0

        selected = False # select a unique terrain type and pass all arguments
        terrain_kwargs = None # Dict of arguments for selected terrain
        max_init_terrain_level = 5 # starting curriculum state
        terrain_length = 4.
        terrain_width = 4
        num_rows= 10 # number of terrain rows (levels)  # spreaded is benifitiall !
        num_cols = 40 # number of terrain cols (types)
        
        # goal_vec_ranges = [[0., 0.], [0., 0.], [-0.5, 0.5]]
        goal_vec_ranges = [[0., 0.], [0., 0.], [0., 0.]]
        # trimesh only:
        slope_treshold = 1.5# slopes above this threshold will be corrected to vertical surfaces
        origin_zero_z = True
        num_goals = 1

    class commands:
        curriculum = False
        max_curriculum = 1.
        num_commands = 4 # default: lin_vel_x, lin_vel_y, ang_vel_yaw, heading (in heading mode ang_vel_yaw is recomputed from heading error)
        resampling_time = 6. # time before command are changed[s]
        heading_command = True # if true: compute ang vel command from heading error
        
        lin_vel_clip = 0.2
        ang_vel_clip = 0.4
        # Easy ranges
        class ranges:
            lin_vel_x = [0., 1.5] # min max [m/s]
            lin_vel_y = [0.0, 0.0]   # min max [m/s]
            ang_vel_yaw = [0, 0]    # min max [rad/s]
            heading = [0, 0]

        # Easy ranges
        class max_ranges:
            lin_vel_x = [0.3, 0.8] # min max [m/s]
            lin_vel_y = [-0.3, 0.3]#[0.15, 0.6]   # min max [m/s]
            ang_vel_yaw = [-0, 0]    # min max [rad/s]
            heading = [-1.6, 1.6]

        class crclm_incremnt:
            lin_vel_x = 0.1 # min max [m/s]
            lin_vel_y = 0.1  # min max [m/s]
            ang_vel_yaw = 0.1    # min max [rad/s]
            heading = 0.5

        waypoint_delta = 0.7

    class init_state( LeggedRobotCfg.init_state ):
        pos = [0.0, 0.0, 0.42] # x,y,z [m]
        default_joint_angles = { # = target angles [rad] when action = 0.0
            'FL_hip_joint': 0.1,   # [rad]
            'RL_hip_joint': 0.1,   # [rad]
            'FR_hip_joint': -0.1 ,  # [rad]
            'RR_hip_joint': -0.1,   # [rad]

            'FL_thigh_joint': 0.8,     # [rad]
            'RL_thigh_joint': 1.,   # [rad]
            'FR_thigh_joint': 0.8,     # [rad]
            'RR_thigh_joint': 1.,   # [rad]

            'FL_calf_joint': -1.5,   # [rad]
            'RL_calf_joint': -1.5,    # [rad]
            'FR_calf_joint': -1.5,  # [rad]
            'RR_calf_joint': -1.5,    # [rad]
        }
    
    class control( LeggedRobotCfg.control ):
        # PD Drive parameters:
        control_type = 'P'
        # stiffness = {'joint': 40.}  # [N*m/rad]
        # damping = {'joint': 0.5}     # [N*m*s/rad]
        stiffness = {'joint': 30.}  # [N*m/rad]
        damping = {'joint': 0.6}     # [N*m*s/rad]
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.25
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4

    class asset( LeggedRobotCfg.asset ):
        # file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/go1/urdf/go1.urdf'
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/a1/urdf/a1.urdf'
        foot_name = "foot"
        penalize_contacts_on = ["thigh", "calf"]
        terminate_after_contacts_on = ["base"]#, "thigh", "calf"]
        self_collisions = 1 # 1 to disable, 0 to enable...bitwise filter
  
    class domain_rand:
        randomize_friction = True
        friction_range = [0.6, 2.]
        randomize_base_mass = False
        added_mass_range = [0., 3.]
        randomize_com = True
        added_com_range = [-0.2, 0.2]
        class com_range:
            x = [-0.2, 0.2]
            y = [-0.2, 0.2]
            z = [-0.2, 0.2]
        push_robots = True
        push_interval_s = 8
        max_push_vel_xy = 0.5
        init_base_vel_range = [0, 0.8]
        randomize_motor = True
        leg_motor_strength_range = [0.8, 1.2]

        delay_update_global_steps = 24 * 8000
        action_delay = False
        action_curr_step = [1, 1]
        action_curr_step_scratch = [0, 1]
        action_delay_view = 1
        action_buf_len = 8
        init_base_pos_range = {"x": [-0.2, 0.2], "y": [-0.2, 0.2]}
        
    class rewards:
        class scales:
            # tracking rewards
            # tracking_goal_vel = 1.5
            # tracking_lin_vel = 3.0
            tracking_yaw = 0.5
            # lin_pos_x = 10.0
            # # regularization rewards
            # lin_vel_z = -1.0
            # ang_vel_xy = -0.05
            # orientation = -1.
            # dof_acc = -2.5e-7
            # collision = -10.
            # action_rate = -0.1
            # delta_torques = -1.0e-7
            # torques = -0.00001
            # hip_pos = -0.5
            # dof_error = -0.04
            # feet_stumble = -1
            # feet_edge = -1

            tracking_ang_vel = 0.05
            world_vel_l2norm = -1.
            legs_energy_substeps = -2e-5
            legs_energy = -0.
            alive = 2.
            # penalty for hardware safety
            exceed_dof_pos_limits = -1e-1
            exceed_torque_limits_i = -2e-1
        soft_dof_pos_limit = 0.9
        base_height_target = 0.25
        only_positive_rewards = True # if true negative total rewards are clipped at zero (avoids early termination problems)
        tracking_sigma = 0.2 # tracking reward = exp(-error^2/sigma)
        # soft_dof_pos_limit = 1. # percentage of urdf limits, values above this limit are penalized
        soft_dof_vel_limit = 1
        soft_torque_limit = 0.4
        # base_height_target = 1.
        max_contact_force = 40. # forces above this value are penalized
    # viewer camera:
    class viewer:
        ref_env = 0
        pos = [10, 0, 6]  # [m]
        lookat = [11., 5, 3.]  # [m]

    class sim:
        dt =  0.005
        substeps = 1
        gravity = [0., 0. ,-9.81]  # [m/s^2]
        up_axis = 1  # 0 is y, 1 is z
        no_camera = True
        
        class physx:
            num_threads = 10
            solver_type = 1  # 0: pgs, 1: tgs
            num_position_iterations = 4
            num_velocity_iterations = 0
            contact_offset = 0.01  # [m]
            rest_offset = 0.0   # [m]
            bounce_threshold_velocity = 0.5 #0.5 [m/s]
            max_depenetration_velocity = 1.0
            max_gpu_contact_pairs = 2**23 #2**24 -> needed for 8000 envs and more
            default_buffer_size_multiplier = 5
            contact_collection = 2 # 0: never, 1: last sub-step, 2: all sub-steps (default=2)

    class termination:
        termination_terms = ['roll', 'pitch', 'goal', 'height']
        termination_threshold = {
            'roll': 1.5,
            'pitch': 1.5,
            'height': 0.
        }


class Go1ParkourCfgPPO(BaseConfig):
    seed = 1
    runner_class_name = 'OnPolicyRunner'
 
    class policy:
        init_noise_std = 1.0
        continue_from_last_std = True
        scan_encoder_dims = [128, 64, 32]
        actor_hidden_dims = [512, 256, 128]
        critic_hidden_dims = [512, 256, 128]
        priv_encoder_dims = [64, 20]
        activation = 'elu' # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid
        # only for 'ActorCriticRecurrent':
        rnn_type = 'lstm'
        rnn_hidden_size = 512
        rnn_num_layers = 1

        tanh_encoder_output = True #False
    
    class algorithm:
        # training params
        value_loss_coef = 1.0
        use_clipped_value_loss = True
        clip_param = 0.2
        entropy_coef = 0.01
        num_learning_epochs = 5
        num_mini_batches = 4 # mini batch size = num_envs*nsteps / nminibatches
        learning_rate = 2.e-4 #5.e-4
        schedule = 'adaptive' # could be adaptive, fixed
        gamma = 0.99
        lam = 0.95
        desired_kl = 0.01
        max_grad_norm = 1.
        # dagger params
        dagger_update_freq = 20
        priv_reg_coef_schedual = [0, 0.1, 2000, 3000]
        priv_reg_coef_schedual_resume = [0, 0.1, 0, 1]
    
    class depth_encoder:
        if_depth = Go1ParkourCfg.depth.use_camera
        depth_shape = Go1ParkourCfg.depth.resized
        buffer_len = Go1ParkourCfg.depth.buffer_len
        hidden_dims = 512
        learning_rate = 1.e-3
        num_steps_per_env = Go1ParkourCfg.depth.update_interval * 24

    class estimator:
        train_with_estimated_states = True
        learning_rate = 1.e-4
        hidden_dims = [128, 64]
        priv_states_dim = Go1ParkourCfg.env.n_priv
        num_prop = Go1ParkourCfg.env.n_proprio
        num_scan = Go1ParkourCfg.env.n_scan

    class runner:
        policy_class_name = 'ActorCritic'
        algorithm_class_name = 'PPO'
        num_steps_per_env = 24 # per iteration
        max_iterations = 50000 # number of policy updates

        # logging
        save_interval = 100 # check for potential saves every this many iterations
        experiment_name = 'rough_go1'
        run_name = ''
        # load and resume
        resume = False
        load_run = -1 # -1 = last run
        checkpoint = -1 # -1 = last saved model
        resume_path = None # updated from load_run and chkpt
        
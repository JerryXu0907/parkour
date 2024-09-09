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

import numpy as np
from numpy.random import choice
from scipy import interpolate
import random
from isaacgym import terrain_utils
from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg
from scipy import ndimage
from pydelatin import Delatin
import pyfqmr
from scipy.ndimage import binary_dilation


class Terrain:
    def __init__(self, cfg: LeggedRobotCfg.terrain, num_robots) -> None:
        self.cfg = cfg
        self.num_robots = num_robots
        self.type = cfg.mesh_type
        if self.type in ["none", 'plane']:
            return
        self.env_length = cfg.terrain_length
        self.env_width = cfg.terrain_width

        self.goal_vec_ranges = self.cfg.goal_vec_ranges
        # calculate total sub-terrains
        self.cfg.num_sub_terrains = cfg.num_rows * cfg.num_cols
        self.env_origins = np.zeros((cfg.num_rows, cfg.num_cols, 3))
        self.env_goals = np.zeros((cfg.num_rows, cfg.num_cols, 3))

        # goals should be only one
        self.goals = np.zeros((cfg.num_rows, cfg.num_cols, 1, 3))

        self.width_per_env_pixels = int(self.env_width / cfg.horizontal_scale)
        self.length_per_env_pixels = int(self.env_length / cfg.horizontal_scale)

        self.border = int(cfg.border_size/self.cfg.horizontal_scale)
        self.tot_cols = int(cfg.num_cols * self.width_per_env_pixels) + 2 * self.border
        self.tot_rows = int(cfg.num_rows * self.length_per_env_pixels) + 2 * self.border

        self.height_field_raw = np.zeros((self.tot_rows , self.tot_cols), dtype=np.int16)
        self.curriculum()
        
        self.heightsamples = self.height_field_raw
        if self.type=="trimesh":
            print("Converting heightmap to trimesh...")
            if cfg.hf2mesh_method == "grid":
                self.vertices, self.triangles, self.x_edge_mask = convert_heightfield_to_trimesh(   self.height_field_raw,
                                                                                                self.cfg.horizontal_scale,
                                                                                                self.cfg.vertical_scale,
                                                                                                self.cfg.slope_treshold)
                half_edge_width = int(self.cfg.edge_width_thresh / self.cfg.horizontal_scale)
                structure = np.ones((half_edge_width*2+1, 1))
                self.x_edge_mask = binary_dilation(self.x_edge_mask, structure=structure)
                if self.cfg.simplify_grid:
                    mesh_simplifier = pyfqmr.Simplify()
                    mesh_simplifier.setMesh(self.vertices, self.triangles)
                    mesh_simplifier.simplify_mesh(target_count = int(0.05*self.triangles.shape[0]), aggressiveness=7, preserve_border=True, verbose=10)

                    self.vertices, self.triangles, normals = mesh_simplifier.getMesh()
                    self.vertices = self.vertices.astype(np.float32)
                    self.triangles = self.triangles.astype(np.uint32)
            else:
                assert cfg.hf2mesh_method == "fast", "Height field to mesh method must be grid or fast"
                self.vertices, self.triangles = convert_heightfield_to_trimesh_delatin(self.height_field_raw, self.cfg.horizontal_scale, self.cfg.vertical_scale)
            print("Created {} vertices".format(self.vertices.shape[0]))
            print("Created {} triangles".format(self.triangles.shape[0]))

    def randomized_terrain(self):
        for k in range(self.cfg.num_sub_terrains):
            # Env coordinates in the world
            (i, j) = np.unravel_index(k, (self.cfg.num_rows, self.cfg.num_cols))

            choice = np.random.uniform(0, 1)
            # difficulty = np.random.choice([0.5, 0.75, 0.9])
            difficulty = np.random.uniform(-0.2, 1.2)
            terrain = self.make_terrain(choice, difficulty)
            self.add_terrain_to_map(terrain, i, j)
        
    def curriculum(self, random=False, max_difficulty=False):
        # self.goal_vec_ranges is a 3x2 vector
        for j in range(self.cfg.num_cols):
            height = [(self.goal_vec_ranges[-1][-1] - self.goal_vec_ranges[-1][0]) / self.cfg.num_cols * j + self.goal_vec_ranges[-1][0],]
            # if self.cfg.num_cols // 2 - 2 <= j <= self.cfg.num_cols // 2 + 2:
            if False:
                for i in range(self.cfg.num_rows):
                    terrain = terrain_utils.SubTerrain("terrain",
                                width=self.length_per_env_pixels,
                                length=self.width_per_env_pixels,
                                vertical_scale=self.cfg.vertical_scale,
                                horizontal_scale=self.cfg.horizontal_scale)
                    gap_size = 0.
                    platform_height = parkour_gap_terrain(terrain,
                                        gap_size=gap_size,
                                        gap_depth=[0.2, 1],
                                        pad_height=0,
                                        x_range=[0.8, 1.5],
                                        y_range=self.cfg.y_range,
                                        half_valid_width=[0.6, 1.2],
                                        # flat=True
                                        )
                    self.add_roughness(terrain)
                    self.add_terrain_to_map(terrain, i, j, platform_height)
            else:
                for i in range(self.cfg.num_rows):
                    difficulty = i / (self.cfg.num_rows-1)
                    # if difficulty < 0.1:
                    #     terrain = terrain_utils.SubTerrain("terrain",
                    #             width=self.length_per_env_pixels,
                    #             length=self.width_per_env_pixels,
                    #             vertical_scale=self.cfg.vertical_scale,
                    #             horizontal_scale=self.cfg.horizontal_scale)
                    #     gap_size = 0.
                    #     platform_height = parkour_gap_terrain(terrain,
                    #                         gap_size=gap_size,
                    #                         gap_depth=[0.2, 1],
                    #                         pad_height=0,
                    #                         x_range=[0.8, 1.5],
                    #                         y_range=self.cfg.y_range,
                    #                         half_valid_width=[0.6, 1.2],
                    #                         # flat=True
                    #                         )
                    #     self.add_roughness(terrain)
                    #     self.add_terrain_to_map(terrain, i, j, platform_height)
                    # else:
                    terrain, platform_height = self.make_terrain_goal(height, difficulty)
                    self.add_terrain_to_map(terrain, i, j, platform_height)
                

    def selected_terrain(self):
        terrain_type = self.cfg.terrain_kwargs.pop('type')
        for k in range(self.cfg.num_sub_terrains):
            # Env coordinates in the world
            (i, j) = np.unravel_index(k, (self.cfg.num_rows, self.cfg.num_cols))

            terrain = terrain_utils.SubTerrain("terrain",
                              width=self.width_per_env_pixels,
                              length=self.length_per_env_pixels,
                              vertical_scale=self.vertical_scale,
                              horizontal_scale=self.horizontal_scale)

            eval(terrain_type)(terrain, **self.cfg.terrain_kwargs.terrain_kwargs)
            self.add_terrain_to_map(terrain, i, j)
    
    def add_roughness(self, terrain, difficulty=1):
        max_height = (self.cfg.height[1] - self.cfg.height[0]) * difficulty + self.cfg.height[0]
        height = random.uniform(self.cfg.height[0], max_height)
        terrain_utils.random_uniform_terrain(terrain, min_height=-height, max_height=height, step=0.005, downsampled_scale=self.cfg.downsampled_scale)

    def add_terrain_to_map(self, terrain, row, col, platform_height=0.):
        i = row
        j = col
        # map coordinate system
        start_x = self.border + i * self.length_per_env_pixels
        end_x = self.border + (i + 1) * self.length_per_env_pixels
        start_y = self.border + j * self.width_per_env_pixels
        end_y = self.border + (j + 1) * self.width_per_env_pixels
        self.height_field_raw[start_x: end_x, start_y:end_y] = terrain.height_field_raw

        # env_origin_x = (i + 0.5) * self.env_length
        env_origin_x = i * self.env_length + 1.0
        env_origin_y = (j + 0.5) * self.env_width
        x1 = int((self.env_length/2. - 0.5) / terrain.horizontal_scale) # within 1 meter square range
        x2 = int((self.env_length/2. + 0.5) / terrain.horizontal_scale)
        y1 = int((self.env_width/2. - 0.5) / terrain.horizontal_scale)
        y2 = int((self.env_width/2. + 0.5) / terrain.horizontal_scale)
        # if self.cfg.origin_zero_z:
        #     env_origin_z = 0
        # else:
        env_origin_z = platform_height
        # env_origin_z = np.max(terrain.height_field_raw[x1:x2, y1:y2])*terrain.vertical_scale
        self.env_origins[i, j] = [env_origin_x, env_origin_y, env_origin_z]
        # self.terrain_type[i, j] = terrain.idx
        self.goals[i, j] = terrain.goals + [i * self.env_length, j * self.env_width, 0]
        # self.env_goals[i, j] = self.goals[i, j] - self.env_origins[i, j]
        # self.env_slope_vec[i, j] = terrain.slope_vector

    def make_terrain_goal(self, goal_vec, difficulty):
        terrain = terrain_utils.SubTerrain("terrain",
                                width=self.length_per_env_pixels,
                                length=self.width_per_env_pixels,
                                vertical_scale=self.cfg.vertical_scale,
                                horizontal_scale=self.cfg.horizontal_scale)
        
        if np.abs(goal_vec[-1]) > 0.1:
            platform_height = parkour_step_terrain(terrain, num_stones=1, step_height=goal_vec[-1] * difficulty)
            platform_height = 0. if goal_vec[-1] > 0. else -goal_vec[-1]
            self.add_roughness(terrain)
        else:
            gap_size = 0.1 + difficulty * 1.
            platform_height = parkour_gap_terrain(terrain,
                                gap_size=gap_size,
                                gap_depth=[0.2, 1],
                                pad_height=0,
                                x_range=[0.8, 1.5],
                                y_range=self.cfg.y_range,
                                half_valid_width=[0.6, 1.2],
                                # flat=True
                                )
            self.add_roughness(terrain)
        return terrain, platform_height
    
def parkour_gap_terrain(terrain,
                           platform_len=1.5, 
                           platform_height=0., 
                           gap_size=0.3,
                           x_range=[0.3, 0.6],
                           y_range=[-1.2, 1.2],
                           half_valid_width=[0.6, 1.2],
                           gap_depth=-200,
                           pad_width=0.1,
                           pad_height=0.5,
                           flat=False):
    
    '''
    terrain.width = platform_len + fixed_range + platform_len
    '''

    goals = np.zeros((1, 3))
    mid_y = terrain.length // 2  # length is actually y width
    # dis_y_min = round(y_range[0] / terrain.horizontal_scale)
    # dis_y_max = round(y_range[1] / terrain.horizontal_scale)

    platform_len = round(platform_len / terrain.horizontal_scale)
    platform_height = round(platform_height / terrain.vertical_scale)
    gap_depth = -round(np.random.uniform(gap_depth[0], gap_depth[1]) / terrain.vertical_scale)
    
    half_valid_width = round(np.random.uniform(half_valid_width[0], half_valid_width[1]) / terrain.horizontal_scale)
    # terrain.height_field_raw[:, :mid_y-half_valid_width] = gap_depth
    # terrain.height_field_raw[:, mid_y+half_valid_width:] = gap_depth
    
    terrain.height_field_raw[0:platform_len, :] = platform_height

    gap_size = round(gap_size / terrain.horizontal_scale)
    # dis_x_min = round(x_range[0] / terrain.horizontal_scale) + gap_size
    # dis_x_max = round(x_range[1] / terrain.horizontal_scale) + gap_size

    dis_x_min = gap_size // 2
    dis_x_max = terrain.width - 2*platform_len - gap_size // 2

    dis_x = platform_len
    rand_x = np.random.randint(dis_x_min, dis_x_max)
    dis_x += rand_x
    # discard rand_y now
    # rand_y = np.random.randint(dis_y_min, dis_y_max)
    if not flat:
        terrain.height_field_raw[dis_x-gap_size//2 : dis_x+gap_size//2, :] = gap_depth

    # terrain.height_field_raw[last_dis_x:dis_x, :mid_y+rand_y-half_valid_width] = gap_depth
    # terrain.height_field_raw[last_dis_x:dis_x, mid_y+rand_y+half_valid_width:] = gap_depth
    goals[0] = [(terrain.width + dis_x + gap_size//2) // 2, mid_y, 0]# + rand_y]    
    terrain.goals = goals * terrain.horizontal_scale
    
    # terrain.height_field_raw[:, :] = 0
    # pad edges
    pad_width = int(pad_width // terrain.horizontal_scale)
    pad_height = int(pad_height // terrain.vertical_scale)
    terrain.height_field_raw[:, :pad_width] = pad_height
    terrain.height_field_raw[:, -pad_width:] = pad_height
    terrain.height_field_raw[:pad_width, :] = pad_height
    terrain.height_field_raw[-pad_width:, :] = pad_height

    return platform_height

def parkour_step_terrain(terrain,
                        platform_len=2.5, 
                        platform_height=0., 
                        num_stones=1,
                        x_range=[0.2, 0.4],
                        y_range=[-0.15, 0.15],
                        half_valid_width=[0.45, 0.5],
                        step_height = 0.2,
                        pad_width=0.1,
                        pad_height=0.5):
    
    goals = np.zeros((1, 3))
    mid_y = terrain.length // 2  # length is actually y width

    # dis_x_min = round( (x_range[0] + step_height) / terrain.horizontal_scale)
    # dis_x_max = round( (x_range[1] + step_height) / terrain.horizontal_scale)


    if step_height < 0:
        platform_height = -step_height
    step_height = round(step_height / terrain.vertical_scale)

    half_valid_width = round(np.random.uniform(half_valid_width[0], half_valid_width[1]) / terrain.horizontal_scale)

    platform_len = round(platform_len / terrain.horizontal_scale)
    platform_height = round(platform_height / terrain.vertical_scale)
    terrain.height_field_raw[0:platform_len, :] = platform_height

    dis_x_min = platform_len
    dis_x_max = terrain.width - platform_len

    dis_x = platform_len
    stair_height = 0

    rand_x = np.random.randint(dis_x_min, dis_x_max)
    # rand_y = np.random.randint(dis_y_min, dis_y_max)

    stair_height += step_height
    terrain.height_field_raw[dis_x:rand_x, ] = platform_height
    # make sure the scale of the terrain is the same
    goals[0] = [(rand_x + terrain.width)//2 * terrain.horizontal_scale, mid_y * terrain.horizontal_scale, stair_height * terrain.vertical_scale]
    terrain.height_field_raw[rand_x:, :] = stair_height
    # final_dis_x = dis_x + np.random.randint(dis_x_min, dis_x_max)
    # # import ipdb; ipdb.set_trace()
    # if final_dis_x > terrain.width:
    #     final_dis_x = terrain.width - 0.5 // terrain.horizontal_scale
    
    terrain.goals = goals
    
    # terrain.height_field_raw[:, :max(mid_y-half_valid_width, 0)] = 0
    # terrain.height_field_raw[:, min(mid_y+half_valid_width, terrain.height_field_raw.shape[1]):] = 0
    # terrain.height_field_raw[:, :] = 0
    # pad edges
    pad_width = int(pad_width // terrain.horizontal_scale)
    pad_height = int(pad_height // terrain.vertical_scale)
    terrain.height_field_raw[:, :pad_width] = pad_height
    terrain.height_field_raw[:, -pad_width:] = pad_height
    terrain.height_field_raw[:pad_width, :] = pad_height
    terrain.height_field_raw[-pad_width:, :] = pad_height

    return platform_height

def convert_heightfield_to_trimesh_delatin(height_field_raw, horizontal_scale, vertical_scale, max_error=0.01):
    mesh = Delatin(np.flip(height_field_raw, axis=1).T, z_scale=vertical_scale, max_error=max_error)
    vertices = np.zeros_like(mesh.vertices)
    vertices[:, :2] = mesh.vertices[:, :2] * horizontal_scale
    vertices[:, 2] = mesh.vertices[:, 2]
    return vertices, mesh.triangles

def convert_heightfield_to_trimesh(height_field_raw, horizontal_scale, vertical_scale, slope_threshold=None):
    """
    Convert a heightfield array to a triangle mesh represented by vertices and triangles.
    Optionally, corrects vertical surfaces above the provide slope threshold:

        If (y2-y1)/(x2-x1) > slope_threshold -> Move A to A' (set x1 = x2). Do this for all directions.
                   B(x2,y2)
                  /|
                 / |
                /  |
        (x1,y1)A---A'(x2',y1)

    Parameters:
        height_field_raw (np.array): input heightfield
        horizontal_scale (float): horizontal scale of the heightfield [meters]
        vertical_scale (float): vertical scale of the heightfield [meters]
        slope_threshold (float): the slope threshold above which surfaces are made vertical. If None no correction is applied (default: None)
    Returns:
        vertices (np.array(float)): array of shape (num_vertices, 3). Each row represents the location of each vertex [meters]
        triangles (np.array(int)): array of shape (num_triangles, 3). Each row represents the indices of the 3 vertices connected by this triangle.
    """
    hf = height_field_raw
    num_rows = hf.shape[0]
    num_cols = hf.shape[1]

    y = np.linspace(0, (num_cols-1)*horizontal_scale, num_cols)
    x = np.linspace(0, (num_rows-1)*horizontal_scale, num_rows)
    yy, xx = np.meshgrid(y, x)

    if slope_threshold is not None:

        slope_threshold *= horizontal_scale / vertical_scale
        move_x = np.zeros((num_rows, num_cols))
        move_y = np.zeros((num_rows, num_cols))
        move_corners = np.zeros((num_rows, num_cols))
        move_x[:num_rows-1, :] += (hf[1:num_rows, :] - hf[:num_rows-1, :] > slope_threshold)
        move_x[1:num_rows, :] -= (hf[:num_rows-1, :] - hf[1:num_rows, :] > slope_threshold)
        move_y[:, :num_cols-1] += (hf[:, 1:num_cols] - hf[:, :num_cols-1] > slope_threshold)
        move_y[:, 1:num_cols] -= (hf[:, :num_cols-1] - hf[:, 1:num_cols] > slope_threshold)
        move_corners[:num_rows-1, :num_cols-1] += (hf[1:num_rows, 1:num_cols] - hf[:num_rows-1, :num_cols-1] > slope_threshold)
        move_corners[1:num_rows, 1:num_cols] -= (hf[:num_rows-1, :num_cols-1] - hf[1:num_rows, 1:num_cols] > slope_threshold)
        xx += (move_x + move_corners*(move_x == 0)) * horizontal_scale
        yy += (move_y + move_corners*(move_y == 0)) * horizontal_scale

    # create triangle mesh vertices and triangles from the heightfield grid
    vertices = np.zeros((num_rows*num_cols, 3), dtype=np.float32)
    vertices[:, 0] = xx.flatten()
    vertices[:, 1] = yy.flatten()
    vertices[:, 2] = hf.flatten() * vertical_scale
    triangles = -np.ones((2*(num_rows-1)*(num_cols-1), 3), dtype=np.uint32)
    for i in range(num_rows - 1):
        ind0 = np.arange(0, num_cols-1) + i*num_cols
        ind1 = ind0 + 1
        ind2 = ind0 + num_cols
        ind3 = ind2 + 1
        start = 2*i*(num_cols-1)
        stop = start + 2*(num_cols-1)
        triangles[start:stop:2, 0] = ind0
        triangles[start:stop:2, 1] = ind3
        triangles[start:stop:2, 2] = ind1
        triangles[start+1:stop:2, 0] = ind0
        triangles[start+1:stop:2, 1] = ind2
        triangles[start+1:stop:2, 2] = ind3

    return vertices, triangles, move_x != 0

if __name__ == "__main__":

    class terrain(LeggedRobotCfg.terrain):
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
        terrain_length = 8.
        terrain_width = 4
        num_rows= 10 # number of terrain rows (levels)  # spreaded is benifitiall !
        num_cols = 40 # number of terrain cols (types)
        
        goal_vec_ranges = np.array([[0., 0.], [0., 0.], [-1.0, 1.0]])
        # trimesh only:
        slope_treshold = 1.5# slopes above this threshold will be corrected to vertical surfaces
        origin_zero_z = True

        num_goals = 1

    from isaacgym import gymutil, gymapi
    gym = gymapi.acquire_gym()
    cfg = terrain()
    # parse arguments
    args = gymutil.parse_arguments()

    # configure sim
    sim_params = gymapi.SimParams()
    sim_params.up_axis = gymapi.UpAxis.UP_AXIS_Z
    sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.81)

    if args.physics_engine == gymapi.SIM_FLEX:
        print("WARNING: Terrain creation is not supported for Flex! Switching to PhysX")
        args.physics_engine = gymapi.SIM_PHYSX
    sim_params.substeps = 2
    sim_params.physx.solver_type = 1
    sim_params.physx.num_position_iterations = 4
    sim_params.physx.num_velocity_iterations = 0
    sim_params.physx.num_threads = args.num_threads
    sim_params.physx.use_gpu = args.use_gpu

    sim = gym.create_sim(args.compute_device_id, args.graphics_device_id, args.physics_engine, sim_params)
    if sim is None:
        print("*** Failed to create sim")
        quit()

    # load ball asset
    import os
    # asset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, os.pardir, "assets")
    # asset_root = "/home/jerryxu/Desktop/isaacgym/assets"
    asset_root = "/home/zhengjie/Desktop/isaacgym/assets"
    asset_file = "urdf/ball.urdf"
    asset = gym.load_asset(sim, asset_root, asset_file, gymapi.AssetOptions())

    # set up the env grid
    cfg.num_rows = 10
    cfg.num_cols = 4
    cfg.num_envs = 64

    env_spacing = 0.56
    env_lower = gymapi.Vec3(-env_spacing, -env_spacing, 0.0)
    env_upper = gymapi.Vec3(env_spacing, env_spacing, env_spacing)
    pose = gymapi.Transform()
    pose.r = gymapi.Quat(0, 0, 0, 1)
    pose.p.z = 1.
    pose.p.x = 3.

    envs = []
    # set random seed
    np.random.seed(17)
    for i in range(cfg.num_envs):
        # create env
        env = gym.create_env(sim, env_lower, env_upper, int(np.sqrt(cfg.num_envs)))
        envs.append(env)

        # generate random bright color
        c = 0.5 + 0.5 * np.random.random(3)
        color = gymapi.Vec3(c[0], c[1], c[2])

        ahandle = gym.create_actor(env, asset, pose, None, 0, 0)
        gym.set_rigid_body_color(env, ahandle, 0, gymapi.MESH_VISUAL_AND_COLLISION, color)

    # create a local copy of initial state, which we can send back for reset
    initial_state = np.copy(gym.get_sim_rigid_body_states(sim, gymapi.STATE_ALL))
    t = Terrain(cfg, cfg.num_envs)
    tm_params = gymapi.TriangleMeshParams()
    tm_params.nb_vertices = t.vertices.shape[0]
    tm_params.nb_triangles = t.triangles.shape[0]
    tm_params.transform.p.x = -cfg.border_size 
    tm_params.transform.p.y = -cfg.border_size
    tm_params.transform.p.z = 0.0
    tm_params.static_friction = cfg.static_friction
    tm_params.dynamic_friction = cfg.dynamic_friction
    tm_params.restitution = cfg.restitution

    gym.add_triangle_mesh(sim, t.vertices.flatten(), t.triangles.flatten(), tm_params)

    # create viewer
    viewer = gym.create_viewer(sim, gymapi.CameraProperties())
    if viewer is None:
        print("*** Failed to create viewer")
        quit()

    cam_pos = gymapi.Vec3(-5, -5, 15)
    cam_target = gymapi.Vec3(0, 0, 10)
    gym.viewer_camera_look_at(viewer, None, cam_pos, cam_target)

    # subscribe to spacebar event for reset
    gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_R, "reset")
    gym.subscribe_viewer_keyboard_event(
        viewer, gymapi.KEY_ESCAPE, "QUIT")
    gym.subscribe_viewer_keyboard_event(
        viewer, gymapi.KEY_V, "toggle_viewer_sync")
    gym.subscribe_viewer_keyboard_event(
        viewer, gymapi.KEY_F, "free_cam")
    for i in range(9):
        gym.subscribe_viewer_keyboard_event(
        viewer, getattr(gymapi, "KEY_"+str(i)), "lookat"+str(i))
    gym.subscribe_viewer_keyboard_event(
        viewer, gymapi.KEY_LEFT_BRACKET, "prev_id")
    gym.subscribe_viewer_keyboard_event(
        viewer, gymapi.KEY_RIGHT_BRACKET, "next_id")
    gym.subscribe_viewer_keyboard_event(
        viewer, gymapi.KEY_SPACE, "pause")
    gym.subscribe_viewer_keyboard_event(
        viewer, gymapi.KEY_W, "vx_plus")
    gym.subscribe_viewer_keyboard_event(
        viewer, gymapi.KEY_S, "vx_minus")
    gym.subscribe_viewer_keyboard_event(
        viewer, gymapi.KEY_A, "left_turn")
    gym.subscribe_viewer_keyboard_event(
        viewer, gymapi.KEY_D, "right_turn")

    def _draw_height_samples(i):
        """ Draws visualizations for dubugging (slows down simulation a lot).
            Default behaviour: draws height measurement points
        """
        # draw height lines
        gym.refresh_rigid_body_state_tensor(sim)
        sphere_geom = gymutil.WireframeSphereGeometry(0.5, 4, 4, None, color=(1, 1, 0), color2=(1, 0, 0))
        # base_pos = (t.env_origins[0, i, :3])#.cpu().numpy()
        base_pos = (t.env_goals[0, i, :3])#.cpu().numpy()
        # heights = measured_heights[i].cpu().numpy()
        # height_points = quat_apply_yaw(base_quat[i].repeat(heights.shape[0]), self.height_points[i]).cpu().numpy()
        # for j in range(heights.shape[0]):
        #     x = height_points[j, 0] + base_pos[0]
        #     y = height_points[j, 1] + base_pos[1]
        #     z = heights[j]
        sphere_pose = gymapi.Transform(gymapi.Vec3(base_pos[0], base_pos[1], base_pos[2]), r=None)
        gymutil.draw_lines(sphere_geom, gym, viewer, None, sphere_pose)
        base_pos = (t.env_goals[1, i, :3])#.cpu().numpy()
        # heights = measured_heights[i].cpu().numpy()
        # height_points = quat_apply_yaw(base_quat[i].repeat(heights.shape[0]), self.height_points[i]).cpu().numpy()
        # for j in range(heights.shape[0]):
        #     x = height_points[j, 0] + base_pos[0]
        #     y = height_points[j, 1] + base_pos[1]
        #     z = heights[j]
        sphere_geom = gymutil.WireframeSphereGeometry(0.5, 4, 4, None, color=(1, 1, 0), color2=(1, 0, 0))
        sphere_pose = gymapi.Transform(gymapi.Vec3(base_pos[0], base_pos[1], base_pos[2]), r=None)
        gymutil.draw_lines(sphere_geom, gym, viewer, None, sphere_pose)
    

    _draw_height_samples(1)
    _draw_height_samples(2)
    _draw_height_samples(3)
    _draw_height_samples(0)

    while not gym.query_viewer_has_closed(viewer):

        # Get input actions from the viewer and handle them appropriately
        for evt in gym.query_viewer_action_events(viewer):
            if evt.action == "reset" and evt.value > 0:
                gym.set_sim_rigid_body_states(sim, initial_state, gymapi.STATE_ALL)

        # step the physics
        gym.simulate(sim)
        gym.fetch_results(sim, True)

        # update the viewer
        gym.step_graphics(sim)
        gym.draw_viewer(viewer, sim, True)

        # Wait for dt to elapse in real time.
        # This synchronizes the physics simulation with the rendering rate.
        gym.sync_frame_time(sim)

    
    # for i in range(len(envs)):
    #     print("here")
    #     _draw_height_samples(i)

    gym.destroy_viewer(viewer)
    gym.destroy_sim(sim)
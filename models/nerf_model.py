import torch
import torch.nn.functional as F
from torch import nn


class ObjectNeRF(nn.Module):
    def __init__(
        self,
        model_config,
    ):
        super(ObjectNeRF, self).__init__()
        self.model_config = model_config
        self.use_voxel_embedding = self.model_config.use_voxel_embedding
        # initialize neural model with config
        self.initialize_scene_branch(model_config)
        self.initialize_object_branch(model_config)

    def initialize_scene_branch(self, model_config):
        self.D = model_config["D"]
        self.W = model_config["W"]
        self.N_freq_xyz = model_config["N_freq_xyz"]
        self.N_freq_dir = model_config["N_freq_dir"]
        self.skips = model_config["skips"]
        # embedding size for voxel representation
        if self.use_voxel_embedding:
            self.N_scn_voxel_size = model_config.get("N_scn_voxel_size", 0)
            self.N_freq_voxel = model_config["N_freq_voxel"]
            voxel_emb_size = (
                self.N_scn_voxel_size + self.N_scn_voxel_size * self.N_freq_voxel * 2
            )
        else:
            voxel_emb_size = 0
        # embedding size for NeRF xyz
        xyz_emb_size = 3 + 3 * self.N_freq_xyz * 2
        self.in_channels_xyz = xyz_emb_size + voxel_emb_size
        self.in_channels_dir = 3 + 3 * self.N_freq_dir * 2

        self.activation = nn.LeakyReLU(inplace=True)

        # xyz encoding layers
        for i in range(self.D):
            if i == 0:
                layer = nn.Linear(self.in_channels_xyz, self.W)
            elif i in self.skips:
                layer = nn.Linear(self.W + self.in_channels_xyz, self.W)
            else:
                layer = nn.Linear(self.W, self.W)
            layer = nn.Sequential(layer, self.activation)
            setattr(self, f"xyz_encoding_{i+1}", layer)
        self.xyz_encoding_final = nn.Linear(self.W, self.W)

        # output layers
        self.sigma = nn.Linear(self.W, 1)
        self.rgb = nn.Sequential(nn.Linear(self.W // 2, 3), nn.Sigmoid())
        # direction encoding layers
        self.dir_encoding = nn.Sequential(
            nn.Linear(self.W + self.in_channels_dir, self.W // 2), self.activation
        )

    def initialize_object_branch(self, model_config):
        # instance encoding
        N_obj_code_length = model_config["N_obj_code_length"]
        if self.use_voxel_embedding:
            N_obj_voxel_size = model_config.get("N_obj_voxel_size", 0)
            inst_voxel_emb_size = (
                N_obj_voxel_size + N_obj_voxel_size * self.N_freq_voxel * 2
            )
        else:
            inst_voxel_emb_size = 0
        self.inst_channel_in = (
            self.in_channels_xyz + N_obj_code_length + inst_voxel_emb_size
        )
        self.inst_D = model_config["inst_D"]
        self.inst_W = model_config["inst_W"]
        self.inst_skips = model_config["inst_skips"]

        for i in range(self.inst_D):
            if i == 0:
                layer = nn.Linear(self.inst_channel_in, self.inst_W)
            elif i in self.inst_skips:
                layer = nn.Linear(self.inst_W + self.inst_channel_in, self.inst_W)
            else:
                layer = nn.Linear(self.inst_W, self.inst_W)
            layer = nn.Sequential(layer, self.activation)
            setattr(self, f"instance_encoding_{i+1}", layer)
        self.instance_encoding_final = nn.Sequential(
            nn.Linear(self.inst_W, self.inst_W),
        )
        self.instance_sigma = nn.Linear(self.inst_W, 1)

        self.inst_dir_encoding = nn.Sequential(
            nn.Linear(self.inst_W + self.in_channels_dir, self.inst_W // 2),
            self.activation,
        )
        self.inst_rgb = nn.Sequential(nn.Linear(self.inst_W // 2, 3), nn.Sigmoid())

    def forward(self, inputs, sigma_only=False):
        output_dict = {}
        input_xyz = inputs["emb_xyz"]
        input_dir = inputs.get("emb_dir", None)

        xyz_ = input_xyz
        for i in range(self.D):
            if i in self.skips:
                xyz_ = torch.cat([input_xyz, xyz_], -1)
            xyz_ = getattr(self, f"xyz_encoding_{i+1}")(xyz_)

        sigma = self.sigma(xyz_)
        output_dict["sigma"] = sigma

        if sigma_only:
            return output_dict

        xyz_encoding_final = self.xyz_encoding_final(xyz_)

        dir_encoding_input = torch.cat([xyz_encoding_final, input_dir], -1)
        dir_encoding = self.dir_encoding(dir_encoding_input)
        rgb = self.rgb(dir_encoding)
        output_dict["rgb"] = rgb

        return output_dict

    def forward_instance(self, inputs, sigma_only=False):
        output_dict = {}
        emb_xyz = inputs["emb_xyz"]
        input_dir = inputs.get("emb_dir", None)
        obj_code = inputs["obj_code"]
        if self.use_voxel_embedding:
            obj_voxel = inputs["obj_voxel"]
            input_x = torch.cat([emb_xyz, obj_voxel, obj_code], -1)
        else:
            input_x = torch.cat([emb_xyz, obj_code], -1)

        x_ = input_x

        for i in range(self.inst_D):
            if i in self.inst_skips:
                x_ = torch.cat([input_x, x_], -1)
            x_ = getattr(self, f"instance_encoding_{i+1}")(x_)
        inst_sigma = self.instance_sigma(x_)
        output_dict["inst_sigma"] = inst_sigma

        if sigma_only:
            return output_dict

        x_final = self.instance_encoding_final(x_)
        dir_encoding_input = torch.cat([x_final, input_dir], -1)
        dir_encoding = self.inst_dir_encoding(dir_encoding_input)
        rgb = self.inst_rgb(dir_encoding)
        output_dict["inst_rgb"] = rgb

        return output_dict

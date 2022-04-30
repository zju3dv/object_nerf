import sys
import os

sys.path.append(os.getcwd())  # noqa
import torch
import torch.nn.functional as F
from torch import nn
import numpy as np
import itertools
import einops
from einops import rearrange
from utils.util import write_point_cloud


def dump_voxel_occupancy_map(
    voxel_occupancy_map, voxel_size, scale_factor, scene_center
):
    idx_occu = torch.nonzero(voxel_occupancy_map)
    voxel_xyz = (
        idx_occu.float().detach().cpu().numpy() * voxel_size * scale_factor
        + scene_center
    )
    write_point_cloud(voxel_xyz, "voxel.ply")


def randomly_set_occupancy_mask_to_true(occupancy_mask, ratio=0.1):
    # which may help with empty embedding learning?
    assert len(occupancy_mask.shape) == 1
    N = occupancy_mask.shape
    idx_empty = torch.nonzero(occupancy_mask == False)
    if idx_empty.shape[0] < 10:
        return occupancy_mask
    # randomly set occupancy which has been marked as False to True
    N_empty = len(idx_empty)
    idx_rand_choice = torch.randperm(N_empty)[: int(N_empty * ratio)]
    occupancy_mask[idx_empty[idx_rand_choice]] = True
    return occupancy_mask


class Embedding(nn.Module):
    def __init__(self, in_channels, N_freqs, logscale=True):
        """
        Defines a function that embeds x to (x, sin(2^k x), cos(2^k x), ...)
        in_channels: number of input channels (3 for both xyz and direction)
        """
        super(Embedding, self).__init__()
        self.N_freqs = N_freqs
        self.in_channels = in_channels
        self.funcs = [torch.sin, torch.cos]
        self.out_channels = in_channels * (len(self.funcs) * N_freqs + 1)

        if logscale:
            self.freq_bands = 2 ** torch.linspace(0, N_freqs - 1, N_freqs)
        else:
            self.freq_bands = torch.linspace(1, 2 ** (N_freqs - 1), N_freqs)

    def forward(self, x):
        """
        Embeds x to (x, sin(2^k x), cos(2^k x), ...)
        Different from the paper, "x" is also in the output
        See https://github.com/bmild/nerf/issues/12

        Inputs:
            x: (B, self.in_channels)

        Outputs:
            out: (B, self.out_channels)
        """
        out = [x]
        for freq in self.freq_bands:
            for func in self.funcs:
                out += [func(freq * x)]

        return torch.cat(out, -1)


class EmbeddingVoxel(nn.Module):
    def __init__(self, channels, N_freqs, max_voxels, dataset_extra_config):
        super(EmbeddingVoxel, self).__init__()
        self.embedding_final = Embedding(channels, N_freqs)
        self.embedding_space_ftr = nn.Embedding(max_voxels, channels)
        self.set_pointclouds(dataset_extra_config)
        self.channels = channels
        self.embedding_xyz_classical = Embedding(3, 10)

    def set_pointclouds(self, dataset_extra_config):
        # load pointclouds from file
        import open3d as o3d
        from utils.util import read_yaml

        self.conf = dataset_extra_config
        pcd = o3d.io.read_point_cloud(self.conf["pcd_path"])

        pcd_xyz = np.asarray(pcd.points)

        scene_center = np.array(self.conf["scene_center"])
        scale_factor = self.conf["scale_factor"]
        pcd_xyz = (pcd_xyz - scene_center) / scale_factor

        # assert pcd_xyz.shape[0] <= self.embedding_space_ftr.num_embeddings
        # if isinstance(pcd_xyz, np.ndarray):
        pcd_xyz = torch.from_numpy(pcd_xyz).float()
        pcd_xyz = pcd_xyz.float().cuda()

        # quantize and normalize voxels
        voxel_size = self.conf["voxel_size"] / scale_factor
        self.register_buffer("voxel_size", torch.scalar_tensor(voxel_size))
        self.register_buffer(
            "bounds", torch.stack([pcd_xyz.min(dim=0)[0], pcd_xyz.max(dim=0)[0]])
        )
        # self.bounds = [pcd_xyz.min(dim=0)[0], pcd_xyz.max(dim=0)[0]]
        self.register_buffer("voxel_offset", -self.bounds[0])
        # self.voxel_offset = -self.bounds[0]
        # print(self.bounds)
        self.register_buffer(
            "voxel_shape",
            torch.tensor(
                [
                    ((self.bounds[1][i] - self.bounds[0][i]) / self.voxel_size)
                    .int()
                    .item()
                    + 3
                    for i in range(3)
                ]
            ).cuda(),
        )
        self.register_buffer(
            "voxel_count", torch.scalar_tensor(self.voxel_shape.prod())
        )

        # mark voxel_occupancy voxels
        self.register_buffer(
            "voxel_occupancy",
            torch.zeros(
                (self.voxel_shape[0], self.voxel_shape[1], self.voxel_shape[2])
            ).bool(),
        )

        # quantize xyz to start from index 0
        pcd_quantize = ((pcd_xyz + self.voxel_offset) / self.voxel_size).round().long()

        # mark center and nearby voxels
        print("Filling the voxel_occupancy...")
        invalid_mask = torch.logical_or(
            ((pcd_quantize < 0).sum(1) > 0),
            ((pcd_quantize >= self.voxel_shape).sum(1) > 0),
        )
        pcd_quantize = pcd_quantize[~invalid_mask]
        self.voxel_occupancy[
            pcd_quantize[:, 0], pcd_quantize[:, 1], pcd_quantize[:, 2]
        ] = True

        # marker neighbors by Conv3d
        MARK_NEIGHBOR = self.conf["neighbor_marks"]
        conv_neighbor = nn.Conv3d(
            1,
            1,
            kernel_size=MARK_NEIGHBOR,
            bias=False,
            padding=(MARK_NEIGHBOR - 1) // 2,
        )
        conv_neighbor.weight.data[:, :, :] = 1
        conv_neighbor = conv_neighbor.cuda()
        orig_shape = self.voxel_occupancy.shape
        self.voxel_occupancy = (
            conv_neighbor(self.voxel_occupancy[None, None, ...].float().cuda())
            .bool()
            .squeeze()
        )
        # mark border surface
        assert self.voxel_occupancy.shape == orig_shape
        # check voxel occupancy
        # dump_voxel_occupancy_map(self.voxel_occupancy, self.voxel_size, scale_factor, scene_center)
        print(
            "Voxel generated:",
            self.voxel_occupancy.shape,
            "Voxel occupancy ratio:",
            self.voxel_occupancy.sum() / self.voxel_occupancy.numel(),
        )
        print("Voxel used:", self.voxel_occupancy.sum())

        # construct voxel idx map for storing sparse voxel
        self.generate_voxel_idx_map()

        self.instance_ftr_C = 8

    def generate_voxel_idx_map(self):
        # construct voxel idx map for storing sparse voxel
        self.register_buffer(
            "voxel_idx_map",
            -torch.ones((self.voxel_shape[0], self.voxel_shape[1], self.voxel_shape[2]))
            .long()
            .cuda(),
        )
        # self.voxel_idx_map =
        idx_occu = torch.nonzero(self.voxel_occupancy)
        assert self.embedding_space_ftr.num_embeddings >= idx_occu.shape[0]
        self.voxel_idx_map[
            idx_occu[:, 0], idx_occu[:, 1], idx_occu[:, 2]
        ] = torch.arange(idx_occu.shape[0]).cuda()

    def self_pruning_empty_voxels(self, model, max_alpha_th=0.5):
        idx_occu = torch.nonzero(self.voxel_occupancy)
        voxel_xyz = idx_occu.float() * self.voxel_size - self.voxel_offset
        N_occu = voxel_xyz.shape[0]
        # TODO(ybbbbt): this consume too much of the memory
        N_samples_per_voxel = 16**3
        N_points_per_batch = 32
        empty_mask = torch.ones(N_occu, dtype=bool, device=voxel_xyz.device)
        empty_mask = []
        cnt_prune = 0
        for i in range(0, N_occu, N_points_per_batch):
            voxel_samples = einops.repeat(
                voxel_xyz[i : i + N_points_per_batch],
                "n1 c -> (n1 n2) c",
                n2=N_samples_per_voxel,
            )
            voxel_samples += (
                torch.rand_like(voxel_samples) * self.voxel_size - self.voxel_size / 2
            )
            # voxel_ftrs, _ = self.compute_voxel_features_sparse(voxel_samples, True)
            voxel_ftrs, _ = self.forward(voxel_samples)
            sigmas, _ = model(voxel_ftrs, sigma_only=True)
            # this alpha means the transmitance consuming when go through 1 meters of the field
            alphas = 1 - torch.exp(-torch.relu(sigmas.squeeze()))
            alphas = rearrange(alphas, "(n1 n2) -> n1 n2", n2=N_samples_per_voxel).max(
                -1
            )[0]
            # empty_mask = alphas < max_alpha_th
            empty_mask_chunk = alphas < max_alpha_th
            cnt_prune += empty_mask_chunk.sum()
            empty_mask += [empty_mask_chunk]
            print("\r{:04d} Pruning voxels ".format(i), cnt_prune, end="")
        print("")
        empty_mask = torch.cat(empty_mask, 0)
        idx_empty = idx_occu[empty_mask, :]
        print(
            "Self pruning {} voxels in {} voxels, pruing ratio {:05f}".format(
                idx_empty.shape[0],
                idx_occu.shape[0],
                idx_empty.shape[0] / idx_occu.shape[0],
            )
        )
        self.voxel_occupancy[idx_empty[:, 0], idx_empty[:, 1], idx_empty[:, 2]] = False
        self.voxel_idx_map[idx_empty[:, 0], idx_empty[:, 1], idx_empty[:, 2]] = -1

    def voxel_subdivision(self):
        idx_occu = torch.nonzero(self.voxel_occupancy)
        voxel_xyz = idx_occu.float() * self.voxel_size - self.voxel_offset

        print("Voxel subdivision begin with {} voxels".format(idx_occu.shape[0]))

        target_voxel_size = self.voxel_size / 2

        # grow from left down side
        corners = [0, 1]
        new_voxel_xyz = []
        for c in itertools.product(corners, repeat=3):
            new_voxel_xyz += [voxel_xyz + torch.tensor(c).cuda() * target_voxel_size]
        new_voxel_xyz = torch.cat(new_voxel_xyz, 0)
        # new_voxel_xyz = voxel_xyz
        new_voxel_xyz_coord = (
            ((new_voxel_xyz + self.voxel_offset) / target_voxel_size).round().long()
        )

        with torch.no_grad():
            new_voxel_ftrs = self.compute_voxel_features_sparse(
                new_voxel_xyz, trilinear_interpolate=True, positional_embedding=False
            )

        # update voxel informations
        self.voxel_size = target_voxel_size
        self.voxel_shape *= 2
        self.voxel_occupancy = (
            torch.zeros((self.voxel_shape[0], self.voxel_shape[1], self.voxel_shape[2]))
            .bool()
            .cuda()
        )
        self.voxel_occupancy[
            new_voxel_xyz_coord[:, 0],
            new_voxel_xyz_coord[:, 1],
            new_voxel_xyz_coord[:, 2],
        ] = True
        self.voxel_count = (
            self.voxel_shape[0] * self.voxel_shape[1] * self.voxel_shape[2]
        )
        self.generate_voxel_idx_map()

        # update voxel features
        with torch.no_grad():
            assign_idx = self.voxel_idx_map[
                new_voxel_xyz_coord[:, 0],
                new_voxel_xyz_coord[:, 1],
                new_voxel_xyz_coord[:, 2],
            ]
            self.embedding_space_ftr.weight[assign_idx] = new_voxel_ftrs

        print(
            "Voxel subdivision end with {} voxels".format(
                torch.nonzero(self.voxel_occupancy).shape[0]
            )
        )

    def ray_box_intersection(self, rays_o, rays_d, batch_near, batch_far):
        # a brute-force version
        N_samples = 256
        z_steps = torch.linspace(0, 1, N_samples, device=rays_o.device)  # (N_samples)
        z_vals = batch_near * (1 - z_steps) + batch_far * z_steps
        xyz_test = rays_o + rays_d * rearrange(z_vals, "n1 n2 -> n1 n2 1")
        N1, N2, _ = xyz_test.shape
        xyz_test = rearrange(xyz_test, "n1 n2 c -> (n1 n2) c")
        occu = self.check_occupancy(xyz_test)
        occu = rearrange(occu, "(n1 n2) -> n1 n2", n1=N1, n2=N2)

        batch_near_out = (z_vals + (~occu).float() * 1e9).min(-1)[0].unsqueeze(-1)
        m = batch_near_out > 1e5
        batch_near_out[m] = batch_near[m]

        batch_far_out = (z_vals * occu.float()).max(-1)[0].unsqueeze(-1)
        m = batch_far_out == 0
        batch_far_out[m] = batch_far[m]
        # print(batch_near_out.min(), batch_near_out.max(), batch_far_out.min(), batch_far_out.max())
        return batch_near_out, batch_far_out

    def forward(self, xyz):
        voxel_ftr, inst_ftr = self.compute_voxel_features_sparse(xyz, True, True)
        xyz_ftr = self.embedding_xyz_classical(xyz)
        voxel_ftr = torch.cat([voxel_ftr, xyz_ftr], -1)
        return voxel_ftr, inst_ftr

    def get_voxel_feature_sparse_from_quantized(self, xyz_quantize):
        """
        get voxel features from quantized xyz coord
        """
        # remove points out of bound
        invalid_mask = torch.logical_or(
            (xyz_quantize < 0).sum(1) > 0, (xyz_quantize >= self.voxel_shape).sum(1) > 0
        )
        xyz_quantize[invalid_mask] = 0

        # get sparse voxel indices from quantized coord and voxel_idx_map
        embedding_idx = self.voxel_idx_map[
            xyz_quantize[:, 0], xyz_quantize[:, 1], xyz_quantize[:, 2]
        ]
        # remove idx==-1, which means empty
        empty_mask = embedding_idx < 0
        invalid_mask = torch.logical_or(invalid_mask, empty_mask)
        # just a placeholder idx
        embedding_idx[invalid_mask] = self.embedding_space_ftr.num_embeddings - 1
        voxel_ftr = self.embedding_space_ftr(embedding_idx)
        voxel_ftr[invalid_mask] = 0
        return voxel_ftr, invalid_mask

    def compute_voxel_features_sparse(
        self, xyz, trilinear_interpolate, positional_embedding=True
    ):
        """
        get voxel features with sparse indexing and trilinear interpolation
        """
        N, _ = xyz.shape
        xyz_scaled = (xyz + self.voxel_offset) / self.voxel_size
        if trilinear_interpolate:
            xyz_quantize = xyz_scaled.floor().long()
            corners = [0, 1]
            xyz_quantize_all = []
            for c in itertools.product(corners, repeat=3):
                xyz_quantize_all += [xyz_quantize + torch.tensor(c).cuda()]
            xyz_quantize_all = torch.cat(xyz_quantize_all, 0)
            voxel_ftr, invalid_mask = self.get_voxel_feature_sparse_from_quantized(
                xyz_quantize_all
            )
            p = xyz_scaled - xyz_quantize.float()
            u, v, w = p[:, 0], p[:, 1], p[:, 2]
            l_u, l_v, l_w = 1 - u, 1 - v, 1 - w
            weights = [
                (l_u) * (l_v) * (l_w),
                (l_u) * (l_v) * (w),
                (l_u) * (v) * (l_w),
                (l_u) * (v) * (w),
                (u) * (l_v) * (l_w),
                (u) * (l_v) * (w),
                (u) * (v) * (l_w),
                (u) * (v) * (w),
            ]
            weights = torch.cat(weights, 0)
            # print(voxel_ftr.shape, weights.shape)
            voxel_ftr = (
                (voxel_ftr * weights.view(-1, 1)).view(8, N, -1).sum(0, keepdim=False)
            )
            # only when all eight voxels are marked as invalid
            invalid_mask = invalid_mask.view(8, N).int().sum(0, keepdim=False) == 8
            # if self.training:
            #     invalid_mask = ~randomly_set_occupancy_mask_to_true(~invalid_mask)
            # voxel_ftr[invalid_mask] = 0
        else:
            xyz_quantize = xyz_scaled.round().long()
            voxel_ftr, invalid_mask = self.get_voxel_feature_sparse_from_quantized(
                xyz_quantize
            )
            # if self.training:
            #     invalid_mask = ~randomly_set_occupancy_mask_to_true(~invalid_mask)
            # voxel_ftr[invalid_mask] = 0
        _, C = voxel_ftr.shape
        scene_x, instance_x = torch.split(
            voxel_ftr, [C - self.instance_ftr_C, self.instance_ftr_C], dim=-1
        )
        # return self.embedding_final(voxel_ftr)
        if positional_embedding:
            return self.embedding_final(scene_x), self.embedding_final(instance_x)
        else:
            return voxel_ftr

    def check_occupancy(self, xyz):
        xyz_quantize = ((xyz + self.voxel_offset) / self.voxel_size).round().long()
        # remove points out of bound
        invalid_mask = torch.logical_or(
            ((xyz_quantize < 0).sum(1) > 0),
            (xyz_quantize >= self.voxel_shape).sum(1) > 0,
        )
        # print((xyz_quantize < 0).sum())
        xyz_quantize[invalid_mask] = 0
        # get occupancy mask through pt coord
        occupancy_mask = self.voxel_occupancy[
            xyz_quantize[:, 0], xyz_quantize[:, 1], xyz_quantize[:, 2]
        ]
        occupancy_mask[invalid_mask] = False
        return occupancy_mask

    def forward_voxel_features_dense(self, xyz):
        H, W, D = self.voxel_shape[0], self.voxel_shape[1], self.voxel_shape[2]
        assert self.embedding_space_ftr.num_embeddings >= H * W * D
        # self.pcd_xyz (N, 3)
        # xyz (M, 3)
        M, N = xyz.shape[0], self.pcd_xyz.shape[0]
        # print(self.embedding_space_ftr.weight.shape)

        # remove empty
        occupancy_mask = self.check_occupancy(xyz)
        ind_full = torch.arange(M)
        # ind_occu = ind_full[occupancy_mask]
        ind_occu = ind_full
        ind_empty = ind_full[~occupancy_mask]
        xyz_occu = xyz[ind_occu]

        # trilinear interpolate voxel features
        embedding_3d = rearrange(
            self.embedding_space_ftr(torch.arange(self.voxel_count).cuda()),
            "(n1 n2 n3) c -> 1 c n1 n2 n3",
            n1=H,
            n2=W,
            n3=D,
        )
        xyz_normalized = (xyz_occu - self.bounds[0]) / (
            self.bounds[1] - self.bounds[0]
        )  # normalize to (0, 1)
        grid = xyz_normalized * 2 - 1  # normalize to (-1, 1) # (M_occu, 3)
        # print(grid.min(), grid.max())
        grid = rearrange(grid, "n1 c -> 1 1 1 n1 c")
        # acutally trilinear interpolation if mode = 'bilinear'
        xyz_ftr = (
            F.grid_sample(
                embedding_3d,
                grid,
                padding_mode="border",
                align_corners=True,
                mode="bilinear",
            )
            .squeeze()
            .permute(1, 0)
        )

        # peridic encoding
        xyz_ftr = self.embedding_final(xyz_ftr)

        # put back empty
        x_full = torch.zeros((M, xyz_ftr.shape[1]), device="cuda")
        x_full[ind_occu] = xyz_ftr

        return x_full

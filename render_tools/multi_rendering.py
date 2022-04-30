import ipdb
import contextlib
import torch
from torch import nn
from einops import rearrange, reduce, repeat
from typing import List, Dict, Any, Optional
import numpy as np
import random
from models.nerf_model import ObjectNeRF

from models.rendering import sample_pdf
from utils.bbox_utils import check_in_any_boxes
from utils.util import print_val_range


def inference_from_model(
    model: ObjectNeRF,
    embedding_xyz: torch.nn.Module,
    dir_embedded: torch.Tensor,
    code_library: torch.nn.Module,
    xyz: torch.Tensor,
    z_vals: torch.Tensor,
    chunk: int,
    instance_id: int,
    # kwargs={},
):
    compute_3d_mask = instance_id > 0
    N_rays = xyz.shape[0]
    N_samples_ = xyz.shape[1]
    xyz_ = rearrange(xyz, "n1 n2 c -> (n1 n2) c")  # (N_rays*N_samples_, 3)

    # Perform model inference to get rgb and raw sigma
    B = xyz_.shape[0]
    out_rgb_chunks = []
    out_sigma_chunks = []
    mask_3d_instance_chunk = []
    instance_rgb_chunk = []

    # hack to suppress zero values
    zero_mask = z_vals[:, -1] == 0
    # zero_mask_repeat = repeat(zero_mask, "n1 -> (n1 n2)", n2=N_samples_)

    dir_embedded_ = repeat(dir_embedded, "n1 c -> (n1 n2) c", n2=N_samples_)
    # (N_rays*N_samples_, embed_dir_channels)
    if compute_3d_mask:
        inst_embedded_ = code_library.embedding_instance(
            torch.ones(
                (dir_embedded_.shape[0]), dtype=torch.long, device=dir_embedded_.device
            )
            * instance_id
        )
        assert dir_embedded_.shape[0] == inst_embedded_.shape[0]

    for i in range(0, B, chunk):
        xyz_embedded, inst_voxel_embedded = embedding_xyz(xyz_[i : i + chunk])
        # xyz_embedded[zero_mask_repeat[i : i + chunk], :] = 0

        # xyzdir_embedded = torch.cat([xyz_embedded, dir_embedded_[i : i + chunk]], 1)
        input_dict = {
            "emb_xyz": xyz_embedded,
            "emb_dir": dir_embedded_[i : i + chunk],
        }
        if compute_3d_mask:
            input_dict["obj_code"] = inst_embedded_[i : i + chunk]
            input_dict["obj_voxel"] = inst_voxel_embedded
            output = model.forward_instance(input_dict)
            mask_3d_instance_chunk += [output["inst_sigma"]]
            instance_rgb_chunk += [output["inst_rgb"]]
        else:
            output = model(input_dict, sigma_only=False)
            out_rgb_chunks += [output["rgb"]]
            out_sigma_chunks += [output["sigma"]]

    if compute_3d_mask:
        mask_3d_instance = torch.cat(mask_3d_instance_chunk, 0)
        mask_3d_instance = rearrange(
            mask_3d_instance, "(n1 n2) 1 -> n1 n2", n1=N_rays, n2=N_samples_
        )
        instance_rgb = torch.cat(instance_rgb_chunk, 0)
        instance_rgb = rearrange(
            instance_rgb, "(n1 n2) c -> n1 n2 c", n1=N_rays, n2=N_samples_
        )
        mask_3d_instance[zero_mask] = -1e5
        return instance_rgb, mask_3d_instance
    else:
        rgbs = torch.cat(out_rgb_chunks, 0)
        sigmas = torch.cat(out_sigma_chunks, 0)
        rgbs = rearrange(rgbs, "(n1 n2) c -> n1 n2 c", n1=N_rays, n2=N_samples_, c=3)
        sigmas = rearrange(
            sigmas, "(n1 n2) c -> n1 (n2 c)", n1=N_rays, n2=N_samples_, c=1
        )
        sigmas[zero_mask] = -1e5
        return rgbs, sigmas


def volume_rendering_multi(
    results: Dict[str, Any],
    typ: str,
    z_vals_list: list,
    rgbs_list: list,
    sigmas_list: list,
    noise_std: float,
    white_back: bool,
    obj_ids_list: list = None,
):
    N_objs = len(z_vals_list)
    # order via z_vals
    z_vals = torch.cat(z_vals_list, 1)  # (N_rays, N_samples*N_objs)
    rgbs = torch.cat(rgbs_list, 1)  # (N_rays, N_samples*N_objs, 3)
    sigmas = torch.cat(sigmas_list, 1)  # (N_rays, N_samples*N_objs)

    z_vals, idx_sorted = torch.sort(z_vals, -1)
    # # TODO(ybbbbt): ugly order three axis
    for i in range(3):
        rgbs[:, :, i] = torch.gather(rgbs[:, :, i], dim=1, index=idx_sorted)
    sigmas = torch.gather(sigmas, dim=1, index=idx_sorted)
    # record object ids for recovering weights of each object after sorting
    if obj_ids_list != None:
        obj_ids = torch.cat(obj_ids_list, -1)
        results[f"obj_ids_{typ}"] = torch.gather(obj_ids, dim=1, index=idx_sorted)

    # Convert these values using volume rendering (Section 4)
    deltas = z_vals[:, 1:] - z_vals[:, :-1]  # (N_rays, N_samples_-1)
    # delta_inf = 1e10 * torch.ones_like(deltas[:, :1]) # (N_rays, 1) the last delta is infinity
    delta_inf = torch.zeros_like(
        deltas[:, :1]
    )  # (N_rays, 1) the last delta is infinity
    deltas = torch.cat([deltas, delta_inf], -1)  # (N_rays, N_samples_)

    # compute alpha by the formula (3)
    noise = torch.randn_like(sigmas) * noise_std
    alphas = 1 - torch.exp(-deltas * torch.relu(sigmas + noise))  # (N_rays, N_samples_)

    alphas_shifted = torch.cat(
        [torch.ones_like(alphas[:, :1]), 1 - alphas + 1e-10], -1
    )  # [1, 1-a1, 1-a2, ...]
    weights = alphas * torch.cumprod(alphas_shifted[:, :-1], -1)  # (N_rays, N_samples_)

    weights_sum = reduce(
        weights, "n1 n2 -> n1", "sum"
    )  # (N_rays), the accumulated opacity along the rays
    # equals "1 - (1-a1)(1-a2)...(1-an)" mathematically

    results[f"weights_{typ}"] = weights
    results[f"opacity_{typ}"] = weights_sum
    results[f"z_vals_{typ}"] = z_vals

    rgb_map = reduce(
        rearrange(weights, "n1 n2 -> n1 n2 1") * rgbs, "n1 n2 c -> n1 c", "sum"
    )
    depth_map = reduce(weights * z_vals, "n1 n2 -> n1", "sum")

    if white_back:
        rgb_map = rgb_map + 1 - weights_sum.unsqueeze(-1)

    results[f"rgb_{typ}"] = rgb_map
    results[f"depth_{typ}"] = depth_map


def render_rays_multi(
    models: Dict[str, ObjectNeRF],
    embeddings: Dict[str, torch.nn.Module],
    code_library: torch.nn.Module,
    rays_list: list,
    obj_instance_ids: list,
    N_samples: int = 64,
    use_disp: bool = False,
    perturb: float = 0,
    noise_std: float = 0,
    N_importance: int = 0,
    chunk: int = 1024 * 32,
    white_back: bool = False,
    background_skip_bbox: Dict[str, Any] = None,  # skip rays inside the bbox
    # **kwargs,
):

    embedding_xyz, embedding_dir = embeddings["xyz"], embeddings["dir"]

    assert len(rays_list) == len(obj_instance_ids)

    z_vals_list = []
    xyz_coarse_list = []
    dir_embedded_list = []
    rays_o_list = []
    rays_d_list = []

    for idx, rays in enumerate(rays_list):
        # Decompose the inputs
        N_rays = rays.shape[0]
        rays_o, rays_d = rays[:, 0:3], rays[:, 3:6]  # both (N_rays, 3)
        near, far = rays[:, 6:7], rays[:, 7:8]  # both (N_rays, 1)

        # Embed direction
        dir_embedded = embedding_dir(rays_d)  # (N_rays, embed_dir_channels)

        rays_o = rearrange(rays_o, "n1 c -> n1 1 c")
        rays_d = rearrange(rays_d, "n1 c -> n1 1 c")

        # compute intersection to update near and far
        # near, far = embedding_xyz.ray_box_intersection(rays_o, rays_d, near, far)

        rays_o_list += [rays_o]
        rays_d_list += [rays_d]

        # Sample depth points
        z_steps = torch.linspace(0, 1, N_samples, device=rays.device)  # (N_samples)
        if not use_disp:  # use linear sampling in depth space
            z_vals = near * (1 - z_steps) + far * z_steps
        else:  # use linear sampling in disparity space
            z_vals = 1 / (1 / near * (1 - z_steps) + 1 / far * z_steps)

        z_vals = z_vals.expand(N_rays, N_samples)

        xyz_coarse = rays_o + rays_d * rearrange(z_vals, "n1 n2 -> n1 n2 1")

        # save for each rays batch
        xyz_coarse_list += [xyz_coarse]
        z_vals_list += [z_vals]
        dir_embedded_list += [dir_embedded]

    # inference for each objects
    rgbs_list = []
    sigmas_list = []
    obj_ids_list = []
    for i in range(len(rays_list)):
        rgbs, sigmas = inference_from_model(
            model=models["coarse"],
            embedding_xyz=embedding_xyz,
            dir_embedded=dir_embedded_list[i],
            code_library=code_library,
            xyz=xyz_coarse_list[i],
            z_vals=z_vals_list[i],
            chunk=chunk,
            instance_id=obj_instance_ids[i],
            # kwargs,
        )

        # mute in bound samples to remove objects
        if obj_instance_ids[i] == 0 and background_skip_bbox is not None:
            in_bound_mask = check_in_any_boxes(background_skip_bbox, xyz_coarse_list[i])
            sigmas[in_bound_mask] = -1e5

        rgbs_list += [rgbs]
        sigmas_list += [sigmas]
        obj_ids_list += [torch.ones_like(sigmas) * i]

    results = {}
    volume_rendering_multi(
        results,
        "coarse",
        z_vals_list,
        rgbs_list,
        sigmas_list,
        noise_std,
        white_back,
        obj_ids_list,
    )

    if N_importance > 0:  # sample points for fine model
        rgbs_list = []
        sigmas_list = []
        z_vals_fine_list = []
        for i in range(len(rays_list)):
            z_vals = z_vals_list[i]
            z_vals_mid = 0.5 * (
                z_vals[:, :-1] + z_vals[:, 1:]
            )  # (N_rays, N_samples-1) interval mid points
            # recover weights according to z_vals from results
            weights_ = results["weights_coarse"][results["obj_ids_coarse"] == i]
            assert weights_.numel() == N_rays * N_samples
            weights_ = rearrange(weights_, "(n1 n2) -> n1 n2", n1=N_rays, n2=N_samples)
            z_vals_ = sample_pdf(
                z_vals_mid, weights_[:, 1:-1].detach(), N_importance, det=(perturb == 0)
            )

            z_vals = torch.sort(torch.cat([z_vals, z_vals_], -1), -1)[0]

            # if we have ray mask (e.g. bbox), we clip z values
            rays = rays_list[i]
            if rays.shape[1] == 10:
                bbox_mask_near, bbox_mask_far = rays[:, 8:9], rays[:, 9:10]
                z_val_mask = torch.logical_and(
                    z_vals > bbox_mask_near, z_vals < bbox_mask_far
                )
                z_vals[z_val_mask] = bbox_mask_far.repeat(1, z_vals.shape[1])[
                    z_val_mask
                ]

            # combine coarse and fine samples
            z_vals_fine_list += [z_vals]

            xyz_fine = rays_o_list[i] + rays_d_list[i] * rearrange(
                z_vals, "n1 n2 -> n1 n2 1"
            )

            rgbs, sigmas = inference_from_model(
                model=models["fine"],
                embedding_xyz=embedding_xyz,
                dir_embedded=dir_embedded_list[i],
                code_library=code_library,
                xyz=xyz_fine,
                z_vals=z_vals_fine_list[i],
                chunk=chunk,
                instance_id=obj_instance_ids[i],
                # kwargs,
            )

            # mute in bound samples to remove objects
            if obj_instance_ids[i] == 0 and background_skip_bbox is not None:
                in_bound_mask = check_in_any_boxes(background_skip_bbox, xyz_fine)
                sigmas[in_bound_mask] = -1e5

            rgbs_list += [rgbs]
            sigmas_list += [sigmas]

        volume_rendering_multi(
            results,
            "fine",
            z_vals_fine_list,
            rgbs_list,
            sigmas_list,
            noise_std,
            white_back,
        )
    return results

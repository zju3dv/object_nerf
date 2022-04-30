import sys
import os

os.environ["OMP_NUM_THREADS"] = "1"  # noqa
os.environ["MKL_NUM_THREADS"] = "1"  # noqa
sys.path.append(".")  # noqa

import numpy as np
import numba as nb  # mute some warning
import torch
from torch import nn
from collections import defaultdict
from tqdm import trange, tqdm
from typing import List, Optional, Any, Dict, Union
from omegaconf import OmegaConf

from datasets.geo_utils import center_pose_from_avg
from utils.bbox_utils import BBoxRayHelper
from utils.util import read_json
from train import ObjectNeRFSystem
from datasets.ray_utils import get_ray_directions, get_rays
from render_tools.multi_rendering import render_rays_multi


def read_testing_config():
    conf_cli = OmegaConf.from_cli()
    conf_test_file = OmegaConf.load(conf_cli.config)
    # order: 1. cli; 2. test_file
    conf_merged = OmegaConf.merge(conf_test_file, conf_cli)

    # read training config snapshot
    ckpt_conf_path = os.path.join(
        os.path.dirname(os.path.abspath(conf_merged.ckpt_path)),
        "run_config_snapshot.yaml",
    )
    conf_merged.ckpt_config_path = ckpt_conf_path
    conf_training = OmegaConf.create()
    conf_training.ckpt_config = OmegaConf.load(ckpt_conf_path)
    # order: 1. merged; 2. training
    conf_merged = OmegaConf.merge(conf_training, conf_merged)
    # check if ckpt folder has contain pcd.ply file
    # just to facilitate example demo
    pcd_file = os.path.join(
        os.path.dirname(os.path.abspath(conf_merged.ckpt_path)),
        "pcd.ply",
    )
    if os.path.exists(pcd_file):
        conf_merged.ckpt_config.dataset_extra.pcd_path = pcd_file
    return conf_merged


class EditableRenderer:
    def __init__(self, config):
        # load config
        self.config = config
        self.ckpt_config = config.ckpt_config
        self.load_model(config.ckpt_path, config.ckpt_config)

        # initialize rendering parameters
        dataset_extra = self.ckpt_config.dataset_extra
        self.near = config.get("near", dataset_extra.near)
        self.far = config.get("far", dataset_extra.far)
        self.scale_factor = dataset_extra.scale_factor
        self.pose_avg = np.concatenate(
            [np.eye(3), np.array(dataset_extra["scene_center"])[:, None]], 1
        )

        self.object_to_remove = []
        self.active_object_ids = [0]
        # self.active_object_ids = []
        self.object_pose_transform = {}
        self.object_bbox_ray_helpers = {}
        self.bbox_enlarge = 0.0

    def load_model(self, ckpt_path, ckpt_config):
        self.system = ObjectNeRFSystem.load_from_checkpoint(
            ckpt_path, config=ckpt_config
        ).cuda()
        self.system.eval()

    def load_frame_meta(self):
        dataset_name = self.ckpt_config.dataset_name
        dataset_extra = self.ckpt_config.dataset_extra
        if dataset_name in ["scannet_base", "toydesk"]:
            data_json_path = os.path.join(
                dataset_extra.root_dir, f"transforms_full.json"
            )
            self.dataset_meta = read_json(data_json_path)
            # load fov
            self.fov_x_deg_dataset = self.dataset_meta["camera_angle_x"] * 180 / np.pi
            print("fov x", self.fov_x_deg_dataset)
            # load poses
            self.poses = []
            tmp_index = []
            for frame in self.dataset_meta["frames"]:
                fix_rot = np.array([1, 0, 0, 0, -1, 0, 0, 0, -1]).reshape(3, 3)
                pose = np.array(frame["transform_matrix"])
                pose[:3, :3] = pose[:3, :3] @ fix_rot
                # centralize and rescale
                # pose = center_pose_from_avg(self.pose_avg, pose)
                # pose[:, 3] /= self.scale_factor
                self.poses.append(pose)
                tmp_index.append(frame["idx"])
            sorted_idx = np.argsort(np.array(tmp_index))
            self.poses = np.array(self.poses)[sorted_idx]
        else:
            assert False, "not implemented dataset type: {}".format(dataset_name)

    def get_camera_pose_by_frame_idx(self, frame_idx):
        return self.poses[frame_idx]

    def scene_inference(
        self,
        rays: torch.Tensor,
        show_progress: bool = True,
    ):
        args = {}
        # args["train_config"] = self.ckpt_config.train

        B = rays.shape[0]
        results = defaultdict(list)
        chunk = self.config.chunk
        for i in tqdm(range(0, B, self.config.chunk), disable=not show_progress):
            with torch.no_grad():
                rendered_ray_chunks = render_rays_multi(
                    models=self.system.models,
                    embeddings=self.system.embeddings,
                    code_library=self.system.code_library,
                    # rays=rays[i : i + chunk],
                    rays_list=[rays[i : i + chunk]],
                    obj_instance_ids=[0],
                    N_samples=self.ckpt_config.model.N_samples,
                    use_disp=self.ckpt_config.model.use_disp,
                    perturb=0,
                    noise_std=0,
                    N_importance=self.ckpt_config.model.N_importance,
                    chunk=self.ckpt_config.train.chunk,  # chunk size is effective in val mode
                    white_back=False,
                    **args,
                )

            for k, v in rendered_ray_chunks.items():
                results[k] += [v]

        for k, v in results.items():
            # cat on rays dim
            if len(v[0].shape) <= 2:
                results[k] = torch.cat(v, 0)
            elif len(v[0].shape) == 3:
                results[k] = torch.cat(v, 1)
        return results

    def generate_rays(self, obj_id, rays_o, rays_d):
        near = self.near
        far = self.far
        if obj_id == 0:
            batch_near = near / self.scale_factor * torch.ones_like(rays_o[:, :1])
            batch_far = far / self.scale_factor * torch.ones_like(rays_o[:, :1])
            # rays_o = rays_o / self.scale_factor
            rays = torch.cat([rays_o, rays_d, batch_near, batch_far], 1)  # (H*W, 8)
        else:
            bbox_mask, bbox_batch_near, bbox_batch_far = self.object_bbox_ray_helpers[
                str(obj_id)
            ].get_ray_bbox_intersections(
                rays_o,
                rays_d,
                self.scale_factor,
                # bbox_enlarge=self.bbox_enlarge / self.get_scale_factor(obj_id),
                bbox_enlarge=self.bbox_enlarge,  # in physical world
            )
            # for area which hits bbox, we use bbox hit near far
            # bbox_ray_helper has scale for us, do no need to rescale
            batch_near_obj, batch_far_obj = bbox_batch_near, bbox_batch_far
            # for the invalid part, we use 0 as near far, which assume that (0, 0, 0) is empty
            batch_near_obj[~bbox_mask] = torch.zeros_like(batch_near_obj[~bbox_mask])
            batch_far_obj[~bbox_mask] = torch.zeros_like(batch_far_obj[~bbox_mask])
            rays = torch.cat(
                [rays_o, rays_d, batch_near_obj, batch_far_obj], 1
            )  # (H*W, 8)
        rays = rays.cuda()
        return rays

    def render_origin(
        self,
        h: int,
        w: int,
        camera_pose_Twc: np.ndarray,
        fovx_deg: float = 70,
    ):
        focal = (w / 2) / np.tan((fovx_deg / 2) / (180 / np.pi))
        directions = get_ray_directions(h, w, focal).cuda()  # (h, w, 3)
        Twc = center_pose_from_avg(self.pose_avg, camera_pose_Twc)
        Twc[:, 3] /= self.scale_factor
        # for scene, Two is eye
        Two = np.eye(4)
        Toc = np.linalg.inv(Two) @ Twc
        Toc = torch.from_numpy(Toc).float().cuda()[:3, :4]
        rays_o, rays_d = get_rays(directions, Toc)
        rays = self.generate_rays(0, rays_o, rays_d)
        results = self.scene_inference(rays)
        return results

    def render_edit(
        self,
        h: int,
        w: int,
        camera_pose_Twc: np.ndarray,
        fovx_deg: float = 70,
        show_progress: bool = True,
        render_bg_only: bool = False,
        render_obj_only: bool = False,
        white_back: bool = False,
    ):
        focal = (w / 2) / np.tan((fovx_deg / 2) / (180 / np.pi))
        directions = get_ray_directions(h, w, focal).cuda()  # (h, w, 3)
        Twc = center_pose_from_avg(self.pose_avg, camera_pose_Twc)
        args = {}
        results = {}
        obj_ids = []
        rays_list = []

        # only render background
        if render_bg_only:
            self.active_object_ids = [0]

        # only render objects
        if render_obj_only:
            self.active_object_ids.remove(0)

        processed_obj_id = []
        for obj_id in self.active_object_ids:
            # count object duplication
            obj_duplication_cnt = np.sum(np.array(processed_obj_id) == obj_id)
            if obj_id == 0:
                # for scene, transform is Identity
                Tow = transform = np.eye(4)
            else:
                object_pose = self.object_pose_transform[
                    f"{obj_id}_{obj_duplication_cnt}"
                ]
                # transform in the real world scale
                Tow_orig = self.get_object_bbox_helper(
                    obj_id
                ).get_world_to_object_transform()
                # transform object into center, then apply user-specific object poses
                transform = np.linalg.inv(Tow_orig) @ object_pose @ Tow_orig
                # for X_c = Tcw * X_w, when we applying transformation on X_w,
                # it equals to Tcw * (transform * X_w). So, Tow = inv(transform) * Twc
                Tow = np.linalg.inv(transform)
                # Tow = np.linalg.inv(Tow)  # this move obejct to center
            processed_obj_id.append(obj_id)
            Toc = Tow @ Twc
            # resize to NeRF scale
            Toc[:, 3] /= self.scale_factor
            Toc = torch.from_numpy(Toc).float().cuda()[:3, :4]
            # all the rays_o and rays_d has been converted to NeRF scale
            rays_o, rays_d = get_rays(directions, Toc)
            rays = self.generate_rays(obj_id, rays_o, rays_d)
            # light anchor should also be transformed
            Tow = torch.from_numpy(Tow).float()
            transform = torch.from_numpy(transform).float()
            obj_ids.append(obj_id)
            rays_list.append(rays)

        # split chunk
        B = rays_list[0].shape[0]
        chunk = self.config.chunk
        results = defaultdict(list)
        background_skip_bbox = self.get_skipping_bbox_helper()
        for i in tqdm(range(0, B, self.config.chunk), disable=not show_progress):
            with torch.no_grad():
                rendered_ray_chunks = render_rays_multi(
                    models=self.system.models,
                    embeddings=self.system.embeddings,
                    code_library=self.system.code_library,
                    rays_list=[r[i : i + chunk] for r in rays_list],
                    obj_instance_ids=obj_ids,
                    N_samples=self.ckpt_config.model.N_samples,
                    use_disp=self.ckpt_config.model.use_disp,
                    perturb=0,
                    noise_std=0,
                    N_importance=self.ckpt_config.model.N_importance,
                    chunk=self.ckpt_config.train.chunk,  # chunk size is effective in val mode
                    white_back=white_back,
                    background_skip_bbox=background_skip_bbox,
                    **args,
                )
            for k, v in rendered_ray_chunks.items():
                results[k] += [v.detach().cpu()]

        for k, v in results.items():
            results[k] = torch.cat(v, 0)

        return results

    def remove_scene_object_by_ids(self, obj_ids):
        """
        Create a clean background by removing user specified objects.
        """
        self.object_to_remove = obj_ids
        for obj_id in obj_ids:
            self.initialize_object_bbox(obj_id)

    def reset_active_object_ids(self):
        self.active_object_ids = [0]

    def set_object_pose_transform(
        self,
        obj_id: int,
        pose: np.ndarray,
        obj_dup_id: int = 0,  # for object duplication
    ):
        self.active_object_ids.append(obj_id)
        if obj_id not in self.active_object_ids:
            self.initialize_object_bbox(obj_id)
        self.object_pose_transform[f"{obj_id}_{obj_dup_id}"] = pose

    def initialize_object_bbox(self, obj_id: int):
        self.object_bbox_ray_helpers[str(obj_id)] = BBoxRayHelper(
            self.config.ckpt_config_path, obj_id
        )

    def get_object_bbox_helper(self, obj_id: int):
        return self.object_bbox_ray_helpers[str(obj_id)]

    def get_skipping_bbox_helper(self):
        skipping_bbox_helper = {}
        for obj_id in self.object_to_remove:
            skipping_bbox_helper[str(obj_id)] = self.object_bbox_ray_helpers[
                str(obj_id)
            ]
        return skipping_bbox_helper

import os
import numpy as np
import torch
import copy
from utils.util import read_yaml, read_json
from datasets.geo_utils import bbox_intersection_batch


class BBoxRayHelper:
    def __init__(self, dataset_config, instance_id):
        super().__init__()
        full_conf = read_yaml(dataset_config)
        self.conf = full_conf["dataset_extra"]
        self.scale_factor = self.conf["scale_factor"]
        self.instance_id = instance_id

        self.dataset_name = full_conf["dataset_name"]
        assert self.dataset_name in ["scannet_base", "toydesk"]

        if self.dataset_name == "scannet_base":
            self.scene_id = self.conf["scene_id"]
            self.read_bbox_info_scannet()
        elif self.dataset_name == "toydesk":
            self.read_bbox_info_desk()

    def get_axis_align_mat(self, rescaled=False):
        if rescaled:
            axis_align_mat = copy.deepcopy(self.axis_align_mat)
            axis_align_mat[:3, 3] /= self.scale_factor
            return axis_align_mat
        else:
            return self.axis_align_mat

    def get_world_to_object_transform(self):
        recenter = np.eye(4)
        if self.dataset_name == "scannet_base":
            recenter[:3, 3] = -self.bbox_c
        trans = recenter @ self.axis_align_mat @ self.pose_avg
        return trans  # Tow

    def read_bbox_info_scannet(self):
        # read axis_align_matrix
        scene_info_file = os.path.join(
            self.conf["scans_dir"], "{}/{}.txt".format(self.scene_id, self.scene_id)
        )
        lines = open(scene_info_file).readlines()
        for line in lines:
            if "axisAlignment" in line:
                axis_align_matrix = [
                    float(x) for x in line.rstrip().strip("axisAlignment = ").split(" ")
                ]
                break
        self.axis_align_mat = np.array(axis_align_matrix).reshape(4, 4)

        # read bbox bounds
        scene_bbox = np.load(
            os.path.join(self.conf["bbox_dir"], "{}_bbox.npy".format(self.scene_id))
        )
        for b in scene_bbox:
            # if b[6] != self.conf['val_instance_id']:
            if b[6] != self.instance_id:
                continue
            length = np.array([b[3], b[4], b[5]]) * 0.5
            center = np.array([b[0], b[1], b[2]])
            self.bbox_bounds = np.array([center - length, center + length])
        self.bbox_c = center  # center in ScanNet aligned coordinate
        self.pose_avg = np.eye(4)
        self.pose_avg[:3, 3] = np.array(self.conf["scene_center"])

    def read_bbox_info_desk(self):
        from scipy.spatial.transform import Rotation as R

        j = read_json(self.conf["bbox_dir"])
        labels = j["labels"]

        # print(len(labels), labels)
        for l in labels:
            if int(l["id"]) != self.instance_id:
                continue
            if "position" not in l["data"]:
                continue
            pos = l["data"]["position"]
            quat = l["data"]["quaternion"]
            scale = l["data"]["scale"]
            pos, scale = np.array(pos), np.array(scale)
            r = R.from_quat(quat)
            rmat = r.as_matrix()
            # self.bbox_c = pos - scale / 2
            self.bbox_c = pos
            # bbox = o3d.geometry.OrientedBoundingBox(center=pos, R=rmat, extent=scale)
            self.axis_align_mat = np.eye(4)
            self.axis_align_mat[:3, :3] = rmat
            self.axis_align_mat[:3, 3] = pos
            self.axis_align_mat = np.linalg.inv(self.axis_align_mat)
            self.bbox_bounds = np.array([-scale / 2, scale / 2])
            break

        self.pose_avg = np.eye(4)
        self.pose_avg[:3, 3] = np.array(self.conf["scene_center"])

    def transform_rays_to_bbox_coordinates(self, rays_o, rays_d, scale_factor):
        if type(rays_o) is torch.Tensor:
            rays_o, rays_d = (
                rays_o.detach().cpu().numpy(),
                rays_d.detach().cpu().numpy(),
            )
        # unscale
        rays_o = rays_o * scale_factor
        # de-centralize
        T_orig_avg = self.pose_avg.squeeze()
        rays_o_bbox = (T_orig_avg[:3, :3] @ rays_o.T).T + T_orig_avg[:3, 3]
        rays_d_bbox = (T_orig_avg[:3, :3] @ rays_d.T).T
        # convert to bbox coordinates
        T_box_orig = self.axis_align_mat
        rays_o_bbox = (T_box_orig[:3, :3] @ rays_o_bbox.T).T + T_box_orig[:3, 3]
        rays_d_bbox = (T_box_orig[:3, :3] @ rays_d.T).T
        return rays_o_bbox, rays_d_bbox

    def transform_xyz_to_bbox_coordinates(self, xyz, scale_factor):
        if type(xyz) is torch.Tensor:
            xyz = xyz.detach().cpu().numpy()
        # unscale
        xyz = xyz * scale_factor
        # de-centralize
        T_orig_avg = self.pose_avg.squeeze()
        xyz_bbox = (T_orig_avg[:3, :3] @ xyz.T).T + T_orig_avg[:3, 3]
        # convert to bbox coordinates
        T_box_orig = self.axis_align_mat
        xyz_bbox = (T_box_orig[:3, :3] @ xyz_bbox.T).T + T_box_orig[:3, 3]
        return xyz_bbox

    def get_ray_bbox_intersections(
        self, rays_o, rays_d, scale_factor=None, bbox_enlarge=0
    ):
        if scale_factor is None:
            scale_factor = self.scale_factor
        rays_o_bbox, rays_d_bbox = self.transform_rays_to_bbox_coordinates(
            rays_o, rays_d, scale_factor
        )
        bbox_bounds = copy.deepcopy(self.bbox_bounds)
        if bbox_enlarge > 0:
            # bbox_z_min_orig = bbox_bounds[0][2]
            bbox_bounds[0] -= bbox_enlarge
            bbox_bounds[1] += bbox_enlarge
            # bbox_bounds[0][2] = bbox_z_min_orig

        bbox_mask, batch_near, batch_far = bbox_intersection_batch(
            bbox_bounds, rays_o_bbox, rays_d_bbox
        )
        bbox_mask, batch_near, batch_far = (
            torch.Tensor(bbox_mask).bool(),
            torch.Tensor(batch_near[..., None]),
            torch.Tensor(batch_far[..., None]),
        )
        batch_near, batch_far = batch_near / scale_factor, batch_far / scale_factor
        return bbox_mask.cuda(), batch_near.cuda(), batch_far.cuda()

    def check_xyz_in_bounds(
        self,
        xyz: torch.Tensor,
        scale_factor: float = None,
        bbox_enlarge: float = 0,
    ):
        """
        scale_factor: we should rescale xyz to real size
        """
        if scale_factor is None:
            scale_factor = self.scale_factor
        xyz = self.transform_xyz_to_bbox_coordinates(xyz, scale_factor)
        xyz = torch.from_numpy(xyz).float().cuda()
        bbox_bounds = copy.deepcopy(self.bbox_bounds)
        if bbox_enlarge > 0:
            z_min_orig = bbox_bounds[0][2]  # keep z_min
            bbox_bounds[0] -= bbox_enlarge
            bbox_bounds[1] += bbox_enlarge
            bbox_bounds[0][2] = z_min_orig
        elif bbox_enlarge < 0:
            # make some margin near the ground
            bbox_bounds[0][2] -= bbox_enlarge
        x_min, y_min, z_min = bbox_bounds[0]
        x_max, y_max, z_max = bbox_bounds[1]
        in_x = torch.logical_and(xyz[:, 0] >= x_min, xyz[:, 0] <= x_max)
        in_y = torch.logical_and(xyz[:, 1] >= y_min, xyz[:, 1] <= y_max)
        in_z = torch.logical_and(xyz[:, 2] >= z_min, xyz[:, 2] <= z_max)
        in_bounds = torch.logical_and(in_x, torch.logical_and(in_y, in_z))
        return in_bounds


def check_in_any_boxes(
    boxes,
    xyz: torch.Tensor,
    scale_factor: float = None,
    bbox_enlarge: float = 0.0,
):
    need_reshape = False
    if len(xyz.shape) == 3:
        N1, N2, _ = xyz.shape
        xyz = xyz.view(-1, 3)
        need_reshape = True
    in_bounds = torch.zeros_like(xyz[:, 0]).bool()
    for k, box in boxes.items():
        in_bounds = torch.logical_or(
            box.check_xyz_in_bounds(xyz, scale_factor, bbox_enlarge), in_bounds
        )
    if need_reshape:
        in_bounds = in_bounds.view(N1, N2)
    return in_bounds

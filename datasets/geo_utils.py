import numpy as np
import numba as nb
import torch
import open3d as o3d

import sys
import os

sys.path.append(os.getcwd())  # noqa


def normalize(v):
    """Normalize a vector."""
    return v / np.linalg.norm(v)


def average_poses(poses):
    """
    Calculate the average pose, which is then used to center all poses
    using @center_poses. Its computation is as follows:
    1. Compute the center: the average of pose centers.
    2. Compute the z axis: the normalized average z axis.
    3. Compute axis y': the average y axis.
    4. Compute x' = y' cross product z, then normalize it as the x axis.
    5. Compute the y axis: z cross product x.

    Note that at step 3, we cannot directly use y' as y axis since it's
    not necessarily orthogonal to z axis. We need to pass from x to y.

    Inputs:
        poses: (N_images, 3, 4)

    Outputs:
        pose_avg: (3, 4) the average pose
    """
    # 1. Compute the center
    center = poses[..., 3].mean(0)  # (3)

    # 2. Compute the z axis
    z = normalize(poses[..., 2].mean(0))  # (3)

    # 3. Compute axis y' (no need to normalize as it's not the final output)
    y_ = poses[..., 1].mean(0)  # (3)

    # 4. Compute the x axis
    x = normalize(np.cross(y_, z))  # (3)

    # 5. Compute the y axis (as z and x are normalized, y is already of norm 1)
    y = np.cross(z, x)  # (3)

    pose_avg = np.stack([x, y, z, center], 1)  # (3, 4)

    return pose_avg


def center_poses(poses, pose_avg=None):
    """
    Center the poses so that we can use NDC.
    See https://github.com/bmild/nerf/issues/34

    Inputs:
        poses: (N_images, 3, 4)
        pose_avg: (3, 4) the average pose given by users

    Outputs:
        poses_centered: (N_images, 3, 4) the centered poses
        pose_avg: (3, 4) the average pose
    """
    if pose_avg is None:
        pose_avg = average_poses(poses)  # (3, 4)

    pose_avg_homo = np.eye(4)
    pose_avg_homo[:3] = pose_avg[
        :3, :4
    ]  # convert to homogeneous coordinate for faster computation
    # by simply adding 0, 0, 0, 1 as the last row
    last_row = np.tile(np.array([0, 0, 0, 1]), (len(poses), 1, 1))  # (N_images, 1, 4)
    poses_homo = np.concatenate(
        [poses, last_row], 1
    )  # (N_images, 4, 4) homogeneous coordinate

    poses_centered = np.linalg.inv(pose_avg_homo) @ poses_homo  # (N_images, 4, 4)
    poses_centered = poses_centered[:, :3]  # (N_images, 3, 4)

    return poses_centered, pose_avg


def center_pose_from_avg(pose_avg, pose):
    pose_avg_homo = np.eye(4)
    pose_avg_homo[
        :3
    ] = pose_avg  # convert to homogeneous coordinate for faster computation
    # by simply adding 0, 0, 0, 1 as the last row
    pose_homo = np.eye(4)
    pose_homo[:3] = pose[:3]
    pose_centered = np.linalg.inv(pose_avg_homo) @ pose_homo  # (4, 4)
    # pose_centered = pose_centered[:, :3] # (N_images, 3, 4)
    return pose_centered


def observe_angle_distance(pose, obj_center):
    # we assume the input pose is Twc[3x4], with (x,y,z) -> (right,down, forward)
    view_dir = pose[:3, :3] @ np.array([0, 0, 1])  # camera view direction in world
    c2o_dir = obj_center - pose[:3, 3]  # camera to object ray
    distance = np.linalg.norm(c2o_dir)
    c2o_dir /= distance  # normalize camera to object ray
    view_angle = np.arccos(c2o_dir.dot(view_dir)) * 180 / np.pi
    return view_angle, distance


@nb.jit(nopython=True)
def bbox_intersection_batch(bounds, rays_o, rays_d):
    N_rays = rays_o.shape[0]
    all_hit = np.empty((N_rays))
    all_near = np.empty((N_rays))
    all_far = np.empty((N_rays))
    for idx, (o, d) in enumerate(zip(rays_o, rays_d)):
        hit, near, far = bbox_intersection(bounds, o, d)
        all_hit[idx] = hit
        all_near[idx] = near
        all_far[idx] = far
    # return (h*w), (h*w, 3), (h*w, 3)
    return all_hit, all_near, all_far


@nb.jit(nopython=True)
def bbox_intersection(bounds, orig, dir):
    # FIXME: currently, it is not working properly if the ray origin is inside the bounding box
    # https://www.scratchapixel.com/lessons/3d-basic-rendering/minimal-ray-tracer-rendering-simple-shapes/ray-box-intersection
    # handle divide by zero
    dir[dir == 0] = 1.0e-14
    invdir = 1 / dir
    sign = (invdir < 0).astype(np.int64)

    tmin = (bounds[sign[0]][0] - orig[0]) * invdir[0]
    tmax = (bounds[1 - sign[0]][0] - orig[0]) * invdir[0]

    tymin = (bounds[sign[1]][1] - orig[1]) * invdir[1]
    tymax = (bounds[1 - sign[1]][1] - orig[1]) * invdir[1]

    if tmin > tymax or tymin > tmax:
        return False, 0, 0
    if tymin > tmin:
        tmin = tymin
    if tymax < tmax:
        tmax = tymax

    tzmin = (bounds[sign[2]][2] - orig[2]) * invdir[2]
    tzmax = (bounds[1 - sign[2]][2] - orig[2]) * invdir[2]

    if tmin > tzmax or tzmin > tmax:
        return False, 0, 0
    if tzmin > tmin:
        tmin = tzmin
    if tzmax < tmax:
        tmax = tzmax

    # additionally, when the orig is inside the box, we return False
    if tmin < 0 or tmax < 0:
        return False, 0, 0

    return True, tmin, tmax


if __name__ == "__main__":
    # check AABB ray intersection
    from tools.O3dVisualizer import O3dVisualizer

    visualizer = O3dVisualizer()

    points = [
        [0, 0, 0],
        [1, 0, 0],
        [0, 1, 0],
        [1, 1, 0],
        [0, 0, 1],
        [1, 0, 1],
        [0, 1, 1],
        [1, 1, 1],
    ]
    lines = [
        [0, 1],
        [0, 2],
        [1, 3],
        [2, 3],
        [4, 5],
        [4, 6],
        [5, 7],
        [6, 7],
        [0, 4],
        [1, 5],
        [2, 6],
        [3, 7],
    ]
    colors = [[1, 0, 0] for i in range(len(lines))]
    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(points),
        lines=o3d.utility.Vector2iVector(lines),
    )
    line_set.colors = o3d.utility.Vector3dVector(colors)

    visualizer.add_o3d_geometry(line_set)
    # o3d.visualization.draw_geometries([line_set])
    bbox_bounds = np.array([[0, 0, 0], [1, 1, 1]])
    orig = np.array([0, 0, 3])
    ray = np.array([0.1, 0.1, -1])
    ray /= np.linalg.norm(ray)

    print(bbox_intersection(bbox_bounds, orig, ray))

    _, dmin, dmax = bbox_intersection(bbox_bounds, orig, ray)

    visualizer.add_np_points(orig[None, ...], size=0.05)

    pts = [orig, orig + dmin * ray, orig + dmax * ray]
    pts_idx = [[0, 1], [1, 2]]
    colors = [[1, 0, 0], [0, 1, 0]]

    visualizer.add_line_set(pts, pts_idx, colors=colors, radius=0.003)

    visualizer.run_visualize()

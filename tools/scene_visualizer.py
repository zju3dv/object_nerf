import numpy as np
import argparse
import sys
import os

sys.path.append(os.getcwd())  # noqa
import glob
import copy
import shutil
import json
from pathlib import Path
from PIL import Image
import open3d as o3d
from tools.O3dVisualizer import O3dVisualizer
import matplotlib.pyplot as plt
from datasets.geo_utils import observe_angle_distance
from utils.util import read_json, map_to_color


def draw_poses(visualizer, pose_info_json):
    frame_num = len(pose_info_json["frames"])
    camera_centers = []
    lines_pt, lines_idx, lines_color = [], [], []

    idx = 0
    for frame_id, frame in enumerate(pose_info_json["frames"]):
        Twc = np.array(frame["transform_matrix"])
        # for nerf_synthetic, we need some transformation
        fix_rot = np.array([1, 0, 0, 0, -1, 0, 0, 0, -1]).reshape(3, 3)
        # Twc[:3, :3] = Twc[:3, :3] @ fix_rot

        center = Twc[:3, 3]
        camera_centers.append(center)
        # draw axis
        # RGB -> right, down, forward
        axis_size = 0.1
        # for .T, you can follow https://stackoverflow.com/questions/12148351/
        axis_pts = (Twc[:3, :3] @ (np.eye(3) * axis_size)).T + center
        lines_pt += [center, axis_pts[0, :], axis_pts[1, :], axis_pts[2, :]]
        lines_idx += [
            [idx * 4 + 0, idx * 4 + 1],
            [idx * 4 + 0, idx * 4 + 2],
            [idx * 4 + 0, idx * 4 + 3],
        ]
        lines_color += [(1, 0, 0), (0, 1, 0), (0, 0, 1)]
        idx += 1

    print("Camera center num:", len(camera_centers))
    # draw line via cylinder, which we can control the line thickness
    # visualizer.add_line_set(lines_pt, lines_idx, colors=lines_color, radius=0.003)

    # draw line via LineSet
    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(np.array(lines_pt)),
        lines=o3d.utility.Vector2iVector(np.array(lines_idx)),
    )
    line_set.colors = o3d.utility.Vector3dVector(lines_color)
    visualizer.add_o3d_geometry(line_set)

    camera_centers = np.array(camera_centers)
    visualizer.add_np_points(
        camera_centers,
        color=map_to_color(np.arange(0, frame_num), cmap="plasma"),
        size=0.01,
    )


def draw_bbox(visualizer, bbox, pcd_file):
    lines_pt, lines_idx, lines_color = [], [], []
    text_pts = []
    print(bbox.shape)
    i = 0
    for idx_b, b in enumerate(bbox):
        # if idx_b != 3: continue
        length = np.array([b[3], b[4], b[5]]) * 0.5
        center = np.array([b[0], b[1], b[2]])
        # print('AABB bbox bounds', center - length, center + length)

        # draw semantic id
        text_pos = center + np.array([0, 0, length[2]])
        curr_text_pts = visualizer.text_3d(
            str(round(b[6])),
            text_pos,
            font="/usr/share/fonts/truetype/noto/NotoMono-Regular.ttf",
        )
        text_pts.append(np.asarray(curr_text_pts.points))

        lines_pt += [
            center + length * np.array([-1, -1, -1]),
            center + length * np.array([1, -1, -1]),
            center + length * np.array([-1, 1, -1]),
            center + length * np.array([1, 1, -1]),
            center + length * np.array([-1, -1, 1]),
            center + length * np.array([1, -1, 1]),
            center + length * np.array([-1, 1, 1]),
            center + length * np.array([1, 1, 1]),
        ]
        lines_idx += [
            [i * 8 + 0, i * 8 + 1],
            [i * 8 + 0, i * 8 + 2],
            [i * 8 + 1, i * 8 + 3],
            [i * 8 + 2, i * 8 + 3],
            [i * 8 + 4, i * 8 + 5],
            [i * 8 + 4, i * 8 + 6],
            [i * 8 + 5, i * 8 + 7],
            [i * 8 + 6, i * 8 + 7],
            [i * 8 + 0, i * 8 + 4],
            [i * 8 + 1, i * 8 + 5],
            [i * 8 + 2, i * 8 + 6],
            [i * 8 + 3, i * 8 + 7],
        ]
        lines_color.extend([np.array([0, 1, 0]) for x in range(12)])
        i += 1

    lines_pt = np.array(lines_pt)

    # read axis align
    meta_file = pcd_file.split("_vh_clean")[0] + ".txt"
    print(meta_file)

    lines = open(meta_file).readlines()
    for line in lines:
        if "axisAlignment" in line:
            axis_align_matrix = [
                float(x) for x in line.rstrip().strip("axisAlignment = ").split(" ")
            ]
            break

    axis_align_matrix = np.array(axis_align_matrix).reshape(4, 4)
    axis_align_matrix = np.linalg.inv(axis_align_matrix)
    lines_pt = (axis_align_matrix[:3, :3] @ lines_pt.T).T + axis_align_matrix[:3, 3]

    visualizer.add_line_set(lines_pt, lines_idx, colors=lines_color, radius=0.003)

    text_pts = np.concatenate(text_pts, axis=0)
    text_pts = (axis_align_matrix[:3, :3] @ text_pts.T).T + axis_align_matrix[:3, 3]
    visualizer.add_np_points(text_pts, color=[1, 0, 0], size=0.01)


if __name__ == "__main__":
    # params
    parser = argparse.ArgumentParser()
    # data paths
    parser.add_argument("--pcd", default=None)
    parser.add_argument("--transform_json", default=None)
    parser.add_argument("--bbox", default=None)
    args = parser.parse_args()

    visualizer = O3dVisualizer()
    mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=1.0, origin=[0, 0, 0]
    )
    visualizer.add_o3d_geometry(mesh_frame)
    if args.pcd:
        pcd = o3d.io.read_point_cloud(args.pcd)
        visualizer.add_o3d_geometry(pcd)

    if args.transform_json:
        draw_poses(visualizer, read_json(args.transform_json))

    if args.bbox:
        # Nx(cx, cy, cz, dx, dy, dz, semantic_label)
        draw_bbox(visualizer, np.load(args.bbox), args.pcd)

    visualizer.run_visualize()

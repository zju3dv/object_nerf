import sys
import os

os.environ["OMP_NUM_THREADS"] = "1"  # noqa
os.environ["MKL_NUM_THREADS"] = "1"  # noqa
sys.path.append(".")  # noqa

import imageio
import numpy as np
from tqdm import tqdm
from render_tools.editable_renderer import EditableRenderer, read_testing_config
from utils.util import get_timestamp
from scipy.spatial.transform import Rotation


def move_camera_pose(pose, progress):
    # control the camera move (spiral pose)
    t = progress * np.pi * 4
    radii = 0.01
    center = np.array([np.cos(t), -np.sin(t), -np.sin(0.5 * t)]) * radii
    pose[:3, 3] += pose[:3, :3] @ center
    return pose


def get_pure_rotation(progress_11: float, max_angle: float = 180):
    trans_pose = np.eye(4)
    trans_pose[:3, :3] = Rotation.from_euler(
        "z", progress_11 * max_angle, degrees=True
    ).as_matrix()
    return trans_pose


def get_transformation_with_duplication_offset(progress, duplication_id: int):
    trans_pose = get_pure_rotation(np.sin(progress * np.pi * 2), max_angle=10)
    offset = 0.05
    if duplication_id > 0:
        trans_pose[0, 3] -= np.sin(progress * np.pi * 2) * offset
        trans_pose[1, 3] -= 0.2
    else:
        trans_pose[0, 3] += np.sin(progress * np.pi * 2) * offset
        trans_pose[1, 3] += 0.55
    return trans_pose


def main(config):
    render_path = f"debug/rendered_view/render_{get_timestamp()}_{config.prefix}/"
    os.makedirs(render_path, exist_ok=True)
    # intialize room optimizer
    renderer = EditableRenderer(config=config)
    renderer.load_frame_meta()
    obj_id_list = config.obj_id_list  # e.g. [4, 6]
    for obj_id in obj_id_list:
        renderer.initialize_object_bbox(obj_id)
    renderer.remove_scene_object_by_ids(obj_id_list)
    W, H = config.img_wh
    total_frames = config.total_frames
    pose_frame_idx = config.test_frame

    for idx in tqdm(range(total_frames)):
        # an example to set object pose
        # trans_pose = get_transformation(0.2)
        processed_obj_id = []
        for obj_id in obj_id_list:
            # count object duplication, which is generally to be zero,
            # but can be increased if duplication operation happened
            obj_duplication_cnt = np.sum(np.array(processed_obj_id) == obj_id)
            progress = idx / total_frames

            if config.edit_type == "duplication":
                trans_pose = get_transformation_with_duplication_offset(
                    progress, obj_duplication_cnt
                )
            elif config.edit_type == "pure_rotation":
                trans_pose = get_pure_rotation(progress_11=(progress * 2 - 1))

            renderer.set_object_pose_transform(obj_id, trans_pose, obj_duplication_cnt)
            processed_obj_id.append(obj_id)

        # Note: uncomment this to render original scene
        # results = renderer.render_origin(
        #     h=H,
        #     w=W,
        #     camera_pose_Twc=move_camera_pose(
        #         renderer.get_camera_pose_by_frame_idx(pose_frame_idx), idx / total_frames
        #     ),
        #     fovx_deg=getattr(renderer, "fov_x_deg_dataset", 60),
        # )

        # render edited scene
        results = renderer.render_edit(
            h=H,
            w=W,
            camera_pose_Twc=move_camera_pose(
                renderer.get_camera_pose_by_frame_idx(pose_frame_idx),
                idx / total_frames,
            ),
            fovx_deg=getattr(renderer, "fov_x_deg_dataset", 60),
        )
        image_out_path = f"{render_path}/render_{idx:04d}.png"
        image_np = results["rgb_fine"].view(H, W, 3).detach().cpu().numpy()
        imageio.imwrite(image_out_path, (image_np * 255).astype(np.uint8))

        renderer.reset_active_object_ids()


if __name__ == "__main__":
    config = read_testing_config()
    main(config)

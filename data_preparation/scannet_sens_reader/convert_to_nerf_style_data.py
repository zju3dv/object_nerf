import numpy as np
import argparse
import os
import glob
import copy
import shutil
import json
import cv2
from pathlib import Path
from PIL import Image


def ensure_dir(dirname):
    dirname = Path(dirname)
    if not dirname.is_dir():
        dirname.mkdir(parents=True, exist_ok=False)


def write_json(content, fname):
    fname = Path(fname)
    with fname.open("wt") as handle:
        json.dump(content, handle, indent=4, sort_keys=False)


if __name__ == "__main__":
    # params
    parser = argparse.ArgumentParser()
    # data paths
    parser.add_argument("--input", required=True, help="ScanNet sens unpack dir")
    parser.add_argument(
        "--output", required=True, help="Output nerf-synthetic style data"
    )
    parser.add_argument("--instance_dir", default=None, help="Instance dir")
    parser.add_argument(
        "--instance_filt_dir", default=None, help="Filtered instance dir"
    )
    args = parser.parse_args()

    tags = ["train", "test", "val", "full"]

    # read color image intrinsics
    K_color = np.loadtxt(f"{args.input}/intrinsic/intrinsic_color.txt")

    # read a image for widht and height
    img = Image.open(f"{args.input}/color/0.jpg")
    W, H = img.size
    print(W, H)

    # TODO: currently we assume no difference of fx and fy
    focal = (K_color[0, 0] + K_color[1, 1]) / 2
    fov_x = np.arctan(W / 2 / focal) * 2
    print("camera fov_x =", fov_x * 180 / np.pi)

    ensure_dir(os.path.join(args.output, "full"))
    # for tag in tags:
    # ensure_dir(os.path.join(args.output, tag))

    info_train = {"camera_angle_x": fov_x, "frames": []}
    info_test = copy.deepcopy(info_train)
    info_val = copy.deepcopy(info_train)
    info_full = copy.deepcopy(info_train)

    img_files_list = glob.glob(f"{args.input}/color/*.jpg")
    print("file count =", len(img_files_list))

    for i in range(len(img_files_list)):
        print("\r{:05d} : {:05d}".format(i, len(img_files_list)), end="")
        # test setting is similar to https://github.com/daipengwa/Neural-Point-Cloud-Rendering-via-Multi-Plane-Projection/blob/93c3515ddca9518d9de80c7d56b69352c834780d/utils.py#L319
        test_step = 100
        test_bound = 20
        val_bound = 10
        active_tags = ["full"]
        if i % 100 == 0:
            active_tags.append("test")
        elif np.abs(i - (round(i / 100.0) * 100)) == val_bound:
            active_tags.append("val")
        elif np.abs(i - (round(i / 100.0) * 100)) > test_bound:
            active_tags.append("train")

        color_file = os.path.join(args.input, "color", "{:d}.jpg".format(i))
        depth_file = os.path.join(args.input, "depth", "{:d}.png".format(i))
        if not os.path.exists(color_file):
            break

        # we store all the images to 'full'
        tag = "full"
        # resize to the same size as depth image
        image = cv2.resize(cv2.imread(color_file, -1), (640, 480))
        cv2.imwrite(os.path.join(args.output, tag, "{:d}.png".format(i)), image)

        # copy depth
        shutil.copy2(
            depth_file, os.path.join(args.output, tag, "{:d}.depth.png".format(i))
        )

        # copy instance
        if args.instance_dir:
            shutil.copy2(
                os.path.join(args.instance_dir, "{:d}.png".format(i)),
                os.path.join(args.output, tag, "{:d}.instance.png".format(i)),
            )
        if args.instance_filt_dir:
            shutil.copy2(
                os.path.join(args.instance_filt_dir, "{:d}.png".format(i)),
                os.path.join(args.output, tag, "{:d}.instance-filt.png".format(i)),
            )

        pose_Twc = np.loadtxt(os.path.join(args.input, "pose/{:d}.txt".format(i)))

        # write info for each tag
        for tag in active_tags:
            meta_info = {
                # 'file_path': './{}/{:d}'.format(tag, i),
                "file_path": "./{}/{:d}".format("full", i),
                "transform_matrix": pose_Twc.tolist(),
                "idx": i,
            }
            locals()[f"info_{tag}"]["frames"].append(meta_info)

    for tag in tags:
        write_json(
            locals()[f"info_{tag}"],
            os.path.join(args.output, "transforms_{}.json".format(tag)),
        )

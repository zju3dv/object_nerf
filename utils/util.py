import json
import yaml
import io
import pickle
from datetime import datetime
import pandas as pd
from pathlib import Path
from itertools import repeat
from collections import OrderedDict
from shutil import copyfile, copytree, ignore_patterns
import matplotlib.pyplot as plt
import open3d as o3d


def ensure_dir(dirname):
    dirname = Path(dirname)
    if not dirname.is_dir():
        dirname.mkdir(parents=True, exist_ok=False)


def read_json(fname):
    fname = Path(fname)
    with fname.open("rt") as handle:
        return json.load(handle, object_hook=OrderedDict)


def write_json(content, fname):
    fname = Path(fname)
    with fname.open("wt") as handle:
        json.dump(content, handle, indent=4, sort_keys=False)


def read_pkl(fname):
    fname = Path(fname)
    with fname.open("rb") as handle:
        return pickle.load(handle)


def write_pkl(content, fname):
    fname = Path(fname)
    with fname.open("wb") as handle:
        pickle.dump(content, handle)


def read_yaml(fname):
    with open(fname, "r") as stream:
        return yaml.safe_load(stream)


def write_yaml(content, fname):
    with io.open(fname, "w", encoding="utf8") as outfile:
        yaml.dump(content, outfile, default_flow_style=False, allow_unicode=True)


def inf_loop(data_loader):
    """wrapper function for endless data loader."""
    for loader in repeat(data_loader):
        yield from loader


def copy_files(src_dir, dst_dir, *ignores):
    copytree(src_dir, dst_dir, ignore=ignore_patterns(*ignores))


def make_source_code_snapshot(log_dir):
    copy_files(
        ".",
        f"{log_dir}/source",
        "saved",
        "__pycache__",
        "data",
        "logs",
        "scans",
        ".vscode",
        "*.so",
        "*.a",
        ".ipynb_checkpoints",
        "build",
        "bin",
        "*.ply",
        "eigen",
        "pybind11",
        "*.npy",
        "*.pth",
        ".git",
        "debug",
    )


def get_timestamp():
    return datetime.now().strftime(r"%y%m%d_%H%M%S")


def map_to_color(x, cmap="coolwarm", vmin=None, vmax=None):
    if vmin == None or vmax == None:
        vmin = min(x)
        vmax = max(x)
    colors = plt.cm.get_cmap(cmap)((x - vmin) / (vmax - vmin))[:, :3]
    return colors


def write_point_cloud(pcd_np, path):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pcd_np)
    o3d.io.write_point_cloud(path, pcd)


def write_idx(idx, filename):
    f = open(filename, "w")
    for i in idx:
        f.write(str(i) + "\n")
    f.close()


def print_val_range(val, name: str):
    print(name, float(val.min()), float(val.max()), float(val.median()))

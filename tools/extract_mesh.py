import os
import sys

os.environ["OMP_NUM_THREADS"] = "1"  # noqa
os.environ["MKL_NUM_THREADS"] = "1"  # noqa
sys.path.append(".")  # noqa

import torch
import numpy as np
from tqdm import tqdm
import mcubes
import open3d as o3d
from omegaconf import OmegaConf

from train import ObjectNeRFSystem

torch.backends.cudnn.benchmark = True


def script_specific_conf():
    return {
        # size of the grid on 1 side, larger=higher resolution
        # "N_grid": 256,
        "N_grid": 512,
        "x_range": [-1.5, 1.5],
        "y_range": [-1.5, 1.5],
        "z_range": [-1.5, 1.5],
        # threshold to consider a location is occupied
        "sigma_threshold": 20.0,
        "chunk": 32 * 2014,
        # "predict_color": True,
        "predict_color": False,
        "ckpt_path": None,
        "prefix": "",
    }


if __name__ == "__main__":
    # args = get_opts()
    conf_cli = OmegaConf.from_cli()
    conf_script_specific = OmegaConf.create(script_specific_conf())
    conf_dataset = OmegaConf.load(conf_cli.dataset_config)
    conf_default = OmegaConf.load("config/default_conf.yml")
    # merge conf with the priority
    config = OmegaConf.merge(conf_default, conf_dataset, conf_script_specific, conf_cli)

    assert config.ckpt_path is not None, "ckpt_path not set"

    system = ObjectNeRFSystem.load_from_checkpoint(config.ckpt_path, config=config)
    system = system.cuda().eval()

    # define the dense grid for query
    N = config.N_grid
    xmin, xmax = config.x_range
    ymin, ymax = config.y_range
    zmin, zmax = config.z_range

    # import ipdb

    # ipdb.set_trace()

    # assert xmax-xmin == ymax-ymin == zmax-zmin, 'the ranges must have the same length!'
    x = np.linspace(xmin, xmax, N)
    y = np.linspace(ymin, ymax, N)
    z = np.linspace(zmin, zmax, N)

    xyz_ = torch.FloatTensor(np.stack(np.meshgrid(x, y, z), -1).reshape(-1, 3)).cuda()

    use_voxel_embedding = config.model.use_voxel_embedding
    obj_id = config.get("obj_id", 0)

    # "obj_code": obj_codes[i : i + chunk],

    # obj_codes = repeat(embedding_instance, "n1 c -> (n1 n2) c", n2=N_samples_)

    # predict sigma (occupancy) for each grid location
    print("Predicting occupancy ...")
    with torch.no_grad():
        B = xyz_.shape[0]
        out_chunks = []
        for i in tqdm(range(0, B, config.chunk)):
            # xyz embedding
            obj_voxel_embbeded = None
            if use_voxel_embedding:
                xyz_embedded, obj_voxel_embbeded = system.embedding_xyz(
                    xyz_[i : i + config.chunk]
                )  # (N, embed_xyz_channels)
            else:
                xyz_embedded = system.embedding_xyz(
                    xyz_[i : i + config.chunk]
                )  # (N, embed_xyz_channels)

            input_dict = {
                "emb_xyz": xyz_embedded,
                "obj_voxel": obj_voxel_embbeded,
            }
            if obj_id > 0:
                N_local_rays = xyz_embedded.shape[0]
                input_dict["obj_code"] = system.code_library.embedding_instance(
                    torch.ones((N_local_rays)).long().cuda() * obj_id
                )
                out = system.nerf_fine.forward_instance(input_dict, sigma_only=True)[
                    "inst_sigma"
                ].cpu()
            else:
                out = system.nerf_fine.forward(input_dict, sigma_only=True)[
                    "sigma"
                ].cpu()
            out_chunks += [out]

        sigma = torch.cat(out_chunks, 0)

    sigma = sigma[:, -1].numpy()
    sigma = np.maximum(sigma, 0).reshape(N, N, N)

    np.save("debug/sigma.npy", sigma)

    # perform marching cube algorithm to retrieve vertices and triangle mesh
    print("Extracting mesh ...")
    vertices, triangles = mcubes.marching_cubes(-sigma, -config.sigma_threshold)

    ##### Until mesh extraction here, it is the same as the original repo. ######

    vertices_ = (vertices / N).astype(np.float64)
    ## invert x and y coordinates (WHY? maybe because of the marching cubes algo)
    x_ = (ymax - ymin) * vertices_[:, 1] + ymin
    y_ = (xmax - xmin) * vertices_[:, 0] + xmin
    vertices_[:, 0] = x_
    vertices_[:, 1] = y_
    vertices_[:, 2] = (zmax - zmin) * vertices_[:, 2] + zmin

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(
        vertices_.astype(np.float64) * config.dataset_extra.scale_factor
    )
    mesh.triangles = o3d.utility.Vector3iVector(triangles.astype(np.int32))

    mesh.compute_vertex_normals()

    if config.predict_color:
        print("Predicting color ...")
        rays_d = -torch.FloatTensor(np.asarray(mesh.vertex_normals)).cuda()
        with torch.no_grad():
            vertices = torch.from_numpy(vertices_).float().cuda()
            B = vertices.shape[0]
            out_color_chunks = []
            for i in tqdm(range(0, B, config.chunk)):
                xyz_embedded = system.embedding_xyz(
                    vertices[i : i + config.chunk]
                )  # (N, embed_xyz_channels)
                dir_embedded = system.embedding_dir(
                    rays_d[i : i + config.chunk]
                )  # (N, embed_dir_channels)
                input_dict = {
                    "emb_xyz": xyz_embedded,
                    "emb_dir": dir_embedded,
                }
                res_chunk = system.nerf_fine.forward(input_dict)
                out_color_chunks += [res_chunk["rgb"].cpu()]
            colors = torch.cat(out_color_chunks, 0)

        mesh.vertex_colors = o3d.utility.Vector3dVector(
            colors.numpy().astype(np.float64)
        )

    o3d.io.write_triangle_mesh(f"debug/extracted_mesh_{config.prefix}.ply", mesh)

    # remove noise in the mesh by keeping only the biggest cluster
    # print('Removing noise ...')
    # mesh = o3d.io.read_triangle_mesh(f"debug/extracted_mesh.ply")

    idxs, count, _ = mesh.cluster_connected_triangles()
    max_cluster_idx = np.argmax(count)
    triangles_to_remove = [
        i for i in range(len(triangles)) if idxs[i] != max_cluster_idx
    ]
    mesh.remove_triangles_by_index(triangles_to_remove)
    mesh.remove_unreferenced_vertices()
    print(
        f"Mesh has {len(mesh.vertices)/1e6:.2f} M vertices and {len(mesh.triangles)/1e6:.2f} M faces."
    )

    bbox = mesh.get_axis_aligned_bounding_box()
    print(bbox)
    o3d.io.write_triangle_mesh(f"debug/extracted_mesh_clean_{config.prefix}.ply", mesh)

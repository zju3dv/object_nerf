import open3d as o3d
from scipy.spatial.transform import Rotation as R
import json


def main():
    mesh_f = "data/our_desk/img_desk_reconstructions/2/model_2.ply"
    with open("datasets/desk_bbox/desk1/bbox.json", "r") as f:
        j = json.load(f)
        labels = j["labels"]

        print(len(labels))
        for l in labels:
            glist = []
            glist.append(o3d.io.read_triangle_mesh(mesh_f))
            if "position" not in l["data"]:
                continue
            pos = l["data"]["position"]
            quat = l["data"]["quaternion"]
            scale = l["data"]["scale"]
            r = R.from_quat(quat)
            rmat = r.as_matrix()
            bbox = o3d.geometry.OrientedBoundingBox(center=pos, R=rmat, extent=scale)
            glist.append(bbox)
            print(pos)
            o3d.visualization.draw_geometries(glist)


if __name__ == "__main__":
    main()

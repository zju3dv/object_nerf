import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
import random


def colorize_open3d_pcd(pcd):
    # use z axis to map color
    pts = np.asarray(pcd.points)
    pts_z_min = np.min(pts[:, 2])
    pts_z_max = np.max(pts[:, 2])
    colors = plt.cm.viridis((pts[:, 2] - pts_z_min) / (pts_z_max - pts_z_min))[:, :3]
    pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd


def map_to_color(x, cmap="coolwarm", vmin=None, vmax=None):
    if vmin == None or vmax == None:
        vmin = min(x)
        vmax = max(x)
    colors = plt.cm.get_cmap(cmap)((x - vmin) / (vmax - vmin))[:, :3]
    return colors


class O3dVisualizer:
    def __init__(self):
        self.geometries = []

    def add_o3d_geometry(self, geometry):
        self.geometries.append(geometry)

    def add_line_set(self, points, lines, colors=None, radius=0.008):
        # line_set = o3d.geometry.LineSet(
        #     points=o3d.utility.Vector3dVector(points),
        #     lines=o3d.utility.Vector2iVector(lines)
        # )
        if colors is None:
            colors = [
                [random.uniform(0, 1), random.uniform(0, 1), random.uniform(0, 1)]
                for i in range(len(lines))
            ]
        # line_set.colors = o3d.utility.Vector3dVector(colors)
        # self.geometries.append(line_set)
        mesh = LineMesh(points, lines, colors, radius=radius)
        self.geometries.extend(mesh.cylinder_segments)

    def add_np_points(
        self, points, color=None, size=None, resolution=3, with_normal=False
    ):
        if size == None:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points[:, :3])
            pcd = colorize_open3d_pcd(pcd)
            self.geometries.append(pcd)
        else:
            points = points[:, :3]
            mesh = o3d.geometry.TriangleMesh()
            for idx, pt in enumerate(points):
                mesh_sphere = o3d.geometry.TriangleMesh.create_sphere(
                    radius=size, resolution=resolution
                )
                if with_normal:
                    mesh_sphere.compute_vertex_normals()
                transform = np.eye(4)
                transform[0:3, 3] = pt
                mesh_sphere.transform(transform)
                if type(color) == np.ndarray:
                    if color.size == 3:
                        mesh_sphere.paint_uniform_color(color)
                    else:
                        mesh_sphere.paint_uniform_color(color[idx, :])
                else:
                    mesh_sphere.paint_uniform_color([1.0, 0.0, 0.0])
                mesh += mesh_sphere
            self.geometries.append(mesh)

    def text_3d(
        self,
        text,
        pos,
        direction=None,
        degree=0.0,
        font="DejaVu Sans Mono for Powerline.ttf",
        font_size=16,
    ):
        """
        Generate a 3D text point cloud used for visualization.
        :param text: content of the text
        :param pos: 3D xyz position of the text upper left corner
        :param direction: 3D normalized direction of where the text faces
        :param degree: in plane rotation of text
        :param font: Name of the font - change it according to your system
        :param font_size: size of the font
        :return: o3d.geoemtry.PointCloud object
        """
        if direction is None:
            direction = (0.0, 0.0, 1.0)

        from PIL import Image, ImageFont, ImageDraw
        from pyquaternion import Quaternion

        font_obj = ImageFont.truetype(font, font_size)
        font_dim = font_obj.getsize(text)

        img = Image.new("RGB", font_dim, color=(255, 255, 255))
        draw = ImageDraw.Draw(img)
        draw.text((0, 0), text, font=font_obj, fill=(0, 0, 0))
        img = np.asarray(img)
        img_mask = img[:, :, 0] < 128
        indices = np.indices([*img.shape[0:2], 1])[:, img_mask, 0].reshape(3, -1).T

        pcd = o3d.geometry.PointCloud()
        pcd.colors = o3d.utility.Vector3dVector(img[img_mask, :].astype(float) / 255.0)
        pcd.points = o3d.utility.Vector3dVector(indices / 100.0)

        raxis = np.cross([0.0, 0.0, 1.0], direction)
        if np.linalg.norm(raxis) < 1e-6:
            raxis = (0.0, 0.0, 1.0)
        trans = (
            Quaternion(axis=raxis, radians=np.arccos(direction[2]))
            * Quaternion(axis=direction, degrees=degree)
        ).transformation_matrix
        trans[0:3, 3] = np.asarray(pos)
        pcd.transform(trans)
        return pcd

    def run_visualize(self):
        o3d.visualization.draw_geometries(self.geometries)


def align_vector_to_another(a=np.array([0, 0, 1]), b=np.array([1, 0, 0])):
    """
    Aligns vector a to vector b with axis angle rotation
    """
    if np.array_equal(a, b):
        return None, None
    axis_ = np.cross(a, b)
    if np.linalg.norm(axis_) == 0:
        return None, None
    axis_ = axis_ / np.linalg.norm(axis_)
    angle = np.arccos(np.dot(a, b))

    return axis_, angle


def normalized(a, axis=-1, order=2):
    """Normalizes a numpy array of points"""
    l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
    l2[l2 == 0] = 1
    return a / np.expand_dims(l2, axis), l2


class LineMesh(object):
    def __init__(self, points, lines=None, colors=[0, 1, 0], radius=0.15):
        """Creates a line represented as sequence of cylinder triangular meshes

        Arguments:
            points {ndarray} -- Numpy array of ponts Nx3.

        Keyword Arguments:
            lines {list[list] or None} -- List of point index pairs denoting line segments. If None, implicit lines from ordered pairwise points. (default: {None})
            colors {list} -- list of colors, or single color of the line (default: {[0, 1, 0]})
            radius {float} -- radius of cylinder (default: {0.15})
        """
        self.points = np.array(points)
        self.lines = (
            np.array(lines)
            if lines is not None
            else self.lines_from_ordered_points(self.points)
        )
        self.colors = np.array(colors)
        self.radius = radius
        self.cylinder_segments = []

        self.create_line_mesh()

    @staticmethod
    def lines_from_ordered_points(points):
        lines = [[i, i + 1] for i in range(0, points.shape[0] - 1, 1)]
        return np.array(lines)

    def create_line_mesh(self):
        first_points = self.points[self.lines[:, 0], :]
        second_points = self.points[self.lines[:, 1], :]
        line_segments = second_points - first_points
        line_segments_unit, line_lengths = normalized(line_segments)

        z_axis = np.array([0, 0, 1])
        # Create triangular mesh cylinder segments of line
        for i in range(line_segments_unit.shape[0]):
            line_segment = line_segments_unit[i, :]
            line_length = line_lengths[i]
            # get axis angle rotation to allign cylinder with line segment
            axis, angle = align_vector_to_another(z_axis, line_segment)
            # Get translation vector
            translation = first_points[i, :] + line_segment * line_length * 0.5
            # create cylinder and apply transformations
            cylinder_segment = o3d.geometry.TriangleMesh.create_cylinder(
                self.radius, line_length, resolution=4
            )
            cylinder_segment = cylinder_segment.translate(translation, relative=False)
            if axis is not None:
                axis_a = axis * angle
                cylinder_segment = cylinder_segment.rotate(
                    R=o3d.geometry.get_rotation_matrix_from_axis_angle(axis_a)
                )
                # cylinder_segment = cylinder_segment.rotate(
                #   axis_a, center=True, type=o3d.geometry.RotationType.AxisAngle)
            # color cylinder
            color = self.colors if self.colors.ndim == 1 else self.colors[i, :]
            cylinder_segment.paint_uniform_color(color)

            self.cylinder_segments.append(cylinder_segment)

    def add_line(self, vis):
        """Adds this line to the visualizer"""
        for cylinder in self.cylinder_segments:
            vis.add_geometry(cylinder)

    def remove_line(self, vis):
        """Removes this line from the visualizer"""
        for cylinder in self.cylinder_segments:
            vis.remove_geometry(cylinder)

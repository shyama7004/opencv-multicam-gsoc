#!/usr/bin/env python3
"""
OpenCV sample: compare and visualize camera frustums from OpenCV and MATLAB outputs,
then export combined view as a PLY mesh.
"""

import argparse
import glob
import yaml
import numpy as np
import cv2 as cv
from plyfile import PlyData, PlyElement

INPUT_YAML      = '/Users/sankarsanbisoyi/Desktop/OCV/opencv_results.yaml'
MAT_YAML_GLOB   = '/Users/sankarsanbisoyi/Documents/yaml/cam*_gt.yaml'
OUTPUT_PLY      = 'camera_frustums_solid.ply'
VIEW_IDX        = 5    # same snapshot index in all MATLAB files

FRUSTUM_DEPTH        = 0.10   # depth of each frustum from the image plane in meters
AXIS_LENGTH          = 0.03   # length of each camera-local axis in meters
CYLINDER_SEGMENTS    = 12     # number of segments around each tube
FRUSTUM_RADIUS_OCV   = 0.001  # tube radius for OpenCV frustum edges
FRUSTUM_RADIUS_MAT   = 0.003  # tube radius for MATLAB frustum edges
AXIS_RADIUS          = 0.003  # tube radius for all axes
DOT_COLOR_OCV        = (255, 255,   0)
CAMERA_COLOR_OCV     = (128,   0, 128)
DOT_COLOR_MAT        = (  0, 255, 255)
CAMERA_COLOR_MAT     = (255, 128,   0)
AXIS_COLORS          = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]

def mat_to_array(node):
    """Read an OpenCV FileStorage node into a NumPy array."""
    mat = node.mat()
    if mat is None:
        raise RuntimeError(f"Could not read node '{node.name()}'")
    return np.array(mat)

def build_frustum_world(K, Rwc, C, img_size, depth=FRUSTUM_DEPTH):
    """Compute 3D frustum corner positions in world coordinates."""
    fx, fy = K[0,0], K[1,1]
    cx, cy = K[0,2], K[1,2]
    w, h   = img_size
    # image plane corners in pixels
    pix = np.array([[0,0], [w,0], [w,h], [0,h]], dtype=float)
    # project to camera frame at given depth
    x_cam = (pix[:,0] - cx) / fx * depth
    y_cam = (pix[:,1] - cy) / fy * depth
    corners_cam = np.stack([x_cam, y_cam, np.full(4, depth)], axis=1)
    # transform to world frame
    corners_w = (Rwc @ corners_cam.T).T + C
    return C, corners_w

def cylinder(p1, p2, radius, color, segs=CYLINDER_SEGMENTS):
    """Generate vertices and face indices for a colored cylinder between two points."""
    axis = p2 - p1
    length = np.linalg.norm(axis)
    if length < 1e-6:
        return [], []
    axis /= length
    # choose an arbitrary perpendicular direction
    ref = np.array([1,0,0]) if abs(axis[0]) < 0.9 else np.array([0,1,0])
    v = np.cross(axis, ref)
    v /= np.linalg.norm(v)
    u = np.cross(axis, v)
    verts, faces = [], []
    # create ring at each end
    for center in (p1, p2):
        for i in range(segs):
            theta = 2 * np.pi * i / segs
            pt = center + radius * (np.cos(theta) * u + np.sin(theta) * v)
            verts.append((*pt, *color))
    # connect rings with quads (two triangles each)
    for i in range(segs):
        i0 = i
        i1 = (i + 1) % segs
        faces += [
            (i0, segs + i0, i1),
            (i1, segs + i0, segs + i1)
        ]
    return verts, faces

def load_opencv_data(input_yaml):
    """Load intrinsics and extrinsics from an OpenCV-generated YAML file."""
    fs = cv.FileStorage(input_yaml, cv.FILE_STORAGE_READ)
    if not fs.isOpened():
        raise IOError(f"Cannot open '{input_yaml}'")
    Ks_raw = mat_to_array(fs.getNode("Ks"))
    n_cam  = Ks_raw.size // 9
    Ks_all = Ks_raw.reshape(n_cam, 3, 3)
    rvecs  = mat_to_array(fs.getNode("Rs")).reshape(n_cam, 3)
    tvecs  = mat_to_array(fs.getNode("Ts")).reshape(n_cam, 3)
    img_sz = mat_to_array(fs.getNode("image_sizes")).reshape(-1, 2).astype(int)
    fs.release()

    centers, Rwcs = [], []
    for i in range(n_cam):
        R, _ = cv.Rodrigues(rvecs[i])    # world → camera
        C     = -R.T @ tvecs[i]          # camera center in world
        centers.append(C)
        Rwcs.append(R.T)                 # camera → world
    return Ks_all, np.vstack(centers), Rwcs, img_sz

def load_matlab_data(mat_yaml_glob, view_idx):
    """Load poses from MATLAB YAML files at a specific view index."""
    files = sorted(glob.glob(mat_yaml_glob))
    E = np.diag([1, -1, 1])  # flip Y axis to match OpenCV convention

    centers, Rwcs = [], []
    for path in files:
        # clean up YAML and parse
        lines = open(path).read().splitlines()
        clean = [
            L.replace('!!opencv-matrix','')
            for L in lines
            if not L.startswith('%') and not L.strip().startswith(('rows:','cols:','dt:'))
        ]
        data = yaml.safe_load('\n'.join(clean))

        # read rotation and translation for the requested index
        r = np.array(data[f"rvecs_{view_idx}"], dtype=float)
        t = np.array(data[f"tvecs_{view_idx}"], dtype=float) * 1e-3  # mm → m

        R, _ = cv.Rodrigues(r)
        Rwc  = E @ R.T @ E
        C    = E @ (-R.T @ t)
        centers.append(C)
        Rwcs.append(Rwc)
    return np.vstack(centers), Rwcs

def write_ply(output_ply, vertices, faces):
    """Write vertices and faces arrays to a PLY file in ASCII format."""
    v_dtype = [('x','f4'),('y','f4'),('z','f4'),
               ('red','u1'),('green','u1'),('blue','u1')]
    f_dtype = [('vertex_indices','i4',(3,))]
    v_arr = np.zeros(len(vertices), v_dtype)
    for i, (x,y,z,r,g,b) in enumerate(vertices):
        v_arr[i] = (x,y,z,r,g,b)
    f_arr = np.zeros(len(faces), f_dtype)
    for i,(a,b,c) in enumerate(faces):
        f_arr[i] = ([a,b,c],)
    PlyData([
        PlyElement.describe(v_arr, 'vertex'),
        PlyElement.describe(f_arr, 'face')
    ], text=True).write(output_ply)

def main():
    """Parse arguments, load data, build frustums, and export to PLY."""
    parser = argparse.ArgumentParser(
        description="Compare OpenCV and MATLAB camera frustums and write a combined PLY."
    )
    parser.add_argument(
        "--input_yaml", default=INPUT_YAML,
        help="Path to OpenCV results YAML file."
    )
    parser.add_argument(
        "--mat_yaml_glob", default=MAT_YAML_GLOB,
        help="Glob pattern for MATLAB pose YAML files."
    )
    parser.add_argument(
        "--output_ply", default=OUTPUT_PLY,
        help="Filename for the output PLY mesh."
    )
    parser.add_argument(
        "--view_idx", type=int, default=VIEW_IDX,
        help="Snapshot index to use from MATLAB files."
    )
    args = parser.parse_args()

    Ks_all, ocv_centers, ocv_Rwcs, img_sz = load_opencv_data(args.input_yaml)
    mat_centers, mat_Rwcs              = load_matlab_data(args.mat_yaml_glob, args.view_idx)

    # compute mean centers for recentering
    ocv_mean = ocv_centers.mean(axis=0)
    mat_mean = mat_centers.mean(axis=0)

    all_vertices, all_faces = [], []
    edges = [(0,j) for j in range(1,5)] + [(j,(j%4)+1) for j in range(1,5)]

    # build OpenCV frustums
    for i in range(len(ocv_centers)):
        C   = ocv_centers[i] - ocv_mean
        _, corners = build_frustum_world(Ks_all[i], ocv_Rwcs[i], C, img_sz[i])
        # add frustum edges
        for a, b in edges:
            p1 = C if a == 0 else corners[a-1]
            p2 = C if b == 0 else corners[b-1]
            verts, faces = cylinder(p1, p2, FRUSTUM_RADIUS_OCV, CAMERA_COLOR_OCV)
            offset = len(all_vertices)
            all_vertices += verts
            all_faces    += [(x+offset, y+offset, z+offset) for x,y,z in faces]
        # add coordinate axes
        for k, axis in enumerate([(1,0,0), (0,1,0), (0,0,1)]):
            end = C + ocv_Rwcs[i] @ (np.array(axis) * AXIS_LENGTH)
            verts, faces = cylinder(C, end, AXIS_RADIUS, AXIS_COLORS[k])
            offset = len(all_vertices)
            all_vertices += verts
            all_faces    += [(x+offset, y+offset, z+offset) for x,y,z in faces]
        # add camera center point
        all_vertices.append((*C, *DOT_COLOR_OCV))

    # build MATLAB frustums
    for i in range(len(mat_centers)):
        C   = mat_centers[i] - mat_mean
        # reuse image size from first OpenCV camera
        _, corners = build_frustum_world(Ks_all[0], mat_Rwcs[i], C, img_sz[0])
        # Aad frustum edges
        for a, b in edges:
            p1 = C if a == 0 else corners[a-1]
            p2 = C if b == 0 else corners[b-1]
            verts, faces = cylinder(p1, p2, FRUSTUM_RADIUS_MAT, CAMERA_COLOR_MAT)
            offset = len(all_vertices)
            all_vertices += verts
            all_faces    += [(x+offset, y+offset, z+offset) for x,y,z in faces]
        # add coordinate axes
        for k, axis in enumerate([(1,0,0), (0,1,0), (0,0,1)]):
            end = C + mat_Rwcs[i] @ (np.array(axis) * AXIS_LENGTH)
            verts, faces = cylinder(C, end, AXIS_RADIUS, AXIS_COLORS[k])
            offset = len(all_vertices)
            all_vertices += verts
            all_faces    += [(x+offset, y+offset, z+offset) for x,y,z in faces]
        # add camera center point
        all_vertices.append((*C, *DOT_COLOR_MAT))

    # write out the combined mesh
    write_ply(args.output_ply, all_vertices, all_faces)
    print(f"Wrote combined frustums (OCV + MATLAB) to '{args.output_ply}'")

if __name__ == "__main__":
    main()

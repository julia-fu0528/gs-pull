import os
import numpy as np
import math
from typing import NamedTuple

def read_params(params_path):
    params = np.loadtxt(
        params_path,
        dtype=[
            ("cam_id", int),
            ("width", int),
            ("height", int),
            ("fx", float),
            ("fy", float),
            ("cx", float),
            ("cy", float),
            ("k1", float),
            ("k2", float),
            ("p1", float),
            ("p2", float),
            ("cam_name", "<U22"),
            ("qvecw", float),
            ("qvecx", float),
            ("qvecy", float),
            ("qvecz", float),
            ("tvecx", float),
            ("tvecy", float),
            ("tvecz", float),
        ]
    )
    params = np.sort(params, order="cam_name")

    return params


def focal2fov(focal, pixels):
    return 2*math.atan(pixels/(2*focal))

    
def qvec2rotmat(qvec):
    return np.array(
        [
            [
                1 - 2 * qvec[2] ** 2 - 2 * qvec[3] ** 2,
                2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
                2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2],
            ],
            [
                2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
                1 - 2 * qvec[1] ** 2 - 2 * qvec[3] ** 2,
                2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1],
            ],
            [
                2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
                2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
                1 - 2 * qvec[1] ** 2 - 2 * qvec[2] ** 2,
            ],
        ]
    )


def get_rot_trans(param):
    qvec = np.asarray([param["qvecw"], param["qvecx"], param["qvecy"], param["qvecz"]])
    tvec = np.asarray([param["tvecx"], param["tvecy"], param["tvecz"]])
    # r = qvec2rotmat(-qvec)
    r = np.transpose(qvec2rotmat(-qvec))
    return r, tvec

def get_extr(param):
    r, tvec = get_rot_trans(param)
    extr = np.vstack([np.hstack([r, tvec[:, None]]), np.zeros((1, 4))])
    extr[3, 3] = 1
    # extr = np.linalg.inv(extr)
    extr = extr[:3]

    return extr

class CameraInfo(NamedTuple):
    # uid: int
    # R: np.array
    # T: np.array
    extr: np.array
    focal_x: np.array
    focal_y: np.array
    cx: np.array
    cy: np.array
    FovY: np.array
    FovX: np.array
    # image: np.array
    # image_path: str
    # image_name: str
    width: int
    height: int

def readBRICSCameras(params, cam_mapper):
    cam_infos = []
    for param in params:
        if param["cam_name"] not in cam_mapper:
            continue
        
        height = param["height"]
        width = param["width"]
        
        extr = get_extr(param)
        
        focal_length_x = param["fx"]
        focal_length_y = param["fy"]
        cx = param["cx"]
        cy = param["cy"]
        FovY = focal2fov(focal_length_y, height)
        FovX = focal2fov(focal_length_x, width)

        intr = np.eye(3)
        intr[0, 0] = param["fx"]
        intr[1, 1] = param["fy"]
        intr[0, 2] = cx
        intr[1, 2] = cy
        
        dist = np.asarray([param["k1"], param["k2"], param["p1"], param["p2"]]) 
        
        cam_infos.append(CameraInfo(extr, focal_length_x, focal_length_y, cx, cy, FovX, FovY, width, height))
    
    return cam_infos



def get_intr(param, undistort=False):
    intr = np.eye(3)
    intr[0, 0] = param["fx_undist" if undistort else "fx"]
    intr[1, 1] = param["fy_undist" if undistort else "fy"]
    intr[0, 2] = param["cx_undist" if undistort else "cx"]
    intr[1, 2] = param["cy_undist" if undistort else "cy"]

    # TODO: Make work for arbitrary dist params in opencv
    dist = np.asarray([param["k1"], param["k2"], param["p1"], param["p2"]])

    return intr, dist

def get_projections(undistort, params, cam_names, cam_mapper):
    # Gets the projection matrices and distortion parameters
    projs = []
    intrs = []
    dist_intrs = []
    dists = []
    rot = []
    trans = []

    for param in params:
        if (param["cam_name"] in cam_names) and (param["cam_name"] in cam_mapper):
            extr = get_extr(param)
            intr, dist = get_intr(param)
            r, t = get_rot_trans(param)

            rot.append(r)
            trans.append(t)

            intrs.append(intr.copy())
            
            dist_intrs.append(intr.copy())

            projs.append(intr @ extr)
            dists.append(dist)
    if undistort:
        cameras = { 'K': np.asarray(dist_intrs),
                    'R': np.asarray(rot), 
                    'T': np.asarray(trans) }
    else:
        cameras = { 'K': np.asarray(intrs),
                    'R': np.asarray(rot), 
                    'T': np.asarray(trans) }
    
    return intrs, np.asarray(projs), dist_intrs, dists, cameras


def get_ngp_cameras(params, cam_names, cam_mapper):
    cam2idx = {}
    pos = []
    rot = []
    intrs = []
    dists = []
    c2ws = []

    for idx, param in enumerate(params):
        if (param["cam_name"] in cam_names) and (param["cam_name"] in cam_mapper):
            w2c = get_extr(param)
            intr, dist = get_intr(param)
            w2c = np.vstack((w2c, np.asarray([[0, 0, 0, 1]])))
            c2w = np.linalg.inv(w2c)
            cam2idx[param["cam_name"]]= idx
            intrs.append(intr)
            dists.append(dist)
            pos.append(c2w[:3, 3])
            rot.append(c2w[:3, :3])
            c2ws.append(c2w)
    extrs = np.array(c2ws)

    return intrs, extrs, dists


def map_camera_names(base_dir, name_list):
    """
    Maps each name in the name_list to a subdirectory in base_dir that starts with the name.

    :param base_dir: The directory to search for subdirectories.
    :param name_list: A list of names to map to subdirectories.
    :return: A dictionary mapping each name in name_list to a matching subdirectory in base_dir.
    """
    # Find all subdirectories in the base directory
    subdirs = [d.split('.')[0] for d in os.listdir(base_dir)]

    # Create a dictionary to map names in name_list to subdirectories
    name_map = {}

    for name in name_list:
        # Find a subdirectory that starts with the name
        matched_subdir = next((subdir for subdir in subdirs if subdir.startswith(name)), None)
        
        if matched_subdir:
            name_map[name] = matched_subdir

    return name_map

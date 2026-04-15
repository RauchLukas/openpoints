import os
import os.path as osp
import numpy as np
import torch
from torch.utils.data import Dataset
from ..build import DATASETS
from ..data_util import crop_pc, voxelize
from ...transforms.point_transform_cpu import PointsToTensor
import glob
from tqdm import tqdm
import logging
import pickle
from concurrent.futures import ThreadPoolExecutor


def hex_to_rgb(hex_string):
    value = hex_string.lstrip('#')
    lv = len(value)
    return [int(value[i:i + lv // 3], 16) for i in range(0, lv, lv // 3)]


VALID_CLASSES = {
    0:	"None",
    1:	"Ceiling",
    2:	"Slanted Ceiling",
    3:	"Ceiling Cutout",
    4:	"Floor",
    5:	"Wall",
    6:	"Drywall",
    7:	"Wall Cutout",
    8:	"Parapet",
    9:	"Beam",
    10:	"Column",
    11:	"Staircase",
    12:	"Window Cutout",
    13:	"Window",
    14:	"Door Rough Opening",
    15:	"Facade Element",
    16:	"Elevator Doors",
    17:	"TGA",
}

ROHBAU3D_COLOR_MAP = {
    0: "#727274",
    1: "#FDA9A9",
    2: "#C78282",
    3: "#007AFF",
    4: "#3B9148",
    5: "#8686C0",
    6: "#89D6E4",
    7: "#911EB4",
    8: "#FF7300",
    9: "#2121E2",
    10: "#FF2D5A",
    11: "#93E09B",
    12: "#990000",
    13: "#FAA940",
    14: "#FF18E0",
    15: "#FFE119",
    16: "#B6CA00",
    17: "#00FF0D",
}

ROHBAU3D_COLOR_MAP_RGB = {k: hex_to_rgb(v) for k, v in ROHBAU3D_COLOR_MAP.items()}

def load_data(in_path, features):
    """Load required arrays or return None on any failure."""
    try:
        coord = np.load(os.path.join(in_path, "coord.npy"))   # (N, 3)
        segment = np.load(os.path.join(in_path, "class.npy"))   # (N, 1)
    except Exception as e:
        logging.error(f"[ERROR] Loading arrays in {in_path}: {e}")
        return None, None, None

    num_points = coord.shape[0]
    if segment.shape[0] != num_points:
        logging.error(f"[ERROR] Label length mismatch in {in_path}. Expected {num_points}, got {segment.shape[0]}")
        return None, None, None

    feat_list = []
    feat_names = []

    # print a warning if the featurs list is empty or not defined [None]
    if not features or features == [None]:
        logging.warning(f"[WARNING] No additional features specified for {in_path}. Only coordinates and labels will be loaded.")
        return coord, np.array([]).reshape(num_points, 0), segment

    if any(f.lower() in ['rgb', 'r', 'g', 'b'] for f in features if f is not None):
        try:
            color = np.load(os.path.join(in_path, "color.npy"))
            if color.shape[0] != num_points:
                logging.error(f"[ERROR] Color length mismatch in {in_path}")
                return None, None, None
            feat_list.append((color / 255.0).astype(np.float32))
            feat_names.extend(['R', 'G', 'B'])
        except Exception as e:
            logging.error(f"[ERROR] Loading color in {in_path}: {e}")
            return None, None, None

    if any(f.lower() in ['intensity', 'i'] for f in features if f is not None):
        try:
            intensity = np.load(os.path.join(in_path, "intensity.npy"))
            if intensity.shape[0] != num_points:
                logging.error(f"[ERROR] Intensity length mismatch in {in_path}")
                return None, None, None
            feat_list.append(np.expand_dims(intensity, -1).astype(np.float32))
            feat_names.append('I')
        except Exception as e:
            logging.error(f"[ERROR] Loading intensity in {in_path}: {e}")
            return None, None, None

    if any(f.lower() in ['normal', 'normals', 'n'] for f in features if f is not None):
        try:
            normal = np.load(os.path.join(in_path, "normal.npy"))
            if normal.shape[0] != num_points:
                logging.error(f"[ERROR] Normal length mismatch in {in_path}")
                return None, None, None
            feat_list.append(normal.astype(np.float32))
            feat_names.extend(['Nx', 'Ny', 'Nz'])
        except Exception as e:
            logging.error(f"[ERROR] Loading normal in {in_path}: {e}")
            return None, None, None

    feat = np.hstack(feat_list) if feat_list else np.array([]).reshape(num_points, 0)

    #DEBUG 
    # logging.info(f"Features loaded for {in_path}: {feat_names} with shape {feat.shape}")

    return coord, feat, segment


@DATASETS.register_module()
class Rohbau3D(Dataset):
    classes = list(VALID_CLASSES.values())
    num_classes = len(classes)
    cmap = [*ROHBAU3D_COLOR_MAP_RGB.values()]
    gravity_dim = 2


    def __init__(self,
                 data_root: str = 'data/Rohnbau3D',
                 split: str = 'train',
                 voxel_size: float = 0.04,
                 voxel_max = None,
                 transform = None,
                 features: list = [None],
                 loop: int = 1, 
                 presample: bool = False, 
                 variable: bool = False,
                 shuffle: bool = True,
                 n_shifted: int = 1,
                 ):
        super().__init__()
        self.split = split
        self.voxel_size = voxel_size
        self.voxel_max = voxel_max
        self.transform = transform
        self.presample = presample
        self.variable = variable
        self.loop = loop
        self.n_shifted = n_shifted
        self.pipe_transform = PointsToTensor() 
        self.features = features

        logging.info(f"Features used for {split}: {self.features}")

        if split == "train" or split == 'val':
            self.data_list = glob.glob(os.path.join(data_root, split, "site_*", 'scene_*'))
            # self.data_list = glob.glob(os.path.join(data_root, split, 'scene_*'))   # list all scan folders. Not the files itself!
        elif split == 'trainval':
            self.data_list = glob.glob(os.path.join(data_root, 'train', "site_*", 'scene_*')) + \
                             glob.glob(os.path.join(data_root, 'val', "site_*", 'scene_*'))
        elif split == 'test':
            self.data_list = glob.glob(os.path.join(data_root, split, "site_*", 'scene_*'))
        else:
            raise ValueError("no such split: {}".format(split))
        
        logging.info("Totally {} samples in {} set.".format(
            len(self.data_list), split))
        
        processed_root = os.path.join(data_root, 'processed')
        filename = os.path.join(
            processed_root, f'rohbau3d_{split}_{voxel_size:.3f}.pkl')
        if presample and not os.path.exists(filename):
            np.random.seed(0)
            self.data = []

            def process_item(item):
                coord, feat, label = load_data(item, self.features)

                if coord is None:
                    return None

                coord, feat, label = crop_pc(
                    coord, feat, label, split=self.split, 
                    voxel_size=self.voxel_size, voxel_max=None, downsample=True, variable=self.variable)
                return np.hstack(
                    (coord, feat, np.expand_dims(label, -1))).astype(np.float32)
            
            with ThreadPoolExecutor(max_workers=16) as executor:
                results = list(tqdm(executor.map(process_item, self.data_list), 
                                   total=len(self.data_list), 
                                   desc=f'Loading Rohbau3D {split} split'))

            self.data = [r for r in results if r is not None]

            npoints = np.array([len(data) for data in self.data])
            logging.info('split: %s, median npoints %.1f, avg num points %.1f, std %.1f' % (
                self.split, np.median(npoints), np.average(npoints), np.std(npoints)))
            os.makedirs(processed_root, exist_ok=True)
            with open(filename, 'wb') as f:
                pickle.dump(self.data, f)
                logging.info(f"{filename} saved successfully")
        elif presample:
            with open(filename, 'rb') as f:
                self.data = pickle.load(f)
                logging.info(f"{filename} load successfully")


    def __getitem__(self, idx):
        data_idx = idx % len(self.data_list)

        if self.presample:
            coord, feat, label = np.split(self.data[data_idx], [3, -1], axis=1)
            # print(f"Shaps: coords:{coord.shape}, feat: {feat.shape}, label: {label.shape}")
        else:
            data_path = self.data_list[data_idx]

            # load RB3D data from numpy files 
            coord, feat, label = load_data(data_path, self.features)

        
        label = label.astype(np.long).squeeze()
        data = {'pos': coord.astype(np.float32), 'x': feat.astype(np.float32), 'y': label}

        # must be bevore pc_crop to ensure vector size after drop_out transforms
        if self.transform is not None:
            data = self.transform(data)

        # if not self.presample: 
        data['pos'], data['x'], data['y'] = crop_pc(
            data['pos'], data['x'], data['y'], self.split, self.voxel_size, self.voxel_max,
            downsample=False, variable=self.variable)

        data = self.pipe_transform(data)
        
        # always try to sphere crop for RB3D
        if 'heights' not in data.keys():
            data['heights'] =  data['pos'][:, self.gravity_dim:self.gravity_dim+1] - data['pos'][:, self.gravity_dim:self.gravity_dim+1].min()

        return data

    def __len__(self):
        return len(self.data_list) * self.loop

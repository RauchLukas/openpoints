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


def hex_to_rgb(hex_string):
    value = hex_string.lstrip('#')
    lv = len(value)
    return [int(value[i:i + lv // 3], 16) for i in range(0, lv, lv // 3)]


VALID_CLASSES = {
            0: "none",
            1: "ceiling",
            2: "floor",
            3: "wall",
            4: "beam",
            5: "column",
            6: "window",
            7: "door",
            8: "stairs",
            9: "equipment",
            10: "installation",
        }

ROHBAU3D_COLOR_MAP = {
            0: '#2f4f4f', 
            1: '#228b22',
            2: '#7f0000',
            3: '#ff8c00', 
            4: '#000080',
            5: '#ffff00',
            6: '#00ff00',
            7: '#00ffff', 
            8: '#ff00ff',
            9: '#eee8aa',
            10: '#ff0000',
}
ROHBAU3D_COLOR_MAP_RGB = {k: hex_to_rgb(v) for k, v in ROHBAU3D_COLOR_MAP.items()}




def load_data(in_path):
    """Load required arrays or return None on any failure."""
    try:
        coord = np.load(os.path.join(in_path, "coord.npy"))   # (N, 3)
        color = np.load(os.path.join(in_path, "color.npy"))   # (N, 3)
        segment = np.load(os.path.join(in_path, "segment.npy"))   # (N, 1)
        # intensity = np.load(os.path.join(in_path, "intensity.npy"))  # (N,)
        # normal = np.load(os.path.join(in_path, "normal.npy")) # (N, 3)
    except Exception as e:
        print(f"[ERROR] Loading arrays in {in_path}: {e}")
        return None

    # sanity check: all arrays must have the same length
    num_points = coord.shape[0]
    if color.shape[0] != num_points:
        print(f"[ERROR] Color length mismatch in {in_path}. Expected {num_points}, got {color.shape[0]}")
        return None
    # if intensity.shape[0] != num_points:
    #     print(f"[ERROR] Intensity length mismatch in {in_path}. Expected {num_points}, got {intensity.shape[0]}")
    #     return None    # original returned; keep same policy
    # if normal.shape[0] != num_points:
    #     print(f"[ERROR] Normal length mismatch in {in_path}. Expected {num_points}, got {normal.shape[0]}")
    #     return None
    if segment.shape[0] != num_points:
        print(f"[ERROR] LabeL length mismatch in {in_path}. Expected {num_points}, got {segment.shape[0]}")
        return None

    return coord, color, segment


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

        if split == "train" or split == 'val':
            self.data_list = glob.glob(os.path.join(data_root, split, 'scan_*'))   # list all scan folders. Not the files itself!
        elif split == 'trainval':
            self.data_list = glob.glob(os.path.join(data_root, 'train', 'scan_*')) + \
                             glob.glob(os.path.join(data_root, 'val', 'scan_*'))
        elif split == 'test':
            self.data_list = glob.glob(os.path.join(data_root, split, 'scan_*'))
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
            for item in tqdm(self.data_list, desc=f'Loading Rohbau3D {split} split'):

                # load RB3D data from numpy files 
                coord, feat, label = load_data(item)

                # coord, feat, label = crop_pc(
                #     coord, feat, label, self.split, self.voxel_size, None, variable=self.variable)
                coord, feat, label = crop_pc(
                    coord, feat, label, split=self.split, 
                    voxel_size=self.voxel_size, voxel_max=None, downsample=True, variable=self.variable)
                cdata = np.hstack(
                    (coord, feat, np.expand_dims(label, -1))).astype(np.float32)
                self.data.append(cdata)
            npoints = np.array([len(data) for data in self.data])
            logging.info('split: %s, median npoints %.1f, avg num points %.1f, std %.1f' % (
                self.split, np.median(npoints), np.average(npoints), np.std(npoints)))
            os.makedirs(processed_root, exist_ok=True)
            with open(filename, 'wb') as f:
                pickle.dump(self.data, f)
                print(f"{filename} saved successfully")
        elif presample:
            with open(filename, 'rb') as f:
                self.data = pickle.load(f)
                print(f"{filename} load successfully")


    def __getitem__(self, idx):
        data_idx = idx % len(self.data_list)
        if self.presample:
            coord, feat, label = np.split(self.data[data_idx], [3, 6], axis=1)
        else:
            data_path = self.data_list[data_idx]
            # data = torch.load(data_path)
            # coord, feat, label = data[0:3]

            # load RB3D data from numpy files 
            coord, feat, label = load_data(data_path)

        feat = (feat + 1) * 127.5
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






# @DATASETS.register_module()
# class ScanNet(Dataset):
#     num_classes = 20
#     classes = ['wall', 'floor', 'cabinet', 'bed', 'chair', 'sofa', 'table', 'door', 'window', 'bookshelf', 'picture',
#                'counter', 'desk', 'curtain', 'refrigerator', 'shower curtain', 'toilet', 'sink', 'bathtub', 'otherfurniture']
#     gravity_dim = 2
    
#     color_mean = [0.46259782, 0.46253258, 0.46253258]
#     color_std =  [0.693565  , 0.6852543 , 0.68061745]
#     """ScanNet dataset, loading the subsampled entire room as input without block/sphere subsampling.
#     number of points per room in average, median, and std: (145841.0, 158783.87179487178, 84200.84445829492)
#     """  
#     def __init__(self,
#                  data_root='data/ScanNet',
#                  split='train',
#                  voxel_size=0.04,
#                  voxel_max=None,
#                  transform=None,
#                  loop=1, presample=False, variable=False,
#                  n_shifted=1
#                  ):
#         super().__init__()
#         self.split = split
#         self.voxel_size = voxel_size
#         self.voxel_max = voxel_max
#         self.transform = transform
#         self.presample = presample
#         self.variable = variable
#         self.loop = loop
#         self.n_shifted = n_shifted
#         self.pipe_transform = PointsToTensor() 

#         if split == "train" or split == 'val':
#             self.data_list = glob.glob(os.path.join(data_root, split, "*.pth"))
#         elif split == 'trainval':
#             self.data_list = glob.glob(os.path.join(
#                 data_root, "train", "*.pth")) + glob.glob(os.path.join(data_root, "val", "*.pth"))
#         elif split == 'test':
#             self.data_list = glob.glob(os.path.join(data_root, split, "*.pth"))
#         else:
#             raise ValueError("no such split: {}".format(split))

#         logging.info("Totally {} samples in {} set.".format(
#             len(self.data_list), split))

#         processed_root = os.path.join(data_root, 'processed')
#         filename = os.path.join(
#             processed_root, f'scannet_{split}_{voxel_size:.3f}.pkl')
#         if presample and not os.path.exists(filename):
#             np.random.seed(0)
#             self.data = []
#             for item in tqdm(self.data_list, desc=f'Loading ScanNet {split} split'):
#                 data = torch.load(item)
#                 coord, feat, label = data[0:3]
#                 coord, feat, label = crop_pc(
#                     coord, feat, label, self.split, self.voxel_size, self.voxel_max, variable=self.variable)
#                 cdata = np.hstack(
#                     (coord, feat, np.expand_dims(label, -1))).astype(np.float32)
#                 self.data.append(cdata)
#             npoints = np.array([len(data) for data in self.data])
#             logging.info('split: %s, median npoints %.1f, avg num points %.1f, std %.1f' % (
#                 self.split, np.median(npoints), np.average(npoints), np.std(npoints)))
#             os.makedirs(processed_root, exist_ok=True)
#             with open(filename, 'wb') as f:
#                 pickle.dump(self.data, f)
#                 print(f"{filename} saved successfully")
#         elif presample:
#             with open(filename, 'rb') as f:
#                 self.data = pickle.load(f)
#                 print(f"{filename} load successfully")
#             # median, average, std of number of points after voxel sampling for val set.
#             # (100338.5, 109686.1282051282, 57024.51083415437)
#             # before voxel sampling
#             # (145841.0, 158783.87179487178, 84200.84445829492)
#     def __getitem__(self, idx):
#         data_idx = idx % len(self.data_list)
#         if self.presample:
#             coord, feat, label = np.split(self.data[data_idx], [3, 6], axis=1)
#         else:
#             data_path = self.data_list[data_idx]
#             data = torch.load(data_path)
#             coord, feat, label = data[0:3]

#         feat = (feat + 1) * 127.5
#         label = label.astype(np.long).squeeze()
#         data = {'pos': coord.astype(np.float32), 'x': feat.astype(np.float32), 'y': label}
#         """debug 
#         from openpoints.dataset import vis_multi_points
#         import copy
#         old_data = copy.deepcopy(data)
#         if self.transform is not None:
#             data = self.transform(data)
#         data['pos'], data['x'], data['y'] = crop_pc(
#             data['pos'], data['x'], data['y'], self.split, self.voxel_size, self.voxel_max,
#             downsample=not self.presample, variable=self.variable)
            
#         vis_multi_points([old_data['pos'][:, :3], data['pos'][:, :3]], colors=[old_data['x'][:, :3]/255.,data['x'][:, :3]])
#         """
#         if self.transform is not None:
#             data = self.transform(data)
        
#         if not self.presample: 
#             data['pos'], data['x'], data['y'] = crop_pc(
#                 data['pos'], data['x'], data['y'], self.split, self.voxel_size, self.voxel_max,
#                 downsample=not self.presample, variable=self.variable)
        
#         data = self.pipe_transform(data)
         
#         if 'heights' not in data.keys():
#             data['heights'] =  data['pos'][:, self.gravity_dim:self.gravity_dim+1] - data['pos'][:, self.gravity_dim:self.gravity_dim+1].min()
#         return data

#     def __len__(self):
#         return len(self.data_list) * self.loop
import random
import cv2
import torch
import SimpleITK as sitk
import numpy as np
import os
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import yaml
from scipy.ndimage.interpolation import zoom
import matplotlib.pyplot as plt
from torch.nn import functional as F
import json
from scipy.ndimage import gaussian_filter, binary_fill_holes


class prln_dataset(Dataset):
    """
    ***请注意***
    3D影像的格式: D x H x W
    候选框的格式: z, x, y, d, h, w
    """

    def __init__(self, cfg):
        super(prln_dataset, self).__init__()
        self.roi_size = [cfg['roi_z'], cfg['roi_y'], cfg['roi_x']]
        self.spacing = [cfg['space_z'], cfg['space_y'], cfg['space_x']]
        self.root_dir = cfg['root_dir']
        self.amin = cfg['a_min']
        self.amax = cfg['a_max']
        self.bmin = cfg['b_min']
        self.bmax = cfg['b_max']
        self.num_samples = cfg['num_samples']
        self.list_dir = os.path.join(self.root_dir, 'PRLN', 'downstream.json')
        self.target_dir = os.path.join(self.root_dir, 'PRLN', 'prln_target.csv')

        self.check_data_integrity = cfg['check_data_integrity']
        self.visualize = cfg['visualize']
        self.margin = 2

        self.__load_data__()

    def __load_data__(self):
        # load label info list
        with open(self.target_dir, 'r') as f:
            lines = f.readlines()
        caselist = [line.strip().replace(' ', '').split(',') for line in lines]

        with open(self.list_dir, 'r') as f:
            target = json.load(f)

        tr = target['tr']
        ts = target['ts']

        if self.check_data_integrity:
            print('check data integrity...')
            for case in caselist:
                casename = case[0]
                assert casename in tr or casename in ts, f'{casename} not in target list'
            print('data check done.')
        self.caselist = [x for x in caselist if x[0] in tr]

    def __len__(self):
        return len(self.caselist)
    

    def __output__(self):
        os.makedirs('prln_cache', exist_ok=True)
        for idx in tqdm(range(len(self))):
            imgs, lbls = self.__makeitem__(idx)
            np.savez_compressed(f'prln_cache/{idx}.npz', imgs=imgs.numpy(), lbls=lbls.numpy().astype('uint8'))

    def __getitem__(self, idx):
        try:
            npz = np.load(f'prln_cache/{idx}.npz')
            imgs, lbls = npz['imgs'], npz['lbls']
            return torch.tensor(imgs).float(), torch.tensor(lbls).float()
        except:
            imgs, lbls = self.__makeitem__(idx)
            os.makedirs('prln_cache', exist_ok=True)
            np.savez_compressed(f'prln_cache/{idx}.npz', imgs=imgs.numpy(), lbls=lbls.numpy().astype('uint8'))
            return imgs, lbls

    def __get_dist__(self, idx):
        case = self.caselist[idx]
        ct_name, label_id, vol, z1, y1, x1, z2, y2, x2 = case
        z1, y1, x1, z2, y2, x2 = int(z1), int(y1), int(x1), int(z2)+1, int(y2)+1, int(x2)+1
        label_id = int(label_id)

        # load image and label
        npz = np.load(os.path.join(self.root_dir, 'npz', ct_name + '.npz'))
        input_spacing = npz['spacing']

        npz = np.load(os.path.join(self.root_dir, 'clean_labels', ct_name + '.npz'))
        lbl_np = npz['arr_0'].astype('uint8')

        lbl_np = lbl_np[z1:z2+1, y1:y2+1, x1:x2+1]
        scale = input_spacing / np.array(self.spacing)
        lbl_np = zoom(lbl_np, scale, order=0)

        non_zero_points = np.argwhere(lbl_np==label_id)
        distances = np.sqrt(np.sum((non_zero_points[:, np.newaxis] - non_zero_points[np.newaxis, :])**2, axis=-1))
        max_dist = distances.max()
        return max_dist
    
    
    def __makeitem__(self, idx):
        case = self.caselist[idx]
        ct_name, label_id, vol, z1, y1, x1, z2, y2, x2 = case
        z1, y1, x1, z2, y2, x2 = int(z1), int(y1), int(x1), int(z2)+1, int(y2)+1, int(x2)+1
        label_id = int(label_id)
        vol = int(vol)
        z, y, x = (z1 + z2) / 2, (y1 + y2) / 2, (x1 + x2) / 2
        d, h, w = z2 - z1, y2 - y1, x2 - x1
        orig = np.array([z, y, x, d, h, w])

        # load image and label
        npz = np.load(os.path.join(self.root_dir, 'npz', ct_name + '.npz'))
        input_spacing = npz['spacing']
        img_np = npz['img'].astype(float)

        npz = np.load(os.path.join(self.root_dir, 'clean_labels', ct_name + '.npz'))
        lbl_np = npz['arr_0'].astype('float32')

        # calulate max dist
        non_zero_points = np.argwhere(lbl_np==label_id)
        distances = np.sqrt(np.sum((non_zero_points[:, np.newaxis] - non_zero_points[np.newaxis, :])**2, axis=-1))
        max_dist = distances.max()
        REAL_DIST_MIN = 1.73
        REAL_DIST_MAX = 25.0
        MIN_UPSCALE = max(0.2, REAL_DIST_MIN / max_dist)
        MAX_UPSCALE = min(5.0, REAL_DIST_MAX / max_dist)
        

        # upscale = random.uniform(1.0, 3.0)
        upscale = random.uniform(MIN_UPSCALE, MAX_UPSCALE)
        scale = input_spacing / np.array(self.spacing) * upscale
        if (orig[3:] * scale - np.array(self.roi_size) + self.margin >= 0).any():
            scale = ((np.array(self.roi_size) - self.margin) / orig[3:]).min()
        elif (orig[3:] * scale < 2).any():
            scale = 2 / orig[3:].min()

        l = np.array([z1, y1, x1])
        r = np.array([z2, y2, x2])
        ll = np.maximum(0, np.floor(r-np.array(self.roi_size)/scale)).astype(int)
        rr = np.minimum(img_np.shape, np.ceil(l+np.array(self.roi_size)/scale)).astype(int)
        
        img_np = img_np[ll[0]:rr[0], ll[1]:rr[1], ll[2]:rr[2]]
        lbl_np = lbl_np[ll[0]:rr[0], ll[1]:rr[1], ll[2]:rr[2]]


        img_np = zoom(img_np, scale, order=1)
        lbl_np[lbl_np != label_id] = 0
        lbl_np[lbl_np == label_id] = 1
        lbl_np = zoom(lbl_np, scale, order=1)

        orig[:3] -= ll
        orig[:3] *= scale
        orig[3:] *= scale
        orig = orig.astype(int)

        # random crop by target
        imgs = []
        lbls = []

        for i in range(self.num_samples):
            img_crop, lbl_crop = self.RandCropByTarget(img_np, lbl_np, self.roi_size, orig, training=True)

            # clamp & normalize
            img_crop = np.clip(img_crop, self.amin, self.amax)
            img_crop = (img_crop - self.amin) / (self.amax - self.amin) * (self.bmax - self.bmin) + self.bmin
            # lbl_crop = gaussian_filter(lbl_crop, sigma=1.0) # Deprecated, gaussian filter may erase small targets
            lbl_crop = binary_fill_holes(lbl_crop > 0.5)
            # lbl_crop = cv2.dilate(lbl_crop, kernel, iterations=2)
            imgs.append(torch.tensor(img_crop).float().unsqueeze(0))
            lbls.append(torch.tensor(lbl_crop).float().unsqueeze(0))
        imgs = torch.stack(imgs).permute(0, 1, 4, 3, 2)
        lbls = torch.stack(lbls).permute(0, 1, 4, 3, 2)

        # img_crop[lbl_crop > 0] = -1.0
        # visualize and save image
        if self.visualize:
            os.makedirs(f'visualize/{ct_name}/{label_id}', exist_ok=True)
            for i in range(img_crop.shape[0]):
                plt.figure()
                plt.subplot(121)
                plt.imshow(img_crop[i], cmap='gray')
                plt.subplot(122)
                plt.imshow(lbl_crop[i], cmap='gray')
                plt.show()
                plt.savefig(f'visualize/{ct_name}/{label_id}/{ct_name}_{label_id}_{i}.png')
                plt.close()

        return imgs, lbls

    def RandCropByTarget(self, img_np, lbl_np, roi, orig, training=True):
        # pad if nessesary
        if img_np.shape[0] < roi[0] or img_np.shape[1] < roi[1] or img_np.shape[2] < roi[2]:
            pad = np.maximum(np.array(roi) - img_np.shape, 0)
            pad = np.array([[0, pad[0]], [0, pad[1]], [0, pad[2]]])
            img_np = np.pad(img_np, pad, 'constant', constant_values=0)
            lbl_np = np.pad(lbl_np, pad, 'constant', constant_values=0)

        margin = np.array([self.margin, self.margin, self.margin])
        roi = np.array(roi)
        fulsize = np.maximum(img_np.shape, roi)
        pos0 = orig[:3]
        vol = orig[3:] + margin  # vol can be decimal
        pos1 = pos0 - roi / 2
        pos2 = pos0 + roi / 2
        shift_min = np.maximum(-pos1, pos0 + vol / 2 - pos2)
        shift_max = np.minimum(fulsize - pos2, pos0 - vol / 2 - pos1)
        shift_range = shift_max - shift_min

        assert (shift_range >= 0).all(), (
        'Upper bound must be greater than lower bound!', orig, img_np.shape, shift_min, shift_max, shift_range)
        shift = np.zeros(3, dtype=np.int32)
        if training:
            # compatible with low version of numpy
            # shift = np.random.randint(shift_range + 1)   # +1 cause upper bound is exclusive
            for i in range(3):
                shift[i] = np.random.randint(shift_range[i] + 1)  # +1 cause upper bound is exclusive
        else:
            # for eval, shift must be certain
            shift = np.maximum(-shift_min, 0)

        cor1 = (pos1 + shift_min + shift).astype('int32')
        cor2 = (pos2 + shift_min + shift).astype('int32')
        cor1 = np.maximum(cor1, 0)
        cor2 = np.minimum(cor2, img_np.shape)
        img_np = img_np[cor1[0]:cor2[0], cor1[1]:cor2[1], cor1[2]:cor2[2]]
        lbl_np = lbl_np[cor1[0]:cor2[0], cor1[1]:cor2[1], cor1[2]:cor2[2]]

        return img_np, lbl_np


def read_yaml_config(config_file):
    with open(config_file, 'r') as file:
        try:
            return yaml.safe_load(file)
        except yaml.YAMLError as e:
            print(e)
            return None


def data_collate(batch):
    imgs = [item[0] for item in batch]
    lbls = [item[1] for item in batch]

    dim = len(imgs[0].shape)
    batch_dict = {
        'image': torch.stack(imgs) if dim == 4 else torch.cat(imgs),
        'label': torch.stack(lbls) if dim == 4 else torch.cat(lbls),
    }
    return batch_dict


def get_loader(cfg):
    dataset = prln_dataset(cfg)
    train_loader = DataLoader(dataset, batch_size=cfg['batch_size'], shuffle=True, num_workers=cfg['num_workers'],
                              collate_fn=data_collate)
    return train_loader


if __name__ == '__main__':
    torch.cuda.set_device(5)
    cfg = read_yaml_config('config/dataset/data_config_prln.yaml')

    # debug
    cfg['visualize'] = True
    cfg['num_samples'] = 1
    print(cfg)

    ds = prln_dataset(cfg)

    for i in range(len(ds)):
        ds.__makeitem__(i)


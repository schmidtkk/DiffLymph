import io
import random
import imageio.v2 as imageio
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
import sys
import os
sys.path.append(os.getcwd())
from utils.sdf_fn import compute_sdf, visualize_sdf

class prlnmask_dataset(Dataset):
    """
    ***请注意***
    3D影像的格式: D x H x W
    候选框的格式: z, x, y, d, h, w
    """

    def __init__(self, cfg):
        super(prlnmask_dataset, self).__init__()
        self.roi_size = [cfg['roi_z'], cfg['roi_y'], cfg['roi_x']]
        self.spacing = [cfg['space_z'], cfg['space_y'], cfg['space_x']]
        self.root_dir = cfg['root_dir']
        self.amin = cfg['a_min']
        self.amax = cfg['a_max']
        self.bmin = cfg['b_min']
        self.bmax = cfg['b_max']
        self.num_samples = cfg['num_samples']
        self.list_dir = os.path.join(self.root_dir, 'PRLN', 'target.json')
        self.target_dir = os.path.join(self.root_dir, 'PRLN', 'prln_target.csv')
        self.margin_ratio = 0.5

        self.check_data_integrity = cfg['check_data_integrity']
        self.visualize = cfg['visualize']
        self.use_sdf = cfg['use_sdf']
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
        # self.caselist = [x for x in caselist if x[0] in tr]
        self.tr = tr

    def __len__(self):
        return len(self.tr)
    
    def __statistic__(self):
        rois = []

        for idx in tqdm(range(len(self.tr))):
            ct_name = self.tr[idx]
            # load image and label
            npz = np.load(os.path.join(self.root_dir, 'npz', ct_name + '.npz'))
            input_spacing = npz['spacing']
            npz = np.load(os.path.join(self.root_dir, 'clean_labels', ct_name + '.npz'))
            lbl_np = npz['arr_0'].astype('uint8')
            label_ids = np.unique(lbl_np)
            label_ids = label_ids[label_ids != 0]
            scale = input_spacing / np.array(self.spacing)
            for label_id in label_ids:
                lbl_roi = lbl_np == label_id
                z_inds, y_inds, x_inds = np.where(lbl_roi > 0)
                zmin, zmax = z_inds.min(), z_inds.max()
                ymin, ymax = y_inds.min(), y_inds.max()
                xmin, xmax = x_inds.min(), x_inds.max()
                lbl_crop = lbl_roi[zmin:zmax + 1, ymin:ymax + 1, xmin:xmax + 1]
                lbl_roi = zoom(lbl_crop, scale, order=0)     
                d, h, w = lbl_roi.shape
                pad_z0 = (self.roi_size[0] - d) // 2
                pad_z1 = self.roi_size[0] - d - pad_z0
                pad_y0 = (self.roi_size[1] - h) // 2
                pad_y1 = self.roi_size[1] - h - pad_y0
                pad_x0 = (self.roi_size[2] - w) // 2
                pad_x1 = self.roi_size[2] - w - pad_x0
                lbl_roi = np.pad(lbl_roi, ((pad_z0, pad_z1), (pad_y0, pad_y1), (pad_x0, pad_x1)), 'constant', constant_values=0)
                if self.use_sdf:
                    lbl_roi = compute_sdf(lbl_roi)
                else:
                    lbl_roi = lbl_roi * 2.0 - 1.0
                assert lbl_roi.shape == tuple(self.roi_size), f'{lbl_roi.shape} != {tuple(self.roi_size)}'
                rois.append(lbl_roi)
        rois = np.stack(rois)
        print(f'ROI shape: {rois.shape}')
        np.savez_compressed('rois.npz', arr_0=rois)
        print('file saved.')
                

    def __getitem__(self, idx):
        ct_name = self.tr[idx]
        # load image and label
        npz = np.load(os.path.join(self.root_dir, 'npz', ct_name + '.npz'))
        input_spacing = npz['spacing']
        npz = np.load(os.path.join(self.root_dir, 'clean_labels', ct_name + '.npz'))
        lbl_np = npz['arr_0'].astype('uint8')
        label_ids = np.unique(lbl_np)
        label_ids = label_ids[label_ids != 0]
        label_id = np.random.choice(label_ids)
        lbl_np = lbl_np == label_id
        z_inds, y_inds, x_inds = np.where(lbl_np > 0)
        zmin, zmax = z_inds.min(), z_inds.max()
        ymin, ymax = y_inds.min(), y_inds.max()
        xmin, xmax = x_inds.min(), x_inds.max()
        lbl_crop = lbl_np[zmin:zmax + 1, ymin:ymax + 1, xmin:xmax + 1]
        scale = input_spacing / np.array(self.spacing)
        lbl_np = zoom(lbl_crop, scale, order=0)        
        scale = self.roi_size / np.array(lbl_np.shape)
        scale = np.min(scale) * (1 - self.margin_ratio)
        lbl_np = zoom(lbl_np, scale, order=0)
        d, h, w = lbl_np.shape
        pad_z0 = (self.roi_size[0] - d) // 2
        pad_z1 = self.roi_size[0] - d - pad_z0
        pad_y0 = (self.roi_size[1] - h) // 2
        pad_y1 = self.roi_size[1] - h - pad_y0
        pad_x0 = (self.roi_size[2] - w) // 2
        pad_x1 = self.roi_size[2] - w - pad_x0
        lbl_np = np.pad(lbl_np, ((pad_z0, pad_z1), (pad_y0, pad_y1), (pad_x0, pad_x1)), 'constant', constant_values=0)

        if self.use_sdf:
            lbl_np = compute_sdf(lbl_np)
            # lbl_np = np.clip(lbl_np, -5, 5) / 5.0
        else:
            lbl_np = lbl_np * 2.0 - 1.0
        assert lbl_np.shape == tuple(self.roi_size), f'{lbl_np.shape} != {tuple(self.roi_size)}'
        mask_tensor = torch.from_numpy(lbl_np).float().unsqueeze(0).permute(0, -1, -2, -3)
        all_one_tensor = torch.ones_like(mask_tensor)

        # visualize and save image
        if self.visualize:
            os.makedirs(f'visualize/{ct_name}/{label_id}', exist_ok=True)
            for i in range(lbl_np.shape[0]):
                plt.figure()
                plt.imshow(lbl_np[i], cmap='gray')
                plt.show()
                plt.savefig(f'visualize/{ct_name}/{label_id}/{ct_name}_{label_id}_{i}.png')
                plt.close()

        return mask_tensor, all_one_tensor
    

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
    dataset = prlnmask_dataset(cfg)
    train_loader = DataLoader(dataset, batch_size=cfg['batch_size'], shuffle=True, num_workers=cfg['num_workers'],
                              collate_fn=data_collate)
    return train_loader



def save_3d_gif(voxel, gif_name):
    angles = np.linspace(0, 360, 36)
    images = []
    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(111, projection='3d')
    filled = np.argwhere(voxel)
    ax.scatter(filled[:, 0], filled[:, 1], filled[:, 2], color='gray', edgecolors='k', alpha=0.9)
    
    # 设置坐标轴范围
    ax.set_xlim(0, voxel.shape[0])
    ax.set_ylim(0, voxel.shape[1])
    ax.set_zlim(0, voxel.shape[2])
    ax.axis('off')
    
    for angle in angles:
        ax.view_init(elev=20., azim=angle)
        # Save the current figure to a bytes buffer
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0.1)
        buf.seek(0)
        images.append(imageio.imread(buf))
        buf.close()
        
    plt.close(fig)
    # Create GIF
    imageio.mimsave(gif_name, images, duration=0.1)
    print(f"GIF generated: {gif_name}")


if __name__ == '__main__':
    cfg = read_yaml_config('config/dataset/data_config_prlnmask.yaml')
    print(cfg)
    ds = prlnmask_dataset(cfg)
    ds.__statistic__()

    # dl = get_loader(cfg)
    # for data in dl:
    #     image = data['image']
    #     label = data['label']
    #     print(image.shape, label.shape)

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


class meply_dataset(Dataset):
    """
    ***请注意***
    3D影像的格式: D x H x W
    候选框的格式: z, x, y, d, h, w
    """

    def __init__(self, cfg):
        super(meply_dataset, self).__init__()
        self.roi_size = [cfg['roi_z'], cfg['roi_y'], cfg['roi_x']]
        self.spacing = [cfg['space_z'], cfg['space_y'], cfg['space_x']]
        self.root_dir = cfg['root_dir']
        self.amin = cfg['a_min']
        self.amax = cfg['a_max']
        self.bmin = cfg['b_min']
        self.bmax = cfg['b_max']
        self.num_samples = cfg['num_samples']
        self.list_dir = os.path.join(self.root_dir, 'clean_label_info_spacing.csv')
        self.img_dir = os.path.join(self.root_dir, 'imagesTr')
        self.lbl_dir = os.path.join(self.root_dir, 'clean_labelsTr')

        self.check_data_integrity = cfg['check_data_integrity']
        self.visualize = cfg['visualize']
        self.__load_data__()

    def __load_data__(self):
        # load label info list
        with open(self.list_dir, 'r') as f:
            lines = f.readlines()
        caselist = [line.strip().replace(' ', '').split(',') for line in lines]
        self.caselist = caselist

        if self.check_data_integrity:
            print('check data integrity...')
            check_total = {}
            for case in caselist:
                ct_name = case[0]
                if ct_name not in check_total:
                    check_total[ct_name] = 1
                else:
                    check_total[ct_name] += 1

            for ct_name in tqdm(check_total.keys()):
                label_name = os.path.join(self.lbl_dir, ct_name + '.nii.gz')
                label = sitk.ReadImage(label_name)
                label_np = sitk.GetArrayFromImage(label)

                image_name = os.path.join(self.img_dir, ct_name + '_0000.nii.gz')
                image = sitk.ReadImage(image_name)
                image_np = sitk.GetArrayFromImage(image)

                label_target_num = len(np.unique(label_np)) - 1

                if image_np.shape != label_np.shape:
                    print('shape unmatched:', ct_name, image_np.shape, label_np.shape)
                if label_target_num != check_total[ct_name]:
                    print('index unmatched:', label_name, label_target_num, check_total[ct_name])
            print('data check done.')

    def __len__(self):
        return len(self.caselist)

    def __getitem__(self, idx):
        case = self.caselist[idx]
        ct_name, label_id, vol, z1, y1, x1, z2, y2, x2 = case
        z1, y1, x1, z2, y2, x2 = int(z1), int(y1), int(x1), int(z2), int(y2), int(x2)
        label_id = int(label_id)
        vol = int(vol)
        z, y, x = (z1 + z2) / 2, (y1 + y2) / 2, (x1 + x2) / 2
        d, h, w = z2 - z1, y2 - y1, x2 - x1
        orig = np.array([z, y, x, d, h, w])

        # load image and label
        npz = np.load(os.path.join(self.root_dir, 'npz', ct_name + '.npz'))
        img_np = npz['img']
        lbl_np = npz['lbl']
        # img_name = os.path.join(self.img_dir, ct_name + '_0000.nii.gz')
        # lbl_name = os.path.join(self.lbl_dir, ct_name + '.nii.gz')
        # img = sitk.ReadImage(img_name)
        # lbl = sitk.ReadImage(lbl_name)
        # img_np = sitk.GetArrayFromImage(img)
        # lbl_np = sitk.GetArrayFromImage(lbl).astype(int)
        # spacing = img.GetSpacing()[::-1]
        # img_np = zoom(img_np, spacing, order=1)
        # lbl_np = zoom(lbl_np, spacing, order=0)

        # random crop by target
        img_np, lbl_np = RandCropByTarget(img_np, lbl_np, self.roi_size, orig, training=True)
        # clip & normalize
        img_np = np.clip(img_np, self.amin, self.amax)
        img_np = (img_np - self.amin) / (self.amax - self.amin) * (self.bmax - self.bmin) + self.bmin
        lbl_np[lbl_np != label_id] = 0

        # visualize and save image
        if self.visualize:
            os.makedirs(f'visualize/{ct_name}/{label_id}', exist_ok=True)
            for i in range(img_np.shape[0]):
                plt.figure()
                plt.subplot(121)
                plt.imshow(img_np[i], cmap='gray')
                plt.subplot(122)
                plt.imshow(lbl_np[i], cmap='gray')
                plt.show()
                plt.savefig(f'visualize/{ct_name}/{label_id}/{ct_name}_{label_id}_{i}.png')
                plt.close()

        # convert to float tensor
        img = torch.tensor(img_np).float().unsqueeze(0)
        lbl = torch.tensor(lbl_np).float().unsqueeze(0)

        return img, lbl


def RandCropByTarget(img_np, lbl_np, roi, orig, training=True):
    # pad if nessesary
    if img_np.shape[0] < roi[0] or img_np.shape[1] < roi[1] or img_np.shape[2] < roi[2]:
        pad = np.maximum(np.array(roi) - img_np.shape, 0)
        pad = np.array([[0, pad[0]], [0, pad[1]], [0, pad[2]]])
        img_np = np.pad(img_np, pad, 'constant', constant_values=0)
        lbl_np = np.pad(lbl_np, pad, 'constant', constant_values=0)

    margin = 2
    margin = np.array([margin, margin, margin])
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

    batch_dict = {
        'image': torch.stack(imgs),
        'label': torch.stack(lbls)
    }
    return batch_dict


def get_loader(cfg):
    dataset = meply_dataset(cfg)
    train_loader = DataLoader(dataset, batch_size=cfg['batch_size'], shuffle=True, num_workers=cfg['num_workers'],
                              collate_fn=data_collate)
    return train_loader


if __name__ == '__main__':
    cfg = read_yaml_config('config/dataset/data_config_meply.yaml')
    print(cfg)
    dl = get_loader(cfg)
    for data in dl:
        print(data['image'].shape)
        pass

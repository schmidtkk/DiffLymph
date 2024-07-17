import copy
import json
import cv2

from vq_gan_3d.model import VQGAN
import torch
import random
import SimpleITK as sitk
from scipy.ndimage import zoom, gaussian_filter
import numpy as np
import matplotlib.pyplot as plt
import os
from omegaconf import DictConfig
from ddpm import Unet3D, GaussianDiffusion, Tester
import hydra
from dataset.lidc_dataloader import lidc_dataset
from glob import glob
import math
import yaml


def read_yaml_config(config_file):
    with open(config_file, 'r') as file:
        try:
            return yaml.safe_load(file)
        except yaml.YAMLError as e:
            print(e)
            return None


def select_mask(root_dir, caselist):
    case = random.choice(caselist)
    ct_name, label_id, _, z1, y1, x1, z2, y2, x2 = case
    z1, y1, x1, z2, y2, x2 = int(z1), int(y1), int(x1), int(z2) + 1, int(y2) + 1, int(x2) + 1
    label_id = int(label_id)
    npz = np.load(f'{root_dir}/npz/{ct_name}.npz')
    lbl_np = npz['lbl']
    lbl_crop = lbl_np[z1:z2, y1:y2, x1:x2]
    lbl_crop = lbl_crop == label_id

    return lbl_crop


# def visualize(img, mask, origin=None, hist=None, dir='visualize'):
#     os.makedirs(dir, exist_ok=True)
#     if len(img.shape) == 3:
#         for z in range(img.shape[-1]):
#             plt.figure()
#             plt.subplot(121)
#             plt.imshow(img[:, :, z], cmap='gray')
#             plt.subplot(122)
#             plt.imshow(mask[:, :, z], cmap='gray')
#             plt.show()
#             plt.savefig(f'{dir}/{z}.png')
#             plt.close()
#     elif len(img.shape) == 4:
#         assert len(mask.shape) == 3
#         assert hist is not None
#         assert origin is not None
#         batch = img.shape[0]
#         depth = img.shape[-1]
#         row = 1 if hist is None else 2
#         for z in range(depth):
#             plt.figure()
#             # plot volume
#             for b in range(batch):
#                 plt.subplot(row, batch+2, b+1)
#                 plt.imshow(img[b, :, :, z], cmap='gray')
#             # plot origin
#                 plt.subplot(row, batch+2, batch+1)
#                 plt.imshow(origin[:, :, z], cmap='gray')
#             # plot mask
#             plt.subplot(row, batch+2, batch+2)
#             plt.imshow(mask[:, :, z], cmap='gray')
#             # plot histgram
#             if hist is not None:
#                 bins = np.arange(1, hist.shape[1]+1)
#                 for b in range(batch):
#                     plt.subplot(row, batch+2, batch+b+3)
#                     plt.bar(bins, hist[b], width=1, edgecolor='black')
#             plt.show()
#             plt.savefig(f'{dir}/{z}.png')
#             plt.close()
#     else:
#         raise NotImplementedError


def visualize(img, mask, origin=None, hist=None, dir='visualize'):
    os.makedirs(dir, exist_ok=True)
    if len(img.shape) == 3:
        for z in range(img.shape[-1]):
            plt.figure()
            plt.subplot(121)
            plt.imshow(img[:, :, z], cmap='gray')
            plt.subplot(122)
            plt.imshow(mask[:, :, z], cmap='gray')
            plt.show()
            plt.savefig(f'{dir}/{z}.png')
            plt.close()
    elif len(img.shape) == 4:
        assert len(mask.shape) == 3
        assert hist is not None
        assert origin is not None
        batch = img.shape[0]
        depth = img.shape[-1]
        row = 1 if hist is None else 2
        for z in range(depth):
            fig = plt.figure(figsize=(15, 4 * row))  # 调整图像的宽度和高度
            gs = fig.add_gridspec(row, batch + 2, height_ratios=[1, 0.3] if hist is not None else [1])
            # plot volume
            for b in range(batch):
                ax = fig.add_subplot(gs[0, b])
                ax.imshow(img[b, :, :, z], cmap='gray')
                ax.set_title(f'Volume {b+1}')
            # plot origin
            ax = fig.add_subplot(gs[0, batch])
            ax.imshow(origin[:, :, z], cmap='gray')
            ax.set_title('Origin')
            # plot mask
            ax = fig.add_subplot(gs[0, batch + 1])
            ax.imshow(mask[:, :, z], cmap='gray')
            ax.set_title('Mask')
            # plot histogram
            if hist is not None:
                bins = np.arange(1, hist.shape[1] + 1)
                for b in range(batch):
                    ax = fig.add_subplot(gs[1, b])
                    ax.bar(bins, hist[b], width=1, edgecolor='black')
                    ax.set_title(f'Histogram {b+1}')
            plt.subplots_adjust(top=0.95, hspace=0.05)  # 调整顶部留白和子图之间的间距
            plt.show()
            plt.savefig(f'{dir}/{z}.png')
            plt.close()
    else:
        raise NotImplementedError


@hydra.main(config_path='config', config_name='base_cfg', version_base=None)
def run(cfg: DictConfig):
    torch.cuda.set_device(7)
    if cfg.model.denoising_fn == 'Unet3D':
        model = Unet3D(
            dim=cfg.model.diffusion_img_size,
            dim_mults=cfg.model.dim_mults,
            channels=cfg.model.diffusion_num_channels,
            out_dim=cfg.model.out_dim,
            cond_dim=cfg['dataset'].get('cond_dim')
        ).cuda()
    else:
        raise ValueError(f"Model {cfg.model.denoising_fn} doesn't exist")

    diffusion = GaussianDiffusion(
        model,
        vqgan_ckpt=cfg.model.vqgan_ckpt,
        image_size=cfg.model.diffusion_img_size,
        num_frames=cfg.model.diffusion_depth_size,
        channels=cfg.model.diffusion_num_channels,
        timesteps=cfg.model.timesteps,
        loss_type=cfg.model.loss_type,
    ).cuda()

    tester = Tester(diffusion)
    diffusion_ckpt = 'weight/ddpm_lidc_hist.pt'
    tester.load(diffusion_ckpt, map_location='cpu', strict=False)

    vqgan_ckpt = 'pretrained_models/AutoencoderModel.ckpt'
    vqgan = VQGAN.load_from_checkpoint(vqgan_ckpt).cuda()
    vqgan.eval()

    ds_cfg = read_yaml_config('config/dataset/data_config_lidc.yaml')
    ds = lidc_dataset(ds_cfg)
    ds_len = len(ds)

    clean = ('/mnt/889cdd89-1094-48ae-b221-146ffe543605/gwd/lefusion/data_conversion/'
             'LIDC-IDRI-Preprocessing-master/3d_data/Clean/Image/*/*.nii.gz')
    clean_crop = glob(clean)

    with open('hist_clusters/clusters.json', 'r') as f:
        clusters = json.load(f)
    cluster_centers = clusters[0]['centers']

    loop_num = 16
    hu_min = cfg.dataset.a_min
    hu_max = cfg.dataset.a_max

    for t in range(loop_num):
        idx = random.choice(range(ds_len))
        img, mask, _ = ds[idx]
        batch = len(cluster_centers) + 2
        hist = torch.tensor(cluster_centers)
        maxh = torch.zeros((1, 16))
        maxh[0, -1] = 1.0
        minh = torch.zeros((1, 16))
        minh[0, 0] = 1.0
        hist = torch.cat((minh, hist, maxh), dim=0).cuda()

        bg_img_file = random.choice(clean_crop)
        bg_img = sitk.ReadImage(bg_img_file)
        volume = sitk.GetArrayFromImage(bg_img)
        volume = np.clip(volume, hu_min, hu_max)
        volume = (volume - hu_min) / (hu_max - hu_min) * 2.0 - 1.0

        origin = copy.deepcopy(volume)
        mask_ = mask[0].clone().cpu().numpy()

        volume = torch.from_numpy(volume).unsqueeze(0).unsqueeze(0).expand(batch, -1, -1, -1, -1).cuda()
        mask = mask.unsqueeze(0).expand(batch, -1, -1, -1, -1).cuda()

        masked_volume = volume * (1 - mask)
        mask = mask * 2.0 - 1.0

        volume = volume.permute(0, 1, -1, -3, -2)
        masked_volume = masked_volume.permute(0, 1, -1, -3, -2)
        mask = mask.permute(0, 1, -1, -3, -2)

        # vqgan encoder inference
        masked_volume_feat = vqgan.encode(masked_volume, quantize=False, include_embeddings=True)
        masked_volume_feat = ((masked_volume_feat - vqgan.codebook.embeddings.min()) /
                              (vqgan.codebook.embeddings.max() - vqgan.codebook.embeddings.min())) * 2.0 - 1.0

        cc = torch.nn.functional.interpolate(mask, size=masked_volume_feat.shape[-3:])
        cond = torch.cat((masked_volume_feat, cc), dim=1)
        cond = (cond, hist)

        # diffusion inference and decoder
        tester.ema_model.eval()
        sample = tester.ema_model.sample(batch_size=volume.shape[0], cond=cond)

        # post-process
        mask_01 = torch.clamp((mask + 1.0) / 2.0, min=0.0, max=1.0)
        sigma = np.random.uniform(0, 4)
        mask_01_np_blur = gaussian_filter(mask_01.cpu().numpy().astype(float), sigma=[0, 0, sigma, sigma, sigma])

        volume_ = torch.clamp((volume + 1.0) / 2.0, min=0.0, max=1.0)
        sample_ = torch.clamp((sample + 1.0) / 2.0, min=0.0, max=1.0)

        mask_01_blur = torch.from_numpy(mask_01_np_blur).cuda()
        final_volume_ = (1 - mask_01_blur) * volume_ + mask_01_blur * sample_
        final_volume_ = final_volume_.permute(0, 1, -2, -1, -3)
        final_volume_ = torch.clamp(final_volume_, min=0.0, max=1.0).squeeze(1).cpu().numpy()
        hist = hist.cpu().numpy()
        visualize(final_volume_, mask_, origin, hist=hist, dir=f'visualize/{t}/generated')


if __name__ == '__main__':
    run()

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

ROI_SIZE = [32, 32, 32]


def visualize(img, mask, dir='visualize'):
    os.makedirs(dir, exist_ok=True)
    for z in range(img.shape[-1]):
        plt.figure()
        plt.subplot(121)
        plt.imshow(img[:, :, z], cmap='jet')
        plt.colorbar()
        plt.subplot(122)
        plt.imshow(mask[:, :, z], cmap='gray')
        plt.show()
        plt.savefig(f'{dir}/{z}.png')
        plt.close()


@hydra.main(config_path='config', config_name='base_cfg', version_base=None)
def run(cfg: DictConfig):
    torch.cuda.set_device(7)
    if cfg.model.denoising_fn == 'Unet3D':
        model = Unet3D(
            dim=cfg.model.diffusion_img_size,
            dim_mults=cfg.model.dim_mults,
            channels=cfg.model.diffusion_num_channels,
            out_dim=cfg.model.out_dim
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
    diffusion_ckpt = 'weight/ddpm_prlnmask_trick.pt'
    tester.load(diffusion_ckpt, map_location='cpu')

    vqgan_ckpt = 'pretrained_models/AutoencoderModel.ckpt'
    vqgan = VQGAN.load_from_checkpoint(vqgan_ckpt).cuda()
    vqgan.eval()

    root_dir = '/mnt/889cdd89-1094-48ae-b221-146ffe543605/gwd/reln'
    with open(f'{root_dir}/clean_label_info_spacing.csv', 'r') as f:
        lines = f.readlines()
    caselist = [line.strip().replace(' ', '').split(',') for line in lines]

    for i in range(10):
        volume = np.zeros(ROI_SIZE)
        mask = np.ones(ROI_SIZE)
        mask_ = mask.copy()

        visualize(volume, mask, f'visualize/{i}/selected')

        volume = torch.from_numpy(volume).cuda().unsqueeze(0).unsqueeze(0).float()
        mask = torch.from_numpy(mask).cuda().unsqueeze(0).unsqueeze(0).float()
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
        final_volume_ = torch.clamp(final_volume_, min=0.0, max=1.0).cpu().numpy().squeeze()
        visualize(final_volume_, mask_, dir=f'visualize/{i}/generated')


if __name__ == '__main__':
    run()

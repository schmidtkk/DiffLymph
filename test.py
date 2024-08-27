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
import itertools
from scipy.ndimage import binary_fill_holes



ROI_SIZE = [64, 64, 64]
SPACING = np.array([0.7, 0.7, 1.25])

def select_context(root_dir, mask, caselist):
    status, img_crop, mask_crop = select_context_loop(root_dir, mask, caselist)
    while not status:
        status, img_crop, mask_crop = select_context_loop(root_dir, mask, caselist)
    return img_crop, mask_crop

def select_context_loop(root_dir, mask, caselist):
    case = random.choice(caselist)
    ct_name = case[0]
    npz = np.load(f'{root_dir}/npz/{ct_name}.npz')
    img = npz['img']
    input_spacing = npz['spacing']
    npz = np.load(f'{root_dir}/clean_labels/{ct_name}.npz')
    lbl = npz['arr_0']
    scale = input_spacing / SPACING
    img = zoom(img, scale, order=1)
    lbl = zoom(lbl, scale, order=0)

    if img.shape[0] < ROI_SIZE[0]:
        return False, None, None
    
    crop_size = mask.shape
    img_size = img.shape
    hu_min = -175
    hu_max = 250

    # sliding window search
    crops = [t for t in itertools.product(range(0, img_size[0]-ROI_SIZE[0], ROI_SIZE[0]//2), 
                                          range(0, img_size[1]-ROI_SIZE[1], ROI_SIZE[1]//2),
                                          range(0, img_size[2]-ROI_SIZE[2], ROI_SIZE[2]//2))]
    
    random.shuffle(crops)
    
    mask_crop = np.zeros(ROI_SIZE)
    zz1 = ROI_SIZE[0] // 2 - crop_size[0] // 2
    yy1 = ROI_SIZE[1] // 2 - crop_size[1] // 2
    xx1 = ROI_SIZE[2] // 2 - crop_size[2] // 2
    zz2 = zz1 + crop_size[0]
    yy2 = yy1 + crop_size[1]
    xx2 = xx1 + crop_size[2]
    mask_crop[zz1:zz2, yy1:yy2, xx1:xx2] = mask
    kernel = np.ones((3, 3), np.uint8)
    mask_dilated = cv2.dilate(mask_crop, kernel, iterations=10)

    
    for z1, y1, x1 in crops:
        z2 = z1 + ROI_SIZE[0]
        y2 = y1 + ROI_SIZE[1]
        x2 = x1 + ROI_SIZE[2]

        img_crop = img[z1:z2, y1:y2, x1:x2]
        img_crop = np.clip(img_crop, hu_min, hu_max)
        img_crop = (img_crop - hu_min) / (hu_max - hu_min) * 2.0 - 1.0

        if np.mean(img_crop) >= 0 and np.sum(img_crop * mask_dilated) / np.sum(mask_dilated) <= -0.5:
            return True, img_crop, mask_crop

    return False, None, None
    

def select_mask(root_dir, caselist):
    case = random.choice(caselist)
    ct_name, label_id, _, z1, y1, x1, z2, y2, x2 = case
    z1, y1, x1, z2, y2, x2 = int(z1), int(y1), int(x1), int(z2)+1, int(y2)+1, int(x2)+1
    label_id = int(label_id)

    npz = np.load(f'{root_dir}/npz/{ct_name}.npz')
    input_spacing = npz['spacing']
    scale = input_spacing / SPACING * 2
    
    npz = np.load(f'{root_dir}/clean_labels/{ct_name}.npz')
    lbl_np = npz['arr_0'].astype('uint8')
    lbl_crop = lbl_np[z1:z2, y1:y2, x1:x2]
    lbl_crop = lbl_crop == label_id

    lbl_crop = binary_fill_holes(lbl_crop.astype(bool)).astype(int)

    lbl_crop = zoom(lbl_crop, scale, order=0)

    return lbl_crop


def visualize(img, mask, dir='visualize'):
    os.makedirs(dir, exist_ok=True)
    for z in range(img.shape[0]):
        plt.figure()
        plt.subplot(121)
        plt.imshow(img[z], cmap='gray')
        plt.subplot(122)
        plt.imshow(mask[z], cmap='gray')
        plt.show()
        plt.savefig(f'{dir}/{z}.png')
        plt.close()


@hydra.main(config_path='config', config_name='base_cfg', version_base=None)
def run(cfg: DictConfig):
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
    diffusion_ckpt = 'weight/model_best.pt'
    tester.load(diffusion_ckpt, map_location='cpu')

    vqgan_ckpt = 'pretrained_models/AutoencoderModel.ckpt'
    vqgan = VQGAN.load_from_checkpoint(vqgan_ckpt).cuda()
    vqgan.eval()

    root_dir = cfg.dataset.root_dir
    with open(f'{root_dir}/PRLN/prln_target.csv', 'r') as f:
        lines = f.readlines()
    caselist = [line.strip().replace(' ', '').split(',') for line in lines]

    for i in range(10):
        mask = select_mask(root_dir, caselist)
        volume, mask = select_context(root_dir, mask, caselist)
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
        mask_01 = torch.clamp((mask+1.0)/2.0, min=0.0, max=1.0)
        # sigma = np.random.uniform(0, 4)
        sigma = 1.0
        mask_01_np_blur = gaussian_filter(mask_01.cpu().numpy().astype(float), sigma=[0,0,sigma,sigma,sigma])

        volume_ = torch.clamp((volume+1.0)/2.0, min=0.0, max=1.0)
        sample_ = torch.clamp((sample+1.0)/2.0, min=0.0, max=1.0)

        mask_01_blur = torch.from_numpy(mask_01_np_blur).cuda()
        final_volume_ = (1-mask_01_blur)*volume_+mask_01_blur*sample_
        final_volume_ = final_volume_.permute(0, 1, -2, -1, -3)
        final_volume_ = torch.clamp(final_volume_, min=0.0, max=1.0).cpu().numpy().squeeze()
        visualize(final_volume_, mask_, dir=f'visualize/{i}/generated')


if __name__ == '__main__':
    torch.cuda.set_device(7)
    run()

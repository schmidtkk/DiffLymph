import copy
import json
import os
import time
import hydra
import numpy as np
import random
from scipy.ndimage import zoom, binary_fill_holes, gaussian_filter, median_filter, binary_dilation
from scipy.spatial.distance import pdist
import itertools
import cv2
import matplotlib.pyplot as plt
import torch

from ddpm.diffusion import GaussianDiffusion, Tester, Unet3D
from vq_gan_3d.model.vqgan import VQGAN
import SimpleITK as sitk
from glob import glob
from skimage.measure import marching_cubes
from tqdm import tqdm

# ignore warnings
import warnings
warnings.filterwarnings("ignore")


ROI_SIZE = [64, 64, 64]
SPACING = np.array([1.25, 0.7, 0.7])

def select_real_mask(root_dir, caselist):
    case = random.choice(caselist)
    ct_name, label_id, _, z1, y1, x1, z2, y2, x2 = case
    z1, y1, x1, z2, y2, x2 = int(z1), int(y1), int(x1), int(z2)+1, int(y2)+1, int(x2)+1
    label_id = int(label_id)

    npz = np.load(f'{root_dir}/npz/{ct_name}.npz')
    input_spacing = npz['spacing']
    
    npz = np.load(f'{root_dir}/clean_labels/{ct_name}.npz')
    lbl_np = npz['arr_0'].astype('float32')
    lbl_crop = lbl_np[z1:z2, y1:y2, x1:x2]
    lbl_crop[lbl_crop != label_id] = 0
    lbl_crop[lbl_crop == label_id] = 1

    mask_ = torch.from_numpy(lbl_crop).cuda().unsqueeze(0).unsqueeze(0).float()
    non_zero_points_indices = torch.nonzero(mask_ > 0.5, as_tuple=True)
    non_zero_points = torch.stack(non_zero_points_indices, dim=-1)
    expanded_points = non_zero_points.unsqueeze(1)
    squared_diff = (expanded_points - non_zero_points.unsqueeze(0)) ** 2
    distances = squared_diff.sum(dim=-1)
    max_dist = torch.sqrt(distances.max()).item()
    torch.cuda.empty_cache()

    REAL_DIST_MIN = 1.7
    REAL_DIST_MAX = 30.0
    target_dist = random.uniform(REAL_DIST_MIN, REAL_DIST_MAX)
    scale = input_spacing / SPACING * target_dist / max_dist

    # scale = input_spacing / SPACING
    lbl_crop = zoom(lbl_crop, scale, order=1)
    lbl_crop = gaussian_filter(lbl_crop, sigma=1.0)
    lbl_crop = binary_fill_holes(lbl_crop > 0.5)

    return lbl_crop

def select_fake_mask(fake_mask_list):
    # fake_mask_list: use glob to get all fake mask cases
    mask_case = random.choice(fake_mask_list)
    mask = np.load(mask_case).transpose(2, 1, 0)

    mask_ = torch.from_numpy(mask).cuda().unsqueeze(0).unsqueeze(0).float()
    mask_ = torch.nn.functional.interpolate(mask_, size=(32, 32, 32), mode='trilinear').squeeze()
    non_zero_points_indices = torch.nonzero(mask_ > 0.5, as_tuple=True)
    non_zero_points = torch.stack(non_zero_points_indices, dim=-1)
    expanded_points = non_zero_points.unsqueeze(1)
    squared_diff = (expanded_points - non_zero_points.unsqueeze(0)) ** 2
    distances = squared_diff.sum(dim=-1)
    max_dist = torch.sqrt(distances.max()).item() * 2.0
    torch.cuda.empty_cache()

    REAL_DIST_MIN = 1.7
    REAL_DIST_MAX = 30.0
    # REAL_DIST_MIN = 20.0
    # REAL_DIST_MAX = 30.0
    target_dist = random.uniform(REAL_DIST_MIN, REAL_DIST_MAX)
    scale = target_dist / max_dist

    mask = zoom(mask.astype('float32'), scale, order=1)
    mask = (mask > 0.5).astype('uint8')

    z_inds, y_inds, x_inds = np.where(mask > 0)
    if len(z_inds) == 0:
        return None
    z1, z2 = z_inds.min(), z_inds.max() + 1
    y1, y2 = y_inds.min(), y_inds.max() + 1
    x1, x2 = x_inds.min(), x_inds.max() + 1
    mask = mask[z1:z2, y1:y2, x1:x2]

    return mask


def anchor_mask_to_img(img, lbl, mask):
    crop_size = mask.shape
    img_size = img.shape

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
    mask_dilated = cv2.dilate(copy.deepcopy(mask_crop), kernel, iterations=10)

    
    for z1, y1, x1 in crops:
        z2 = z1 + ROI_SIZE[0]
        y2 = y1 + ROI_SIZE[1]
        x2 = x1 + ROI_SIZE[2]

        img_crop = img[z1:z2, y1:y2, x1:x2]
        # img_crop = np.clip(img_crop, hu_min, hu_max)
        # img_crop = (img_crop - hu_min) / (hu_max - hu_min) * 2.0 - 1.0

        lbl_crop = lbl[z1:z2, y1:y2, x1:x2]

        # avoid overlap with real mask
        if np.sum(mask_dilated * lbl_crop) > 0:
            continue

        # roi should be empty
        intensity = np.sum(img_crop * mask_dilated) / np.sum(mask_dilated)
        variance = np.var(img_crop[mask_dilated > 0])

        if intensity <= -0.6 and intensity >= -0.8 and variance <= 0.03:
            lbl[z1+zz1: z1+zz2, y1+yy1: y1+yy2, x1+xx1: x1+xx2] += np.uint8(mask * 2)
            shape = [np.sum(mask), z1+zz1, y1+yy1, x1+xx1, z1+zz2-1, y1+yy2-1, x1+xx2-1]
            pos = [z1, y1, x1]

            return True, img_crop, mask_crop, shape, pos

    return False, None, None, None, None



def prepare_volume(root_dir, img_list, mask_list, k, t):
    img_name = random.choice(img_list)

    npz = np.load(f'{root_dir}/npz/{img_name}.npz')
    img = npz['img']
    input_spacing = npz['spacing']
    npz = np.load(f'{root_dir}/clean_labels/{img_name}.npz')
    lbl = npz['arr_0'].astype('float32')

    label_base = int(lbl.max())

    MARGIN = 20
    _, y_inds, x_inds = np.where(lbl > 0)
    y1, y2 = y_inds.min(), y_inds.max() + 1
    x1, x2 = x_inds.min(), x_inds.max() + 1
    y1 = max(0, y1 - MARGIN)
    x1 = max(0, x1 - MARGIN)
    y2 = min(lbl.shape[1], y2 + MARGIN)
    x2 = min(lbl.shape[2], x2 + MARGIN)
    lbl = lbl[:, y1:y2, x1:x2]
    lbl[lbl > 0] = 1
    img = img[:, y1:y2, x1:x2]
    scale = input_spacing / SPACING * 2
    img = zoom(img, scale, order=1)
    hu_min = -175
    hu_max = 250
    img = np.clip(img, hu_min, hu_max)
    img = (img - hu_min) / (hu_max - hu_min) * 2.0 - 1.0
    
    lbl = zoom(lbl, scale, order=1)
    lbl = gaussian_filter(lbl, sigma=1)
    lbl = binary_fill_holes(lbl > 0.5).astype('uint8')
    pre_lbl = copy.deepcopy(lbl)

    img_crops = []
    mask_crops = []
    gen_info_list = []
    roi_pos = []
    success = 0
    fail = 0
    MAX_FAIL = 100

    while success < k and fail < MAX_FAIL:
        # mask = select_real_mask(root_dir, mask_list)
        mask = select_fake_mask(mask_list)
        while mask is None:
            mask = select_fake_mask(mask_list)
        
        # tip: anchor_mask_to_img will add new samples to lbl
        status, img_crop, mask_crop, shape, pos = anchor_mask_to_img(img, lbl, mask)
        if status:
            success += 1
            label_base += 1
            img_crops.append(img_crop)
            mask_crops.append(mask_crop)
            roi_pos.append(pos)
            info = [img_name+f'G{t:03d}', label_base] + shape
            info_str = ', '.join([str(x) for x in info]) + '\n'
            gen_info_list.append(info_str)
        else:
            fail += 1
        print('success:', success, 'fail:', fail)


    if success == 0:
        print('fail to find enough samples')
        return None, None, None, None, None, None, None
    
    img_crops = np.stack(img_crops)
    mask_crops = np.stack(mask_crops)
    roi_pos = np.array(roi_pos)

    print(img_crops.shape, mask_crops.shape)
    return img_crops, mask_crops, gen_info_list, img, lbl, roi_pos, pre_lbl


def synthesize_sample(vqgan, tester, imgs, masks):
    # split imgs and masks into batches
    BATCH_SIZE = 5
    TOTAL = imgs.shape[0]
    sample_masks = []
    sample_blur_masks = []
    sample_textures = []

    structure = np.ones((1, 3, 3, 3), dtype=int)

    for i in range(0, TOTAL, BATCH_SIZE):
        bs = BATCH_SIZE if i + BATCH_SIZE < TOTAL else TOTAL - i
        volume = torch.from_numpy(imgs[i:i+bs]).cuda().unsqueeze(1).float()
        mask_ = copy.deepcopy(masks[i:i+bs])
        mask = binary_dilation(masks[i:i+bs], structure=structure, iterations=5)
        mask = torch.from_numpy(masks[i:i+bs]).cuda().unsqueeze(1).float()  
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
        
        sample_ = torch.clamp(sample, min=-1.0, max=1.0).permute(0, 1, -2, -1, -3).squeeze(1).cpu().numpy()
        sample_ = median_filter(sample_, size=(1,3,3,3))
        sigma = 2.0
        mask_01_blur = gaussian_filter(mask_.astype('float32'), sigma=[0,sigma,sigma,sigma])

        sample_masks.append(mask_)
        sample_blur_masks.append(mask_01_blur)
        sample_textures.append(sample_)
    
    sample_masks = np.concatenate(sample_masks, axis=0)
    sample_blur_masks = np.concatenate(sample_blur_masks, axis=0)
    sample_textures = np.concatenate(sample_textures, axis=0)
    return sample_textures, sample_masks, sample_blur_masks
 

def synthesize_volume(img, lbl, pos, textures, masks, blur_masks):
    assert len(textures) == len(masks) and len(textures) == len(pos)

    D, H, W = textures[0].shape
    for i in range(len(textures)):
        z1, y1, x1 = pos[i]
        t = textures[i]
        m = masks[i]
        bm = blur_masks[i]
        img[z1:z1+D, y1:y1+H, x1:x1+W] = (1-bm) * img[z1:z1+D, y1:y1+H, x1:x1+W] + bm * t
        lbl[z1:z1+D, y1:y1+H, x1:x1+W][m > 0] = 3
        
    return img, lbl


def build_model(cfg):
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
    diffusion_ckpt = 'weight/ddpm_zoom.pt'
    tester.load(diffusion_ckpt, map_location='cpu')

    vqgan_ckpt = 'pretrained_models/AutoencoderModel.ckpt'
    vqgan = VQGAN.load_from_checkpoint(vqgan_ckpt).cuda()
    vqgan.eval()
    
    return vqgan, tester

@hydra.main(config_path='config', config_name='base_cfg', version_base=None)
def main(cfg):
    torch.cuda.set_device(7)
    root_dir = cfg.dataset.root_dir

    list_dir = os.path.join(root_dir, 'PRLN', 'downstream.json')
    with open(list_dir, 'r') as f:
        target = json.load(f)
    tr = target['tr']
    img_list = tr

    # ## use real masks
    # with open(f'{root_dir}/PRLN/prln_target.csv', 'r') as f:
    #     lines = f.readlines()
    # caselist = [line.strip().replace(' ', '').split(',') for line in lines]
    # mask_list = [x for x in caselist if x[0] in tr]
    
    ## use generated masks
    mask_list = sorted(glob('/mnt/889cdd89-1094-48ae-b221-146ffe543605/gwd/medicaldiffusion/generated/*.npy'))

    vqgan, tester = build_model(cfg)

    # gen_info = []
    for t in tqdm(range(300)):
        img_crops = None
        while img_crops is None:
            img_crops, mask_crops, gen_info_str, img, lbl, pos, pre_lbl = prepare_volume(root_dir, img_list, mask_list, 5, t)
        
        ct_name = gen_info_str[0].split(',')[0]
        # gen_info.extend(gen_info_str)
        textures, masks, blur_masks = synthesize_sample(vqgan, tester, img_crops, mask_crops)

        lbl[lbl==2] = 0
        post_img, post_lbl = synthesize_volume(img, lbl, pos, textures, masks, blur_masks)

        OUTPUT = 'synthesis_visualization'
        os.makedirs(OUTPUT, exist_ok=True)
        img_out = sitk.GetImageFromArray(post_img)
        lbl_out = sitk.GetImageFromArray(post_lbl)
        sitk.WriteImage(img_out, f'{OUTPUT}/{ct_name}.nii.gz')
        sitk.WriteImage(lbl_out, f'{OUTPUT}/{ct_name}_mask.nii.gz')
        # np.savez_compressed(f'{OUTPUT}/{ct_name}.npz', img=post_img, lbl=post_lbl)
        # with open(f'{OUTPUT}/gen_info.csv', 'a') as f:
        #     f.writelines(gen_info_str)
    
    # with open('synthesis/gen_info.csv', 'w') as f:
    #     f.writelines(gen_info)

if __name__ == '__main__':
    main()
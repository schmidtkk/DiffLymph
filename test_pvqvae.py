import random
import torch
import numpy as np
from tqdm import tqdm
import yaml
from pvqvae import PVQVAE_diff
from pvqvae.distributions import DiagonalGaussianDistribution
from dataset.prlnmask_dataloader import prlnmask_dataset
from matplotlib import pyplot as plt
from skimage.measure import marching_cubes
import os
import trimesh
from scipy.ndimage import binary_fill_holes, label
from utils.sdf_fn import compute_sdf
import open3d as o3d


def fill_holes_along_axis(volume, axis):
    filled_volume = np.copy(volume)
    for i in range(volume.shape[axis]):
        if axis == 0:
            slice_2d = volume[i, :, :]
        elif axis == 1:
            slice_2d = volume[:, i, :]
        else:
            slice_2d = volume[:, :, i]

        filled_slice = binary_fill_holes(slice_2d)

        # 确保单连通性
        labeled_array, num_features = label(filled_slice)
        if num_features > 1:
            max_label = max(range(1, num_features + 1), key=lambda x: np.sum(labeled_array == x))
            filled_slice = labeled_array == max_label

        if axis == 0:
            filled_volume[i, :, :] = filled_slice
        elif axis == 1:
            filled_volume[:, i, :] = filled_slice
        else:
            filled_volume[:, :, i] = filled_slice

    return filled_volume


def sdf_to_filled_mesh(sdf):
    vertices, faces, normals, values = marching_cubes(sdf, level=0.02)
    voxel_grid_size = sdf.shape
    voxel_grid = np.zeros(voxel_grid_size, dtype=bool)
    vertex_indices = np.round(vertices).astype(int)
    for idx in vertex_indices:
        voxel_grid[idx[0], idx[1], idx[2]] = True

    filled_voxel = voxel_grid
    while True:
        prev_filled_voxel = filled_voxel.copy()
        filled_voxel = fill_holes_along_axis(filled_voxel, 1)
        filled_voxel = fill_holes_along_axis(filled_voxel, 2)
        filled_voxel = fill_holes_along_axis(filled_voxel, 0)

        if np.array_equal(filled_voxel, prev_filled_voxel):
            break

    vertices_filled, faces_filled, normals_filled, values_filled = marching_cubes(filled_voxel, level=0.02)

    # 将网格转换为 Open3D 的三角网格对象
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices_filled)
    mesh.triangles = o3d.utility.Vector3iVector(faces_filled)
    mesh.compute_vertex_normals()

    # 拉普拉斯平滑
    mesh = mesh.filter_smooth_simple(number_of_iterations=3)
    mesh.compute_vertex_normals()

    return np.asarray(mesh.vertices), np.asarray(mesh.triangles)


def visualize_and_save_slices(original, reconstructed, save_dir='slices'):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Number of slices along the z-axis
    num_slices = original.shape[1]

    for i in range(num_slices):
        plt.figure(figsize=(12, 6))

        # Original image slice
        plt.subplot(1, 2, 1)
        plt.imshow(original[:, i, :], vmin=-0.2, vmax=0.2, cmap='gray')
        plt.title(f'Original Slice {i+1}')

        # Reconstructed image slice
        plt.subplot(1, 2, 2)
        plt.imshow(reconstructed[:, i, :], vmin=-0.2, vmax=0.2, cmap='gray')
        plt.title(f'Reconstructed Slice {i+1}')

        # Save the figure
        plt.savefig(os.path.join(save_dir, f'slice_{i+1}.png'))
        plt.close()


def read_yaml_config(config_file):
    with open(config_file, 'r') as file:
        try:
            return yaml.safe_load(file)
        except yaml.YAMLError as e:
            print(e)
            return None


def test_pvqvae_diff():
    # Define the configuration for the PVQVAE_diff model
    ddconfig = {
        "z_channels": 8,
        "resolution": 64,
        "in_channels": 1,
        "out_ch": 1,
        "ch": 32,
        "ch_mult": [1, 2, 4, 4],
        "num_res_blocks": 2,
        "attn_resolutions": [],
        "dropout": 0.0
    }
    n_embed = 512
    embed_dim = 8

    # Instantiate the PVQVAE_diff model
    model = PVQVAE_diff(ddconfig, n_embed, embed_dim)
    # load check point from pretrained_models/pvqvae.pth
    chkpt = torch.load('pretrained_models/pvqvae.pth', map_location='cpu')
    state_dict = chkpt['model']
    model.load_state_dict(state_dict)
    model.cuda()

    torch.manual_seed(202407)
    random.seed(202407)

    # Generate a random input tensor with shape (batch_size, in_channels, depth, height, width)
    cfg = read_yaml_config('config/dataset/data_config_prlnmask.yaml')
    ds = prlnmask_dataset(cfg)
    bs = 12
    inds = random.sample(range(len(ds)), bs)
    sdfs = []
    for i in inds:
        mask, _ = ds[i]
        x = mask.unsqueeze(0).cuda()
        sdfs.append(x)
    x = torch.cat(sdfs, 0)
    print(x.shape)

    # x = np.load('pvqvae/sample.npy')
    # x = torch.from_numpy(x).cuda()
    # bs = x.shape[0]
    
    # Test forward pass
    with torch.no_grad():
        posterior = model.encode_whole_fold(x)
        z = posterior.sample()
        dec = model.decode(z)
        assert dec.shape == x.shape, f"Forward pass failed: reconstructed shape {dec.shape} does not match input shape {x.shape}"

    # visualize original and reconstructed mesh
    x = x.cpu().detach().numpy().clip(-0.2, 0.2)
    dec = dec.cpu().detach().numpy()
 
    for i in tqdm(range(bs)):
        # Generate meshes for original and reconstructed data
        x_mesh = marching_cubes(x[i, 0], level=0.02)
        # dec_mesh = marching_cubes(dec[i, 0], level=0.02)
        dec_mesh = sdf_to_filled_mesh(dec[i, 0])

        # Create a figure with 4 subplots for each view, for both original and reconstructed
        fig = plt.figure(figsize=(20, 10))

        # Original - Top view
        ax1 = fig.add_subplot(241, projection='3d')
        ax1.plot_trisurf(x_mesh[0][:, 0], x_mesh[0][:, 1], x_mesh[0][:, 2], triangles=x_mesh[1], cmap='viridis')
        ax1.view_init(elev=90, azim=0)
        ax1.set_title('Original - Top View')

        # Original - Side view
        ax2 = fig.add_subplot(242, projection='3d')
        ax2.plot_trisurf(x_mesh[0][:, 0], x_mesh[0][:, 1], x_mesh[0][:, 2], triangles=x_mesh[1], cmap='viridis')
        ax2.view_init(elev=0, azim=0)
        ax2.set_title('Original - Side View')

        # Original - Front view
        ax3 = fig.add_subplot(243, projection='3d')
        ax3.plot_trisurf(x_mesh[0][:, 0], x_mesh[0][:, 1], x_mesh[0][:, 2], triangles=x_mesh[1], cmap='viridis')
        ax3.view_init(elev=0, azim=90)
        ax3.set_title('Original - Front View')

        # Original - Isometric view
        ax4 = fig.add_subplot(244, projection='3d')
        ax4.plot_trisurf(x_mesh[0][:, 0], x_mesh[0][:, 1], x_mesh[0][:, 2], triangles=x_mesh[1], cmap='viridis')
        ax4.view_init(elev=30, azim=45)
        ax4.set_title('Original - Isometric View')

        # Repeat the same for the reconstructed data
        # Reconstructed - Top view
        ax5 = fig.add_subplot(245, projection='3d')
        ax5.plot_trisurf(dec_mesh[0][:, 0], dec_mesh[0][:, 1], dec_mesh[0][:, 2], triangles=dec_mesh[1], cmap='viridis')
        ax5.view_init(elev=90, azim=0)
        ax5.set_title('Reconstructed - Top View')

        # Reconstructed - Side view
        ax6 = fig.add_subplot(246, projection='3d')
        ax6.plot_trisurf(dec_mesh[0][:, 0], dec_mesh[0][:, 1], dec_mesh[0][:, 2], triangles=dec_mesh[1], cmap='viridis')
        ax6.view_init(elev=0, azim=0)
        ax6.set_title('Reconstructed - Side View')

        # Reconstructed - Front view
        ax7 = fig.add_subplot(247, projection='3d')
        ax7.plot_trisurf(dec_mesh[0][:, 0], dec_mesh[0][:, 1], dec_mesh[0][:, 2], triangles=dec_mesh[1], cmap='viridis')
        ax7.view_init(elev=0, azim=90)
        ax7.set_title('Reconstructed - Front View')

        # Reconstructed - Isometric view
        ax8 = fig.add_subplot(248, projection='3d')
        ax8.plot_trisurf(dec_mesh[0][:, 0], dec_mesh[0][:, 1], dec_mesh[0][:, 2], triangles=dec_mesh[1], cmap='viridis')
        ax8.view_init(elev=30, azim=45)
        ax8.set_title('Reconstructed - Isometric View')

        plt.tight_layout()
        plt.show()
        plt.savefig(f'sdf/{i}.png')
        plt.close()

        # Visualize and save slices
        # visualize_and_save_slices(x[i, 0], dec[i, 0], save_dir=f'sdf/slices_{i}')



if __name__ == '__main__':
    torch.cuda.set_device(7)
    test_pvqvae_diff()
import random
import torch
import SimpleITK as sitk
import numpy as np
import os

from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from skimage import measure
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import yaml
from scipy.ndimage.interpolation import zoom
import matplotlib.pyplot as plt
from torch.nn import functional as F
import json
import open3d as o3d

import sys
import os

sys.path.append(os.getcwd())


def extract_and_sample_surface_points(matrix, num_samples=1024):
    # 将3D 0-1矩阵转换为浮点型矩阵
    float_matrix = matrix.astype(np.float32)
    # 使用Marching Cubes算法提取表面
    verts, faces, normals, values = measure.marching_cubes(float_matrix, level=0.5)
    # 创建Open3D的三角网格对象
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(verts)
    mesh.triangles = o3d.utility.Vector3iVector(faces)
    mesh.vertex_normals = o3d.utility.Vector3dVector(normals)
    # 平滑处理
    mesh = mesh.filter_smooth_taubin(number_of_iterations=20)
    # 提取表面点
    surface_points = np.asarray(mesh.vertices)
    # 使用最远点采样算法将表面点云降采样到指定数量的点
    downsampled_point_cloud = farthest_point_sampling(surface_points, num_samples)

    return downsampled_point_cloud, mesh

def farthest_point_sampling(point_cloud, num_samples):
    N, D = point_cloud.shape
    centroids = np.zeros((num_samples, D))
    distances = np.ones(N) * 1e10
    farthest = np.random.randint(0, N)

    for i in range(num_samples):
        centroids[i] = point_cloud[farthest]
        dist = np.linalg.norm(point_cloud - centroids[i], axis=1)
        mask = dist < distances
        distances[mask] = dist[mask]
        farthest = np.argmax(distances)

    return centroids

def normalize_point_cloud(point_cloud):
    centroid = np.mean(point_cloud, axis=0)
    point_cloud -= centroid
    furthest_distance = np.max(np.sqrt(np.sum(point_cloud ** 2, axis=1)))
    point_cloud /= furthest_distance
    return point_cloud


class PointcloudScaleAndTranslate(object):
    def __init__(self, scale_low=2. / 3., scale_high=3. / 2., translate_range=0.2):
        self.scale_low = scale_low
        self.scale_high = scale_high
        self.translate_range = translate_range

    def __call__(self, pc):
        xyz1 = np.random.uniform(low=self.scale_low, high=self.scale_high, size=[3])
        xyz2 = np.random.uniform(low=-self.translate_range, high=self.translate_range, size=[3])
        pc[:, 0:3] = torch.mul(pc[:, 0:3], torch.from_numpy(xyz1).float()) + torch.from_numpy(xyz2).float()

        return pc


class prlnmask_pcd_dataset(Dataset):
    """
    ***请注意***
    3D影像的格式: D x H x W
    候选框的格式: z, x, y, d, h, w
    """

    def __init__(self, cfg):
        super(prlnmask_pcd_dataset, self).__init__()
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
        self.margin_ratio = 0.2

        self.check_data_integrity = cfg['check_data_integrity']
        self.visualize = cfg['visualize']
        # self.use_sdf = cfg['use_sdf']
        self.preproc_dir = cfg['preproc_dir']
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

    def output_pcd(self):
        os.makedirs(self.preproc_dir, exist_ok=True)
        for idx in tqdm(range(len(self.tr))):
            pcd_list = []
            for _ in range(10):
                points = self.__makeitem__(idx)
                pcd_list.append(points)
            pcd = torch.stack(pcd_list)
            ct_name = self.tr[idx]
            np.save(os.path.join(self.preproc_dir, f'{ct_name}.npy'), pcd.numpy())

    def __getitem__(self, idx):
        return [self.__makeitem__(idx)]
        # npy = np.load(os.path.join(self.preproc_dir, f'{self.tr[idx]}.npy'))
        # idj = random.randint(0, npy.shape[0] - 1)
        # return [torch.from_numpy(npy[idj]).float()]

    def __makeitem__(self, idx):
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

        # random sample point cloud from lbl_np
        points = lbl_np
        # points = np.array(np.where(points > 0)).T.astype(float)
        # indices = np.random.choice(points.shape[0], self.num_samples, replace=True)
        # points = points[indices]

        points, mesh = extract_and_sample_surface_points(points)

        # normalize points
        # points = normalize_point_cloud(points)
        # transform = PointcloudScaleAndTranslate()
        # points = torch.from_numpy(points).float()
        # points = transform(points)

        visualize(self.visualize, ct_name, label_id, points, mesh, self.roi_size)

        return points


def read_yaml_config(config_file):
    with open(config_file, 'r') as file:
        try:
            return yaml.safe_load(file)
        except yaml.YAMLError as e:
            print(e)
            return None


def data_collate(batch):
    pcd = [item[0] for item in batch]

    batch_dict = {
        'pcd': torch.stack(pcd),
    }
    return batch_dict


def get_loader(cfg):
    dataset = prlnmask_pcd_dataset(cfg)
    train_loader = DataLoader(dataset, batch_size=cfg['batch_size'], shuffle=True, num_workers=cfg['num_workers'],
                              collate_fn=data_collate)
    return train_loader


def visualize(vis, ct_name, label_id, points, mesh, size):
    if not vis:
        return
    os.makedirs('visualize', exist_ok=True)
    fig = plt.figure(figsize=(24, 12))

    views = [(20, 30), (30, 40), (40, 50), (50, 60)]
    x_size, y_size, z_size = size

    # 点云视图
    for i, view in enumerate(views, start=1):
        ax = fig.add_subplot(2, 4, i, projection='3d')
        colors = points[:, 2]
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=colors, cmap='viridis', marker='o', s=5, alpha=0.75)
        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')
        ax.set_title(f'Point Cloud View {i}')
        ax.grid(True)
        ax.set_xlim(0, x_size)
        ax.set_ylim(0, y_size)
        ax.set_zlim(0, z_size)
        ax.view_init(elev=view[0], azim=view[1])

    # 网格视图
    for i, view in enumerate(views, start=1):
        ax = fig.add_subplot(2, 4, i + 4, projection='3d')
        triangles = np.asarray(mesh.triangles)
        vertices = np.asarray(mesh.vertices)
        ax.add_collection3d(Poly3DCollection(vertices[triangles], alpha=0.75, edgecolor='k'))
        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')
        ax.set_title(f'Mesh View {i}')
        ax.grid(True)
        ax.set_xlim(0, x_size)
        ax.set_ylim(0, y_size)
        ax.set_zlim(0, z_size)
        ax.view_init(elev=view[0], azim=view[1])

    plt.tight_layout()
    plt.savefig(f'visualize/{ct_name}_{label_id}.png')
    plt.close(fig)

if __name__ == '__main__':
    cfg = read_yaml_config('config/dataset/data_config_prlnmask_pcd.yaml')
    print(cfg)
    ds = prlnmask_pcd_dataset(cfg)
    # ds.output_pcd()

    for i in range(len(ds)):
        data = ds[i]

    # dl = get_loader(cfg)
    # for data in dl:
    #     image = data['image']
    #     label = data['label']
    #     print(image.shape, label.shape)

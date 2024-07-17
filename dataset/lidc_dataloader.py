"""
    先不进行预处理操作
    处理好的3d roi进行相应的扩散
    处理成npy的lidc数据集
    修改一下dataset,直接读配对的npy文件

    修改成读取nii.gz的操作
"""
import json
import numpy as np
import torch
import yaml
from torch.utils.data import Dataset, DataLoader, DistributedSampler
import os
from torchvision import transforms
import glob
import cv2
import SimpleITK as sitk
import torchio as tio
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from scipy.special import rel_entr
import matplotlib.pyplot as plt


class lidc_dataset(Dataset):
    def __init__(self, cfg):
        self.root_dir = cfg['root_dir']
        self.amin = cfg['a_min']
        self.amax = cfg['a_max']
        self.bmin = cfg['b_min']
        self.bmax = cfg['b_max']
        self.remove_test_path = cfg['test_path']
        self.target_shape = (cfg['roi_z'], cfg['roi_y'], cfg['roi_x'])
        self.file_names = self.get_file_names()

        self.preprocessing_img = tio.Compose([
            tio.Clamp(out_min=self.amin, out_max=self.amax),
            tio.RescaleIntensity(in_min_max=(self.amin, self.amax),
                                 out_min_max=(self.bmin, self.bmax)),
            tio.CropOrPad(target_shape=self.target_shape)
        ])

        self.preprocessing_mask = tio.Compose([
            tio.CropOrPad(target_shape=self.target_shape)
        ])

    def train_transform(self, image, label, p):
        TRAIN_TRANSFORMS = tio.Compose([
            tio.RandomFlip(axes=(1), flip_probability=p),
        ])
        image = TRAIN_TRANSFORMS(image)
        label = TRAIN_TRANSFORMS(label)
        return image, label

    def get_file_names(self):
        all_file_names = glob.glob(os.path.join(
            self.root_dir, './**/*.nii.gz'), recursive=True)

        # Set for quick lookup
        test_file_names = set()
        # Read the file names from the text file
        with open(self.remove_test_path, 'r') as file:
            for line in file:
                test_file_name = line.strip()  # We only need the name part, not the full path
                test_file_names.add(test_file_name)

        # Filter out the files that are listed in the test.txt
        # Only include files whose base name (without the path and extension) is not in test_file_names
        filtered_file_names = [
            f for f in all_file_names
            # Remove '.nii.gz' and check
            if os.path.basename(f)[:-7] not in test_file_names
        ]
        return filtered_file_names

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, index):
        path = self.file_names[index]

        img = tio.ScalarImage(path)
        # print(path)
        mask_path = path.replace("/Image/", "/Mask/")

        # 更改文件名中的 'NI' 为 'MA'
        filename = mask_path.split('/')[-1]

        # new_filename = filename.replace("Vol", "Mask")

        # clean
        new_filename = filename.replace("Vol", "Mask")

        # print(mask_path)

        mask_path = mask_path.replace(filename, new_filename)

        mask = tio.LabelMap(mask_path)  # （C,W,H,D）

        # print min and max value of mask
        assert mask.data.min() == 0 and mask.data.max() == 1, "mask value error"

        img = self.preprocessing_img(img)
        mask = self.preprocessing_mask(mask)

        p = 0.5
        img, mask = self.train_transform(img, mask, p)
        img = img.data.permute(0, -1, -2, -3)
        mask = mask.data.permute(0, -1, -2, -3)

        # calculate histogram of img in the mask region
        hist = torch.histc(img[mask > 0], bins=16, min=-1, max=1) / mask.sum()
        if torch.sum(hist) == 0 or torch.isnan(hist).any():
            print(index, mask.sum(), "----", hist)
            print(img[mask > 0])

        return img, mask, hist


def data_collate(batch):
    imgs = [item[0] for item in batch]
    lbls = [item[1] for item in batch]
    hists = [item[2] for item in batch]

    dim = len(imgs[0].shape)
    batch_dict = {
        'image': torch.stack(imgs) if dim == 4 else torch.cat(imgs),
        'label': torch.stack(lbls) if dim == 4 else torch.cat(lbls),
        'hist': torch.stack(hists),
    }
    return batch_dict


def get_loader(cfg, rank):

    dataset = lidc_dataset(cfg)

    if cfg.get('dist', False):
        # 如果启用了分布式训练，使用 DistributedSampler
        sampler = DistributedSampler(dataset, rank=rank)
        shuffle = False  # 使用 DistributedSampler 时无需 shuffle
    else:
        sampler = None
        shuffle = True

    train_loader = DataLoader(
        dataset,
        batch_size=cfg['batch_size'],
        shuffle=shuffle,
        num_workers=cfg['num_workers'],
        collate_fn=data_collate,
        sampler=sampler
    )
    return train_loader


def read_yaml_config(config_file):
    with open(config_file, 'r') as file:
        try:
            return yaml.safe_load(file)
        except yaml.YAMLError as e:
            print(e)
            return None


def kl_divergence(P, Q):
    P = np.asarray(P, dtype=np.float32)
    Q = np.asarray(Q, dtype=np.float32)

    # 确保两个向量是概率分布
    P /= np.sum(P)
    Q /= np.sum(Q)

    # 计算 KL 散度
    return np.sum(rel_entr(P, Q))


def js_divergence(P, Q):
    P = np.asarray(P, dtype=np.float32)
    Q = np.asarray(Q, dtype=np.float32)

    # 确保两个向量是概率分布
    P /= np.sum(P)
    Q /= np.sum(Q)

    # 计算 M = (P + Q) / 2
    M = 0.5 * (P + Q)

    # 计算 JSD
    return 0.5 * np.sum(rel_entr(P, M)) + 0.5 * np.sum(rel_entr(Q, M))


def custom_kmeans_js(X, n_clusters, max_iter=100):
    # 初始化聚类中心
    centers = X[np.random.choice(X.shape[0], n_clusters, replace=False)]

    for iteration in range(max_iter):
        # 计算每个样本到聚类中心的 js 散度
        labels = np.zeros(X.shape[0])
        for i, x in enumerate(X):
            js_distances = [js_divergence(x, center) for center in centers]
            labels[i] = np.argmin(js_distances)

        # 更新聚类中心
        new_centers = np.zeros((n_clusters, X.shape[1]))
        for j in range(n_clusters):
            cluster_points = X[labels == j]
            if len(cluster_points) > 0:
                new_centers[j] = np.mean(cluster_points, axis=0)

        # 检查是否收敛
        if np.allclose(centers, new_centers):
            break

        centers = new_centers

    return labels, centers


def convert_to_native(obj):
    """
    将 NumPy 数据类型转换为 Python 原生数据类型。
    """
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.generic):
        return obj.item()
    return obj


def plot_hist():
    cfg = read_yaml_config('config/dataset/data_config_lidc.yaml')
    ds = lidc_dataset(cfg)
    hists = []
    num_case = len(ds)
    # num_case = 100
    from tqdm import tqdm
    print('loading data...')
    for i in tqdm(range(num_case)):
        *_, hist = ds[i]
        hists.append(hist)
    hists = torch.stack(hists).numpy()
    print('data loaded!')

    tsne = TSNE(n_components=2, random_state=0, metric=js_divergence)
    hists_tsne = tsne.fit_transform(hists)

    n_class_list = [3, 4, 5, 6, 7, 8]
    cluster_output = []
    for n_class in n_class_list:
        labels, centers = custom_kmeans_js(hists, n_class)
        plt.figure()
        plt.scatter(hists_tsne[:, 0], hists_tsne[:, 1], c=labels)
        plt.show()
        plt.savefig(f'hists_JS_{n_class}.png')
        plt.close()
        # print cluster scores
        silhouette_avg = silhouette_score(hists, labels)
        cluster_output.append(
            {'n_class': n_class, 
             'score': silhouette_avg, 
             'centers': centers})
        
    cluster_output = sorted(cluster_output, key=lambda x: x['score'], reverse=True)
    with open('cluster_output.json', 'w') as f:
        json.dump(cluster_output, f, default=convert_to_native)


def test():
    cfg = read_yaml_config('config/dataset/data_config_lidc.yaml')
    dl = get_loader(cfg)
    for data in dl:
        print(data['image'].shape)


if __name__ == '__main__':
    plot_hist()

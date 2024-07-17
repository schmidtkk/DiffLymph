import os
import random
import numpy as np
import torch
from tqdm import tqdm
from pvqvae.pvqvae import PVQVAE_diff
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


def cluster_and_visualize():
    features = np.load('features.npz')['arr_0']
    features = features.reshape(features.shape[0], -1)
    print(features.shape)

    # t-SNE for visualization
    tsne = TSNE(n_components=2, random_state=0)
    features_2d = tsne.fit_transform(features)

    # 为每个点生成渐变颜色，这里使用点的第二个t-SNE坐标作为颜色的依据
    colors = features_2d[:, 1]  # 选择第二个坐标作为颜色值

    # 标准化颜色值以适应颜色映射
    norm = plt.Normalize(min(colors), max(colors))
    colors_mapped = plt.cm.viridis(norm(colors))

    # Plotting the points with gradient colors
    plt.figure(figsize=(8, 6))
    plt.scatter(features_2d[:, 0], features_2d[:, 1], c=colors_mapped, marker='o')
    plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap='viridis'))
    plt.title('t-SNE visualization with gradient colors')
    plt.xlabel('t-SNE component 1')
    plt.ylabel('t-SNE component 2')
    plt.show()
    plt.savefig('tsne_visualization.png')
    plt.close()


# def cluster_and_visualize():
#     features = np.load('features.npz')['arr_0']
#     features = features.reshape(features.shape[0], -1)
#     print(features.shape)

#     NUM_CLS = 3
#     # K-means clustering
#     kmeans = KMeans(n_clusters=NUM_CLS, random_state=0).fit(features)
#     labels = kmeans.labels_

#     # t-SNE for visualization
#     tsne = TSNE(n_components=NUM_CLS, random_state=0)
#     features_2d = tsne.fit_transform(features)

#     # Plotting the clusters
#     plt.figure(figsize=(8, 6))
#     plt.scatter(features_2d[:, 0], features_2d[:, 1], c=labels, cmap='viridis', marker='o')
#     plt.colorbar()
#     plt.title('t-SNE visualization of K-means clusters')
#     plt.xlabel('t-SNE component 1')
#     plt.ylabel('t-SNE component 2')
#     plt.show()
#     plt.savefig('clusters.png')
#     plt.close()


def encode():
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
    model.eval()

    torch.manual_seed(202407)
    random.seed(202407)

    rois = np.load('rois.npz')['arr_0']

    BATCH_SIZE = 24
    TOTAL = rois.shape[0]

    features = []
    for i in tqdm(range(0, TOTAL, BATCH_SIZE)):
        bs = BATCH_SIZE if i + BATCH_SIZE < TOTAL else TOTAL - i

        x = rois[i:i+bs]
        x = torch.from_numpy(x).cuda().unsqueeze(1).float()
        print(x.shape)

        with torch.no_grad():
            z = model.encode_whole_fold(x).mode()
            print(z.shape)
        features.append(z)
    
    features = torch.cat(features, dim=0).cpu().numpy()

    np.savez_compressed('features.npz', arr_0=features)



if __name__ == '__main__':
    torch.cuda.set_device(7)
    # encode()
    cluster_and_visualize()

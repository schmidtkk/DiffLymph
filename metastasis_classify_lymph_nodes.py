import gc
import io
import json
import os
import random
import hydra
import numpy as np
import torch
from scipy.ndimage import zoom, gaussian_filter, binary_fill_holes
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from PIL import Image
from skimage.measure import marching_cubes
import SimpleITK as sitk
from tqdm import tqdm


def check_metastasis(root_dir):
    print('Checking metastasis...')
    
    with open(f'{root_dir}/PRLN/prln_target.csv', 'r') as f:
        lines = f.readlines()
    caselist = [line.strip().replace(' ', '').split(',') for line in lines]
    
    LABEL_DIR = os.path.join(root_dir, 'label')
    OUTPUT_FILE = os.path.join(root_dir, 'PRLN', 'prln_target_with_type.csv')
    all_cases = []
    pre_ct = None
    for case in tqdm(caselist):
        ct_name, label_id, _, z1, y1, x1, z2, y2, x2 = case
        z1, y1, x1, z2, y2, x2 = int(z1), int(y1), int(x1), int(z2)+1, int(y2)+1, int(x2)+1
        cz, cy, cx = (z1+z2)//2, (y1+y2)//2, (x1+x2)//2

        if pre_ct != ct_name:
            pre_ct = ct_name
            label = sitk.ReadImage(os.path.join(LABEL_DIR, f'{ct_name}.nii.gz'))
            lbl_np = sitk.GetArrayFromImage(label)
            z_inds, *_ = np.where((lbl_np==6)|(lbl_np==7))
            z_min = max(0, z_inds.min() - 10)
        cz += z_min
        real_label_id = lbl_np[cz, cy, cx]
        if real_label_id not in [6, 7]:
            crop = lbl_np[z1+z_min:z2+z_min, y1:y2, x1:x2]
            real_label_id = 6 if np.sum(crop==6) > np.sum(crop==7) else 7
        assert real_label_id in [6, 7], f'Label id {real_label_id} is not a lymph node'
        case[2] = real_label_id
        all_cases.append(','.join(map(str, case)))

    with open(OUTPUT_FILE, 'w') as f:
        f.write('\n'.join(all_cases))

    print('Done.')


def distance_matrix(points):
    points_expanded = points.unsqueeze(1)  # (N, 1, 3)
    distances = torch.norm(points_expanded - points, dim=2)  # (N, N)
    return distances

def find_longest_axis(points):
    distances = distance_matrix(points)
    max_dist, max_idx = torch.max(distances.view(-1), 0)
    max_idx = torch.div(max_idx, distances.size(0), rounding_mode='floor'), max_idx % distances.size(0)
    point_a, point_b = points[max_idx[0]], points[max_idx[1]]
    return point_a, point_b, max_dist

def find_shortest_axis(points, point_a, point_b, eps=0.01):
    """
    寻找垂直于AB且OC与OD夹角为180度的最短轴。
    """
    ab_vector = point_b - point_a
    mid_point = (point_a + point_b) / 2

    n = points.size(0)
    i_indices, j_indices = torch.tril_indices(n, n, offset=-1)  # 获取点对的索引
    p1 = points[i_indices]
    p2 = points[j_indices]

    # 计算CD向量和OC、OD向量
    cd_vectors = p2 - p1
    oc_vectors = p1 - mid_point
    od_vectors = p2 - mid_point

    # 归一化向量
    cd_vectors_normalized = cd_vectors / torch.norm(cd_vectors, dim=1, keepdim=True)
    ab_vector_normalized = ab_vector / torch.norm(ab_vector)
    oc_vectors_normalized = oc_vectors / torch.norm(oc_vectors, dim=1, keepdim=True)
    od_vectors_normalized = od_vectors / torch.norm(od_vectors, dim=1, keepdim=True)

    # 计算AB与CD的点积，判断是否垂直
    ab_cd_dot_products = torch.abs(torch.sum(cd_vectors_normalized * ab_vector_normalized, dim=1))
    perpendicular_mask = ab_cd_dot_products < eps

    # 计算OC与OD的点积，判断是否夹角为180度
    oc_od_dot_products = torch.sum(oc_vectors_normalized * od_vectors_normalized, dim=1)
    collinear_mask = torch.abs(oc_od_dot_products + 1) < eps

    valid_mask = perpendicular_mask & collinear_mask

    if torch.sum(valid_mask) == 0:
        return find_shortest_axis(points, point_a, point_b, eps=eps+0.05)  # 尝试更小的eps值

    valid_i_indices = i_indices[valid_mask]
    valid_j_indices = j_indices[valid_mask]

    # 计算符合条件的点对之间的距离
    valid_distances = torch.norm(points[valid_i_indices] - points[valid_j_indices], dim=1)
    min_dist_idx = torch.argmin(valid_distances)

    point_c = points[valid_i_indices[min_dist_idx]]
    point_d = points[valid_j_indices[min_dist_idx]]
    min_dist = valid_distances[min_dist_idx]

    return point_c, point_d, min_dist




def select_real_mask(root_dir, case, visualize=False):
    SPACING = np.array([1.25, 0.7, 0.7])
    MARGIN = 10
    AMP = 5.0
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

    amp_max = 64 * SPACING / input_spacing / np.array([z2-z1, y2-y1, x2-x1])
    AMP = min(AMP, amp_max.min())
    scale = input_spacing / SPACING * AMP
    lbl_crop = np.pad(lbl_crop, ((MARGIN, MARGIN), (MARGIN, MARGIN), (MARGIN, MARGIN)), mode='constant', constant_values=0)
    lbl_crop = zoom(lbl_crop, scale, order=3)

    # marching cubes
    points, *_ = marching_cubes(lbl_crop, level=0.5)
    points = points.copy()

    # points = np.argwhere(lbl_crop > 0.5)
    
    if points.shape[0] < 2:
        print(f'Case: {ct_name}, Label: {label_id}, No points found.')
        return None
    point_a, point_b, max_dist = find_longest_axis(torch.from_numpy(points).float())
    point_c, point_d, min_dist = find_shortest_axis(torch.from_numpy(points).float(), point_a, point_b)
    ratio = min_dist / max_dist
    max_dist /= AMP
    min_dist /= AMP
    print(f'Case: {ct_name}, Label: {label_id}, Max Dist: {max_dist}, Min Dist: {min_dist}, Ratio: {ratio}')

    # visualize a b c d in 3d
    if visualize:
        point_o = (point_a + point_b) / 2
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        # 绘制点集
        # downsample points by ramdom choice
        points = points[np.random.choice(points.shape[0], 1000, replace=False)]
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], c='r', marker='o', s=5, alpha=0.25, label='Points')
        
        # 绘制长轴两端点
        ax.scatter(point_a[0], point_a[1], point_a[2], c='b', marker='^', s=100, label='Point A')
        ax.scatter(point_b[0], point_b[1], point_b[2], c='b', marker='^', s=100, label='Point B')
        
        # 绘制短轴两端点
        if point_c is not None and point_d is not None:
            ax.scatter(point_c[0], point_c[1], point_c[2], c='g', marker='s', s=100, label='Point C')
            ax.scatter(point_d[0], point_d[1], point_d[2], c='g', marker='s', s=100, label='Point D')
        
        # 绘制向量 AB
        ax.plot([point_a[0], point_b[0]], [point_a[1], point_b[1]], [point_a[2], point_b[2]], color='blue', linestyle='--', linewidth=2, label='Vector AB')
        
        # 绘制向量 OC 和 OD (从 O 到 C 和 D)
        if point_c is not None and point_d is not None:
            ax.plot([point_o[0], point_c[0]], [point_o[1], point_c[1]], [point_o[2], point_c[2]], color='purple', linestyle='-', linewidth=2, label='Vector OC')
            ax.plot([point_o[0], point_d[0]], [point_o[1], point_d[1]], [point_o[2], point_d[2]], color='orange', linestyle='-', linewidth=2, label='Vector OD')
        # 设置背景颜色
        ax.set_facecolor('white')
        plt.show()

        OUPTUT_DIR = 'short-axis'
        os.makedirs(OUPTUT_DIR, exist_ok=True)
        filename=f'{OUPTUT_DIR}/{ct_name}-{label_id:02d}.gif'
        images = []

        for angle in range(0, 360, 10):
            ax.view_init(elev=10., azim=angle)
            plt.draw()
            # 保存到内存缓冲区
            buf = io.BytesIO()
            plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
            buf.seek(0)
            # 从缓冲区加载图像
            img = Image.open(buf)
            images.append(img.copy())

            buf.close()  # 关闭缓冲区
        
        plt.close()  # 关闭图像窗口

        images[0].save(
            filename,
            save_all=True,
            append_images=images[1:],
            duration=500,
            loop=0
        )

    return max_dist, min_dist, ratio



@hydra.main(config_path='config', config_name='base_cfg', version_base=None)
def main(cfg):
    torch.cuda.set_device(7)
    root_dir = cfg.dataset.root_dir

    # check_metastasis(root_dir)

    with open(f'{root_dir}/PRLN/prln_target_with_type.csv', 'r') as f:
        lines = f.readlines()
    caselist = [line.strip().replace(' ', '').split(',') for line in lines]

    all_cases = []
    for case in tqdm(caselist):
        min_dist, max_dist, ratio = select_real_mask(root_dir, case)
        case.append('{:.3f}'.format(min_dist.item()))
        case.append('{:.3f}'.format(max_dist.item()))
        case.append('{:.3f}'.format(ratio.item()))
        all_cases.append(','.join(map(str, case)))
        with open('metastasis-short-long-axis-tmp.csv', 'a') as f:
            f.write(all_cases[-1] + '\n')
        gc.collect()

    with open('metastasis-short-long-axis.csv', 'w') as f:
        f.writelines('\n'.join(all_cases))

    # check_metastasis(root_dir, mask_list)
    
if __name__ == '__main__':
    main()
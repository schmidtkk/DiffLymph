import numpy as np
from scipy.ndimage import distance_transform_edt, gaussian_filter
import matplotlib.pyplot as plt
from skimage import measure
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.spatial import Delaunay


def compute_sdf(voxel_matrix):
    inside_dist = distance_transform_edt(voxel_matrix)
    outside_dist = distance_transform_edt(1 - voxel_matrix)
    sdf = outside_dist - inside_dist
    sdf = sdf / 32.0
    sdf = gaussian_filter(sdf, sigma=3.0)
    sdf = sdf.clip(-0.2, 0.2)
    return sdf

def visualize_sdf(voxel_matrix, sdf, slice_index, prefix='slice'):
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.title("Voxel Matrix Slice")
    plt.imshow(voxel_matrix[slice_index], cmap='gray')
    plt.subplot(1, 2, 2)
    plt.title("SDF Slice")
    plt.imshow(sdf[slice_index], cmap='jet')
    plt.colorbar()
    plt.show()
    plt.savefig(f'{prefix}_{slice_index}.png')
    plt.close()


def extract_surface(sdf):
    vertices, faces, normals, values = measure.marching_cubes(sdf, level=0)
    return vertices, faces


def visualize_surface(vertices, faces):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    mesh = Poly3DCollection(vertices[faces])
    mesh.set_edgecolor('k')
    ax.add_collection3d(mesh)
    ax.set_xlim(0, 32)
    ax.set_ylim(0, 32)
    ax.set_zlim(0, 32)
    plt.show()
    plt.savefig('surface.png')
    plt.close()


def create_3d_mask(vertices, faces, shape):
    mask = np.zeros(shape, dtype=np.uint8)
    
    # Create a Delaunay triangulation for the vertices
    delaunay = Delaunay(vertices)
    
    # Generate grid points
    x = np.arange(0, shape[0])
    y = np.arange(0, shape[1])
    z = np.arange(0, shape[2])
    grid = np.array(np.meshgrid(x, y, z)).T.reshape(-1, 3)
    
    # Find grid points that are inside the convex hull
    inside = delaunay.find_simplex(grid) >= 0
    
    # Convert the grid points back to 3D index
    grid_points = grid[inside]
    
    # Set the mask values to 1 where points are inside the convex hull
    mask[grid_points[:, 0], grid_points[:, 1], grid_points[:, 2]] = 1
    
    return mask
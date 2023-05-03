import numpy as np
import open3d as o3d
from sklearn.cluster import KMeans
import matplotlib as plt


def main():
    show_by_image('./cow_and_lady.pcd')
    removed_outliers_pcd = remove_outliers('./cow_and_lady.pcd')
    show_by_pcd(removed_outliers_pcd)
    dbscan_segmentation(removed_outliers_pcd)
    # k_means(removed_outliers_pcd)


def show_by_image(image_name):
    pcd = o3d.io.read_point_cloud(image_name)
    show_by_pcd(pcd)


def show_by_pcd(pcd):
    o3d.visualization.draw_geometries([pcd])


def remove_outliers(image):
    # http://www.open3d.org/docs/latest/tutorial/Advanced/pointcloud_outlier_removal.html
    pcd = o3d.io.read_point_cloud(image)
    cl, ind = pcd.remove_statistical_outlier(nb_neighbors=100,
                                             std_ratio=1)
    inlier_cloud = pcd.select_by_index(ind)

    return inlier_cloud


def k_means(pcd):
    # https://towardsdatascience.com/3d-point-cloud-clustering-tutorial-with-k-means-and-python-c870089f3af8
    # Convert point cloud data to numpy array
    xyz = np.asarray(pcd.points)

    # n_clusters is the number of clusters we want.
    # random_state makes the results reproducible and can be useful for debugging
    kmeans = KMeans(n_clusters=4, random_state=1, n_init=5).fit(xyz)

    # Get the cluster labels and the centroids
    labels = kmeans.labels_
    centroids = kmeans.cluster_centers_

    # random color for each cluster
    colors = np.random.rand(len(centroids), 3)

    # Assign each point in the point cloud to a cluster based on the labels
    pcd.colors = o3d.utility.Vector3dVector(np.zeros_like(xyz))
    for i, label in enumerate(labels):
        pcd.colors[i] = colors[label]
    # Visualize the segmented point cloud
    o3d.visualization.draw_geometries([pcd])


def dbscan_segmentation(pcd):
    # http://www.open3d.org/docs/latest/tutorial/Basic/pointcloud.html
    labels = np.array(
        pcd.cluster_dbscan(eps=0.05, min_points=10, print_progress=False))
    max_label = labels.max()
    colors = plt.colormaps.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
    colors[labels < 0] = 0
    pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])
    o3d.visualization.draw_geometries([pcd])


main()

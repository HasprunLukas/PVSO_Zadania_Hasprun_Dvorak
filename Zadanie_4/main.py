import numpy as np
import open3d as o3d
from sklearn.cluster import KMeans


def main():
    # show_by_image(
    #     "/home/pdvorak/school/pvso/PVSO_Zadania_Hasprun_Dvorak/Zadanie_4/cow_and_lady.pcd")
    # show_by_image("./output.pcd")
    removed_outliers_pcd = remove_outliers(
        '/home/pdvorak/school/pvso/PVSO_Zadania_Hasprun_Dvorak/Zadanie_4/cow_and_lady.pcd')
    k_means(removed_outliers_pcd)

    # show_by_pcd(removed_outliers_pcd)


def show_by_image(image_name):
    print("Loading " + image_name)
    pcd = o3d.io.read_point_cloud(image_name)
    print(pcd)
    print(np.asarray(pcd.points))
    show_by_pcd(pcd)


def show_by_pcd(pcd):
    o3d.visualization.draw_geometries([pcd])


def remove_outliers(image):
    # http://www.open3d.org/docs/latest/tutorial/Advanced/pointcloud_outlier_removal.html
    pcd = o3d.io.read_point_cloud(image)
    cl, ind = pcd.remove_statistical_outlier(nb_neighbors=20,
                                             std_ratio=2.0)
    inlier_cloud = pcd.select_by_index(ind)

    return inlier_cloud


def k_means(pcd):
    # https://towardsdatascience.com/3d-point-cloud-clustering-tutorial-with-k-means-and-python-c870089f3af8
    # Convert point cloud data to numpy array
    xyz = np.asarray(pcd.points)

    # n_clusters is the number of clusters we want.
    # random_state makes the results reproducible and can be useful for debugging
    kmeans = KMeans(n_clusters=20, random_state=1).fit(xyz)

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


main()

import numpy as np
import open3d as o3d
import open3d.visualization


def main():
    show_by_image("./cow_and_lady.pcd")
    # show_by_image("./output.pcd")
    removed_outliers_pcd = remove_outliers('./cow_and_lady.pcd')
    show_by_pcd(removed_outliers_pcd)


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


main()

import numpy as np
import open3d as o3d
import matplotlib as plt


def main():
    # show_pcd("./cow_and_lady.pcd")
    # show_pcd("./output.pcd")
    pcd = o3d.io.read_point_cloud('./cow_and_lady.pcd')
    ransac(pcd)


def show_pcd(image_name):
    print("Loading " + image_name)
    pcd = o3d.io.read_point_cloud(image_name)
    print(pcd)
    print(np.asarray(pcd.points))
    o3d.visualization.draw_geometries([pcd])


def ransac(pcd):
    plane_model, inliers = pcd.segment_plane(distance_threshold=0.01, ransac_n=3, num_iterations=1000)
    inlier_cloud = pcd.select_by_index(inliers)
    outlier_cloud = pcd.select_by_index(inliers, invert=True)
    # inlier_cloud.paint_uniform_color([1, 0, 0])
    # outlier_cloud.paint_uniform_color([0.6, 0.6, 0.6])

    # o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud])

    labels = np.array(pcd.cluster_dbscan(eps=0.05, min_points=10))

    max_label = labels.max()
    colors = plt.colormaps.get_cmap("tab20")(labels / (max_label
                                             if max_label > 0 else 1))
    colors[labels < 0] = 0
    pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])
    # o3d.visualization.draw_geometries([pcd])

    segment_models = {}
    segments = {}

    max_plane_idx = 20

    rest = pcd
    for i in range(max_plane_idx):
        colors = plt.colormaps.get_cmap("tab20")(i)
        segment_models[i], inliers = rest.segment_plane(
            distance_threshold=0.01, ransac_n=3, num_iterations=1000)
        segments[i] = rest.select_by_index(inliers)
        segments[i].paint_uniform_color(list(colors[:3]))
        rest = rest.select_by_index(inliers, invert=True)
        print("pass", i, "/", max_plane_idx, "done.")
    # o3d.visualization.draw_geometries([segments[i] for i in range(max_plane_idx)] + [rest])

    labels = np.array(segments[i].cluster_dbscan(eps=1*10, min_points=10))

    candidates = [len(np.where(labels == j)[0]) for j in np.unique(labels)]

    best_candidate = int(np.unique(labels)[np.where(candidates == np.max(candidates))[0]])

    rest = rest.select_by_index(inliers, invert=True) + segments[i].select_by_index(
        list(np.where(labels != best_candidate)[0]))
    segments[i] = segments[i].select_by_index(list(np.where(labels == best_candidate)[0]))

    labels = np.array(rest.cluster_dbscan(eps=0.05, min_points=5))
    max_label = labels.max()
    print(f"point cloud has {max_label + 1} clusters")
    colors = plt.colormaps.get_cmap("tab10")(labels / (max_label if max_label > 0 else 1))
    colors[labels < 0] = 0
    rest.colors = o3d.utility.Vector3dVector(colors[:, :3])

    o3d.visualization.draw_geometries([rest])


main()

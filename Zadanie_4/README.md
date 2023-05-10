# PVSO Zadanie 4

## Tasks
 Cieľom zadania, je oboznámiť sa s pracou s mračnom bodov (point cloud) a segmentácia
objektov v priestore. Študent si vyskúša vytvorenie vlastného mračna bodov a aplikáciu metód
na získanie segmentovaného priestoru. Použitie externých knižníc ako open3d, sklearn,
opencv, a iných je dovolené a odporúčané. Zadanie pozostáva z viacerých úloh:
1. Vytvorenie mračna bodov pomocou Kinect v2 pre testovanie. Nájdite online na webe
mračno bodov popisujúce väčší priestor (väčší objem dát aspoň 4x4 metre) pre
testovanie algoritmov a načítajte mračno dostupného datasetu (2B)
2. Pomocou knižnice (open3d - python) načítate vytvorené mračno bodov a zobrazíte.
(2B)
3. Mračná bodov očistite od okrajových bodov. Pre tuto úlohu je vhodne použiť algoritmus
RANSAC. (5B)
4. Segmentujete priestor do klastrov pomocou vhodne zvolených algoritmov (K-means,
DBSCAN, BIRCH, Gausian mixture, mean shift ...). Treba si zvoliť aspoň 2 algoritmy a
porovnať ich výsledky. (5+5B)
5. Detailne vysvetlite fungovanie zvolených algoritmov. (4B) (Keďže neimplementujete
konkrétny algoritmus ale používate funkcie tretích strán je potrebné rozumieť aj ako sú
funkcie implementované)
6. Vytvorte dokumentáciu zadania (popis implementovaných algoritmov, Grafické
porovnanie výstupov, vysvetlite rozdiel v kvalite výstupov pre rozdielne typy algortimov)
(2B).

## Task 2

We downloaded a point cloud file from the internet.

![Original point cloud](/Zadanie_4/original.png)


## Task 3

Open3d library have a built in function, **remove_statistical_outlier**. THis function removes points that are further away from their neighbors compared to the average for the point cloud. It takes two input parameters
 - **nb_neighbors** - This specfies how many neightbours are taken into account in order to calculate the average distances in a given point.
 - **std_ration** - Allows setting the threshold level based on the standard deviation of the average distances across the point cloud. The lower this number is the more aggresive the filter will be.

![After using remove static outliners](/Zadanie_4/ransac.png)

## Task 4

For the segmentation we selected the **K-Means** and the **DBSCAN** algorithm.

### **K-Mean Clustering**

It represents all the data points with K representatives, which gave the algorithm its name. So K is a user-defined number that we put into the system.

For K-Means we used the **sklearn** library. The library have a predefinied function for K-Means clustering. This function takes three parameters:
 - **n_cluster** - The number of clusters to form as well as the number of centroids to generate.
 - **random_state** - Determines random number generation for centroid initialization. Use an int to make the randomness deterministic.
 - **n_init** - Number of times the k-means algorithm is run with different centroid seeds. The final results is the best output of n_init consecutive runs in terms of inertia. Several runs are recommended for sparse high-dimensional problems
  
![K-Means clustering](/Zadanie_4/k_means.png)

### **DBSCAN Clustering**

Open3d have a predefinied function for DBSCAN. This fnction is **cluster_dbscan** and it takes two parameters as input:
 - **eps** - Defines the distance to neighbour in a cluster.
 - **min_points** - Defines the number of minimum points to define a cluster.



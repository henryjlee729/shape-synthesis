import numpy as np
import open3d as o3d

# Load .pts file
points = np.loadtxt("./test_chair_data/1a6f615e8b1b5ae4dbbc9440457e303e.pts")   # shape: (N, 3)

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points)

pcd.estimate_normals(
    search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.15, max_nn=50)
)
pcd.orient_normals_consistent_tangent_plane(30)

distances = pcd.compute_nearest_neighbor_distance()
avg_dist = np.mean(distances)

radii = [avg_dist * 2.5, avg_dist * 4.0, avg_dist * 6.5, avg_dist * 8.0]
mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
    pcd, o3d.utility.DoubleVector(radii)
)

pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)

mesh.compute_vertex_normals()
o3d.visualization.draw_geometries([mesh], mesh_show_back_face=True)

# Evaluations:
# Poisson: Not very good
# Ball Pivoting: Better, but still has holes and artifacts.

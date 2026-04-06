import numpy as np
import open3d as o3d

points = np.loadtxt("./test_chair_data/1a6f615e8b1b5ae4dbbc9440457e303e.pts")

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points)

o3d.visualization.draw_geometries([pcd])
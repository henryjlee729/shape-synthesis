from pathlib import Path
import numpy as np
import open3d as o3d


def generate_mesh(input_path: str, output_path: str | None = None) -> str:
    points = np.loadtxt(input_path, dtype=np.float32)
    if points.ndim == 1:
        points = points.reshape(1, -1)
    points = points[:, :3]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
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

    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_duplicated_vertices()
    mesh.remove_non_manifold_edges()
    mesh.compute_vertex_normals()

    if output_path is None:
        output_dir = Path('./output')
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = str(output_dir / f'{Path(input_path).stem}_mesh.ply')

    o3d.io.write_triangle_mesh(output_path, mesh)
    return output_path


if __name__ == '__main__':
    output = generate_mesh('./test_chair_data/1a6f615e8b1b5ae4dbbc9440457e303e.pts')
    print(f'Saved mesh to {output}')

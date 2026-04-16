from pathlib import Path
import numpy as np
import open3d as o3d

def generate_mesh(input_path: str, output_path: str | None = None) -> str:

    # Load point cloud
    points = np.loadtxt(input_path, dtype=np.float32)

    # Handle edge case: single point
    if points.ndim == 1:
        points = points.reshape(1, -1)

    # Keep only XYZ coordinates
    points = points[:, :3]

    if len(points) < 50:
        raise ValueError("Not enough points to build a mesh.")

    # Normalize point cloud (center + scale)
    center = points.mean(axis=0)
    points = points - center

    scale = np.max(np.linalg.norm(points, axis=1))
    if scale > 0:
        points = points / scale

    # Create Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    # Downsample slightly to reduce noise
    pcd = pcd.voxel_down_sample(voxel_size=0.015)

    # Remove statistical outliers
    pcd, _ = pcd.remove_statistical_outlier(
        nb_neighbors=20,
        std_ratio=1.5
    )

    # Estimate normals (important for reconstruction)
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=0.06,
            max_nn=30
        )
    )

    # Compute average neighbor distance
    distances = pcd.compute_nearest_neighbor_distance()
    avg_dist = np.mean(distances)

    # Alpha controls mesh tightness
    # smaller = sharper, larger = smoother
    alpha = avg_dist * 3.0

    # Alpha shape reconstruction
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(
        pcd, alpha
    )

    # Cleanup mesh artifacts
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_duplicated_vertices()
    mesh.remove_non_manifold_edges()

    # Light smoothing (avoid over-smoothing)
    mesh = mesh.filter_smooth_taubin(number_of_iterations=1)

    # Compute normals for rendering
    mesh.compute_vertex_normals()

    # Save output
    if output_path is None:
        base_dir = Path(__file__).resolve().parent
        output_dir = base_dir.parent / "test"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = str(output_dir / f"{Path(input_path).stem}_mesh.ply")

    o3d.io.write_triangle_mesh(output_path, mesh)
    return output_path
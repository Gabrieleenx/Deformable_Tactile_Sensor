import trimesh
import numpy as np
import matplotlib.pyplot as plt

class ThicknessMapper:
    """
    A class to calculate and visualize the intersection thickness between two meshes.
    All public-facing units (inputs and outputs) are in meters (m).
    """
    def __init__(self, mesh1_path, mesh2_path, resolution=0.0002, nbins=60, xy_range=0.02):
        """
        Initializes the mapper by loading meshes and setting up parameters.
        All arguments are expected to be in meters.

        Args:
            mesh1_path (str): File path to the first mesh (e.g., the object).
            mesh2_path (str): File path to the second mesh (e.g., the sensor).
            resolution (float): The voxel size for the intersection calculation (in m).
            nbins (int): The number of bins for the 2D thickness map.
            xy_range (float): The range (in m) for the x and y axes of the thickness map.
        """
        self.mesh1 = trimesh.load(mesh1_path)
        self.mesh2 = trimesh.load(mesh2_path)
        self.mesh1.units = 'mm'
        self.mesh2.units = 'mm'
        
        self.resolution_mm = resolution * 1000
        self.nbins = nbins
        self.xy_range_mm = xy_range * 1000
        self.xy_range = xy_range 

    def get_thickness_map(self, pose, z_offset=0.0, show_scene=False):
        """
        Calculates the thickness map based on the provided pose and z-offset.

        Args:
            pose (np.ndarray): A 4x4 homogeneous transformation matrix representing the pose
                               of mesh2 in the world frame. The translation component is assumed
                               to be in meters.
            z_offset (float): An additional translation along the Z-axis of the pose frame (in m).
            show_scene (bool): If True, a 3D visualization of the meshes and intersection
                               is displayed.

        Returns:
            np.ndarray: A 2D numpy array representing the thickness map (in m).
        """
        # Create a deep copy to avoid modifying the original mesh
        mesh2_transformed = self.mesh2.copy()

        # Extract rotation and translation from the pose matrix
        R = pose[:3, :3]
        t_m = pose[:3, 3]  # Assumed to be in meters
        
        # Create the offset vector in the pose frame
        z_offset_vector_m = np.array([0, 0, z_offset])
        
        # Apply the rotation to the z_offset vector to get its direction in the world frame
        z_offset_rotated_m = R @ z_offset_vector_m
        
        # Add the rotated offset to the original pose translation
        t_final_m = t_m + z_offset_rotated_m
        
        # Construct the final transformation matrix for the mesh
        T_mesh2 = np.eye(4)
        T_mesh2[:3, :3] = R
        T_mesh2[:3, 3] = t_final_m * 1000 # Convert to mm
        
        mesh2_transformed.apply_transform(T_mesh2)

        # Compute intersection volume, handling the case of no intersection gracefully
        try:
            inter = self.mesh1.intersection(mesh2_transformed)
        except trimesh.exceptions.InvalidMesh:
            print("Warning: Mesh intersection failed due to invalid mesh. Returning an empty map.")
            inter = trimesh.Trimesh()

        is_empty = inter.is_empty or inter.vertices.shape[0] == 0
        if is_empty:
            print("Warning: No intersection found.")
            thickness_map_m = np.zeros((self.nbins, self.nbins))
        else:
            voxelized = inter.voxelized(pitch=self.resolution_mm)
            points_world = voxelized.points

            # Transform voxel centers into mesh2 local frame
            T_world_to_mesh2 = np.linalg.inv(T_mesh2)
            points_local = trimesh.transform_points(points_world, T_world_to_mesh2)
            x_mm, y_mm, z_mm = points_local.T

            # Accumulate thickness into a 2D grid
            x_min_mm, x_max_mm = -self.xy_range_mm, self.xy_range_mm
            y_min_mm, y_max_mm = -self.xy_range_mm, self.xy_range_mm
            xedges_mm = np.linspace(x_min_mm, x_max_mm, self.nbins + 1)
            yedges_mm = np.linspace(y_min_mm, y_max_mm, self.nbins + 1)

            ix = np.searchsorted(xedges_mm, x_mm, side='right') - 1
            iy = np.searchsorted(yedges_mm, y_mm, side='right') - 1
            valid = (ix >= 0) & (ix < self.nbins) & (iy >= 0) & (iy < self.nbins)
            ix, iy, z_mm_valid = ix[valid], iy[valid], z_mm[valid]

            z_min_map_mm = np.full((self.nbins, self.nbins), np.inf)
            z_max_map_mm = np.full((self.nbins, self.nbins), -np.inf)
            np.minimum.at(z_min_map_mm, (ix, iy), z_mm_valid)
            np.maximum.at(z_max_map_mm, (ix, iy), z_mm_valid)
            
            thickness_map_mm = z_max_map_mm - z_min_map_mm
            thickness_map_mm[~np.isfinite(thickness_map_mm)] = 0

            # Convert thickness map to meters for the return value and plotting
            thickness_map_m = thickness_map_mm / 1000.0

        # Plot heatmap for debugging and visualization
        x_min_m, x_max_m = -self.xy_range, self.xy_range
        y_min_m, y_max_m = -self.xy_range, self.xy_range
        
        
        # Optional: Visualize meshes + intersection in 3D
        if show_scene:
        


            self.mesh1.visual.face_colors = [200, 100, 100, 255]
            mesh2_transformed.visual.face_colors = [100, 100, 200, 155]
            
            if is_empty:
                scene = trimesh.Scene([self.mesh1, mesh2_transformed])
            else:
                inter.visual.face_colors = [30, 200, 30, 255]
                scene = trimesh.Scene([self.mesh1, mesh2_transformed, inter])

            scene.show()

        total_sum = np.sum(thickness_map_m)
        if total_sum > 0:
            thickness_map_m = thickness_map_m / total_sum

        return thickness_map_m.tolist()

# ------------------------------------------------------------
# Example Usage
# ------------------------------------------------------------
if __name__ == '__main__':
    # Define mesh file paths
    mesh1_file = "data_creation/mesh_data/sensor_playground v3.stl"
    mesh2_file = "data_creation/mesh_data/sensor.stl"

    # Define a sample pose transformation matrix for mesh2
    theta = np.deg2rad(180)
    Ry = np.array([
        [np.cos(theta), 0, np.sin(theta), 0],
        [0, 1, 0, 0],
        [-np.sin(theta), 0, np.cos(theta), 0],
        [0, 0, 0, 1]
    ])
    T_mesh2_base = np.eye(4)
    # Pose translation in meters
    T_mesh2_base[:3, 3] = [-0.025, 0.025, 0.031]
    sample_pose = T_mesh2_base @ Ry

    # Instantiate the mapper
    # All arguments are now in meters
    mapper = ThicknessMapper(mesh1_file, mesh2_file, resolution=0.0002, xy_range=0.02)

    # Example 1: With intersection
    thickness_map_with_inter = mapper.get_thickness_map(
        pose=sample_pose, 
        z_offset=0.000,  # 5mm offset in meters
        show_scene=True
    )


    plt.show()
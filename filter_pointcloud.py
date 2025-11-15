#!/usr/bin/env python3
"""
Script to apply global mask filtering (brightness + DBSCAN clustering) to an existing point cloud
and save the filtered version without retraining.

Usage:
    python filter_pointcloud.py \
        --input_ply /path/to/point_cloud/iteration_20000/point_cloud.ply \
        --output_ply /path/to/point_cloud/iteration_20000/point_cloud_filtered.ply \
        --bright_threshold 0.2 \
        --distance_threshold 0.2 \
        --min_samples 20
"""

import argparse
import torch
import numpy as np
from sklearn.cluster import DBSCAN
from gaussian_splatting.scene.gaussian_model import GaussianModel
from gaussian_splatting.utils.sh_utils import SH2RGB
import os
import sys

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def detect_and_filter_clusters(xyz, opacity, eps=2.0, min_samples=100):
    """
    Detect spatial clusters and filter out outlier clusters.
    
    Args:
        xyz: [N, 3] tensor of gaussian positions
        opacity: [N] tensor of gaussian opacities
        eps: Maximum distance between clusters for DBSCAN
        min_samples: Minimum size for a cluster to be considered valid
    
    Returns:
        cluster_mask: Boolean mask of gaussians to keep
    """
    # Convert to numpy for clustering
    xyz_np = xyz.detach().cpu().numpy()
    opacity_np = opacity.detach().cpu().numpy()
    
    # Use DBSCAN to detect clusters
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(xyz_np)
    cluster_labels = clustering.labels_
    
    # Find the largest cluster (main cluster)
    unique_labels = np.unique(cluster_labels)
    counts = np.array([(cluster_labels == label).sum() for label in unique_labels])
    
    # Find the largest cluster (excluding noise labeled as -1)
    valid_mask = unique_labels != -1
    valid_labels = unique_labels[valid_mask]
    valid_counts = counts[valid_mask]
    
    if len(valid_labels) == 0:
        print("Warning: No valid clusters found, keeping all points")
        cluster_mask = np.ones(len(cluster_labels), dtype=bool)
    else:
        largest_cluster_label = valid_labels[np.argmax(valid_counts)]
        
        print(f"Detected {len(valid_labels)} clusters with sizes: {valid_counts}")
        print(f"Largest cluster: label {largest_cluster_label} with {valid_counts.max()} points")
        
        # Calculate statistics for the largest cluster
        main_mask = cluster_labels == largest_cluster_label
        main_center = xyz_np[main_mask].mean(axis=0)
        main_opacity = opacity_np[main_mask].mean()
        
        print(f"Largest cluster stats:")
        print(f"  - Center: {main_center}")
        print(f"  - Mean opacity: {main_opacity:.3f}")
        
        # Calculate distance from each point in cluster to center
        distances = np.linalg.norm(xyz_np[main_mask] - main_center, axis=1)
        distance_threshold = np.percentile(distances, 95)  # Keep 95% closest points
        
        # Filter outliers within the cluster
        outlier_mask = distances <= distance_threshold  # Keep points within threshold
        filtered_indices = np.where(main_mask)[0][outlier_mask]
        
        # Create final mask: keep only points in largest cluster that are within distance threshold
        cluster_mask = np.zeros(len(cluster_labels), dtype=bool)
        cluster_mask[filtered_indices] = True
        
        print(f"Cluster filtering: keeping {cluster_mask.sum()}/{len(cluster_mask)} gaussians (largest cluster only)")
    
    return torch.tensor(cluster_mask, device=xyz.device)


def filter_pointcloud(input_ply_path, output_ply_path, bright_threshold=0.2, 
                     distance_threshold=0.2, min_samples=20):
    """
    Load a PLY file, apply filtering, and save the filtered version.
    
    Args:
        input_ply_path: Path to input PLY file
        output_ply_path: Path to save filtered PLY file
        bright_threshold: Brightness threshold for filtering dark Gaussians
        distance_threshold: Distance threshold for DBSCAN clustering
        min_samples: Minimum samples for DBSCAN clustering
    """
    print(f"Loading point cloud from: {input_ply_path}")
    
    # Load the Gaussian model from PLY
    gaussians = GaussianModel(sh_degree=3)  # Adjust sh_degree if needed
    gaussians.load_ply(input_ply_path)
    
    print(f"Loaded {gaussians.get_xyz.shape[0]} Gaussians")
    
    # Move to CUDA if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    gaussians._xyz = gaussians._xyz.to(device)
    gaussians._features_dc = gaussians._features_dc.to(device)
    gaussians._features_rest = gaussians._features_rest.to(device)
    gaussians._opacity = gaussians._opacity.to(device)
    gaussians._scaling = gaussians._scaling.to(device)
    gaussians._rotation = gaussians._rotation.to(device)
    
    print("Applying global mask filtering...")
    with torch.no_grad():
        xyz = gaussians.get_xyz
        
        # Filter by brightness
        rgb_dc = torch.sigmoid(gaussians._features_dc)
        if rgb_dc.dim() == 3:  # e.g., [N,1,3]
            rgb_dc = rgb_dc.squeeze(1)
        brightness = 0.2126*rgb_dc[:,0] + 0.7152*rgb_dc[:,1] + 0.0722*rgb_dc[:,2]
        non_black_mask = brightness > bright_threshold
        
        print(f"Brightness filter: {non_black_mask.sum().item()}/{len(non_black_mask)} Gaussians passed")
        
        # Apply clustering filter to remaining gaussians
        global_mask = torch.zeros(len(xyz), dtype=torch.bool, device=xyz.device)
        if non_black_mask.sum() > 0:
            opacity = gaussians.get_opacity.squeeze()
            cluster_mask = detect_and_filter_clusters(
                xyz[non_black_mask],
                opacity[non_black_mask],
                eps=distance_threshold,
                min_samples=min_samples
            )
            # Map cluster mask back to full array
            global_mask[non_black_mask] = cluster_mask
        else:
            print("Warning: No bright Gaussians found, keeping all")
            global_mask = torch.ones(len(xyz), dtype=torch.bool, device=xyz.device)
        
        print(f"Global mask: {global_mask.sum().item()}/{len(global_mask)} Gaussians kept after filtering")
        
        # Save filtered PLY
        print(f"Saving filtered point cloud to: {output_ply_path}")
        os.makedirs(os.path.dirname(output_ply_path), exist_ok=True)
        gaussians.save_ply(output_ply_path, global_mask=global_mask)
        
        print(f"Done! Filtered point cloud saved with {global_mask.sum().item()} Gaussians")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Filter an existing point cloud PLY file")
    parser.add_argument("--input_ply", type=str, required=True,
                       help="Path to input PLY file")
    parser.add_argument("--output_ply", type=str, required=True,
                       help="Path to save filtered PLY file")
    parser.add_argument("--bright_threshold", type=float, default=0.2,
                       help="Brightness threshold for filtering dark Gaussians (default: 0.2)")
    parser.add_argument("--distance_threshold", type=float, default=0.2,
                       help="Distance threshold for DBSCAN clustering (default: 0.2)")
    parser.add_argument("--min_samples", type=int, default=20,
                       help="Minimum samples for DBSCAN clustering (default: 20)")
    
    args = parser.parse_args()
    
    filter_pointcloud(
        args.input_ply,
        args.output_ply,
        bright_threshold=args.bright_threshold,
        distance_threshold=args.distance_threshold,
        min_samples=args.min_samples
    )


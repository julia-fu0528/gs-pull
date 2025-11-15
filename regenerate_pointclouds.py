#!/usr/bin/env python3
"""
Script to regenerate point clouds with colors from existing SuGaR checkpoints.
Usage: python regenerate_pointclouds.py --checkpoint_path <path> --iteration <iter>
"""

import argparse
import torch
import os
from sugar_scene.gs_model import GaussianSplattingWrapper
from sugar_scene.sugar_model import SuGaR
from sugar_utils.spherical_harmonics import SH2RGB
from np_utils.train import Runner

def regenerate_pointcloud(checkpoint_path, scene_path, iteration, dataset_name='brics', gpu=0):
    """Regenerate point cloud with colors from a SuGaR checkpoint."""
    
    torch.cuda.set_device(gpu)
    device = torch.device(f'cuda:{gpu}')
    
    # Load Gaussian Splatting model
    print(f"Loading 3DGS model from {scene_path}...")
    nerfmodel = GaussianSplattingWrapper(
        source_path=scene_path,
        output_path=scene_path,  # Assuming checkpoint is in scene_path
        iteration_to_load=20000,  # Adjust if needed
        load_gt_images=False,
        eval_split=False,
        dataset_name=dataset_name,
    )
    
    # Load SuGaR checkpoint
    checkpoint_file = os.path.join(checkpoint_path, f'{iteration}.pt')
    print(f"Loading SuGaR checkpoint from {checkpoint_file}...")
    checkpoint = torch.load(checkpoint_file, map_location=device)
    
    colors = SH2RGB(checkpoint['state_dict']['_sh_coordinates_dc'][:, 0, :])
    sugar = SuGaR(
        nerfmodel=nerfmodel,
        points=checkpoint['state_dict']['_points'],
        colors=colors,
        initialize=True,
        sh_levels=nerfmodel.gaussians.active_sh_degree + 1,
        keep_track_of_knn=True,
        knn_to_track=16,
        beta_mode='average',
        primitive_types='diamond',
        surface_mesh_to_bind=None,
    )
    ck_state_dict = checkpoint['state_dict']
    if '_weights' in ck_state_dict:
        del ck_state_dict['_weights']
    sugar.load_state_dict(ck_state_dict, strict=False)
    
    # Note: neus is not needed for visual_point_cloud, but may be needed for other operations
    # If you get errors, you may need to load neus:
    # scene_name = checkpoint_path.split('/')[-1]
    # if scene_name in ['Barn', 'Meetingroom', 'Courthouse']:
    #     sugar.part_num = 4
    # else:
    #     sugar.part_num = 1
    # sugar.neus = Runner(checkpoint_path, None, part_num=sugar.part_num)
    
    # Generate point cloud with colors
    print(f"Generating point cloud for iteration {iteration}...")
    sugar.visual_point_cloud(iteration=iteration, checkpoint_path=checkpoint_path)
    print(f"Point cloud saved to {checkpoint_path}/meshes/points_{iteration}.ply")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Regenerate point clouds with colors from SuGaR checkpoints.')
    parser.add_argument('--checkpoint_path', type=str, required=True,
                        help='Path to SuGaR checkpoint directory')
    parser.add_argument('--scene_path', type=str, required=True,
                        help='Path to scene data (for loading 3DGS model)')
    parser.add_argument('--iteration', type=int, required=True,
                        help='Iteration number of checkpoint to load')
    parser.add_argument('--dataset_name', type=str, default='brics',
                        help='Dataset name (default: brics)')
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU device index (default: 0)')
    
    args = parser.parse_args()
    regenerate_pointcloud(
        checkpoint_path=args.checkpoint_path,
        scene_path=args.scene_path,
        iteration=args.iteration,
        dataset_name=args.dataset_name,
        gpu=args.gpu
    )


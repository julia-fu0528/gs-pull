import os
import torch
from random import randint
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
import trimesh
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
import zarr
from PIL import Image
import numpy as np
import wandb
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

def detect_and_filter_clusters(xyz, opacity, eps=2.0, min_samples=100):
    """
    Detect spatial clusters and filter out outlier clusters.
    
    Args:
        xyz: [N, 3] tensor of gaussian positions
        opacity: [N] tensor of gaussian opacities
        distance_threshold: Maximum distance between clusters
        min_cluster_size: Minimum size for a cluster to be considered valid
    
    Returns:
        cluster_mask: Boolean mask of gaussians to keep
    """
    from sklearn.cluster import DBSCAN
    import numpy as np
    
    # Convert to numpy for clustering
    xyz_np = xyz.detach().cpu().numpy()
    opacity_np = opacity.detach().cpu().numpy()
    
    # Use DBSCAN to detect clusters
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(xyz_np)
    cluster_labels = clustering.labels_
    
    # Find the largest cluster (main cluster) - use older NumPy syntax
    print(f"cluster_labels: {cluster_labels}")
    print(f"cluster labels: {len(cluster_labels)}")
    unique_labels = np.unique(cluster_labels)
    print(f"unique_labels: {unique_labels}")
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
        # main_cluster_label = unique_labels[np.argmax(counts)]
    
        print(f"Detected {len(valid_labels)} clusters with sizes: {valid_counts}")
        print(f"Largest cluster: label {largest_cluster_label} with {valid_counts.max()} points")
    
        # # Calculate cluster statistics
        # cluster_stats = {}
        # for label in unique_labels:
        #     if label == -1:  # Noise points
        #         continue
        #     mask = cluster_labels == label
        #     cluster_stats[label] = {
        #         'size': mask.sum(),
        #         'mean_opacity': opacity_np[mask].mean(),
        #         'center': xyz_np[mask].mean(axis=0),
        #         'is_main': (label == main_cluster_label)
        #     }
        # Calculate statistics for the largest cluster
        main_mask = cluster_labels == largest_cluster_label
        main_center = xyz_np[main_mask].mean(axis=0)
        main_opacity = opacity_np[main_mask].mean()
        
        # Find clusters that are too far from main cluster
        # main_center = cluster_stats[main_cluster_label]['center']
        # clusters_to_keep = [main_cluster_label]  # Always keep main cluster
        
        print(f"Largest cluster stats:")
        print(f"  - Center: {main_center}")
        print(f"  - Mean opacity: {main_opacity:.3f}")
        
        # Calculate distance from each point in cluster to center
        distances = np.linalg.norm(xyz_np[main_mask] - main_center, axis=1)
        distance_threshold = np.percentile(distances, 95)  # Keep 95% closest points
        
        # Filter outliers within the cluster
        outlier_mask = distances <= distance_threshold
        filtered_indices = np.where(main_mask)[0][outlier_mask]
        
        # Create final mask
        cluster_mask = np.zeros(len(cluster_labels), dtype=bool)
        cluster_mask[filtered_indices] = True
    
        # for label, stats in cluster_stats.items():
        #     if label == main_cluster_label:
        #         continue
                
        #     # Calculate distance to main cluster
        #     distance_to_main = np.linalg.norm(stats['center'] - main_center)
            
        #     # Keep cluster if it's close enough OR has high average opacity
        #     if distance_to_main <= eps * 2:  # Within 2x threshold
        #         clusters_to_keep.append(label)
        #         print(f"Keeping cluster {label} (distance: {distance_to_main:.3f}, opacity: {stats['mean_opacity']:.3f})")
        #     else:
        #         print(f"Filtering out cluster {label} (distance: {distance_to_main:.3f}, opacity: {stats['mean_opacity']:.3f})")
        
        # # Create final mask
        # cluster_mask = np.isin(cluster_labels, clusters_to_keep)
        print(f"Cluster filtering: keeping {cluster_mask.sum()}/{len(cluster_mask)} gaussians (largest cluster only)")
        # print(f"Cluster filtering: keeping {cluster_mask.sum()}/{len(cluster_mask)} gaussians")
    return torch.tensor(cluster_mask, device=xyz.device)




def to_wandb_img(t: torch.Tensor):
    """
    t: [C,H,W] or [1,3,H,W] float in [0,1] on CUDA
    returns a wandb.Image
    """
    if t.dim() == 4: t = t[0]
    t = t.detach().clamp(0,1).permute(1,2,0).cpu().numpy()  # HWC
    return wandb.Image(t)

def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from
             , wandb_run=None, brics=True, frame_idx=0):
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, brics=brics, frame_idx=frame_idx)
    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    for iteration in range(first_iter, opt.iterations + 1):        
        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam != None:
                    net_image = render(custom_cam, gaussians, pipe, background, scaling_modifer)["render"]
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception as e:
                network_gui.conn = None

        iter_start.record()

        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        bg = torch.rand((3), device="cuda") if opt.random_background else background

        render_pkg = render(viewpoint_cam, gaussians, pipe, bg)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

        # Loss
        mask = zarr.open(viewpoint_cam.mask_path, mode="r")
        mask = mask[viewpoint_cam.time_idx, :, :]
        mask = torch.from_numpy(mask).float() 
        mask = mask.unsqueeze(0)  # (1, 4, H, W)
        gt_image = Image.open(viewpoint_cam.image_path).convert("RGB")
        gt_image = gt_image.resize((viewpoint_cam.image_width, viewpoint_cam.image_height))
        gt_image = torch.from_numpy(np.array(gt_image)).float() / 255.0  # Convert to tensor [0,1]
        gt_image = gt_image.permute(2, 0, 1).unsqueeze(0)  # (D1, 4, H, W)
        gt_image[0, :3, :, :] = gt_image[0, :3, :, :] * mask
        # convert to rgb
        gt_image = gt_image[:, :3, :, :]
        gt_image = gt_image.cuda()
        
        
        Ll1 = l1_loss(image, gt_image)
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))

        if iteration > 4000:
            scaling = gaussians.get_scaling
            scaling_loss = scaling.min(1)[0].mean()
            loss += 100 * scaling_loss

        if iteration > 1000:
            opacities = gaussians.get_opacity
            opacity_loss = (1 - opacities).abs().mean()
            loss += 0.03 * opacity_loss

        if iteration % 1000 == 0:
            scaling = gaussians.get_scaling
            print(scaling.min(1)[0].mean().item(), scaling.median(1)[0].mean().item(), scaling.max(1)[0].mean().item())
        loss.backward()
        if iteration % 10 == 0 and wandb_run is not None:
            wandb_run.log({
                "train/l1": Ll1.item(),
                "train/loss": loss.item(),
                "train/ema_loss": ema_loss_for_log,
                "train/iter_time_ms": iter_start.elapsed_time(iter_end),
                "train/num_points": gaussians.get_xyz.shape[0],
            }, step=iteration)

        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            training_report(tb_writer, wandb_run, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background))
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                
                # Collect unique time values from cameras (for static model, this will typically be just one value)
                train_cams = scene.getTrainCameras()
                test_cams = scene.getTestCameras()
                unique_times = set()
                
                # Collect timestamps from ALL training cameras
                for viewpoint_cam in train_cams:
                    cams = viewpoint_cam if isinstance(viewpoint_cam, list) else [viewpoint_cam]
                    for cam in cams:
                        time_value = getattr(cam, 'time', None)
                        if torch.is_tensor(time_value):
                            time_value = time_value.item()
                        if time_value is not None:
                            unique_times.add(float(time_value))
                
                # Also collect timestamps from test cameras
                for test_cam in test_cams:
                    cams = test_cam if isinstance(test_cam, list) else [test_cam]
                    for cam in cams:
                        time_value = getattr(cam, 'time', None)
                        if torch.is_tensor(time_value):
                            time_value = time_value.item()
                        if time_value is not None:
                            unique_times.add(float(time_value))
                
                unique_times = sorted(list(unique_times))
                print(f"Found {len(unique_times)} unique time values: {unique_times}")
                
                # Compute global mask using brightness filter
                xyz = gaussians.get_xyz
                # Filter by brightness
                rgb_dc = torch.sigmoid(gaussians._features_dc)
                if rgb_dc.dim() == 3:  # e.g., [N,1,3]
                    rgb_dc = rgb_dc.squeeze(1)
                brightness = 0.2126*rgb_dc[:,0] + 0.7152*rgb_dc[:,1] + 0.0722*rgb_dc[:,2]
                bright_th = 0.3
                non_black_mask = brightness > bright_th
                
                # For static model, we can optionally apply clustering filter here
                # Uncomment if you want to use clustering:
                distance_threshold = 0.1
                if non_black_mask.sum() > 0:
                    cluster_mask = detect_and_filter_clusters(
                        xyz[non_black_mask],
                        gaussians.get_opacity[non_black_mask],
                        eps=distance_threshold,
                        min_samples=35
                    )
                    global_mask = torch.zeros(len(xyz), dtype=torch.bool, device=xyz.device)
                    global_mask[non_black_mask] = cluster_mask
                else:
                    global_mask = torch.zeros(len(xyz), dtype=torch.bool, device=xyz.device)
                
                print(f"Global mask: {global_mask.sum().item()}/{len(global_mask)} gaussians kept after filtering")
                
                # For static model, we don't compute deformations, but we can save the mask
                # If you have a deformation network, uncomment below:
                # deformed_params = {}
                # with torch.no_grad():
                #     N_all = gaussians.get_xyz.shape[0]
                #     for time_value in unique_times:
                #         time_tensor = torch.full((N_all, 1), time_value, device=gaussians.get_xyz.device)
                #         means3D, scales, rotations, opacity, _ = gaussians._deformation.forward(
                #             gaussians.get_xyz,
                #             gaussians._scaling,
                #             gaussians._rotation,
                #             gaussians._opacity,
                #             None,
                #             time_tensor
                #         )
                #         d_xyz = means3D - gaussians.get_xyz
                #         d_rotation = rotations - gaussians._rotation
                #         d_scaling = scales - gaussians._scaling
                #         deformed_params[time_value] = (d_xyz, d_rotation, d_scaling)
                
                # Save with optional parameters (scene.save needs to be updated to accept these)
                scene.save(iteration, global_mask=global_mask)
                # If scene.save supports it: scene.save(iteration, deformed_params=deformed_params, global_mask=global_mask)
                
                snap_dir = os.path.join(args.model_path, "point_cloud/iteration_{}".format(iteration))
                if wandb_run is not None:
                    art = wandb.Artifact(f"gaussians-{iteration:06d}", type="point_cloud")
                    p = os.path.join(snap_dir, "point_cloud.ply")
                    if os.path.isfile(p):
                        print(f"Adding file: {p}")
                        art.add_file(p)
                    else:
                        print(f"File not found: {p}")
                    wandb_run.log_artifact(art)

            # Densification
            if iteration < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold)

                # if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                #     gaussians.reset_opacity(ratio=0.1)

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, wandb_run, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                    
                    mask = zarr.open(viewpoint.mask_path, mode="r")
                    mask_frame_idx = getattr(viewpoint, 'mask_frame_idx', None) or getattr(viewpoint, 'time_idx', None)
                    mask = mask[mask_frame_idx, :, :]
                    mask = torch.from_numpy(mask).float() 
                    mask = mask.unsqueeze(0)  # (1, 4, H, W)
                    gt_image = Image.open(viewpoint.image_path).convert("RGB")
                    # gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    gt_image = gt_image.resize((viewpoint.image_width, viewpoint.image_height))
                    gt_image = torch.from_numpy(np.array(gt_image)).float() / 255.0  # Convert to tensor [0,1]
                    gt_image = gt_image.permute(2, 0, 1).unsqueeze(0)  # (D1, 4, H, W)
                    gt_image[0, :3, :, :] = gt_image[0, :3, :, :] * mask
                    # convert to rgb
                    gt_image = gt_image[:, :3, :, :]
                    gt_image = gt_image.cuda()
                    if idx < 5 and wandb_run is not None:   # don't spam
                        wandb_run.log({
                            f"{config['name']}/render_{idx}": to_wandb_img(image),
                            f"{config['name']}/gt_{idx}":     to_wandb_img(gt_image),
                        }, step=iteration)
                        
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])          
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[3000, 7_000, 15_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    parser.add_argument("--frame_idx", type=int, default=0, help="Frame index to train on (single frame)")

    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)
    run = wandb.init(
        project="gs-pull",
        # entity="wanjia_fu",
        name=args.source_path.split("/")[-2] or "run",
        # dir=args.model_path,                # put run files next to your outputs
        # config=vars(args),                  # log all CLI args
        # tags=["3DGS", "deform", "train"],
    )
    print(f"W&B URL:{run.url}")
    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from, 
             wandb_run=run, frame_idx=args.frame_idx)

    # All done
    print("\nTraining complete.")

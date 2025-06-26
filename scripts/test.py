# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.


import os,sys
code_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(f'{code_dir}/../')
import glob
import time
from omegaconf import OmegaConf
from core.utils.utils import InputPadder
from Utils import *
from core.foundation_stereo import *


def load_model(args):
    """Loads the stereo model and checkpoint.

    Args:
        args: Command-line arguments.

    Returns:
        model: The loaded stereo model.
    """
    ckpt_dir = args.ckpt_dir
    cfg = OmegaConf.load(f'{os.path.dirname(ckpt_dir)}/cfg.yaml')
    if 'vit_size' not in cfg:
        cfg['vit_size'] = 'vitl'
    for k in args.__dict__:
        if k not in ['left_file', 'right_file']: # Avoid overwriting constructed paths
             cfg[k] = args.__dict__[k]
    current_args = OmegaConf.create(cfg) # Use a different name to avoid confusion
    logging.info(f"args for model loading:\n{current_args}")
    logging.info(f"Using pretrained model from {ckpt_dir}")

    model = FoundationStereo(current_args)

    ckpt = torch.load(ckpt_dir)
    logging.info(f"ckpt global_step:{ckpt['global_step']}, epoch:{ckpt['epoch']}")
    model.load_state_dict(ckpt['model'])

    model.cuda()
    model.eval()
    return model


def run_inference(model, img0_path, img1_path, args):
    """Runs inference on a pair of images.

    Args:
        model: The stereo model.
        img0_path: Path to the left image.
        img1_path: Path to the right image.
        args: Command-line arguments.

    Returns:
        disp: The disparity map.
        img0_ori: The original left image.
        H: Height of the original image.
        W: Width of the original image.
        inference_time: The time taken for inference.
    """
    img0 = imageio.imread(img0_path)
    img1 = imageio.imread(img1_path)
    if img0.shape[-1] == 4:
        img0 = img0[..., :3]
    if img1.shape[-1] == 4:
        img1 = img1[..., :3]
    scale = args.scale
    assert scale <= 1, "scale must be <=1"
    img0 = cv2.resize(img0, fx=scale, fy=scale, dsize=None)
    img1 = cv2.resize(img1, fx=scale, fy=scale, dsize=None)
    H, W = img0.shape[:2]
    img0_ori = img0.copy()
    logging.info(f"img0: {img0.shape}")
    img0_torch = torch.as_tensor(img0).cuda().float()[None].permute(0, 3, 1, 2)
    img1_torch = torch.as_tensor(img1).cuda().float()[None].permute(0, 3, 1, 2)
    padder = InputPadder(img0_torch.shape, divis_by=32, force_square=False)
    img0_padded, img1_padded = padder.pad(img0_torch, img1_torch)   

    start_time = time.time()
    with torch.cuda.amp.autocast(True):
        if not args.hiera:
            disp = model.forward(img0_padded, img1_padded, iters=args.valid_iters, test_mode=True)
        else:
            disp = model.run_hierachical(img0_padded, img1_padded, iters=args.valid_iters, test_mode=True, small_ratio=0.5)
    inference_time = time.time() - start_time

    disp = padder.unpad(disp.float())
    disp = disp.data.cpu().numpy().reshape(H, W)
    return disp, img0_ori, H, W, inference_time


def save_results(args, disp, img0_ori, H, W):
    """Saves the disparity map visualization and point cloud.

    Args:
        args: Command-line arguments.
        disp: The disparity map.
        img0_ori: The original left image.
        H: Height of the original image.
        W: Width of the original image.
    """
    vis = vis_disparity(disp)
    vis = np.concatenate([img0_ori, vis], axis=1)
    imageio.imwrite(f'{args.out_dir}/{args.basename}_vis.png', vis)
    logging.info(f"Output saved to {args.out_dir}/{args.basename}_vis.png")

    current_disp = disp.copy()
    if args.remove_invisible:
        yy, xx = np.meshgrid(np.arange(current_disp.shape[0]), np.arange(current_disp.shape[1]), indexing='ij')
        us_right = xx - current_disp
        invalid = us_right < 0
        current_disp[invalid] = np.inf

    if args.get_pc:
        with open(args.intrinsic_file, 'r') as f:
            lines = f.readlines()
            K = np.array(list(map(float, lines[0].rstrip().split()))).astype(np.float32).reshape(3, 3)
            baseline = float(lines[1])
        K[:2] *= args.scale
        depth = K[0, 0] * baseline / current_disp
        np.save(f'{args.out_dir}/{args.basename}_depth_meter.npy', depth)
        xyz_map = depth2xyzmap(depth, K)
        pcd = toOpen3dCloud(xyz_map.reshape(-1, 3), img0_ori.reshape(-1, 3))
        keep_mask = (np.asarray(pcd.points)[:, 2] > 0) & (np.asarray(pcd.points)[:, 2] <= args.z_far)
        keep_ids = np.arange(len(np.asarray(pcd.points)))[keep_mask]
        pcd = pcd.select_by_index(keep_ids)
        o3d.io.write_point_cloud(f'{args.out_dir}/{args.basename}_cloud.ply', pcd)
        logging.info(f"PCL saved to {args.out_dir}/{args.basename}_cloud.ply")

        if args.denoise_cloud:
            logging.info("[Optional step] denoise point cloud...")
            cl, ind = pcd.remove_radius_outlier(nb_points=args.denoise_nb_points, radius=args.denoise_radius)
            inlier_cloud = pcd.select_by_index(ind)
            o3d.io.write_point_cloud(f'{args.out_dir}/{args.basename}_cloud_denoise.ply', inlier_cloud)


if __name__=="__main__":
  code_dir = os.path.dirname(os.path.realpath(__file__))
  parser = argparse.ArgumentParser()
  parser.add_argument('--basename_dir', default=f'{code_dir}/../data/', type=str, help='directory of input images, e.g. xxx_left/right.png')
  parser.add_argument('--intrinsic_file', default=f'{code_dir}/../assets/K.txt', type=str, help='camera intrinsic matrix and baseline file')
  parser.add_argument('--ckpt_dir', default=f'{code_dir}/../pretrained_models/23-51-11/model_best_bp2.pth', type=str, help='pretrained model path')
  parser.add_argument('--out_dir', default=f'{code_dir}/../output/', type=str, help='the directory to save results')
  parser.add_argument('--scale', default=1, type=float, help='downsize the image by scale, must be <=1')
  parser.add_argument('--hiera', default=0, type=int, help='hierarchical inference (only needed for high-resolution images (>1K))')
  parser.add_argument('--z_far', default=10, type=float, help='max depth to clip in point cloud')
  parser.add_argument('--valid_iters', type=int, default=32, help='number of flow-field updates during forward pass')
  parser.add_argument('--get_pc', type=int, default=1, help='save point cloud output')
  parser.add_argument('--remove_invisible', default=1, type=int, help='remove non-overlapping observations between left and right images from point cloud, so the remaining points are more reliable')
  parser.add_argument('--denoise_cloud', type=int, default=0, help='whether to denoise the point cloud')
  parser.add_argument('--denoise_nb_points', type=int, default=30, help='number of points to consider for radius outlier removal')
  parser.add_argument('--denoise_radius', type=float, default=0.03, help='radius to use for outlier removal')
  args = parser.parse_args()

  print("Starting test...")
  set_logging_format()
  set_seed(0)
  torch.autograd.set_grad_enabled(False)
  os.makedirs(args.out_dir, exist_ok=True)

  model = load_model(args)

  image_files = sorted(glob.glob(os.path.join(args.basename_dir, "*_left.png")))
  if not image_files:
    logging.warning(f"No XXX_left.png files found in {args.basename_dir}")

  total_inference_time = 0
  inference_count = 0

  for left_file_path in image_files:
    basename = os.path.basename(left_file_path).replace('_left.png', '')
    right_file_path = os.path.join(args.basename_dir, f"{basename}_right.png")

    if not os.path.exists(right_file_path):
        logging.warning(f"Right image {right_file_path} not found for {left_file_path}. Skipping.")
        continue

    # Update args for current pair
    args.left_file = left_file_path
    args.right_file = right_file_path
    args.basename = basename # for save_results

    logging.info(f"Processing pair: {args.left_file} and {args.right_file}")

    disp, img0_ori, H, W, current_inference_time = run_inference(model, args.left_file, args.right_file, args)
    logging.info(f"Inference time for {args.basename}: {current_inference_time:.4f} seconds")
    total_inference_time += current_inference_time
    inference_count += 1
    save_results(args, disp, img0_ori, H, W)

  if inference_count > 0:
    average_inference_time = total_inference_time / inference_count
    logging.info(f"Average inference time over {inference_count} pairs: {average_inference_time:.4f} seconds")
  else:
    logging.info("No image pairs were processed.")

  logging.info("Processing complete.")

import time

from shapely import points

t0 = time.time()
import os,sys
import argparse
import warnings
warnings.filterwarnings("ignore", message="xFormers is not available*")
import logging
#import imageio
import cv2 as cv
import open3d as o3d
from matplotlib import pyplot as plt
import torch
import numpy as np
code_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(f'{code_dir}/../')
from omegaconf import OmegaConf
#from core.utils.utils import InputPadder
from Utils import set_logging_format, set_seed, vis_disparity, depth2xyzmap, toOpen3dCloud
from core.foundation_stereo import FoundationStereo

args = None
model = None
mono_model_type = "MiDaS_small"
mono_model = torch.hub.load("intel-isl/MiDaS", mono_model_type, trust_repo=True)
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
mono_model.to(device)
mono_model.eval()

mono_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
if mono_model_type == "DPT_Large" or mono_model_type == "DPT_Hybrid":
    mono_transform = mono_transforms.dpt_transform
else:
    mono_transform = mono_transforms.small_transform

block_match_model =cv.StereoSGBM_create(numDisparities=3*16, blockSize=15)

index = 1
file_name = ""
src_name = "DepthImages"

calib = np.load('stereo_params_2.npz')
K1, dist1 = calib['K1'], calib['dist1']
K2, dist2 = calib['K2'], calib['dist2']
R1, R2 = calib['R1'], calib['R2']
P1, P2 = calib['P1'], calib['P2']
Q = calib['Q']
ROI1, ROI2 = calib['ROI1'], calib['ROI2']

BASELINE = 0.109

dim_divis_factor = 32

x = max(ROI1[0], ROI2[0])
y = max(ROI1[1], ROI2[1])
w = min(ROI1[0] + ROI1[2], ROI2[0] + ROI2[2]) - x
h = min(ROI1[1] + ROI1[3], ROI2[1] + ROI2[3]) - y

print(f"ROI1: {ROI1}, ROI2: {ROI2}, x: {x}, y: {y}, w: {w}, h: {h}")

w, h = 1280, 720  # Assuming a fixed resolution for the cameras

map1L, map2L = cv.initUndistortRectifyMap(K1, dist1, R1, P1, (w, h), cv.CV_16SC2)
map1R, map2R = cv.initUndistortRectifyMap(K2, dist2, R2, P2, (w, h), cv.CV_16SC2)

key_pressed = {'pressed': False}

def on_key(event):
    # When a key is pressed, set the flag True
    print(f"Key pressed: {event.key}")
    key_pressed['pressed'] = True

fig, ax = plt.subplots(4,2, figsize=(15, 7))
fig.canvas.mpl_connect('key_press_event', on_key)
visualization_image_placeholder = np.zeros((h, w, 3), dtype=np.uint8)
visualization_image = ax[0, 0].imshow(visualization_image_placeholder)
depth_image_placeholder = np.zeros((h, w), dtype=np.uint8)
depth_image = ax[1, 0].imshow(depth_image_placeholder)
disp_image_placeholder = np.zeros((h, w), dtype=np.uint8)
disp_image = ax[0, 1].imshow(disp_image_placeholder, cmap='plasma')
mono_depth_image_placeholder = np.zeros((h, w), dtype=np.uint8)
mono_depth_image = ax[1, 1].imshow(mono_depth_image_placeholder, cmap='plasma')
hist_norm_depth_image = ax[2, 0].imshow(depth_image_placeholder, cmap='plasma')
hist_norm_mono_image = ax[2, 1].imshow(mono_depth_image_placeholder, cmap='plasma')
stereo_edges_image = ax[3, 0].imshow(np.zeros((h, w), dtype=np.uint8), cmap='gray')
mono_edges_image = ax[3, 1].imshow(np.zeros((h, w), dtype=np.uint8), cmap='gray')
plt.show(block=False)

def parse_args():
    global args
    parser = argparse.ArgumentParser()
    parser.add_argument('--left_file', default=f'{code_dir}/../assets/left.png', type=str)
    parser.add_argument('--right_file', default=f'{code_dir}/../assets/right.png', type=str)
    parser.add_argument('--intrinsic_file', default=f'{code_dir}/../assets/K.txt', type=str, help='camera intrinsic matrix and baseline file')
    parser.add_argument('--ckpt_dir', default=f'{code_dir}/../pretrained_models/11-33-40/model_best_bp2.pth', type=str, help='pretrained model path')
    parser.add_argument('--out_dir', default=f'{code_dir}/../output/', type=str, help='the directory to save results')
    parser.add_argument('--scale', default=0.5, type=float, help='downsize the image by scale, must be <=1')
    parser.add_argument('--hiera', default=0, type=int, help='hierarchical inference (only needed for high-resolution images (>1K))')
    parser.add_argument('--z_far', default=10, type=float, help='max depth to clip in point cloud')
    parser.add_argument('--valid_iters', type=int, default=32, help='number of flow-field updates during forward pass')
    parser.add_argument('--get_pc', type=int, default=1, help='save point cloud output')
    parser.add_argument('--remove_invisible', default=1, type=int, help='remove non-overlapping observations between left and right images from point cloud, so the remaining points are more reliable')
    parser.add_argument('--denoise_cloud', type=int, default=1, help='whether to denoise the point cloud')
    parser.add_argument('--denoise_nb_points', type=int, default=30, help='number of points to consider for radius outlier removal')
    parser.add_argument('--denoise_radius', type=float, default=0.03, help='radius to use for outlier removal')
    parser.add_argument('--live', type=bool, default=False, help='whether to run in live mode')
    args = parser.parse_args()
    return args

def pad_to_multiple(img, multiple=32):
    h, w = img.shape[:2]
    pad_h = (multiple - h % multiple) % multiple
    pad_w = (multiple - w % multiple) % multiple
    padded_img = cv.copyMakeBorder(img, 0, pad_h, 0, pad_w, cv.BORDER_REPLICATE)
    return padded_img

def crop_to_multiple(img, multiple=32):
    h, w = img.shape[:2]
    crop_h = h - (h % multiple)
    crop_w = w - (w % multiple)
    cropped_img = img[:crop_h, :crop_w]
    return cropped_img

def rectify_images(left_frame, right_frame):
    global map1L, map2L, map1R, map2R, x, y, w, h
    rectifiedL = cv.remap(left_frame, map1L, map2L, cv.INTER_LINEAR)
    rectifiedR = cv.remap(right_frame, map1R, map2R, cv.INTER_LINEAR)
    # Crop the images to the valid ROI
    rectifiedL = rectifiedL[y:y+h, x:x+w]
    rectifiedR = rectifiedR[y:y+h, x:x+w]

    return rectifiedL, rectifiedR


def get_frame(index = 1):
    global left_frame, right_frame, file_name
    path = "testImages\\"
    #src = "CameraCalibration\\"
    src = "DepthImages\\"
    path = os.path.join(path, src)
    #path = "testImages\\DepthImages\\"
    left = cv.imread(f"{path}right{index}.jpg")
    right = cv.imread(f"{path}left{index}.jpg")

    src_name = src.replace("\\", "")
    file_name = f"{src_name}_frame_{index}"

    if left is None or right is None:
        raise FileNotFoundError(f"Images for index {index} not found.")
    
    left = cv.cvtColor(left, cv.COLOR_BGR2RGB)
    right = cv.cvtColor(right, cv.COLOR_BGR2RGB)
    
    left, right = rectify_images(left, right)
    
    left = cv.resize(left, None, fx=args.scale, fy=args.scale, interpolation=cv.INTER_LINEAR)
    right = cv.resize(right, None, fx=args.scale, fy=args.scale, interpolation=cv.INTER_LINEAR)

    left = crop_to_multiple(left)
    right = crop_to_multiple(right)

    return left, right

def get_precalced(stereo=True):
    global left_frame, right_frame, file_name
    # Load the pre-calibrated disparity map
    src = "test_outputs/foundation/" if stereo else "test_outputs/mono/"
    
    try:
        saved = np.load(f"{src}{src_name}_frame_{index-1}.npy")
    except FileNotFoundError:
        return None

    return saved

def setup_model():  
    global model, args
    ckpt_dir = args.ckpt_dir
    cfg = OmegaConf.load(f'{os.path.dirname(ckpt_dir)}/cfg.yaml')
    if 'vit_size' not in cfg:
        cfg['vit_size'] = 'vitl'
    for k in args.__dict__:
        cfg[k] = args.__dict__[k]
    args = OmegaConf.create(cfg)
    logging.info(f"Using pretrained model from {ckpt_dir}")

    model = FoundationStereo(args)
    print("loading model from", ckpt_dir)
    ckpt = torch.load(ckpt_dir, weights_only=False)
    print("loaded")
    logging.info(f"ckpt global_step:{ckpt['global_step']}, epoch:{ckpt['epoch']}")
    model.load_state_dict(ckpt['model'])

    model.cuda()
    model.eval()

def calc_depth(disp):
    global P1, BASELINE
    depth = P1[0, 0] * BASELINE / (disp + 1e-6)  # Avoid division by zero
    return depth

def create_mesh(pcd):
    # Create a mesh from the point cloud
    pcd.estimate_normals()
    pcd.orient_normals_consistent_tangent_plane(k=30)

    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=9)
    densities = np.asarray(densities)
    density_threshold = np.percentile(densities, 5)
    vertices_to_keep = densities > density_threshold
    mesh.remove_vertices_by_mask(~vertices_to_keep)
    mesh.compute_vertex_normals()
    return mesh, densities

def reverse_normals(mesh):
    normals = np.asarray(mesh.vertex_normals)
    mesh.vertex_normals = o3d.utility.Vector3dVector(-normals)

    # Optionally reverse triangle winding too
    mesh.triangles = o3d.utility.Vector3iVector(np.asarray(mesh.triangles)[:, ::-1])
    return mesh

def create_point_cloud(disp, left_frame, de_noise=True, max_depth=10):
    global P1
    depth = calc_depth(disp)
    xyz = depth2xyzmap(depth, P1)
    pcd = toOpen3dCloud(xyz.reshape(-1, 3), colors=left_frame.reshape(-1, 3))

    keep_mask = (np.asarray(pcd.points)[:,2]>0) & (np.asarray(pcd.points)[:,2]<= max_depth)
    keep_ids = np.arange(len(np.asarray(pcd.points)))[keep_mask]
    pcd = pcd.select_by_index(keep_ids)

    if de_noise:
      cl, ind = pcd.remove_radius_outlier(nb_points=args.denoise_nb_points, radius=args.denoise_radius)
      inlier_cloud = pcd.select_by_index(ind)
      pcd = inlier_cloud

    return pcd

def clean(data, min, max):
    data[data < min] = 0
    data[data > max] = 0  # Remove outliers

    return data

def summary_statistics(data):
    flat = data.flatten()
    mean = np.mean(flat)
    std = np.std(flat)
    min_val = np.min(flat)
    max_val = np.max(flat)
    median = np.median(flat)
    q1 = np.percentile(flat, 25)
    q3 = np.percentile(flat, 75)
    iqr = q3 - q1
    var = np.var(flat)

    print(f"Summary Statistics:\n"
          f"Mean: {mean:.2f}, Std: {std:.2f}, Min: {min_val:.2f}, Max: {max_val:.2f}, "
          f"Median: {median:.2f}, Q1: {q1:.2f}, Q3: {q3:.2f}, IQR: {iqr:.2f}, Variance: {var:.2f}")
    
if __name__ == "__main__":
    
    args = parse_args()
    set_logging_format()
    set_seed(0)
    torch.autograd.set_grad_enabled(False)
    os.makedirs(args.out_dir, exist_ok=True)

    setup_model()

    scale = args.scale
    P1[:2] *= args.scale

    assert scale <= 1, "scale must be <=1"
    
    with torch.no_grad():
        while True:
            print(f"Processing frame index: {index}")
            left_frame, right_frame = get_frame(index)
            if left_frame is None or right_frame is None:
                logging.error("Failed to retrieve frames. Exiting.")
                break
            index += 1
                
            H,W = left_frame.shape[:2]
            left_frame_ori = left_frame.copy()
            right_frame_ori = right_frame.copy()
            # Run the model
            saved_disp = get_precalced(stereo=True)
            if saved_disp is not None:
                disp = saved_disp
                print(f"Using pre-calculated disparity for {file_name}")
            else:
                disp = model.forward(
                    torch.as_tensor(left_frame).cuda().float()[None].permute(0, 3, 1, 2),
                    torch.as_tensor(right_frame).cuda().float()[None].permute(0, 3, 1, 2),
                    iters=args.valid_iters,
                    test_mode=True
                )

                disp = disp.data.cpu().numpy().reshape(H, W)

                np.save(f"test_outputs/foundation/{file_name}", disp)
            saved_mono = get_precalced(stereo=False)
            if saved_mono is not None:
                mono_depth = saved_mono
                print(f"Using pre-calculated mono depth for {file_name}")
            else:
                mono_input = mono_transform(left_frame_ori).to(device)
                mono_depth = mono_model(mono_input)
                mono_depth = torch.nn.functional.interpolate(
                    mono_depth.unsqueeze(1),
                    size=left_frame_ori.shape[:2],
                    mode="bicubic",
                    align_corners=False,
                ).squeeze().cpu().numpy()

                np.save(f"test_outputs/mono/{file_name}", mono_depth)

            # reverse mono depth
            #mono_depth = mono_depth.max() - mono_depth

            #block_match_disp = block_match_model.compute(
            #    cv.cvtColor(left_frame_ori, cv.COLOR_RGB2GRAY),
            #    cv.cvtColor(right_frame_ori, cv.COLOR_RGB2GRAY)
            #).astype(np.float32) / 16.0

            #block_match_disp = cv.resize(block_match_disp, (W, H), interpolation=cv.INTER_LINEAR)

            # show disparity with matplotlib
            depth = calc_depth(disp.copy())
            depth = clean(depth, 0, 20)

            depth_average = np.mean(depth)
            depth_std = np.std(depth)

            depth = clean(depth, depth_average - 2 * depth_std, depth_average + 2 * depth_std)

            mono_average = np.mean(mono_depth)
            mono_std = np.std(mono_depth)
            mono_depth = clean(mono_depth, mono_average - 2 * mono_std, mono_average + 2 * mono_std)

            hist_depth = cv.equalizeHist((depth * 255 / np.max(depth)).astype(np.uint8))
            hist_mono = cv.equalizeHist((mono_depth * 255 / np.max(mono_depth)).astype(np.uint8))

            mono_edges = cv.Canny(mono_depth.astype(np.uint8), 100, 200)
            stereo_edges = cv.Canny(hist_depth.astype(np.uint8), 100, 200)

            visualization_image.set_data(left_frame_ori)
            #disp_image.set_data(block_match_disp.astype(np.uint8))
            mono_depth_image.set_data(mono_depth)
            depth_image.set_data(depth)
            hist_norm_depth_image.set_data(hist_depth)
            hist_norm_mono_image.set_data(hist_mono)
            stereo_edges_image.set_data(stereo_edges.astype(np.uint8))
            mono_edges_image.set_data(mono_edges.astype(np.uint8))

            depth_image.set_clim(vmin=depth.min(), vmax=depth.max())
            disp_image.set_clim(vmin=disp.min(), vmax=disp.max())
            mono_depth_image.set_clim(vmin=mono_depth.min(), vmax=mono_depth.max())
            hist_norm_depth_image.set_clim(vmin=0, vmax=255)
            hist_norm_mono_image.set_clim(vmin=0, vmax=255)
            stereo_edges_image.set_clim(vmin=0, vmax=255)
            mono_edges_image.set_clim(vmin=0, vmax=255)

            #pcd = create_point_cloud(disp, left_frame_ori, de_noise=False, max_depth=10)
            #mesh, densities = create_mesh(pcd)
            #mesh = reverse_normals(mesh)
            #o3d.visualization.draw_geometries([mesh])

            plt.draw()

            key_pressed['pressed'] = False
            while not key_pressed['pressed']:
                plt.pause(0.1)      
            #pcd_vis = o3d.visualization.Visualizer()
            #pcd_vis.create_window()
            #pcd_vis.add_geometry(pcd)
            #pcd_vis.get_render_option().point_size = 1.0
            #pcd_vis.get_render_option().background_color = np.array([0.5, 0.5, 0.5])
            #ctr = pcd_vis.get_view_control()
            #ctr.set_front([0, 0, -1])  # camera looks along -Z
            #ctr.set_up([0, -1, 0])     # optional flip Y
            #ctr.set_lookat([0, 0, 0])  # look at center
            #ctr.set_zoom(0.5)    
            #pcd_vis.run()
            #pcd_vis.destroy_window()



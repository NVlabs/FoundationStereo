"""Rectify a Slamcore EuRoC-style stereo sequence and run FoundationStereo on it, in one go.

Rectification follows slam_utils/scripts/depth_with_autocalibration.py: camera intrinsics and
extrinsics come from the dataset's capture_info.json, and, if the trajectory directory has a
refined_calibration.csv, the camera rotations R_SC are replaced by the last refined ones.

Frames are sub-sampled along the SLAM trajectory with slam_utils' subsample_trajectory()
(a new frame is kept once the body moved more than --subsample_distance_m or rotated more than
--subsample_rotation_deg since the last kept frame), as in integrate_depth_o3d.py.

Output layout (compatible with integrate_depth_o3d.py --depth_dataset_path <out_dir>
--depth_suffix <depth_subdir>):
    <out_dir>/K.txt, <out_dir>/T_CP.txt      rectified intrinsics + baseline, rectification rotation
    <out_dir>/cam0, <out_dir>/cam1           rectified grayscale images (png; --save_cam0 / --save_cam1)
    <out_dir>/<depth_subdir>                 FoundationStereo depth maps (npy, metres)
    <out_dir>/<depth_subdir>_vis             disparity visualisations (if --save_vis)
    <out_dir>/<depth_subdir>_ply             point clouds (if --get_pc 1)
"""
import os, sys
import types

code_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(f"{code_dir}/../")
from omegaconf import OmegaConf
from core.utils.utils import InputPadder
from Utils import *
import pandas
from pathlib import Path
from scipy.spatial.transform import Rotation
from core.foundation_stereo import *
import cv2


def import_slam_utils(slam_utils_dir):
    """Make the slam_utils repo importable from this (foundation_stereo) environment.

    slam_utils modules import python_utils.app_utils.inform_about_app_extras at load time; that
    internal package is usually not installed here, so a no-op stand-in is registered if needed
    (it is only ever called when numpy/cv2 themselves are missing)."""
    sys.path.insert(0, str(slam_utils_dir))
    try:
        import python_utils.app_utils  # noqa: F401
    except ImportError:
        pkg = types.ModuleType("python_utils")
        app_utils = types.ModuleType("python_utils.app_utils")
        app_utils.inform_about_app_extras = lambda *a, **k: None
        pkg.app_utils = app_utils
        sys.modules["python_utils"] = pkg
        sys.modules["python_utils.app_utils"] = app_utils


def read_refined_calibration(calibration_path):
    """Last refined R_SC for cam0 and cam1 (same as in depth_with_autocalibration.py)."""
    df = pandas.read_csv(calibration_path)
    cam0_CS = []
    cam1_CS = []
    for row in df.values:
        sid_index = row[2]
        q_SC_wxyz = row[3:7]
        R_SC = Rotation.from_quat((*q_SC_wxyz[1:], q_SC_wxyz[0])).as_matrix()
        if sid_index == 0:
            cam0_CS.append(R_SC)
        else:
            cam1_CS.append(R_SC)
    return cam0_CS[-1], cam1_CS[-1]


def read_trajectory(trajectory_csv):
    """{timestamp: 4x4 T_WB} from a Slamcore trajectory csv. Uses the FiducialWorld_0 frame if
    present, World_0 otherwise (as integrate_depth_o3d.py does)."""
    df = pandas.read_csv(trajectory_csv, skipinitialspace=True)
    ref_col = df.columns[8]
    ref_frames = df[ref_col].astype(str).str.strip()
    ref = "FiducialWorld_0" if (ref_frames == "FiducialWorld_0").any() else "World_0"
    df = df[ref_frames == ref]
    poses = {}
    for row in df.itertuples(index=False):
        T = np.eye(4)
        T[0:3, 3] = (row[1], row[2], row[3])
        T[0:3, 0:3] = Rotation.from_quat((row[5], row[6], row[7], row[4])).as_matrix()
        poses[int(row[0])] = T
    return poses


def read_euroc_camera(dataset_path, cam_id):
    """{timestamp: image path} from <dataset>/cam<id>/data.csv."""
    cam_dir = Path(dataset_path) / f"cam{cam_id}"
    df = pandas.read_csv(cam_dir / "data.csv", skipinitialspace=True)
    return {int(ts): cam_dir / "data" / str(fname) for ts, fname in zip(df.iloc[:, 0], df.iloc[:, 1])}


def to_gray(img):
    if img.ndim == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


if __name__ == "__main__":
    code_dir = os.path.dirname(os.path.realpath(__file__))
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, type=str,
                        help="EuRoC-style dataset dir with capture_info.json, cam0/, cam1/")
    parser.add_argument("--trajectory_dir", default="", type=str,
                        help="SLAM output dir; refined_calibration.csv is used for rectification "
                        "if present, and the trajectory csv for frame sub-sampling")
    parser.add_argument("--trajectory_file", default="optimised_trajectory_0.csv", type=str,
                        help="trajectory csv name inside --trajectory_dir")
    parser.add_argument("--no_refined_calibration", action="store_true",
                        help="use capture_info.json extrinsics even if refined_calibration.csv exists")
    parser.add_argument("--subsample_distance_m", default=2.0, type=float,
                        help="keep a frame once the body moved more than this since the last kept one")
    parser.add_argument("--subsample_rotation_deg", default=15.0, type=float,
                        help="keep a frame once the body rotated more than this since the last kept one")
    parser.add_argument("--first_n_frames", default=0, type=int,
                        help="if >0, only process the first N frames after sub-sampling")
    parser.add_argument("--dry_run", action="store_true",
                        help="only report how many frames would be processed")
    parser.add_argument("--slam_utils_dir", default=f"{code_dir}/../../slam_utils", type=str,
                        help="path to the slam_utils repository")
    parser.add_argument("--out_dir", required=True, type=str, help="the directory to save results")
    parser.add_argument("--depth_subdir", default="fs_depth", type=str,
                        help="sub-directory of --out_dir for the depth maps")
    parser.add_argument("--save_cam0", action="store_true",
                        help="also save rectified cam0 images (integrate_depth_o3d.py reads them "
                        "for colours)")
    parser.add_argument("--save_cam1", action="store_true",
                        help="also save rectified cam1 images")
    parser.add_argument("--save_vis", action="store_true",
                        help="also save disparity visualisations")
    parser.add_argument(
        "--ckpt_dir",
        default=f"{code_dir}/../pretrained_models/23-51-11/model_best_bp2.pth",
        type=str,
        help="pretrained model path",
    )
    parser.add_argument("--scale", default=1, type=float,
                        help="downsize the image by scale, must be <=1")
    parser.add_argument("--hiera", default=0, type=int,
                        help="hierarchical inference (only needed for high-resolution images (>1K))")
    parser.add_argument("--z_far", default=20, type=float, help="max depth to clip in point cloud")
    parser.add_argument("--valid_iters", type=int, default=32,
                        help="number of flow-field updates during forward pass")
    parser.add_argument("--get_pc", type=int, default=0, help="save point cloud output")
    parser.add_argument(
        "--remove_invisible",
        default=1,
        type=int,
        help="remove non-overlapping observations between left and right images, so the remaining points are more reliable",
    )
    parser.add_argument("--denoise_cloud", type=int, default=1,
                        help="whether to denoise the point cloud")
    parser.add_argument("--denoise_nb_points", type=int, default=5,
                        help="number of points to consider for radius outlier removal")
    parser.add_argument("--denoise_radius", type=float, default=0.2,
                        help="radius to use for outlier removal")
    args = parser.parse_args()

    set_logging_format()
    set_seed(0)
    torch.autograd.set_grad_enabled(False)
    assert args.scale <= 1, "scale must be <=1"

    import_slam_utils(args.slam_utils_dir)
    from slam_utils.dataset_tools.read_capture_info import read_capture_info
    from slam_utils.depth.top_view_generator import get_K_and_distortion
    from slam_utils.depth.stereo_provider import StereoProvider
    from slam_utils.colmap.algorithms import subsample_trajectory

    # ---------------------------------------------------------------- rectification set-up
    dataset_path = Path(args.dataset)
    cameras_params, T_SC0, T_SC1, T_BS, imu_to_cam_shift = read_capture_info(
        f"{dataset_path}/capture_info.json"
    )
    trajectory_dir = Path(args.trajectory_dir) if args.trajectory_dir else None
    if trajectory_dir is not None and not args.no_refined_calibration:
        calib_path = trajectory_dir / "refined_calibration.csv"
        if calib_path.exists():
            R_SC_0, R_SC_1 = read_refined_calibration(calib_path)
            T_SC0[0:3, 0:3] = R_SC_0
            T_SC1[0:3, 0:3] = R_SC_1
            logging.info(f"Using refined camera rotations from {calib_path}")
        else:
            logging.warning(f"{calib_path} not found, using capture_info.json extrinsics")

    K0, distortion0 = get_K_and_distortion(cameras_params[0])
    K1, distortion1 = get_K_and_distortion(cameras_params[1])
    w = cameras_params[0]["w"]
    h = cameras_params[0]["h"]
    stereo_provider = StereoProvider(
        K0, distortion0, K1, distortion1, T_SC0, T_SC1, w, h, 1, is_filtering=False
    )
    K_rect = stereo_provider.K_depth.astype(np.float32)
    baseline = float(stereo_provider.stereo_baseline)
    logging.info(f"Rectified K:\n{K_rect}\nbaseline: {baseline}")

    # ---------------------------------------------------------------- frame selection
    cam0_paths = read_euroc_camera(dataset_path, 0)
    cam1_paths = read_euroc_camera(dataset_path, 1)
    cam1_timestamps = np.array(sorted(cam1_paths.keys()))
    left_timestamps = sorted(cam0_paths.keys())
    if trajectory_dir is not None:
        T_WBs = read_trajectory(trajectory_dir / args.trajectory_file)
        pairs = [(ts, ts) for ts in left_timestamps if ts in T_WBs]
        n_before = len(pairs)
        pairs = subsample_trajectory(
            pairs, T_WBs, args.subsample_distance_m, args.subsample_rotation_deg
        )
        left_timestamps = [ts for ts, _ in pairs]
        logging.info(
            f"Sub-sampled {n_before} frames with poses to {len(left_timestamps)} "
            f"(distance_step={args.subsample_distance_m}m, rotation_step={args.subsample_rotation_deg}deg)"
        )
    if args.first_n_frames > 0:
        left_timestamps = left_timestamps[: args.first_n_frames]
    print(f"To process: {len(left_timestamps)}")
    if args.dry_run:
        sys.exit(0)

    # ---------------------------------------------------------------- outputs
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    depth_dir = out_dir / args.depth_subdir
    depth_dir.mkdir(exist_ok=True)
    cam0_dir = out_dir / "cam0"
    cam1_dir = out_dir / "cam1"
    if args.save_cam0:
        cam0_dir.mkdir(exist_ok=True)
    if args.save_cam1:
        cam1_dir.mkdir(exist_ok=True)
    vis_dir = out_dir / f"{args.depth_subdir}_vis"
    if args.save_vis:
        vis_dir.mkdir(exist_ok=True)
    ply_dir = out_dir / f"{args.depth_subdir}_ply"
    if args.get_pc:
        ply_dir.mkdir(exist_ok=True)

    # K.txt / T_CP.txt in the same format as depth_with_autocalibration.py (full resolution K;
    # integrate_depth_o3d.py rescales it if the depth maps are smaller).
    with open(out_dir / "K.txt", "w") as f:
        f.write(" ".join(str(v) for v in stereo_provider.K_depth.reshape(-1)) + " \n")
        f.write(str(baseline))
    with open(out_dir / "T_CP.txt", "w") as f:
        f.write(" ".join(str(v) for v in stereo_provider.T_CP.reshape(-1)) + " ")

    # Pixels outside the rectified left image's valid area (black borders), dilated: invalid depth.
    valid_rect = cv2.remap(
        np.full((h, w), 255, np.uint8),
        stereo_provider.rectification_x,
        stereo_provider.rectification_y,
        cv2.INTER_NEAREST,
    )
    black_mask = cv2.resize(valid_rect, fx=args.scale, fy=args.scale, dsize=None, interpolation=cv2.INTER_NEAREST) == 0
    kernel = np.ones((20, 20), np.uint8)
    black_mask = cv2.dilate(black_mask.astype(np.uint8) * 255, kernel, iterations=1) > 0

    K_scaled = K_rect.copy()
    K_scaled[:2] *= args.scale

    # ---------------------------------------------------------------- model
    ckpt_dir = args.ckpt_dir
    cfg = OmegaConf.load(f"{os.path.dirname(ckpt_dir)}/cfg.yaml")
    if "vit_size" not in cfg:
        cfg["vit_size"] = "vitl"
    for k in args.__dict__:
        cfg[k] = args.__dict__[k]
    args = OmegaConf.create(cfg)
    logging.info(f"args:\n{args}")
    logging.info(f"Using pretrained model from {ckpt_dir}")
    model = FoundationStereo(args)
    ckpt = torch.load(ckpt_dir)
    logging.info(f"ckpt global_step:{ckpt['global_step']}, epoch:{ckpt['epoch']}")
    model.load_state_dict(ckpt["model"])
    model.cuda()
    model.eval()

    # ---------------------------------------------------------------- processing
    for i, left_ts in enumerate(left_timestamps):
        if (depth_dir / f"{left_ts}.npy").exists():
            logging.info(f"Skipping {left_ts}, already processed")
            continue
        right_ts = int(cam1_timestamps[np.argmin(np.abs(left_ts - cam1_timestamps))])
        img_left = cv2.imread(str(cam0_paths[left_ts]), cv2.IMREAD_UNCHANGED)
        img_right = cv2.imread(str(cam1_paths[right_ts]), cv2.IMREAD_UNCHANGED)
        if img_left is None or img_right is None:
            logging.warning(f"Could not read images for {left_ts} / {right_ts}, skipping")
            continue

        left_rect = cv2.remap(to_gray(img_left), stereo_provider.rectification_x,
                              stereo_provider.rectification_y, cv2.INTER_LINEAR)
        right_rect = cv2.remap(to_gray(img_right), stereo_provider.rectification_r_x,
                               stereo_provider.rectification_r_y, cv2.INTER_LINEAR)
        if args.save_cam0:
            cv2.imwrite(str(cam0_dir / f"{left_ts}.png"), left_rect)
        if args.save_cam1:
            cv2.imwrite(str(cam1_dir / f"{left_ts}.png"), right_rect)

        img0 = np.repeat(left_rect[:, :, None], 3, axis=2)
        img1 = np.repeat(right_rect[:, :, None], 3, axis=2)
        img0 = cv2.resize(img0, fx=args.scale, fy=args.scale, dsize=None)
        img1 = cv2.resize(img1, fx=args.scale, fy=args.scale, dsize=None)
        H, W = img0.shape[:2]
        img0_ori = img0.copy()

        img0 = torch.as_tensor(img0).cuda().float()[None].permute(0, 3, 1, 2)
        img1 = torch.as_tensor(img1).cuda().float()[None].permute(0, 3, 1, 2)
        padder = InputPadder(img0.shape, divis_by=32, force_square=False)
        img0, img1 = padder.pad(img0, img1)

        with torch.cuda.amp.autocast(True):
            if not args.hiera:
                disp = model.forward(img0, img1, iters=args.valid_iters, test_mode=True)
            else:
                disp = model.run_hierachical(
                    img0, img1, iters=args.valid_iters, test_mode=True, small_ratio=0.5
                )
        disp = padder.unpad(disp.float())
        disp = disp.data.cpu().numpy().reshape(H, W)
        disp[black_mask] = np.inf

        if args.remove_invisible:
            yy, xx = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
            disp[(xx - disp) < 0] = np.inf

        depth = K_scaled[0, 0] * baseline / disp
        np.save(depth_dir / f"{left_ts}.npy", depth)
        n_valid = int(np.sum((depth > 0.1) & (depth < args.z_far - 0.1)))
        print(f"[{i + 1}/{len(left_timestamps)}] {left_ts}: valid depth pixels {n_valid}")

        if args.save_vis:
            vis = vis_disparity(disp)
            vis = np.concatenate([img0_ori, vis], axis=1)
            imageio.imwrite(vis_dir / f"{left_ts}.png", vis)

        if args.get_pc:
            xyz_map = depth2xyzmap(depth, K_scaled)
            pcd = toOpen3dCloud(xyz_map.reshape(-1, 3), img0_ori.reshape(-1, 3))
            pts_z = np.asarray(pcd.points)[:, 2]
            keep_ids = np.arange(len(pts_z))[(pts_z > 0) & (pts_z <= args.z_far)]
            pcd = pcd.select_by_index(keep_ids)
            if args.denoise_cloud:
                _, ind = pcd.remove_radius_outlier(
                    nb_points=args.denoise_nb_points, radius=args.denoise_radius
                )
                pcd = pcd.select_by_index(ind)
            o3d.io.write_point_cloud(str(ply_dir / f"{left_ts}.ply"), pcd)

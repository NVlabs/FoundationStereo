# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import os
import sys
import json
import logging
import argparse
import numpy as np
import torch
import torch.multiprocessing as mp
import cv2
import imageio
from omegaconf import OmegaConf
import open3d as o3d
from scipy.spatial.transform import Rotation as R
from tqdm.auto import tqdm
import random

code_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(f'{code_dir}/../')

from core.utils.utils import InputPadder
from core.foundation_stereo import FoundationStereo

logging.basicConfig(level=logging.INFO, format='[%(filename)s:%(funcName)s()] %(message)s', datefmt='%m-%d|%H:%M:%S')

def set_seed(random_seed):
    np.random.seed(random_seed)
    random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class CuSFMDataInference:
    def __init__(self, args):
        self.args = args
        self.device = torch.device(f'cuda:{args.rank}' if torch.cuda.is_available() else 'cpu')
        if torch.cuda.is_available():
            torch.cuda.set_device(self.device)
        self.load_metadata()
        self.load_model()

    def load_metadata(self):
        with open(self.args.metadata_file, 'r') as f:
            camera_meta = json.load(f)

        self.camera_params_id_to_camera_params = camera_meta['camera_params_id_to_camera_params']
        self.stereo_pairs = camera_meta["stereo_pair"]
        for stereo_pair in self.stereo_pairs:
            if 'left_camera_param_id' not in stereo_pair:
                stereo_pair['left_camera_param_id'] = '0'

        self.camera_keyframes = {}
        for keyframe in camera_meta['keyframes_metadata']:
            if 'camera_params_id' not in keyframe:
                keyframe['camera_params_id'] = '0'
            if keyframe['camera_params_id'] not in self.camera_keyframes:
                self.camera_keyframes[keyframe['camera_params_id']] = {}
            self.camera_keyframes[keyframe['camera_params_id']][keyframe['synced_sample_id']] = keyframe

    def load_model(self):
        cfg = OmegaConf.load(f'{os.path.dirname(self.args.ckpt_dir)}/cfg.yaml')
        # Set default vit_size if not present in config
        if 'vit_size' not in cfg:
            cfg['vit_size'] = 'vitl'
        self.model = FoundationStereo(cfg)
        ckpt = torch.load(self.args.ckpt_dir, map_location=self.device)
        self.model.load_state_dict(ckpt['model'])
        self.model.to(self.device)
        self.model.eval()

    def get_camera_params(self, camera_param_id: str):
        if camera_param_id not in self.camera_params_id_to_camera_params:
            raise ValueError(f"No camera data found for camera name {camera_param_id}")
        return self.camera_params_id_to_camera_params[camera_param_id]

    def get_projection_matrix(self, camera_params):
        projection_matrix_data = camera_params['calibration_parameters']['projection_matrix']['data']
        projection_matrix = np.array(projection_matrix_data).reshape((3, 4))
        return projection_matrix

    def get_camera_transform(self, camera_param_id: str):
        camera_params = self.camera_params_id_to_camera_params[camera_param_id]
        translation = camera_params['sensor_meta_data']['sensor_to_vehicle_transform']['translation']
        translation_vector = np.array([translation['x'], translation['y'], translation['z']])
        rotation = camera_params['sensor_meta_data']['sensor_to_vehicle_transform'].get('axis_angle')
        if rotation:
            axis = [rotation['x'], rotation['y'], rotation['z']]
            angle = np.deg2rad(rotation['angle_degrees'])
            rotation_matrix = R.from_rotvec(np.array(axis) * angle).as_matrix()
        else:
            rotation_matrix = np.eye(3)
        return translation_vector, rotation_matrix

    def compute_baseline(self, left_id, right_id):
        translation_1, rotation_1 = self.get_camera_transform(left_id)
        translation_2, rotation_2 = self.get_camera_transform(right_id)

        T_camera_1_vehicle = np.eye(4)
        T_camera_1_vehicle[:3, :3] = rotation_1
        T_camera_1_vehicle[:3, 3] = translation_1

        T_camera_2_vehicle = np.eye(4)
        T_camera_2_vehicle[:3, :3] = rotation_2
        T_camera_2_vehicle[:3, 3] = translation_2

        T_left_to_right_transform = np.linalg.inv(T_camera_2_vehicle) @ T_camera_1_vehicle
        baseline = -T_left_to_right_transform[0, 3]

        if baseline == 0:
            raise ValueError("The computed baseline is zero, which is invalid for depth calculation.")

        print(f"Baseline between camera {left_id} and camera {right_id}: {baseline}")
        return baseline

    def get_output_image_file(self, output_folder, left_image_name):
        left_image_name = left_image_name.split('.')[0]
        return f'{output_folder}/{left_image_name}.png'

    def depth2xyzmap(self, depth, K):
        H, W = depth.shape
        x, y = np.meshgrid(np.arange(W), np.arange(H))
        xyz = np.stack((x, y, np.ones_like(x)), axis=-1)
        xyz = (xyz @ np.linalg.inv(K).T) * depth[..., None]
        return xyz

    def toOpen3dCloud(self, xyz, rgb):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyz)
        pcd.colors = o3d.utility.Vector3dVector(rgb / 255.0)
        return pcd

    def process_camera(self, left_param_id: str):
        camera_params = self.get_camera_params(left_param_id)
        camera = camera_params['sensor_meta_data']['sensor_name']
        print('Processing camera_param_id: ', left_param_id, 'sensor_name: ', camera)

        if self.args.get_pc:
            os.makedirs(f'{self.args.out_dir}/pcd/{camera}', exist_ok=True)

        K = self.get_projection_matrix(camera_params)[:3, :3]
        print("Camera matrix for camera: ", camera, " is \n", K)

        baseline = 0
        right_param_id = None
        for stereo_pair in self.stereo_pairs:
            if left_param_id == stereo_pair['left_camera_param_id']:
                right_param_id = stereo_pair['right_camera_param_id']
                baseline = stereo_pair['baseline_meters']
                break

        if not right_param_id:
            raise ValueError(f"No stereo pair found for camera {camera} with id {left_param_id}")

        if baseline == 0:
            baseline = self.compute_baseline(left_param_id, right_param_id)

        left_camera_keyframes = self.camera_keyframes[left_param_id]
        right_camera_keyframes = self.camera_keyframes[right_param_id]

        if not left_camera_keyframes:
            logging.info(f"No keyframes found for camera {camera}")
            return
        if not right_camera_keyframes:
            logging.info(f"No keyframes found for right camera of {camera}")
            return

        create_dir = True
        for synced_sample_id, left_keyframe in tqdm(
                left_camera_keyframes.items(),
                desc=f"Estimating depth for {camera}", leave=True):
            if synced_sample_id not in right_camera_keyframes:
                logging.info(
                    f"No corresponding right keyframe found for {left_keyframe['image_name']} "
                    f"with synced_sample_id {synced_sample_id}"
                )
                continue

            right_keyframe = right_camera_keyframes[synced_sample_id]

            # make sure output folder for camera exists
            if create_dir:
                os.makedirs(
                    os.path.dirname(
                        self.get_output_image_file(
                            f'{self.args.out_dir}/scaled_0_4',
                            left_keyframe['image_name'])),
                    exist_ok=True)
                os.makedirs(
                    os.path.dirname(
                        self.get_output_image_file(
                            f'{self.args.out_dir}/original_size',
                            left_keyframe['image_name'])),
                    exist_ok=True)
                if getattr(self.args, "also_generate_for_right_camera", False):
                    os.makedirs(
                        os.path.dirname(
                            self.get_output_image_file(
                                f'{self.args.out_dir}/scaled_0_4',
                                right_keyframe['image_name'])),
                        exist_ok=True)
                    os.makedirs(
                        os.path.dirname(
                            self.get_output_image_file(
                                f'{self.args.out_dir}/original_size',
                                right_keyframe['image_name'])),
                        exist_ok=True)
                create_dir = False

            left_file = os.path.join(
                self.args.imgdir, left_keyframe['image_name'])
            right_file = os.path.join(
                self.args.imgdir, right_keyframe['image_name'])

            if not os.path.exists(left_file) or not os.path.exists(right_file):
                logging.info(
                    f"Image not found for {left_file} or {right_file}")
                continue

            img0 = imageio.imread(left_file)
            img1 = imageio.imread(right_file)
            H_big, W_big = img0.shape[:2]
            scale = self.args.scale
            img0 = cv2.resize(img0, fx=scale, fy=scale, dsize=None)
            img1 = cv2.resize(img1, fx=scale, fy=scale, dsize=None)
            H, W = img0.shape[:2]

            img0_tensor = torch.as_tensor(img0).to(
                self.device).float()[None].permute(0, 3, 1, 2)
            img1_tensor = torch.as_tensor(img1).to(
                self.device).float()[None].permute(0, 3, 1, 2)

            padder = InputPadder(
                img0_tensor.shape, divis_by=32, force_square=False)
            img0_tensor, img1_tensor = padder.pad(img0_tensor, img1_tensor)

            img0_tensor = img0_tensor.contiguous()
            img1_tensor = img1_tensor.contiguous()

            with torch.cuda.amp.autocast(True):
                disp = self.model.forward(
                    img0_tensor,
                    img1_tensor,
                    iters=self.args.valid_iters,
                    test_mode=True)
                disp = padder.unpad(disp.float())

                if getattr(self.args, "also_generate_for_right_camera", False):
                    img0_flipped_tensor = torch.flip(img0_tensor, dims=[3])
                    img1_flipped_tensor = torch.flip(img1_tensor, dims=[3])
                    disp2 = self.model.forward(
                        img1_flipped_tensor,
                        img0_flipped_tensor,
                        iters=self.args.valid_iters,
                        test_mode=True)
                    disp2 = torch.flip(disp2, dims=[3])
                    disp2 = padder.unpad(disp2.float())

            disp = disp.data.cpu().numpy().reshape(H, W)
            disp_big = cv2.resize(
                disp, (W_big, H_big), interpolation=cv2.INTER_NEAREST)

            # Convert disparity to depth in (mm)
            doffs = 0
            depth = 1000 * scale * K[0, 0] * baseline / (disp + doffs)
            depth_big = 1000 * scale * K[0, 0] * baseline / (disp_big + doffs)

            depth[depth > 65535] = 0
            depth_big[depth_big > 65535] = 0

            imageio.imwrite(
                self.get_output_image_file(
                    f'{self.args.out_dir}/scaled_0_4',
                    left_keyframe['image_name']), depth.astype(np.uint16))
            imageio.imwrite(
                self.get_output_image_file(
                    f'{self.args.out_dir}/original_size',
                    left_keyframe['image_name']), depth_big.astype(np.uint16))

            if getattr(self.args, "also_generate_for_right_camera", False):
                disp2 = disp2.data.cpu().numpy().reshape(H, W)
                disp2_big = cv2.resize(
                    disp2, (W_big, H_big), interpolation=cv2.INTER_NEAREST)

                depth2 = 1000 * scale * K[0, 0] * baseline / (disp2 + doffs)
                depth2_big = 1000 * scale * K[0, 0] * baseline / (
                    disp2_big + doffs)

                depth2[depth2 > 65535] = 0
                depth2_big[depth2_big > 65535] = 0

                imageio.imwrite(
                    self.get_output_image_file(
                        f'{self.args.out_dir}/scaled_0_4',
                        right_keyframe['image_name']),
                    depth2.astype(np.uint16))
                imageio.imwrite(
                    self.get_output_image_file(
                        f'{self.args.out_dir}/original_size',
                        right_keyframe['image_name']),
                    depth2_big.astype(np.uint16))

            if self.args.get_pc:
                K_copy = K.copy()
                K_copy[:2] *= scale
                xyz_map = self.depth2xyzmap(depth, K_copy)
                pcd = self.toOpen3dCloud(
                    xyz_map.reshape(-1, 3), img0.reshape(-1, 3))
                keep_mask = (np.asarray(pcd.points)[:, 2] > 0) & (
                    np.asarray(pcd.points)[:, 2] <= self.args.z_far)
                keep_ids = np.arange(len(np.asarray(pcd.points)))[keep_mask]
                pcd = pcd.select_by_index(keep_ids)
                o3d.io.write_point_cloud(
                    f'{self.args.out_dir}/pcd/{camera}/{os.path.basename(left_file)[:-4]}.ply',
                    pcd)

def is_left_camera(camera_name: str):
    return camera_name.endswith("_left") or camera_name.endswith("left_rgb")

def main(rank, world_size, args):
    args.rank = rank
    set_seed(0)
    torch.autograd.set_grad_enabled(False)

    sfm_inference = CuSFMDataInference(args)

    cameras = []
    if getattr(args, "camera", None) is not None:
        # Support --camera (single camera id) for subprocess usage
        cameras = [args.camera]
    elif getattr(args, "cameras", None) is not None:
        cameras = [cam for cam in args.cameras.split(',') if cam != ""]

    camera_param_ids = []
    for cam_id, cam_params in sfm_inference.camera_params_id_to_camera_params.items():
        cam_name = cam_params['sensor_meta_data']['sensor_name']
        if (len(cameras) == 0 or cam_name in cameras or cam_id in cameras) and is_left_camera(cam_name):
            camera_param_ids.append(cam_id)

    # Distribute cameras across GPUs
    cameras_per_gpu = len(camera_param_ids) // world_size
    start_idx = rank * cameras_per_gpu
    end_idx = start_idx + cameras_per_gpu if rank < world_size - 1 else len(camera_param_ids)
    assigned_camera_param_ids = camera_param_ids[start_idx:end_idx]

    print('Camera param ids to process on rank: ', rank, ' are: ', assigned_camera_param_ids)

    for camera in assigned_camera_param_ids:
        sfm_inference.process_camera(camera)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--imgdir',
        required=True,
        type=str,
        help='the directory to load camera images')
    parser.add_argument(
        '--logdir',
        default='./debug',
        type=str,
        help='the directory to save logs and checkpoints')
    parser.add_argument(
        '--ckpt_dir', default='../pretrained_models/23-51-11/model_best_bp2.pth', type=str)
    parser.add_argument('--run_name', default='2024-09-19-21-36-31', type=str)
    parser.add_argument('--out_dir', required=True, type=str)
    parser.add_argument('--scale', default=0.4, type=float)
    parser.add_argument(
        '--valid_iters',
        type=int,
        default=32,
        help='number of flow-field updates during forward pass')
    parser.add_argument(
        '--get_pc', type=bool, default=False, help='get point cloud output')
    parser.add_argument(
        '--z_far',
        default=10,
        type=float,
        help='max depth to clip in point cloud')
    parser.add_argument(
        '--cameras', default=None, help='comma-separated camera names')
    parser.add_argument(
        '--camera', default=None, type=str, help='single camera id (for subprocess usage)')
    parser.add_argument(
        '--metadata_file',
        required=True,
        type=str,
        help='path to the metadata json file')
    parser.add_argument(
        '--num_gpus', type=int, default=1, help='number of GPUs to use')
    parser.add_argument(
        '--also_generate_for_right_camera',
        type=bool,
        default=False,
        help='whether to generate depth for right camera')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.logdir, exist_ok=True)

    # to avoid error: rate limit exceeded
    torch.hub._validate_not_a_forked_repo = lambda a, b, c: True

    if args.num_gpus > 1:
        mp.spawn(
            main, args=(args.num_gpus, args), nprocs=args.num_gpus, join=True)
    else:
        main(0, 1, args)

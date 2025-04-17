import argparse
import json
from pathlib import Path

from more_itertools import one
import av
import cv2
from PIL import Image
import numpy as np

IMG_DIM = 1024

def parse_arguments():
    parser = argparse.ArgumentParser(description="Extract stereo images from a video.")
    parser.add_argument("--video", type=str, required=True, help="Path to the input video file.")
    parser.add_argument("--ts", type=float, required=True, help="Time interval (in seconds) to extract frames.")
    parser.add_argument("--camera_info", type=str, required=True, help="Camera info file (shm_raw_camera_info.json)")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the extracted images.")
    parser.add_argument("--output_size", type=int, default=IMG_DIM, help="Size of output image.")
    return parser.parse_args()


def info_to_camera_matrix(info):
    width_scale = IMG_DIM / info["image_width"]
    crop_offset = info["image_height"] - info["image_width"]  # NEO is bottom crop

    # Extract camera intrinsic parameters from the info dictionary
    fx = info["camera_matrix"]["f"][0] * width_scale
    fy = info["camera_matrix"]["f"][1] * width_scale
    cx = info["camera_matrix"]["c"][0] * width_scale
    cy = (info["camera_matrix"]["c"][1] - crop_offset) * width_scale

    # Create the camera matrix
    K = np.array([[fx, 0, cx],
                  [0, fy, cy],
                  [0, 0, 1]], dtype=np.float32)

    D = np.array([
        info["distortion_model"]["k"][0],
        info["distortion_model"]["k"][1],
        info["distortion_model"]["k"][2],
        info["distortion_model"]["k"][3],
    ], dtype=np.float32)

    return K, D


def fisheye_to_rectilinear(fisheye_image, fish_K, fish_D, rect_size, rect_camera, rotation_angle=np.pi / 6, rotation_axis=np.array([1, 0, 0])):
    # Create a rotation matrix using Rodrigues' formula
    rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)  # Normalize the rotation vector
    # Rotation between the fisheye and rectilinear camera
    R = cv2.Rodrigues(rotation_angle * rotation_axis)[0]

    # Map the fisheye image to a rectilinear view
    # NOTE: Create this map once and use it for all images of the same size
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(fish_K, fish_D, R, rect_camera, rect_size, cv2.CV_32F)
    rectilinear_image = cv2.remap(fisheye_image, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

    return rectilinear_image


if __name__ == "__main__":
    args = parse_arguments()

    # Decode video frame
    print(f"Decoding {args.video} at timestamp {args.ts} sec...")
    container = av.open(args.video)
    stream = one(container.streams.video)
    ctx = av.Codec(stream.codec_context.codec.name, "r").create()
    ctx.extradata = stream.codec_context.extradata

    image = None
    for packet in container.demux(stream):
        for frame in ctx.decode(packet):
            frame_s = float(frame.pts * stream.time_base)
            if frame_s >= args.ts:
                image = frame.to_image()
                print(f"Extracted frame at {args.ts} seconds.")
                break
        if image is not None:
            break

    if image is None:
        print("No frame found at the specified timestamp.")

    # Split left and right images
    width, height = image.size
    left_image = np.array(image.crop((0, 0, width, height // 2)))
    right_image = np.array(image.crop((0, height // 2, width, height)))

    # Rectify images
    with open(args.camera_info, "r") as f:
        camera_info = json.load(f)

    image_size = (args.output_size, args.output_size)
    K_out = np.array([[args.output_size//2, 0, args.output_size//2],
                        [0, args.output_size//2, args.output_size//2],
                        [0, 0, 1]], dtype=np.float32)
    K_left, D_left = info_to_camera_matrix(camera_info["left"])
    K_right, D_right = info_to_camera_matrix(camera_info["right"])

    left_image_rect = fisheye_to_rectilinear(left_image, K_left, D_left, image_size, K_out)
    right_image_rect = fisheye_to_rectilinear(right_image, K_right, D_right, image_size, K_out)

    # Save to file
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    Image.fromarray(left_image_rect).save(f"{output_path}/left.jpg")
    Image.fromarray(right_image_rect).save(f"{output_path}/right.jpg")
    print(f"Output directory: {args.output_dir}")

    # Save intrinsics file
    K_txt = f"{' '.join(map(str, K_out.flatten()))}"
    camera_dist_txt = str(np.linalg.norm(camera_info["left_to_right_translation"]))
    intrinsics_txt = K_txt + "\n" + camera_dist_txt
    with open(output_path / "intrinsics.txt", "w") as f:
        f.write(intrinsics_txt)
    print(f"Saved intrinsics to {output_path / 'intrinsics.txt'}")

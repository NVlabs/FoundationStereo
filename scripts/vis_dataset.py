import os,sys
code_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(f'{code_dir}/../')
import argparse
import imageio
from Utils import depth_uint8_decoding, vis_disparity


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--dataset_path", type=str, default="./DATA/sample/manipulation_v5_realistic_kitchen_2500_1/dataset/data/")
  args = parser.parse_args()

  root = args.dataset_path
  left = imageio.imread(f'{root}/left/rgb/0000.jpg')
  right = imageio.imread(f'{root}/right/rgb/0000.jpg')
  disp = depth_uint8_decoding(imageio.imread(f'{root}/left/disparity/0000.png'))
  vis = vis_disparity(disp)

  import matplotlib.pyplot as plt

  # Create a figure with 3 subplots
  fig, axes = plt.subplots(1, 3, figsize=(15, 5))

  # Display images
  axes[0].imshow(left)
  axes[0].set_title('Left Image')
  axes[0].axis('off')

  axes[1].imshow(right)
  axes[1].set_title('Right Image')
  axes[1].axis('off')

  axes[2].imshow(vis)
  axes[2].set_title('Disparity Visualization')
  axes[2].axis('off')

  plt.tight_layout()
  plt.show()

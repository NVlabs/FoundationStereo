import cv2 as cv
import numpy as np

from matplotlib import pyplot as plt

key_pressed = {'pressed': False}
calib = np.load('stereo_params_2.npz')
K1, dist1 = calib['K1'], calib['dist1']
K2, dist2 = calib['K2'], calib['dist2']
R1, R2 = calib['R1'], calib['R2']
P1, P2 = calib['P1'], calib['P2']
Q = calib['Q']
ROI1, ROI2 = calib['ROI1'], calib['ROI2']

BASELINE = 0.109

SCALE = 0.5

P1[:2] *= SCALE

def on_key(event):
    # When a key is pressed, set the flag True
    print(f"Key pressed: {event.key}")
    key_pressed['pressed'] = True

def clean(data, min=0, max=20):
    data[data < min] = 0
    data[data > max] = 0  # Remove outliers

    return data

def get_images(index=0, path="testImages/DepthImages/"):
    right = cv.imread(f"{path}left{index}.jpg")
    left = cv.imread(f"{path}right{index}.jpg")

    return left, right

def flat_areas(matrix, threshold=0.1):
    # Compute gradient magnitude
    gy, gx = np.gradient(matrix.astype(np.float32))
    grad_mag = np.sqrt(gx**2 + gy**2)
    return grad_mag

def remove_flat_areas(matrix, threshold=0):
    flat_areas_mask = flat_areas(matrix) <= threshold
    matrix[flat_areas_mask] = 0  # Set flat areas to 0
    return matrix

def get_pre_calced(index=0, src_name="DepthImages"):
    path = "test_outputs/" 
    stereo_path = f"{path}foundation/{src_name}_frame_{index}.npy"
    mono_path = f"{path}mono/{src_name}_frame_{index}.npy"

    stereo_disp = np.load(stereo_path)
    stereo = disp_to_depth(stereo_disp)
    mono = np.load(mono_path)
    mono = mono_to_sudo_depth(mono)
    return stereo, mono

def mono_to_sudo_depth(mono):
    mono = mono.max() - mono
    return mono

def disp_to_depth(disp):
    # Convert disparity map to depth map
    depth = np.zeros_like(disp)
    valid_disp = disp > 0
    depth[valid_disp] = (P1[0, 0] * BASELINE) / disp[valid_disp]
    depth = clean(depth, min=0.1, max=20.0)  # Clean the depth map
    return depth

def normalize(matrix):
    # Normalize the matrix to the range [0, 1]
    min_val = np.nanmin(matrix)
    max_val = np.nanmax(matrix)
    if max_val - min_val > 0:
        normalized_matrix = (matrix - min_val) / (max_val - min_val)
    else:
        normalized_matrix = np.zeros_like(matrix)  # If all values are the same
    return normalized_matrix

def normalize_depths(stereo, mono):
    stereo = normalize(stereo)
    mono = normalize(mono)
    return stereo, mono

def compare_depths(stereo, mono):

    # Calculate the difference
    diff = np.abs(stereo - mono)

    # Normalize the difference for visualization
    diff_normalized = normalize(diff)

    return diff_normalized

def histogram_equalization(stereo, mono):
    flat_stereo = stereo.flatten()
    flat_mono = mono.flatten()

    hist_stereo, bins_stereo = np.histogram(flat_stereo, bins=256, range=(0, 1))
    hist_mono, bins_mono = np.histogram(flat_mono, bins=256, range=(0, 1))
    cdf_stereo = hist_stereo.cumsum()
    cdf_mono = hist_mono.cumsum()
    cdf_stereo_normalized = cdf_stereo / cdf_stereo[-1]
    cdf_mono_normalized = cdf_mono / cdf_mono[-1]
    equalized_stereo = np.interp(flat_stereo, bins_stereo[:-1], cdf_stereo_normalized)
    equalized_mono = np.interp(flat_mono, bins_mono[:-1], cdf_mono_normalized)
    stereo[:] = equalized_stereo.reshape(stereo.shape)
    mono[:] = equalized_mono.reshape(mono.shape)

    return stereo, mono


fig, ax = plt.subplots(4, 2, figsize=(10, 5))
ax[0, 0].set_title("Left Image")
ax[0, 1].set_title("Right Image")
ax[1, 0].set_title("Stereo Depth Map")
ax[1, 1].set_title("Mono Depth Map")
ax[2, 0].set_title("Normalized Stereo")
ax[2, 1].set_title("Normalized Mono")

for axy in ax.ravel():
    axy.axis('off')


plt.connect('key_press_event', on_key)
index = 1
src_name = "DepthImages"
#src_name = "CameraCalibration"
while True:
    try:
        left, right = get_images(index, path=f"testImages/{src_name}/")
        stereo, mono = get_pre_calced(index, src_name=src_name)
        stereo, mono = remove_flat_areas(stereo.copy()), remove_flat_areas(mono.copy())
        stereo_norm, mono_norm = normalize_depths(stereo.copy(), mono.copy())
        diff = compare_depths(stereo_norm.copy(), mono_norm.copy())
        ax[0, 0].imshow(left)
        ax[0, 1].imshow(right)
        ax[1, 0].imshow(stereo, cmap='gray')
        ax[1, 1].imshow(mono, cmap='gray')
        ax[2, 0].imshow(stereo_norm, cmap='plasma')
        ax[2, 1].imshow(mono_norm, cmap='plasma')
        ax[3, 0].imshow(diff, cmap='hot')
        ax[3, 1].imshow(flat_areas(mono_norm) <= 0, cmap='hot')

        index += 1
        plt.draw()

        key_pressed['pressed'] = False
        while not key_pressed['pressed']:
            plt.pause(0.1)
    except FileNotFoundError:
        print(f"Files not found for index {index}. Exiting.")
        break

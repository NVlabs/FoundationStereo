import os
import re
import cv2
import sys

def is_blurry(image_path, threshold=100.0, show=False):
    """
    Checks if an image is blurry based on the Laplacian variance.
    
    Args:
        image_path (str): Path to the input image.
        threshold (float): Threshold below which the image is considered blurry.
        show (bool): Whether to display the image and blur score.

    Returns:
        (bool, float): Tuple of (is_blurry, blur_score)
    """
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Image not found: {image_path}")

    laplacian = cv2.Laplacian(image, cv2.CV_64F)
    variance = laplacian.var()

    if show:
        print(f"Blur score: {variance:.2f}")
        cv2.imshow("Image", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return (variance < threshold), variance

def get_matching_files(directory, pattern):
    """
    Returns a list of file paths in the directory that match the regex pattern.
    
    Args:
        directory (str): Path to the directory.
        pattern (str): Regex pattern to match file names.

    Returns:
        List[str]: List of full file paths.
    """
    regex = re.compile(pattern)
    matching_files = []

    for filename in os.listdir(directory):
        if os.path.isfile(os.path.join(directory, filename)) and regex.match(filename):
            matching_files.append(os.path.join(directory, filename))
    
    return matching_files

def remove_blurry_images(directory, pattern, threshold=100.0):
    """
    Removes blurry images from the directory based on the regex pattern.
    
    Args:
        directory (str): Path to the directory containing images.
        pattern (str): Regex pattern to match file names.
        threshold (float): Threshold below which an image is considered blurry.
    """
    matching_files = get_matching_files(directory, pattern)

    for image_path in matching_files:
        try:
            blurry, score = is_blurry(image_path, threshold=threshold)
            if blurry:
                os.remove(image_path)
                print(f"Removed blurry image: {image_path} (score: {score:.2f})")
            else:
                print(f"Kept sharp image: {image_path} (score: {score:.2f})")
        except ValueError as e:
            print(e)

def remove_unmatched_images(directory, pattern1, pattern2):
    """
    Removes images from the directory that do not have a matching counterpart based on two regex patterns.
    
    Args:
        directory (str): Path to the directory containing images.
        pattern1 (str): Regex pattern for the first set of images.
        pattern2 (str): Regex pattern for the second set of images.
    """
    files1 = get_matching_files(directory, pattern1)
    files2 = get_matching_files(directory, pattern2)

    indices1 = {int(re.search(r'\d+', os.path.basename(f)).group()) for f in files1}
    indices2 = {int(re.search(r'\d+', os.path.basename(f)).group()) for f in files2}

    common_indices = indices1 & indices2

    for file in files1 + files2:
        index = int(re.search(r'\d+', os.path.basename(file)).group())
        if index not in common_indices:
            print(f"Removing unmatched image: {file}")
            os.remove(file)

# Example usage:
if __name__ == "__main__":
    path = "testImages/CameraCalibration/"
    # rename files to match a pattern
    images_right = get_matching_files(path, r"CalImRight\d+\.jpg")
    images_left = get_matching_files(path, r"CalImLeft\d+\.jpg")

    #remove_unmatched_images(path, r"right\d+\.jpg", r"left\d+\.jpg")
    print(f"Found {len(images_right)} right images and {len(images_left)} left images.")
    for index, image_right in enumerate(images_right, start=0):
        print(f"Processing {image_right} at index {index}")
        image_left = images_left[index]
        new_name_right = f"right{index+1}.jpg"
        new_name_left = f"left{index+1}.jpg"
        os.rename(image_right, os.path.join(path, new_name_right))
        os.rename(image_left, os.path.join(path, new_name_left))
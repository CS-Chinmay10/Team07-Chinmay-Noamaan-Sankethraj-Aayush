import numpy as np
import cv2

original_image_paths = [r"original_images\10_left.jpeg", r"original_images\13_left.jpeg", r"original_images\15_left.jpeg", r"original_images\16_left.jpeg", r"original_images\17_left.jpeg"]
enhanced_bd_image_paths = [r"enhanced_images\brighten-darken\10_left_b-d.png", r"enhanced_images\brighten-darken\13_left_b-d.png", r"enhanced_images\brighten-darken\15_left_b-d.png", r"enhanced_images\brighten-darken\16_left_b-d.png", r"enhanced_images\brighten-darken\17_left_b-d.png"]
enhanced_s_image_paths = [r"enhanced_images\sharpen\10_left_s.png", r"enhanced_images\sharpen\13_left_s.png", r"enhanced_images\sharpen\15_left_s.png", r"enhanced_images\sharpen\16_left_s.png", r"enhanced_images\sharpen\17_left_s.png"]
psnr_bd_values = []
psnr_s_values = []
ssim_bd_values = []
ssim_s_values = []

def calculate_psnr(original, processed):
    # Ensure the images are numpy arrays
    original = np.array(original, dtype=np.float32)
    processed = np.array(processed, dtype=np.float32)
    
    # Compute Mean Squared Error (MSE)
    mse = np.mean((original - processed) ** 2)
    
    # Handle the case where MSE is zero (identical images)
    if mse == 0:
        return float('inf')  # Infinite PSNR for identical images
    
    # Define maximum pixel value
    max_pixel = 255.0  # Change if using a different pixel depth
    
    # Compute PSNR
    psnr = 10 * np.log10((max_pixel ** 2) / mse)
    return psnr

# Example usage
# original_image and processed_image should be NumPy arrays
print("Computing the PSNR metrics:")
for img_path_idx in range(len(original_image_paths)):
    original_image = cv2.imread(original_image_paths[img_path_idx])
    enhanced_bd_image = cv2.imread(enhanced_bd_image_paths[img_path_idx])
    enhanced_s_image = cv2.imread(enhanced_s_image_paths[img_path_idx])
    enhanced_bd_image = cv2.resize(enhanced_bd_image, (original_image.shape[1], original_image.shape[0]))
    enhanced_s_image = cv2.resize(enhanced_s_image, (original_image.shape[1], original_image.shape[0]))
    psnr_bd_value = calculate_psnr(original_image, enhanced_bd_image)
    psnr_bd_values.append(psnr_bd_value)
    psnr_s_value = calculate_psnr(original_image, enhanced_s_image)
    psnr_s_values.append(psnr_s_value)

print()
from skimage.metrics import structural_similarity as ssim

# Load two images (original and processed) as grayscale
print("Computing the SSIM metrics:")
for img_path_idx in range(len(original_image_paths)):
    original_image = cv2.imread(original_image_paths[img_path_idx], cv2.IMREAD_GRAYSCALE)
    processed_bd_image = cv2.imread(enhanced_bd_image_paths[img_path_idx], cv2.IMREAD_GRAYSCALE)
    processed_s_image = cv2.imread(enhanced_s_image_paths[img_path_idx], cv2.IMREAD_GRAYSCALE)
    processed_bd_image = cv2.resize(processed_bd_image, (original_image.shape[1], original_image.shape[0]))
    processed_s_image = cv2.resize(processed_s_image, (original_image.shape[1], original_image.shape[0]))
    ssim_bd_value, ssim_bd_map = ssim(original_image, processed_bd_image, full=True)
    ssim_bd_values.append(ssim_bd_value)
    ssim_s_value, ssim_bd_map = ssim(original_image, processed_s_image, full=True)
    ssim_s_values.append(ssim_s_value)



print("Average PSNR for brighten-darken image enhancement method:", (sum(psnr_bd_values) / len(original_image_paths)))
print()
print("Average SSIM for brighten-darken image enhancement method:", (sum(ssim_bd_values) / len(original_image_paths)))
print()
print("Average PSNR for sharpen image enhancement method:", (sum(psnr_s_values) / len(original_image_paths)))
print()
print("Average SSIM for sharpen image enhancement method:", (sum(ssim_s_values) / len(original_image_paths)))
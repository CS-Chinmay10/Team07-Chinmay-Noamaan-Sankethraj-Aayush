import cv2
import numpy as np
import matplotlib.pyplot as plt

def seg(img, t=8, A=200, L=50):  
    """
    Segment the image using thresholding and morphological operations.

    Parameters:
    img: Input image (BGR format).
    t: Threshold value for segmentation.
    A: Minimum area for segments to be retained.
    L: Minimum length for centrelines to be retained.
    """

    # Extract the Green Channel
    g = img[:, :, 1]

    # Creating mask for restricting FOV
    _, mask = cv2.threshold(g, 10, 255, cv2.THRESH_BINARY)  
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    mask = cv2.erode(mask, kernel, iterations=3)

    # CLAHE and background estimation
    clahe = cv2.createCLAHE(clipLimit=3, tileGridSize=(9, 9))
    g_cl = clahe.apply(g)
    g_cl1 = cv2.medianBlur(g_cl, 5)
    bg = cv2.GaussianBlur(g_cl1, (55, 55), 0)

    # Background subtraction
    norm = np.float32(bg) - np.float32(g_cl1)
    norm = norm * (norm > 0)

    # Thresholding for segmentation
    _, t_bin = cv2.threshold(norm, t, 255, cv2.THRESH_BINARY)

    # Removing noise points by coloring the contours
    t_bin = np.uint8(t_bin)
    th = t_bin.copy()
    contours, _ = cv2.findContours(t_bin, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    for c in contours:
        if cv2.contourArea(c) < A:
            cv2.drawContours(th, [c], 0, 0, -1)

    # Apply the mask
    th = th * (mask / 255)
    th = np.uint8(th)

    # Display the segmented image
    plt.figure(figsize=(8, 8))
    plt.imshow(th, cmap='gray')
    plt.title("Segmented Image")
    plt.axis('off')
    plt.show()

# Load the image
img_path = r"10_left_bd.png"
img = cv2.imread(img_path)

# Run the segmentation function
seg(img)

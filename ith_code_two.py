import streamlit as st
from ietk import methods, util
from PIL import Image
import numpy as np
import cv2
import os

# Import the segmentation function
def seg(img, t=8, A=200, L=50):  
    """
    Segment the image using thresholding and morphological operations.
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
    return th

# Set up Streamlit app
st.title("Fundus Image Enhancement and Segmentation")
st.write("Upload an eye fundus image to see the enhanced and segmented results.")

# Upload image
uploaded_file = st.file_uploader("Choose an eye fundus image", type=["png", "jpg", "jpeg"])

if uploaded_file:
    # Save uploaded image temporarily
    temp_path = "./temp_uploaded_image.png"
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Display uploaded image
    st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)

    # Select enhancement and segmentation options
    enhance_brighten_darken = st.checkbox("Apply Brighten/Darken", value=True)
    enhance_sharpen = st.checkbox("Apply Sharpen", value=True)
    apply_segmentation = st.checkbox("Apply Segmentation")

    # Add a button to process the image
    if st.button("Process Image"):
        try:
            # Show the loading spinner while processing
            with st.spinner('Processing image...'):
                # Load the image using PIL and convert to numpy array
                img = np.array(Image.open(temp_path))

                # If the image has multiple channels (RGB), normalize each channel to [0, 1]
                if img.ndim == 3:  # RGB image
                    img = img.astype(np.float32) / 255.0  # Normalize to [0, 1]
                else:  # Grayscale image
                    img = img.astype(np.float32) / 255.0  # Normalize to [0, 1]

                # Clamp the values to ensure they are within the [0.0, 1.0] range
                img = np.clip(img, 0.0, 1.0)

                # Crop image
                I, fg = util.center_crop_and_get_foreground_mask(img)

                # Apply enhancement methods based on user selection
                enhanced_images = []
                titles = []

                if enhance_brighten_darken:
                    enhanced_img = methods.brighten_darken(I, 'A+B+X', focus_region=fg)
                    enhanced_img = np.clip(enhanced_img, 0.0, 1.0)
                    enhanced_images.append(enhanced_img)
                    titles.append("Enhanced (Brightened and Darkened)")

                if enhance_sharpen:
                    enhanced_img2 = methods.sharpen(I, bg=~fg)
                    enhanced_img2 = np.clip(enhanced_img2, 0.0, 1.0)
                    enhanced_images.append(enhanced_img2)
                    titles.append("Enhanced (Sharpened)")

                # Apply segmentation if selected
                if apply_segmentation:
                    segmented_img = seg((img * 255).astype(np.uint8))
                    enhanced_images.append(segmented_img)
                    titles.append("Segmented Image")

                # Display results if any method was selected
                if enhanced_images:
                    st.subheader("Results")
                    for img, title in zip(enhanced_images, titles):
                        st.image(img, caption=title, use_container_width=True, clamp=True)

                        # Sanitize file name to remove invalid characters
                        sanitized_title = title.replace(" ", "_").replace("/", "_").replace(":", "_")

                        # Save image as temporary file for download
                        save_path = f"./{sanitized_title}.png"
                        Image.fromarray((img * 255).astype(np.uint8) if img.ndim == 3 else img).save(save_path)

                        # Provide download button for each image
                        with open(save_path, "rb") as file:
                            st.download_button(
                                label=f"Save {title}",
                                data=file,
                                file_name=sanitized_title + ".png",
                                mime="image/png"
                            )

                        # Remove the temporary save file
                        os.remove(save_path)

                # Remove temp file
                os.remove(temp_path)

        except Exception as e:
            st.error(f"Error processing image: {e}")

# Team07-Chinmay-Noamaan-Sankethraj-Aayush
The code and files present in this repository represent the work we have completed as part of the course project for IT820: Information Technology for Healthcare.

This repository contains tools for processing and analyzing retinal fundus images. It includes modules for image quality assessment, vessel segmentation, and a Streamlit-based web interface for image enhancement and segmentation.
## Features
- __Image quality metrics:__ Compute Peak Signal to Noise Ratio (PSNR) and Structural Similarity Index Measure (SSIM) to evaluate the quality of enhanced fundus images.
- __Blood Vessel Segmentation:__ Perform vessel segmentation on fundus images.
- __Web Interface:__ Enhance retinal fundus images and segment blood vessels via a user-friendly interface.

## Repository Contents
### Files
`qc_metrics.py` - Compute the PSNR and SSIM between two images.
- __Input:__ Two images - orignal and processed.
- __Output:__ PSNR and SSIM score.

`vessel_segment.py` - Performs blood vessel segmentation on a given fundus image.
- __Input:__ A retinal fundus image.
- __Output:__ An image showing segmented blood vessels.

`ith_code_two.py` - Contains the code for the Streamlit-based web interface.
 - __Features:__
   - Upload and process low-quality retinal fundus images.
   - Apply enhancement techniques (Brighten/Darken and Sharpen).
   - Perform vessel segmentation.
 - __Output:__  Enhanced images and segmented blood vessels, which can be saved to the system.

`requirements.txt` - Lists the required Python packages and their versions. Use this file to install dependencies using `pip install -r requirements.txt`

### Folder
`images/` - Contains sample fundus images used in this project to demonstrate image enhancement and segmentation techniques.

## Installation
1. Clone the repository:
   `git clone https://github.com/CS-Chinmay10/Team07-Chinmay-Noamaan-Sankethraj-Aayush.git
cd Team07-Chinmay-Noamaan-Sankethraj-Aayush`

2. Install required dependencies:
   `pip install -r requirements.txt`

3. Run the Streamlit web interface:
   `streamlit run ith_code_two.py`

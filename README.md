# Bulk Image Grid Slicer

This script processes all images in a selected folder, detects grid cells in each image, and saves the largest cell crops (with red mask overlay) into separate subfolders for each image.

## Features
- Command-line folder selection
- Processes all images in the folder (`.png`, `.jpg`, `.jpeg`, `.bmp`, `.tif`, `.tiff`)
- For each image:
  - Detects grid lines
  - Applies a red transparent mask to cell areas
  - Crops and saves the 16 largest cells (with both dimensions > 200px)
  - Saves results in a subfolder named after the image
  - Logs all cell sizes in `cell_sizes.log`

## Requirements
- Python 3.x
- OpenCV (`opencv-python`)
- NumPy

Install dependencies:
```
pip install opencv-python numpy
```

## Usage
1. Place your images in a folder.
2. Run the script:
   ```
   python slice.py
   ```
3. Enter the path to your image folder when prompted.
4. Processed results will appear in subfolders (one per image) in the current directory.

## Output
- For each image, a folder is created with:
  - `masked_grid_red.png`: The image with a red mask overlay
  - `cell_1.png` ... `cell_16.png`: The 16 largest cropped cell images
  - `cell_sizes.log`: Log of all detected cell sizes

---
**Author:** Your Name
**Date:** 2025-12-31

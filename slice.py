
import cv2
import numpy as np
import os
import shutil

def process_image(image_path):
    def group_lines(lines, min_gap=10):
        if len(lines) == 0:
            return []
        groups = []
        current_group = [lines[0]]
        for l in lines[1:]:
            if l - current_group[-1] > min_gap:
                groups.append(int(np.mean(current_group)))
                current_group = [l]
            else:
                current_group.append(l)
        groups.append(int(np.mean(current_group)))
        return groups

    img = cv2.imread(image_path)
    if img is None:
        print(f"❌ Could not read image: {image_path}")
        return
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Detect white grid lines
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 250, 255, cv2.THRESH_BINARY)
    # Invert threshold to get mask for cell (non-grid) areas
    mask = cv2.bitwise_not(thresh)  # cell mask: cells = 255, grid = 0
    # Create a strong red overlay
    red = np.zeros_like(img, dtype=np.uint8)
    red[:, :] = (0, 0, 255)  # Red in BGR
    # Apply red mask to cell areas
    alpha = 0.7  # Transparency
    mask_bool = mask > 0
    blended = img.copy()
    blended[mask_bool] = (
        img[mask_bool] * (1 - alpha) + red[mask_bool] * alpha
    ).astype(np.uint8)
    # Save the red-masked image
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    image_dir = os.path.dirname(image_path)
    output_dir = os.path.join(image_dir, base_name)
    shutil.rmtree(output_dir, ignore_errors=True)
    os.makedirs(output_dir, exist_ok=True)
    masked_grid_path = os.path.join(output_dir, "masked_grid_red.png")
    cv2.imwrite(masked_grid_path, blended)
    # Find grid lines
    vertical_sum = np.sum(thresh, axis=0)
    horizontal_sum = np.sum(thresh, axis=1)
    v_lines = np.where(vertical_sum > 0.9 * np.max(vertical_sum))[0]
    h_lines = np.where(horizontal_sum > 0.9 * np.max(horizontal_sum))[0]
    # Get grid cell boundaries
    v_positions = [0] + group_lines(v_lines) + [img.shape[1]]
    h_positions = [0] + group_lines(h_lines) + [img.shape[0]]
    # Collect all cell info for logging and selection
    cell_info = []
    for i in range(len(h_positions) - 1):
        for j in range(len(v_positions) - 1):
            y1, y2 = h_positions[i], h_positions[i+1]
            x1, x2 = v_positions[j], v_positions[j+1]
            cell = blended[y1:y2, x1:x2]
            size = (y2-y1, x2-x1)
            area = size[0] * size[1]
            cell_info.append({
                'cell': cell,
                'size': size,
                'area': area,
                'coords': (y1, y2, x1, x2)
            })
    # Filter for cells with both dimensions > 200, then sort and select
    large_cells = [c for c in cell_info if c['size'][0] > 200 and c['size'][1] > 200]
    largest_16_by_area = sorted(large_cells, key=lambda c: c['area'], reverse=True)[:16]
    largest_16 = sorted(largest_16_by_area, key=lambda c: (c['coords'][0], c['coords'][2]))
    # Log all cell image dimensions
    with open(os.path.join(output_dir, "cell_sizes.log"), "w") as f:
        for idx, c in enumerate(cell_info, 1):
            log_line = f"cell_{idx}.png: {c['size'][0]}x{c['size'][1]}"
            print(log_line)
            f.write(log_line + "\n")
    # Save the 16 largest, cropping to the red-masked area only (remove white margin)
    for idx, c in enumerate(largest_16, 1):
        y1, y2, x1, x2 = c['coords']
        cell_mask = mask[y1:y2, x1:x2]
        coords = cv2.findNonZero(cell_mask)
        orig_cell_img = img[y1:y2, x1:x2]
        if coords is not None:
            x_min, y_min = coords.min(axis=0)[0]
            x_max, y_max = coords.max(axis=0)[0]
            margin = 5
            x_min = min(x_max, max(0, x_min + margin))
            y_min = min(y_max, max(0, y_min + margin))
            x_max = max(x_min, min(orig_cell_img.shape[1] - 1, x_max - margin))
            y_max = max(y_min, min(orig_cell_img.shape[0] - 1, y_max - margin))
            cropped = orig_cell_img[y_min:y_max+1, x_min:x_max+1]
        else:
            cropped = orig_cell_img
        cv2.imwrite(os.path.join(output_dir, f"cell_{idx}.png"), cropped)
    # Remove the red-masked image after saving cell crops and log
    if os.path.exists(masked_grid_path):
        try:
            os.remove(masked_grid_path)
        except Exception as e:
            print(f"Warning: Could not remove {masked_grid_path}: {e}")
    print(f"✅ Saved {len(largest_16)} largest cell images (>200x200) in folder '{output_dir}' and logged all sizes to cell_sizes.log")


def main():
    import glob
    folder = input("Enter the path to the folder containing images: ").strip('"')
    if not os.path.isdir(folder):
        print("❌ Not a valid folder.")
        return
    # Accept common image formats
    image_files = []
    for ext in ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tif", "*.tiff"):
        image_files.extend(glob.glob(os.path.join(folder, ext)))
    if not image_files:
        print("❌ No images found in the folder.")
        return
    print(f"Found {len(image_files)} images. Processing...")
    for img_path in image_files:
        print(f"\nProcessing: {img_path}")
        process_image(img_path)
    print("\n✅ All images processed.")


if __name__ == "__main__":
    main()

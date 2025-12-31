import cv2
import numpy as np

# Load image
image_path = "oak.png"
img = cv2.imread(image_path)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Detect white grid lines
blur = cv2.GaussianBlur(gray, (5, 5), 0)
_, thresh = cv2.threshold(blur, 250, 255, cv2.THRESH_BINARY)


# Invert threshold to get mask for cell (non-grid) areas
mask = cv2.bitwise_not(thresh)  # cell mask: cells = 255, grid = 0


# Create a strong red overlay
red = np.zeros_like(img, dtype=np.uint8)
red[:, :] = (0, 0, 255)  # Red in BGR


# Only apply red where mask is nonzero (cell areas)
alpha = 0.4  # Transparency factor
mask_bool = mask > 0



# Increase alpha for stronger red mask
alpha = 0.7  # More visible transparency
blended = img.copy()
blended[mask_bool] = (
	img[mask_bool] * (1 - alpha) + red[mask_bool] * alpha
).astype(np.uint8)


# Save the red-masked image
cv2.imwrite("masked_grid_red.png", blended)
print("✅ Red transparent mask applied and saved as 'masked_grid_red.png'")

# --- Cut and save each cell image separately ---
# Use the original grid mask (thresh) to find grid lines
vertical_sum = np.sum(thresh, axis=0)
horizontal_sum = np.sum(thresh, axis=1)

# Detect positions where gaps are large (likely grid lines)
v_lines = np.where(vertical_sum > 0.9 * np.max(vertical_sum))[0]
h_lines = np.where(horizontal_sum > 0.9 * np.max(horizontal_sum))[0]

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

v_positions = group_lines(v_lines)
h_positions = group_lines(h_lines)

# Add image edges as boundaries
v_positions = [0] + v_positions + [img.shape[1]]
h_positions = [0] + h_positions + [img.shape[0]]

import os
base_name = os.path.splitext(os.path.basename(image_path))[0]
output_dir = base_name
os.makedirs(output_dir, exist_ok=True)

count = 1
cell_sizes = []
for i in range(len(h_positions) - 1):
	for j in range(len(v_positions) - 1):
		y1, y2 = h_positions[i], h_positions[i+1]
		x1, x2 = v_positions[j], v_positions[j+1]
		cell = blended[y1:y2, x1:x2]
		if cell.size > 0:
			cv2.imwrite(f"output_cells/cell_{count}.png", cell)
			cell_sizes.append((y2-y1, x2-x1))
			count += 1

# Save the 16 largest cell images (by area) with both dimensions > 200
import shutil
import shutil
shutil.rmtree(output_dir, ignore_errors=True)
os.makedirs(output_dir, exist_ok=True)

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

# Filter for cells with both dimensions > 200
large_cells = [c for c in cell_info if c['size'][0] > 200 and c['size'][1] > 200]
# Sort by area descending and take the top 16
largest_16 = sorted(large_cells, key=lambda c: c['area'], reverse=True)[:16]

# Log all cell image dimensions
with open(f"{output_dir}/cell_sizes.log", "w") as f:
	for idx, c in enumerate(cell_info, 1):
		log_line = f"cell_{idx}.png: {c['size'][0]}x{c['size'][1]}"
		print(log_line)
		f.write(log_line + "\n")


# Save the 16 largest, cropping to the red-masked area only (remove white margin)
for idx, c in enumerate(largest_16, 1):
	y1, y2, x1, x2 = c['coords']
	# Get the corresponding mask region for this cell
	cell_mask = mask[y1:y2, x1:x2]
	# Find bounding box of nonzero (cell) area in mask
	coords = cv2.findNonZero(cell_mask)
	orig_cell_img = img[y1:y2, x1:x2]
	if coords is not None:
		x_min, y_min = coords.min(axis=0)[0]
		x_max, y_max = coords.max(axis=0)[0]
		# Cut 5 extra pixels from each side if possible
		margin = 5
		x_min = min(x_max, max(0, x_min + margin))
		y_min = min(y_max, max(0, y_min + margin))
		x_max = max(x_min, min(orig_cell_img.shape[1] - 1, x_max - margin))
		y_max = max(y_min, min(orig_cell_img.shape[0] - 1, y_max - margin))
		cropped = orig_cell_img[y_min:y_max+1, x_min:x_max+1]
	else:
		cropped = orig_cell_img
	cv2.imwrite(f"{output_dir}/cell_{idx}.png", cropped)

print(f"✅ Saved {len(largest_16)} largest cell images (>200x200) in folder '{output_dir}' and logged all sizes to cell_sizes.log")

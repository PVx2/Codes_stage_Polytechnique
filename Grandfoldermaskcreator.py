# SCRIPT: Grandfoldermaskcreator.py

"""
OBJECTIF: Générer automatiquement des masques à partir de fichiers ROI ImageJ
          Les masques sont créés avec les mêmes dimensions que les images correspondantes
          et sauvegardés au format TIFF avec un fond blanc, les ROI en noir et leurs contours en blanc.

FONCTIONNEMENT:
1. Parcourt récursivement l'arborescence à la recherche de fichiers ROI (.zip)
2. Pour chaque ROI, trouve l'image correspondante par matching de nom
3. Crée un masque aux dimensions exactes de l'image
4. Dessine les régions d'intérêt avec leurs contours
5. Sauvegarde le masque dans le même dossier que l'image

"""

import os
import numpy as np
import matplotlib.pyplot as plt
from read_roi import read_roi_zip
from skimage.draw import polygon
from skimage.segmentation import find_boundaries
from skimage.measure import label
import tifffile
import re
from glob import glob

# ─── CONFIG ───────────────────────────────────────────
root_dir = r'C:\Users\33672\Documents\Stage X\Hom_med\E11'
image_extension = '.tif'
roi_zip_prefixes = ['CORR_RoiSet', 'RoiSet']
output_mask_prefix = 'masque'

def extract_common_name_from_image(full_name):
    """
    Extracts the common part of the image name - same logic as main script
    """
    # Remove the prefix and extension
    common_name = full_name.replace("CORR_", "").replace("LocZP_", "").replace(".tif", "")
    common_name = common_name.replace("EMX1TCyt5_", "")
    # Remove slice information if present
    if "_s" in common_name:
        common_name = common_name.split("_s")[0]
    # Remove 40X information but keep the number
    common_name = re.sub(r'_40X-?(\d+)', r'_\1', common_name)
    common_name = re.sub(r'_MIP|_Mid-Mid|_Mid|hR_Mid|hL_Mid|_1q|_Mid-', '', common_name)
    common_name = re.sub(r'_Crop_|_CROP_', '_crop_', common_name)

    return common_name

def extract_common_name_from_roi(roi_filename):
    """
    Handles the ROI filename structure
    """
    common_name = roi_filename.replace(".zip", "")
    
    # Handle the different ROI prefixes more carefully
    if common_name.startswith("CORR_RoiSet_"):
        common_name = common_name.replace("CORR_RoiSet_", "")
    elif common_name.startswith("RoiSet_CORR_"):
        common_name = common_name.replace("RoiSet_CORR_", "")
    elif common_name.startswith("RoiSet_"):
        common_name = common_name.replace("RoiSet_", "")
    elif common_name.startswith("CORR_"):
        common_name = common_name.replace("CORR_", "")
    
    # Now apply the same cleaning as for images
    common_name = common_name.replace("LocZP_", "")
    common_name = common_name.replace("EMX1TCyt5_", "")
    
    if "_s" in common_name:
        common_name = common_name.split("_s")[0]
    
    common_name = re.sub(r'_40X-?(\d+)', r'_\1', common_name)
    common_name = re.sub(r'_MIP|_Mid-Mid|_Mid|hR_Mid|hL_Mid|_1q|_Mid-', '', common_name)
    common_name = re.sub(r'_Crop_|_CROP_', '_crop_', common_name)

    return common_name

def find_matching_image_for_roi(roi_path, directory_path):
    """
    Find the matching image file for a given ROI file with improved matching logic
    """
    roi_filename = os.path.basename(roi_path)
    roi_common_name = extract_common_name_from_roi(roi_filename)

    # Get all image files in the directory
    image_files = glob(os.path.join(directory_path, "*.tif"))
    image_files = [
        f for f in image_files
        if (("LocZP_" in os.path.basename(f)) or ("CORR_LocZP_" in os.path.basename(f)))
        and not os.path.basename(f).startswith("masque_")
        and not "Segmented" in os.path.basename(f)
        and not "RefSurface" in os.path.basename(f)
        and not "C4-" in os.path.basename(f)
    ]

    print(f"🔍 Looking for match for ROI: {roi_common_name}")
    
    # First pass: Look for exact matches
    for image_file in image_files:
        image_filename = os.path.basename(image_file)
        image_common_name = extract_common_name_from_image(image_filename)
        
        print(f"   Comparing with image: {image_common_name}")

        # Direct exact match
        if roi_common_name == image_common_name:
            print(f"   ✅ Exact match found!")
            return image_file

    # Second pass: Match base name without position suffixes (ANT, MID, POST)
    roi_base = re.sub(r'_(ANT|MID|POST)$', '', roi_common_name)
    
    for image_file in image_files:
        image_filename = os.path.basename(image_file)
        image_common_name = extract_common_name_from_image(image_filename)
        image_base = re.sub(r'_(ANT|MID|POST)$', '', image_common_name)
        
        if roi_base == image_base:
            print(f"   ✅ Base match found: {roi_base} == {image_base}")
            return image_file

    # Third pass: Try partial matching for complex cases
    for image_file in image_files:
        image_filename = os.path.basename(image_file)
        image_common_name = extract_common_name_from_image(image_filename)
        
        # Check if the core parts match (ignoring position markers)
        roi_parts = roi_common_name.split('_')
        image_parts = image_common_name.split('_')
        
        # Find longest common subsequence of parts
        common_parts = []
        for part in roi_parts:
            if part in image_parts and part not in ['ANT', 'MID', 'POST']:
                common_parts.append(part)
        
        # If we have significant overlap, consider it a match
        if len(common_parts) >= 4:  # Adjust threshold as needed
            print(f"   ✅ Partial match found based on common parts: {common_parts}")
            return image_file

    print(f"   ❌ No match found")
    return None

def process_image_and_rois(image_path, zip_path, output_path):
    """
    Create mask with same dimensions as the corresponding image
    """
    # Load image to get exact dimensions
    img = tifffile.imread(image_path)
    if len(img.shape) == 3:  # Multi-channel image
        H, W = img.shape[1], img.shape[2]  # Assuming shape is (C, H, W)
    else:  # Grayscale
        H, W = img.shape

    print(f"📐 Image dimensions: {W}x{H}")
    # Read ROIs
    all_rois = read_roi_zip(zip_path)
    total_rois = len(all_rois)
    if total_rois == 0:
        print(f"⚠️ No ROIs found in {zip_path}")
        return
    # Create label mask with exact image dimensions
    label_mask = np.zeros((H, W), dtype=np.uint16)
    displayed_rois = 0
    for idx, roi in enumerate(all_rois.values(), start=1):
        xs = np.array(roi["x"], dtype=int)
        ys = np.array(roi["y"], dtype=int)

        if len(xs) > 0 and len(ys) > 0:
            # Ensure coordinates are within image bounds
            xs = np.clip(xs, 0, W - 1)
            ys = np.clip(ys, 0, H - 1)

            # Check if ROI is actually within image bounds
            if np.any(xs >= 0) and np.any(xs < W) and np.any(ys >= 0) and np.any(ys < H):
                rr, cc = polygon(ys, xs, shape=label_mask.shape)
                label_mask[rr, cc] = idx
                displayed_rois += 1
            else:
                print(f"⚠️ ROI {idx} is outside image bounds, skipping")

    # Create visual mask (white background with black ROIs and white boundaries)
    visual_mask = np.ones_like(label_mask, dtype=np.uint16) * 65535  # White background
    visual_mask[label_mask > 0] = 0  # Black ROIs
    boundaries = find_boundaries(label_mask, mode='thick')
    visual_mask[boundaries] = 65535  # White boundaries

    # Save TIFF
    tifffile.imwrite(output_path, visual_mask)
    # Summary output
    print(f"✅ Saved: {os.path.basename(output_path)}")
    print(f"   └─ Total ROIs read: {total_rois}")
    print(f"   └─ ROIs successfully displayed: {displayed_rois}")
    print(f"   └─ Mask dimensions: {W}x{H} (matches image)")
    print()

# ─── MAIN AUTOMATION LOOP ─────────────────────────────
print("🔍 Starting mask generation with exact image dimensions...")
print(f"📁 Searching in: {root_dir}")
print()

processed_count = 0
for dirpath, dirnames, filenames in os.walk(root_dir):
    zip_files = []
    for prefix in roi_zip_prefixes:
        zip_files.extend([f for f in filenames if f.lower().endswith('.zip') and f.startswith(prefix)])
    
    if zip_files:
        print(f"📂 Processing directory: {os.path.relpath(dirpath, root_dir)}")
        
        # Debug: show available files
        available_images = [f for f in filenames if f.endswith('.tif') and 'LocZP_' in f and not f.startswith('masque_') and 'C4-' not in f]
        print(f"Available images in directory:")
        for img in available_images:
            print(f"   • {img} -> common: {extract_common_name_from_image(img)}")
        print()
    
    for zip_file in zip_files:
        zip_path = os.path.join(dirpath, zip_file)

        # Find matching image using the improved logic
        matching_image_path = find_matching_image_for_roi(zip_path, dirpath)

        if matching_image_path:
            matching_image_name = os.path.basename(matching_image_path)
            zip_base = os.path.splitext(zip_file)[0]
            output_name = f"{output_mask_prefix}_{zip_base}.tif"
            output_path = os.path.join(dirpath, output_name)

            print(f"🔄 Processing: {zip_file} + {matching_image_name}")
            print(f"   └─ ROI common name: {extract_common_name_from_roi(zip_file)}")
            print(f"   └─ Image common name: {extract_common_name_from_image(matching_image_name)}")

            process_image_and_rois(matching_image_path, zip_path, output_path)
            processed_count += 1
        else:
            print(f"⚠️ No matching image found for {zip_file} in {dirpath}")
            print(f"   └─ ROI common name: {extract_common_name_from_roi(zip_file)}")
            print()
            
print(f"🎉 Processing complete! Generated {processed_count} masks with exact image dimensions.")